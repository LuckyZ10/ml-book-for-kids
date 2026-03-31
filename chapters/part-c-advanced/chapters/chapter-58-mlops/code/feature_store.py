"""
特征存储完整实现
模拟Feast的核心功能：特征定义、物化、服务
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable, Any
from enum import Enum
import pickle
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 数据模型
# =============================================================================

class FeatureType(Enum):
    """特征类型"""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    TIMESTAMP = "timestamp"


@dataclass
class Entity:
    """
    实体定义
    
    实体是特征的主体，如用户、商品、订单等
    """
    name: str  # 如 "user", "product"
    join_key: str  # 如 "user_id", "product_id"
    description: str = ""
    
    def __hash__(self):
        return hash(self.name)


@dataclass  
class Feature:
    """
    特征定义
    
    描述一个特征的元数据
    """
    name: str
    dtype: FeatureType
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class FeatureView:
    """
    特征视图
    
    一组相关的特征，通常来自同一个数据源
    类似于数据库的"视图"
    """
    name: str
    entities: List[Entity]
    features: List[Feature]
    source: str  # 数据源标识
    ttl: Optional[timedelta] = None  # 特征有效期（Time-To-Live）
    online: bool = True  # 是否启用在线服务
    description: str = ""
    
    def get_feature_names(self) -> List[str]:
        return [f.name for f in self.features]
    
    def get_entity_names(self) -> List[str]:
        return [e.name for e in self.entities]


# =============================================================================
# 离线存储实现
# =============================================================================

class OfflineStore:
    """
    离线特征存储
    
    存储历史特征数据，支持时间旅行查询
    """
    
    def __init__(self, storage_path: str = "./feast_offline"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._data: Dict[str, pd.DataFrame] = {}
    
    def write(self, feature_view: FeatureView, df: pd.DataFrame):
        """
        写入特征数据
        
        Args:
            feature_view: 特征视图
            df: 包含特征的数据框，必须包含：
                - 实体列（如 user_id）
                - 时间戳列（event_timestamp）
                - 特征列
        """
        # 确保必要列存在
        required_cols = [e.join_key for e in feature_view.entities] + ['event_timestamp']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 确保时间戳列是datetime类型
        df = df.copy()
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        
        # 存储数据
        self._data[feature_view.name] = df
        
        # 持久化到Parquet（实际生产环境）
        file_path = os.path.join(self.storage_path, f"{feature_view.name}.parquet")
        df.to_parquet(file_path, index=False)
        
        print(f"[OfflineStore] Written {len(df)} rows to {feature_view.name}")
    
    def get_historical_features(
        self,
        feature_views: List[FeatureView],
        entity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        获取历史特征（Point-in-Time Correct Join）
        
        这是特征存储的核心功能！确保不会出现数据泄漏。
        
        数学原理：
        对于每个(entity, timestamp)对，找到特征视图中：
        - entity匹配
        - event_timestamp ≤ timestamp（训练时间）
        - 最新的记录
        
        Args:
            feature_views: 需要的特征视图列表
            entity_df: 实体数据框，包含 entity_id 和 timestamp 列
        
        Returns:
            合并后的特征数据框
        """
        result = entity_df.copy()
        result['request_timestamp'] = pd.to_datetime(result['timestamp'])
        
        for fv in feature_views:
            if fv.name not in self._data:
                raise ValueError(f"Feature view {fv.name} not found")
            
            feature_df = self._data[fv.name].copy()
            
            # Point-in-Time Correct Join
            # 对于每个entity和request_timestamp，找到最新的有效特征
            merged_rows = []
            
            for _, row in result.iterrows():
                entity_filter = True
                for entity in fv.entities:
                    entity_filter = entity_filter & (feature_df[entity.join_key] == row[entity.join_key])
                
                # 筛选：特征时间戳 ≤ 请求时间戳（防止数据泄漏！）
                time_filter = feature_df['event_timestamp'] <= row['request_timestamp']
                
                valid_features = feature_df[entity_filter & time_filter]
                
                if len(valid_features) > 0:
                    # 取最新的记录
                    latest = valid_features.loc[valid_features['event_timestamp'].idxmax()]
                    merged_rows.append(latest[fv.get_feature_names()].to_dict())
                else:
                    # 无有效特征，填充NaN
                    merged_rows.append({f: np.nan for f in fv.get_feature_names()})
            
            # 合并到结果
            feature_cols = pd.DataFrame(merged_rows)
            result = pd.concat([result.reset_index(drop=True), feature_cols], axis=1)
        
        return result


# =============================================================================
# 在线存储实现
# =============================================================================

class OnlineStore:
    """
    在线特征存储
    
    低延迟的特征查询，使用内存数据库（如Redis）
    这里使用SQLite模拟
    """
    
    def __init__(self, db_path: str = "./feast_online.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()
    
    def _init_tables(self):
        """初始化表结构"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                feature_view TEXT,
                entity_key TEXT,
                feature_name TEXT,
                feature_value TEXT,
                event_timestamp TEXT,
                PRIMARY KEY (feature_view, entity_key, feature_name)
            )
        """)
        self.conn.commit()
    
    def materialize(self, feature_view: FeatureView, df: pd.DataFrame):
        """
        物化特征到在线存储
        
        将离线计算好的特征同步到在线存储，供实时推理使用
        """
        cursor = self.conn.cursor()
        
        # 构建实体键（多个实体用#连接）
        df = df.copy()
        df['entity_key'] = df[[e.join_key for e in feature_view.entities]].apply(
            lambda x: '#'.join(map(str, x)), axis=1
        )
        
        # 只保存最新的特征值
        df = df.sort_values('event_timestamp').groupby('entity_key').last().reset_index()
        
        for _, row in df.iterrows():
            for feature in feature_view.features:
                value = row.get(feature.name)
                if pd.notna(value):
                    cursor.execute("""
                        INSERT OR REPLACE INTO feature_values
                        (feature_view, entity_key, feature_name, feature_value, event_timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        feature_view.name,
                        row['entity_key'],
                        feature.name,
                        str(value),
                        row['event_timestamp']
                    ))
        
        self.conn.commit()
        print(f"[OnlineStore] Materialized {len(df)} entities for {feature_view.name}")
    
    def get_online_features(
        self,
        feature_views: List[FeatureView],
        entity_rows: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        获取在线特征
        
        用于实时推理，必须保证低延迟（<10ms）
        
        Args:
            feature_views: 特征视图列表
            entity_rows: 实体列表，如 [{'user_id': 'user_123'}]
        
        Returns:
            特征值数据框
        """
        cursor = self.conn.cursor()
        results = []
        
        for row in entity_rows:
            row_result = row.copy()
            
            for fv in feature_views:
                # 构建实体键
                entity_key = '#'.join(str(row.get(e.join_key, '')) for e in fv.entities)
                
                # 查询所有特征
                feature_names = fv.get_feature_names()
                placeholders = ','.join(['?'] * len(feature_names))
                
                cursor.execute(f"""
                    SELECT feature_name, feature_value
                    FROM feature_values
                    WHERE feature_view = ? AND entity_key = ? AND feature_name IN ({placeholders})
                """, [fv.name, entity_key] + feature_names)
                
                features = {f: None for f in feature_names}  # 默认值
                for fname, fvalue in cursor.fetchall():
                    # 类型转换
                    try:
                        features[fname] = float(fvalue)
                    except:
                        features[fname] = fvalue
                
                row_result.update(features)
            
            results.append(row_result)
        
        return pd.DataFrame(results)
    
    def close(self):
        self.conn.close()


# =============================================================================
# 特征注册中心
# =============================================================================

class FeatureRegistry:
    """
    特征注册中心
    
    管理特征视图的元数据、版本和血缘
    """
    
    def __init__(self, registry_path: str = "./feast_registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        
        self._feature_views: Dict[str, FeatureView] = {}
        self._entities: Dict[str, Entity] = {}
        self._load_registry()
    
    def _load_registry(self):
        """从磁盘加载注册信息"""
        registry_file = os.path.join(self.registry_path, 'registry.json')
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                data = json.load(f)
            # 这里简化处理，实际应该完整反序列化
            print(f"[Registry] Loaded registry from {registry_file}")
    
    def _save_registry(self):
        """保存注册信息到磁盘"""
        registry_file = os.path.join(self.registry_path, 'registry.json')
        # 简化序列化
        data = {
            'feature_views': list(self._feature_views.keys()),
            'entities': list(self._entities.keys()),
            'updated_at': datetime.now().isoformat()
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def apply_entity(self, entity: Entity):
        """注册实体"""
        self._entities[entity.name] = entity
        self._save_registry()
        print(f"[Registry] Registered entity: {entity.name}")
    
    def apply_feature_view(self, feature_view: FeatureView):
        """注册特征视图"""
        # 验证实体已注册
        for entity in feature_view.entities:
            if entity.name not in self._entities:
                raise ValueError(f"Entity {entity.name} not registered")
        
        self._feature_views[feature_view.name] = feature_view
        self._save_registry()
        print(f"[Registry] Registered feature view: {feature_view.name}")
    
    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        """获取特征视图"""
        return self._feature_views.get(name)
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """获取实体"""
        return self._entities.get(name)
    
    def list_feature_views(self) -> List[str]:
        """列出所有特征视图"""
        return list(self._feature_views.keys())


# =============================================================================
# 特征存储主类
# =============================================================================

class FeatureStore:
    """
    特征存储主类
    
    整合离线存储、在线存储和注册中心的完整特征存储实现
    """
    
    def __init__(self, base_path: str = "./feature_store"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        self.registry = FeatureRegistry(os.path.join(base_path, 'registry'))
        self.offline_store = OfflineStore(os.path.join(base_path, 'offline'))
        self.online_store = OnlineStore(os.path.join(base_path, 'online.db'))
    
    def apply(self, obj: Union[Entity, FeatureView]):
        """
        应用实体或特征视图定义
        
        类似于Terraform的apply操作，声明式配置
        """
        if isinstance(obj, Entity):
            self.registry.apply_entity(obj)
        elif isinstance(obj, FeatureView):
            self.registry.apply_feature_view(obj)
        else:
            raise ValueError(f"Unknown object type: {type(obj)}")
    
    def ingest(self, feature_view_name: str, df: pd.DataFrame):
        """
        摄取特征数据
        
        将特征数据写入离线存储，并可选地物化到在线存储
        """
        fv = self.registry.get_feature_view(feature_view_name)
        if fv is None:
            raise ValueError(f"Feature view {feature_view_name} not found")
        
        # 写入离线存储
        self.offline_store.write(fv, df)
        
        # 如果启用在线服务，物化到在线存储
        if fv.online:
            self.online_store.materialize(fv, df)
    
    def get_historical_features(
        self,
        feature_refs: List[str],
        entity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        获取历史特征
        
        Args:
            feature_refs: 特征引用列表，如 ['user_stats:num_purchases', 'user_stats:avg_order_value']
            entity_df: 实体数据框
        """
        # 解析特征引用，按特征视图分组
        fv_names = set()
        for ref in feature_refs:
            if ':' in ref:
                fv_names.add(ref.split(':')[0])
            else:
                fv_names.add(ref)
        
        feature_views = [self.registry.get_feature_view(n) for n in fv_names]
        feature_views = [fv for fv in feature_views if fv is not None]
        
        return self.offline_store.get_historical_features(feature_views, entity_df)
    
    def get_online_features(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        获取在线特征
        
        用于实时推理场景
        """
        fv_names = set(ref.split(':')[0] if ':' in ref else ref for ref in feature_refs)
        feature_views = [self.registry.get_feature_view(n) for n in fv_names]
        feature_views = [fv for fv in feature_views if fv is not None]
        
        return self.online_store.get_online_features(feature_views, entity_rows)


# =============================================================================
# 演示：完整的特征存储工作流
# =============================================================================

def demo_feature_store():
    """
    演示特征存储的完整工作流
    """
    print("="*80)
    print("特征存储演示")
    print("="*80)
    
    # 1. 初始化特征存储
    store = FeatureStore("./demo_feature_store")
    
    # 2. 定义实体
    user_entity = Entity(
        name="user",
        join_key="user_id",
        description="应用用户"
    )
    
    product_entity = Entity(
        name="product",
        join_key="product_id",
        description="商品"
    )
    
    store.apply(user_entity)
    store.apply(product_entity)
    
    # 3. 定义特征视图
    user_stats_fv = FeatureView(
        name="user_stats",
        entities=[user_entity],
        features=[
            Feature(name="num_purchases", dtype=FeatureType.INT, description="购买次数"),
            Feature(name="avg_order_value", dtype=FeatureType.FLOAT, description="平均订单金额"),
            Feature(name="days_since_last_purchase", dtype=FeatureType.INT, description="距上次购买天数"),
            Feature(name="favorite_category", dtype=FeatureType.STRING, description="偏好类目"),
        ],
        source="user_transactions_table",
        ttl=timedelta(days=7),
        description="用户统计特征"
    )
    
    product_stats_fv = FeatureView(
        name="product_stats",
        entities=[product_entity],
        features=[
            Feature(name="num_views", dtype=FeatureType.INT, description="浏览次数"),
            Feature(name="num_purchases", dtype=FeatureType.INT, description="购买次数"),
            Feature(name="conversion_rate", dtype=FeatureType.FLOAT, description="转化率"),
        ],
        source="product_events_table",
        ttl=timedelta(days=1),
        description="商品统计特征"
    )
    
    store.apply(user_stats_fv)
    store.apply(product_stats_fv)
    
    print("\n" + "-"*40)
    print("注册的实体和特征视图:")
    print(f"  实体: {store.registry.list_feature_views()}")
    
    # 4. 生成示例数据
    print("\n" + "-"*40)
    print("生成示例特征数据...")
    
    np.random.seed(42)
    
    # 用户特征数据
    user_data = []
    for user_id in range(100):
        for days_ago in [0, 7, 14, 30]:  # 多个时间点
            user_data.append({
                'user_id': f'user_{user_id}',
                'event_timestamp': datetime.now() - timedelta(days=days_ago),
                'num_purchases': np.random.poisson(5 + days_ago * 0.1),
                'avg_order_value': np.random.normal(100, 30),
                'days_since_last_purchase': days_ago,
                'favorite_category': np.random.choice(['electronics', 'clothing', 'food'])
            })
    user_df = pd.DataFrame(user_data)
    
    # 商品特征数据
    product_data = []
    for product_id in range(50):
        views = np.random.poisson(1000)
        purchases = np.random.poisson(views * 0.05)
        for days_ago in [0, 1, 2]:  # 商品特征更新更频繁
            product_data.append({
                'product_id': f'product_{product_id}',
                'event_timestamp': datetime.now() - timedelta(days=days_ago),
                'num_views': views + np.random.poisson(10),
                'num_purchases': purchases,
                'conversion_rate': purchases / max(views, 1)
            })
    product_df = pd.DataFrame(product_data)
    
    # 5. 摄取数据
    print("\n摄取用户特征...")
    store.ingest("user_stats", user_df)
    
    print("\n摄取商品特征...")
    store.ingest("product_stats", product_df)
    
    # 6. 获取历史特征（用于训练）
    print("\n" + "-"*40)
    print("获取历史特征（用于训练）...")
    
    # 构造训练样本：假设我们要预测用户是否会购买某商品
    training_entities = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(10)],
        'product_id': [f'product_{i % 50}' for i in range(10)],
        'timestamp': [datetime.now() - timedelta(days=1) for _ in range(10)]
    })
    
    historical_features = store.get_historical_features(
        feature_refs=['user_stats:num_purchases', 'user_stats:avg_order_value', 
                     'product_stats:conversion_rate'],
        entity_df=training_entities
    )
    
    print("\n历史特征数据（前5行）:")
    print(historical_features.head().to_string())
    
    # 7. 获取在线特征（用于推理）
    print("\n" + "-"*40)
    print("获取在线特征（用于推理）...")
    
    online_entities = [
        {'user_id': 'user_0', 'product_id': 'product_1'},
        {'user_id': 'user_5', 'product_id': 'product_10'},
        {'user_id': 'user_10', 'product_id': 'product_20'},
    ]
    
    online_features = store.get_online_features(
        feature_refs=['user_stats:num_purchases', 'user_stats:avg_order_value',
                     'product_stats:conversion_rate'],
        entity_rows=online_entities
    )
    
    print("\n在线特征数据:")
    print(online_features.to_string())
    
    # 8. 展示Point-in-Time Correctness的重要性
    print("\n" + "-"*40)
    print("Point-in-Time Correctness 验证...")
    
    # 假设我们在3天前进行预测
    entity_past = pd.DataFrame({
        'user_id': ['user_0'],
        'timestamp': [datetime.now() - timedelta(days=3)]
    })
    
    features_past = store.get_historical_features(
        feature_refs=['user_stats:num_purchases'],
        entity_df=entity_past
    )
    
    print(f"\n3天前的预测，使用的特征:")
    print(features_past[['user_id', 'timestamp', 'num_purchases']].to_string())
    print("\n注意：这是3天前的用户购买次数，不包含之后的数据！")
    print("这就是Point-in-Time Correctness，防止数据泄漏的关键。")
    
    print(f"\n{'='*80}")
    print("特征存储演示完成！")
    print(f"{'='*80}\n")
    
    return store


if __name__ == "__main__":
    store = demo_feature_store()