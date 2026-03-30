"""
模型注册中心完整实现
包含：版本管理、阶段转换、血缘追踪、审批流程
"""

import os
import json
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union
from enum import Enum
import pickle
import sqlite3

import numpy as np


# =============================================================================
# 数据模型
# =============================================================================

class ModelStage(Enum):
    """模型阶段"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelStatus(Enum):
    """模型状态"""
    PENDING = "pending"  # 等待审批
    APPROVED = "approved"  # 已批准
    REJECTED = "rejected"  # 已拒绝
    DEPRECATED = "deprecated"  # 已弃用


@dataclass
class ModelSignature:
    """
    模型签名
    
    描述模型的输入输出格式，类似于函数签名
    """
    inputs: Dict[str, str]  # 如 {'user_id': 'int64', 'features': 'float32[10]'}
    outputs: Dict[str, str]  # 如 {'probability': 'float32', 'prediction': 'int64'}
    
    def to_dict(self):
        return asdict(self)


@dataclass
class RunInfo:
    """
    运行信息
    
    记录模型训练的血缘信息
    """
    run_id: str
    experiment_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    
    def to_dict(self):
        return {
            'run_id': self.run_id,
            'experiment_id': self.experiment_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'metrics': self.metrics,
            'params': {k: str(v) for k, v in self.params.items()},
            'artifacts': self.artifacts,
            'git_commit': self.git_commit,
            'git_branch': self.git_branch
        }


@dataclass
class ModelVersion:
    """
    模型版本
    
    注册中心的核心数据单元
    """
    name: str  # 模型名称，如 "user_churn_predictor"
    version: str  # 语义化版本，如 "v1.2.3"
    stage: ModelStage
    status: ModelStatus
    
    # 模型元数据
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    # 血缘信息
    run_info: Optional[RunInfo] = None
    
    # 模型签名
    signature: Optional[ModelSignature] = None
    
    # 存储路径
    artifact_uri: Optional[str] = None
    
    # 注册信息
    registered_at: datetime = field(default_factory=datetime.now)
    registered_by: str = ""
    
    # 审批信息
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    approval_notes: Optional[str] = None
    
    # 性能指标（缓存）
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'stage': self.stage.value,
            'status': self.status.value,
            'description': self.description,
            'tags': self.tags,
            'run_info': self.run_info.to_dict() if self.run_info else None,
            'signature': self.signature.to_dict() if self.signature else None,
            'artifact_uri': self.artifact_uri,
            'registered_at': self.registered_at.isoformat(),
            'registered_by': self.registered_by,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'approved_by': self.approved_by,
            'approval_notes': self.approval_notes,
            'metrics': self.metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """从字典创建"""
        return cls(
            name=data['name'],
            version=data['version'],
            stage=ModelStage(data['stage']),
            status=ModelStatus(data['status']),
            description=data.get('description', ''),
            tags=data.get('tags', {}),
            run_info=RunInfo(**data['run_info']) if data.get('run_info') else None,
            signature=ModelSignature(**data['signature']) if data.get('signature') else None,
            artifact_uri=data.get('artifact_uri'),
            registered_at=datetime.fromisoformat(data['registered_at']),
            registered_by=data.get('registered_by', ''),
            approved_at=datetime.fromisoformat(data['approved_at']) if data.get('approved_at') else None,
            approved_by=data.get('approved_by'),
            approval_notes=data.get('approval_notes'),
            metrics=data.get('metrics', {})
        )


# =============================================================================
# 模型注册中心
# =============================================================================

class ModelRegistry:
    """
    模型注册中心
    
    核心功能：
    1. 模型版本管理
    2. 阶段转换（Development → Staging → Production → Archived）
    3. 血缘追踪
    4. 审批工作流
    """
    
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        self.models_path = os.path.join(registry_path, 'models')
        self.db_path = os.path.join(registry_path, 'registry.db')
        
        os.makedirs(self.models_path, exist_ok=True)
        
        # 初始化数据库
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        
        # 内存缓存
        self._cache: Dict[str, Dict[str, ModelVersion]] = {}
        self._load_cache()
    
    def _init_db(self):
        """初始化数据库表"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                name TEXT,
                version TEXT,
                stage TEXT,
                status TEXT,
                description TEXT,
                tags TEXT,
                run_info TEXT,
                signature TEXT,
                artifact_uri TEXT,
                registered_at TEXT,
                registered_by TEXT,
                approved_at TEXT,
                approved_by TEXT,
                approval_notes TEXT,
                metrics TEXT,
                PRIMARY KEY (name, version)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stage_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                version TEXT,
                from_stage TEXT,
                to_stage TEXT,
                transitioned_at TEXT,
                transitioned_by TEXT,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS approval_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                version TEXT,
                approver TEXT,
                decision TEXT,
                notes TEXT,
                approved_at TEXT
            )
        """)
        
        self.conn.commit()
    
    def _load_cache(self):
        """从数据库加载到内存缓存"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM model_versions")
        
        for row in cursor.fetchall():
            mv = ModelVersion(
                name=row[0],
                version=row[1],
                stage=ModelStage(row[2]),
                status=ModelStatus(row[3]),
                description=row[4],
                tags=json.loads(row[5]) if row[5] else {},
                run_info=RunInfo(**json.loads(row[6])) if row[6] else None,
                signature=ModelSignature(**json.loads(row[7])) if row[7] else None,
                artifact_uri=row[8],
                registered_at=datetime.fromisoformat(row[9]),
                registered_by=row[10],
                approved_at=datetime.fromisoformat(row[11]) if row[11] else None,
                approved_by=row[12],
                approval_notes=row[13],
                metrics=json.loads(row[14]) if row[14] else {}
            )
            
            if mv.name not in self._cache:
                self._cache[mv.name] = {}
            self._cache[mv.name][mv.version] = mv
    
    def _save_to_db(self, mv: ModelVersion):
        """保存模型版本到数据库"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            mv.name,
            mv.version,
            mv.stage.value,
            mv.status.value,
            mv.description,
            json.dumps(mv.tags),
            json.dumps(mv.run_info.to_dict()) if mv.run_info else None,
            json.dumps(mv.signature.to_dict()) if mv.signature else None,
            mv.artifact_uri,
            mv.registered_at.isoformat(),
            mv.registered_by,
            mv.approved_at.isoformat() if mv.approved_at else None,
            mv.approved_by,
            mv.approval_notes,
            json.dumps(mv.metrics)
        ))
        
        self.conn.commit()
    
    def register_model(
        self,
        name: str,
        version: str,
        model_object: any,
        run_info: Optional[RunInfo] = None,
        signature: Optional[ModelSignature] = None,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        registered_by: str = ""
    ) -> ModelVersion:
        """
        注册新模型版本
        
        这是注册中心的核心操作
        """
        # 检查版本是否已存在
        if name in self._cache and version in self._cache[name]:
            raise ValueError(f"Model version {name}:{version} already exists")
        
        # 保存模型文件
        model_dir = os.path.join(self.models_path, name, version)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_object, f)
        
        # 创建模型版本对象
        mv = ModelVersion(
            name=name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            status=ModelStatus.PENDING,
            description=description,
            tags=tags or {},
            run_info=run_info,
            signature=signature,
            artifact_uri=model_path,
            registered_by=registered_by,
            metrics=run_info.metrics if run_info else {}
        )
        
        # 保存到缓存和数据库
        if name not in self._cache:
            self._cache[name] = {}
        self._cache[name][version] = mv
        self._save_to_db(mv)
        
        print(f"[Registry] Registered model: {name}:{version}")
        print(f"  Stage: {mv.stage.value}")
        print(f"  Status: {mv.status.value}")
        print(f"  Path: {model_path}")
        
        return mv
    
    def transition_stage(
        self,
        name: str,
        version: str,
        new_stage: ModelStage,
        transitioned_by: str,
        notes: str = ""
    ) -> ModelVersion:
        """
        转换模型阶段
        
        例如：从 Staging 提升到 Production
        
        阶段转换规则：
        - DEVELOPMENT → STAGING: 需要审批
        - STAGING → PRODUCTION: 需要审批，且当前Staging模型不能是待审批状态
        - PRODUCTION → ARCHIVED: 自动，不需要审批
        """
        if name not in self._cache or version not in self._cache[name]:
            raise ValueError(f"Model {name}:{version} not found")
        
        mv = self._cache[name][version]
        old_stage = mv.stage
        
        # 验证转换规则
        if new_stage == ModelStage.PRODUCTION:
            if mv.status != ModelStatus.APPROVED:
                raise ValueError(f"Model must be approved before transitioning to PRODUCTION")
            
            # 将其他Production模型降级
            for v in self._cache[name].values():
                if v.stage == ModelStage.PRODUCTION and v.version != version:
                    v.stage = ModelStage.ARCHIVED
                    self._save_to_db(v)
                    print(f"[Registry] Archived previous production model: {name}:{v.version}")
        
        # 执行转换
        mv.stage = new_stage
        self._save_to_db(mv)
        
        # 记录转换历史
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO stage_transitions (model_name, version, from_stage, to_stage, 
                                          transitioned_at, transitioned_by, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, version, old_stage.value, new_stage.value,
              datetime.now().isoformat(), transitioned_by, notes))
        self.conn.commit()
        
        print(f"[Registry] Transitioned {name}:{version} {old_stage.value} → {new_stage.value}")
        
        return mv
    
    def approve_model(
        self,
        name: str,
        version: str,
        approver: str,
        decision: bool = True,
        notes: str = ""
    ) -> ModelVersion:
        """
        审批模型
        
        Args:
            decision: True表示批准，False表示拒绝
        """
        if name not in self._cache or version not in self._cache[name]:
            raise ValueError(f"Model {name}:{version} not found")
        
        mv = self._cache[name][version]
        
        if decision:
            mv.status = ModelStatus.APPROVED
            mv.approved_at = datetime.now()
            mv.approved_by = approver
            mv.approval_notes = notes
        else:
            mv.status = ModelStatus.REJECTED
            mv.approval_notes = notes
        
        self._save_to_db(mv)
        
        # 记录审批历史
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO approval_history (model_name, version, approver, decision, notes, approved_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, version, approver, 'APPROVED' if decision else 'REJECTED',
              notes, datetime.now().isoformat()))
        self.conn.commit()
        
        action = "Approved" if decision else "Rejected"
        print(f"[Registry] {action} {name}:{version} by {approver}")
        
        return mv
    
    def get_model(self, name: str, version: Optional[str] = None, 
                  stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """
        获取模型版本
        
        可以通过版本号或阶段获取
        """
        if name not in self._cache:
            return None
        
        if version:
            return self._cache[name].get(version)
        
        if stage:
            for mv in self._cache[name].values():
                if mv.stage == stage:
                    return mv
            return None
        
        # 默认返回最新的Production模型
        return self.get_model(name, stage=ModelStage.PRODUCTION)
    
    def load_model(self, name: str, version: Optional[str] = None,
                   stage: Optional[ModelStage] = None) -> any:
        """
        加载模型对象
        """
        mv = self.get_model(name, version, stage)
        if not mv:
            raise ValueError(f"Model not found: {name}:{version or stage}")
        
        with open(mv.artifact_uri, 'rb') as f:
            return pickle.load(f)
    
    def list_models(self) -> List[str]:
        """列出所有模型名称"""
        return list(self._cache.keys())
    
    def list_versions(self, name: str) -> List[str]:
        """列出模型的所有版本"""
        if name not in self._cache:
            return []
        return list(self._cache[name].keys())
    
    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """获取生产环境模型"""
        return self.get_model(name, stage=ModelStage.PRODUCTION)
    
    def compare_versions(self, name: str, version_a: str, version_b: str) -> Dict:
        """比较两个模型版本"""
        mv_a = self._cache[name].get(version_a)
        mv_b = self._cache[name].get(version_b)
        
        if not mv_a or not mv_b:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'stage_a': mv_a.stage.value,
            'stage_b': mv_b.stage.value,
            'metrics_comparison': {},
            'params_comparison': {}
        }
        
        # 比较指标
        all_metrics = set(mv_a.metrics.keys()) | set(mv_b.metrics.keys())
        for metric in all_metrics:
            val_a = mv_a.metrics.get(metric)
            val_b = mv_b.metrics.get(metric)
            if val_a and val_b:
                diff = ((val_b - val_a) / val_a * 100) if val_a != 0 else 0
                comparison['metrics_comparison'][metric] = {
                    'v1': val_a,
                    'v2': val_b,
                    'diff_pct': diff
                }
        
        return comparison


# =============================================================================
# 演示
# =============================================================================

def demo_model_registry():
    """演示模型注册中心的完整工作流"""
    print("="*80)
    print("模型注册中心演示")
    print("="*80)
    
    # 1. 初始化注册中心
    registry = ModelRegistry("./demo_model_registry")
    
    # 2. 创建一个简单的模型（这里用随机森林作为示例）
    from sklearn.ensemble import RandomForestClassifier
    
    model_v1 = RandomForestClassifier(n_estimators=50, random_state=42)
    model_v2 = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    # 模拟训练（实际应该有真实数据）
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model_v1.fit(X, y)
    model_v2.fit(X, y)
    
    # 3. 注册模型v1.0.0
    print("\n" + "-"*40)
    print("注册模型 v1.0.0...")
    
    run_info_v1 = RunInfo(
        run_id="run_001",
        experiment_id="exp_churn",
        start_time=datetime.now(),
        end_time=datetime.now(),
        metrics={'accuracy': 0.85, 'f1': 0.83, 'auc': 0.89},
        params={'n_estimators': 50, 'max_depth': None},
        git_commit="abc123",
        git_branch="main"
    )
    
    signature_v1 = ModelSignature(
        inputs={'user_features': 'float32[10]', 'account_age': 'int64'},
        outputs={'churn_probability': 'float32', 'will_churn': 'int64'}
    )
    
    mv1 = registry.register_model(
        name="churn_predictor",
        version="v1.0.0",
        model_object=model_v1,
        run_info=run_info_v1,
        signature=signature_v1,
        description="初始版本用户流失预测模型",
        tags={'team': 'growth', 'priority': 'high'},
        registered_by="data_scientist@company.com"
    )
    
    # 4. 注册模型v1.1.0
    print("\n" + "-"*40)
    print("注册模型 v1.1.0...")
    
    run_info_v2 = RunInfo(
        run_id="run_002",
        experiment_id="exp_churn",
        start_time=datetime.now(),
        end_time=datetime.now(),
        metrics={'accuracy': 0.88, 'f1': 0.87, 'auc': 0.92},
        params={'n_estimators': 100, 'max_depth': 10},
        git_commit="def456",
        git_branch="main"
    )
    
    mv2 = registry.register_model(
        name="churn_predictor",
        version="v1.1.0",
        model_object=model_v2,
        run_info=run_info_v2,
        signature=signature_v1,
        description="优化版本：增加树数量和深度限制",
        tags={'team': 'growth', 'priority': 'high'},
        registered_by="data_scientist@company.com"
    )
    
    # 5. 审批流程
    print("\n" + "-"*40)
    print("审批流程...")
    
    # 批准v1.0.0进入Staging
    registry.approve_model(
        name="churn_predictor",
        version="v1.0.0",
        approver="ml_lead@company.com",
        decision=True,
        notes="性能满足基准要求，批准进入Staging"
    )
    
    # 将v1.0.0提升到Staging
    registry.transition_stage(
        name="churn_predictor",
        version="v1.0.0",
        new_stage=ModelStage.STAGING,
        transitioned_by="ml_engineer@company.com",
        notes="通过Staging测试"
    )
    
    # 批准并提升v1.1.0到Production
    registry.approve_model(
        name="churn_predictor",
        version="v1.1.0",
        approver="ml_lead@company.com",
        decision=True,
        notes="性能显著提升，批准直接投入生产"
    )
    
    registry.transition_stage(
        name="churn_predictor",
        version="v1.1.0",
        new_stage=ModelStage.PRODUCTION,
        transitioned_by="ml_engineer@company.com",
        notes="替换v1.0.0"
    )
    
    # 6. 查看当前状态
    print("\n" + "-"*40)
    print("模型版本状态:")
    
    for version in registry.list_versions("churn_predictor"):
        mv = registry.get_model("churn_predictor", version)
        print(f"  {version}: {mv.stage.value} ({mv.status.value})")
    
    # 7. 获取Production模型
    print("\n" + "-"*40)
    print("获取生产环境模型...")
    prod_model = registry.get_production_model("churn_predictor")
    print(f"  生产模型: {prod_model.name}:{prod_model.version}")
    print(f"  准确率: {prod_model.metrics.get('accuracy')}")
    
    # 8. 版本比较
    print("\n" + "-"*40)
    print("版本比较 (v1.0.0 vs v1.1.0):")
    comparison = registry.compare_versions("churn_predictor", "v1.0.0", "v1.1.0")
    for metric, values in comparison['metrics_comparison'].items():
        print(f"  {metric}: {values['v1']:.4f} → {values['v2']:.4f} ({values['diff_pct']:+.1f}%)")
    
    # 9. 加载模型进行推理
    print("\n" + "-"*40)
    print("加载生产模型进行推理...")
    loaded_model = registry.load_model("churn_predictor", stage=ModelStage.PRODUCTION)
    sample_input = np.random.randn(1, 5)
    prediction = loaded_model.predict(sample_input)
    print(f"  输入: {sample_input[0]}")
    print(f"  预测: {prediction[0]}")
    
    print(f"\n{'='*80}")
    print("模型注册中心演示完成！")
    print(f"{'='*80}\n")
    
    return registry


if __name__ == "__main__":
    registry = demo_model_registry()