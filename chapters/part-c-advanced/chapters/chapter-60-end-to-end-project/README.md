# 第六十章 完整项目：端到端的AI应用

> **本章地位**：全书最终章，毕业设计项目。将整合前59章所有知识，构建一个完整的生产级AI系统。

---

## 本章学习目标

完成本章后，你将能够：
- 从零设计一个完整的AI产品架构
- 构建数据收集、处理、建模的全流程管道
- 实现MLOps最佳实践：版本控制、实验追踪、自动化部署
- 搭建可扩展的API服务和前端界面
- 建立监控系统和模型迭代机制

---

## 60.1 项目概述：智慧购（SmartShop）智能电商助手

### 60.1.1 项目背景与愿景

想象一下这样的场景：

> 小明打开"智慧购"APP，系统根据他过去的浏览和购买记录，在首页精准推荐了他正在寻找的跑步鞋。当他犹豫尺码时，智能客服"小智"主动询问他的脚型，并推荐了最适合的型号。结账后，系统预测小明可能需要运动袜，在下一次推送中贴心地展示了相关产品。

这就是我们要构建的**"智慧购"**——一个融合推荐系统、智能客服、用户行为预测的完整AI电商解决方案。

**项目使命**：
- 让技术服务于真实的商业场景
- 展示从数据到生产环境的完整ML生命周期
- 证明"小学生也能理解世界级AI系统"

### 60.1.2 系统架构全景

```
┌─────────────────────────────────────────────────────────────────┐
│                        SmartShop 系统架构                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   前端界面    │    │   API网关    │    │  管理后台    │     │
│   │  React App   │◄──►│   FastAPI    │◄──►│  Streamlit  │     │
│   └──────────────┘    └──────┬───────┘    └──────────────┘     │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     │
│   │  推荐服务    │     │  客服服务    │     │  分析服务    │     │
│   │ Recommendation│   │   Chatbot   │     │  Analytics  │     │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     │
│          │                   │                   │             │
│          └───────────────────┼───────────────────┘             │
│                              ▼                                 │
│   ┌─────────────────────────────────────────────────────┐     │
│   │                  模型服务层 (MLflow)                  │     │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │     │
│   │  │协同过滤  │  │ 深度推荐 │  │ RAG客服  │          │     │
│   │  │ 模型     │  │  模型    │  │  模型    │          │     │
│   │  └──────────┘  └──────────┘  └──────────┘          │     │
│   └─────────────────────────────────────────────────────┘     │
│                              │                                 │
│   ┌──────────────────────────┼──────────────────────────┐     │
│   │                     数据层                            │     │
│   │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │     │
│   │  │PostgreSQL│ │  Redis  │  │ChromaDB│  │  MinIO  │   │     │
│   │  │(主数据库)│  │(缓存)   │  │(向量库) │  │(对象存储)│   │     │
│   │  └────────┘  └────────┘  └────────┘  └────────┘   │     │
│   └────────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 60.1.3 技术栈选择

| 层级 | 技术选型 | 选择理由 |
|------|---------|---------|
| **前端** | React + TailwindCSS | 组件化开发，响应式设计 |
| **API框架** | FastAPI | 高性能异步，自动文档生成 |
| **推荐算法** | PyTorch + Surprise | 深度学习 + 经典算法结合 |
| **NLP/客服** | LangChain + ChromaDB | RAG架构，本地知识库 |
| **数据存储** | PostgreSQL + Redis | 关系型+缓存，性能均衡 |
| **向量数据库** | ChromaDB | 轻量级，易集成 |
| **对象存储** | MinIO | 兼容S3，本地部署 |
| **实验追踪** | MLflow | 完整的ML生命周期管理 |
| **容器化** | Docker + Compose | 开发环境一致性 |
| **监控** | Prometheus + Grafana | 业界标准监控方案 |

---

## 60.2 需求分析与系统设计

### 60.2.1 功能需求规格

#### 用例1：个性化商品推荐

**用户故事**：
> 作为购物者，我希望看到为我量身推荐的商品，这样我可以更快找到感兴趣的产品。

**验收标准**：
- [ ] 首页展示8个个性化推荐商品
- [ ] 推荐基于用户历史行为和相似用户
- [ ] 新用户看到热门商品（冷启动处理）
- [ ] 推荐结果响应时间 < 200ms

**技术实现**：
```python
# 推荐服务接口设计
class RecommendationService:
    """
    个性化推荐服务
    
    费曼法理解：就像一位熟悉你品味的购物顾问，
    根据你的喜好和"和你相似的人"的喜好来推荐商品
    """
    
    async def get_personalized_recommendations(
        self,
        user_id: str,
        context: RecommendationContext,
        n_items: int = 8
    ) -> List[RecommendedItem]:
        """
        获取个性化推荐
        
        Args:
            user_id: 用户唯一标识
            context: 推荐上下文（时间、设备、位置等）
            n_items: 返回商品数量
            
        Returns:
            推荐商品列表，按置信度排序
        """
        pass
```

#### 用例2：智能客服对话

**用户故事**：
> 作为购物者，我希望随时获得购物帮助，就像有位24小时在线的导购员。

**验收标准**：
- [ ] 理解用户自然语言询问
- [ ] 基于知识库提供准确回答
- [ ] 多轮对话保持上下文
- [ ] 无法回答时优雅转人工

**技术实现**：
```python
# 客服服务接口设计
class CustomerServiceBot:
    """
    智能客服机器人
    
    费曼法理解：就像一位读过所有产品手册的超级店员，
    能立刻回答关于任何商品的问题
    """
    
    async def chat(
        self,
        session_id: str,
        user_message: str,
        conversation_history: List[Message]
    ) -> BotResponse:
        """
        处理用户对话
        
        Args:
            session_id: 会话唯一标识
            user_message: 用户输入消息
            conversation_history: 历史对话记录
            
        Returns:
            包含回答和推荐动作的响应
        """
        pass
```

#### 用例3：用户行为分析

**用户故事**：
> 作为运营人员，我希望了解用户行为模式，以便优化商品策略。

**验收标准**：
- [ ] 实时统计用户活跃度
- [ ] 识别高价值用户群体
- [ ] 预测用户流失风险
- [ ] 生成可视化报表

### 60.2.2 非功能需求

| 类别 | 需求 | 指标 |
|------|------|------|
| **性能** | API响应时间 | P95 < 200ms |
| **性能** | 推荐服务吞吐量 | > 1000 QPS |
| **可用性** | 系统可用性 | 99.9% |
| **可扩展性** | 水平扩展 | 支持10倍流量增长 |
| **安全** | 数据加密 | 传输+存储全加密 |
| **可维护性** | 代码覆盖率 | > 80% |

---

## 60.3 数据架构设计

### 60.3.1 数据模型设计

```python
"""
SmartShop 数据模型

费曼法理解：数据就像商场的各种记录——
- 用户信息 = 会员卡档案
- 商品信息 = 商品目录
- 交互记录 = 购物小票
- 对话记录 = 顾客咨询记录
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    """用户实体"""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    age_group = Column(String(20))  # 18-25, 26-35, etc.
    gender = Column(String(10))
    location = Column(String(50))
    registration_date = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime)
    preferences = Column(JSON)  # 存储用户偏好标签
    
    # 关系
    interactions = relationship("UserItemInteraction", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")

class Item(Base):
    """商品实体"""
    __tablename__ = 'items'
    
    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    category = Column(String(50), nullable=False)
    subcategory = Column(String(50))
    brand = Column(String(50))
    price = Column(Float, nullable=False)
    description = Column(Text)
    features = Column(JSON)  # 商品特性，如颜色、尺码等
    image_url = Column(String(500))
    stock_quantity = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    interactions = relationship("UserItemInteraction", back_populates="item")

class UserItemInteraction(Base):
    """
    用户-商品交互记录
    
    这是推荐系统的核心数据，记录用户的所有行为
    """
    __tablename__ = 'user_item_interactions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    item_id = Column(String(36), ForeignKey('items.id'), nullable=False)
    interaction_type = Column(String(20), nullable=False)  # view, click, cart, purchase
    rating = Column(Float)  # 可选的评分，1-5
    timestamp = Column(DateTime, default=datetime.utcnow)
    context = Column(JSON)  # 交互上下文：设备、位置、时间等
    
    # 关系
    user = relationship("User", back_populates="interactions")
    item = relationship("Item", back_populates="interactions")

class Conversation(Base):
    """客服对话会话"""
    __tablename__ = 'conversations'
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey('users.id'))
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    status = Column(String(20), default='active')  # active, closed, escalated
    
    # 关系
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    """对话消息"""
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey('conversations.id'))
    sender_type = Column(String(10), nullable=False)  # user, bot, human
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    intent = Column(String(50))  # 识别到的用户意图
    confidence = Column(Float)  # 意图识别置信度
    
    # 关系
    conversation = relationship("Conversation", back_populates="messages")
```

### 60.3.2 数据流架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      数据流架构图                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐              │
│   │  用户行为 │────►│  事件总线 │────►│  实时处理 │              │
│   │  采集     │     │  Kafka   │     │  Flink   │              │
│   └──────────┘     └──────────┘     └────┬─────┘              │
│                                          │                      │
│                    ┌─────────────────────┼─────────────────┐   │
│                    ▼                     ▼                 ▼   │
│   ┌──────────┐  ┌──────────┐        ┌──────────┐      ┌────────┐│
│   │ 历史数据 │  │ 实时特征 │        │ 推荐模型 │      │ 监控告警││
│   │  Data Lake│  │  Redis   │        │ 更新     │      │        ││
│   └──────────┘  └──────────┘        └──────────┘      └────────┘│
│        │                                              │        │
│        ▼                                              ▼        │
│   ┌──────────┐                                   ┌──────────┐ │
│   │ 离线训练  │◄─────────────────────────────────►│ 生产环境  │ │
│   │  Pipeline│                                   │  Serving │ │
│   └──────────┘                                   └──────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 60.3.3 数据生成与模拟

由于这是教学项目，我们需要生成模拟数据：

```python
"""
数据生成器

为SmartShop生成真实的模拟数据，用于演示和测试
"""

import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker(['zh_CN'])  # 中文数据

class DataGenerator:
    """
    模拟数据生成器
    
    生成用户、商品和交互数据，模拟真实电商场景
    """
    
    # 商品类别定义
    CATEGORIES = {
        '电子产品': ['手机', '笔记本', '耳机', '平板', '智能手表'],
        '服装': ['T恤', '牛仔裤', '连衣裙', '运动鞋', '外套'],
        '食品': ['零食', '饮料', '保健品', '水果', '茶叶'],
        '家居': ['床上用品', '厨具', '装饰品', '收纳', '灯具'],
        '美妆': ['护肤品', '彩妆', '香水', '洗护', '美容仪']
    }
    
    BRANDS = {
        '电子产品': ['Apple', 'Samsung', 'Xiaomi', 'Huawei', 'Sony'],
        '服装': ['Uniqlo', 'Zara', 'Nike', 'Adidas', 'H&M'],
        '食品': ['三只松鼠', '良品铺子', '雀巢', '可口可乐', '农夫山泉'],
        '家居': ['宜家', '无印良品', '网易严选', '小米', '美的'],
        '美妆': ['兰蔻', '雅诗兰黛', '欧莱雅', 'SK-II', '完美日记']
    }
    
    def __init__(self, seed: int = 42):
        """初始化生成器"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        fake.seed_instance(seed)
        
    def generate_users(self, n_users: int = 10000) -> pd.DataFrame:
        """
        生成用户数据
        
        Args:
            n_users: 用户数量
            
        Returns:
            用户DataFrame
        """
        users = []
        age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
        age_weights = [0.25, 0.35, 0.20, 0.15, 0.05]  # 年轻用户更多
        
        for i in range(n_users):
            user = {
                'user_id': f'U{str(i).zfill(6)}',
                'username': fake.user_name(),
                'email': fake.email(),
                'age_group': np.random.choice(age_groups, p=age_weights),
                'gender': np.random.choice(['M', 'F'], p=[0.45, 0.55]),
                'location': fake.city(),
                'registration_date': fake.date_between(
                    start_date='-2y', 
                    end_date='today'
                ),
                'last_active': fake.date_between(
                    start_date='-30d', 
                    end_date='today'
                )
            }
            users.append(user)
            
        return pd.DataFrame(users)
    
    def generate_items(self, n_items: int = 5000) -> pd.DataFrame:
        """
        生成商品数据
        
        Args:
            n_items: 商品数量
            
        Returns:
            商品DataFrame
        """
        items = []
        
        # 价格区间定义
        price_ranges = {
            '电子产品': (500, 15000),
            '服装': (50, 2000),
            '食品': (10, 500),
            '家居': (30, 3000),
            '美妆': (50, 5000)
        }
        
        for i in range(n_items):
            category = random.choice(list(self.CATEGORIES.keys()))
            subcategory = random.choice(self.CATEGORIES[category])
            brand = random.choice(self.BRANDS[category])
            price_min, price_max = price_ranges[category]
            
            item = {
                'item_id': f'I{str(i).zfill(6)}',
                'name': f'{brand}{subcategory}{random.randint(1, 999)}',
                'category': category,
                'subcategory': subcategory,
                'brand': brand,
                'price': round(np.random.uniform(price_min, price_max), 2),
                'description': fake.text(max_nb_chars=200),
                'stock_quantity': random.randint(0, 1000),
                'created_at': fake.date_between(
                    start_date='-1y', 
                    end_date='today'
                )
            }
            items.append(item)
            
        return pd.DataFrame(items)
    
    def generate_interactions(
        self, 
        users: pd.DataFrame, 
        items: pd.DataFrame,
        n_interactions: int = 100000
    ) -> pd.DataFrame:
        """
        生成用户-商品交互数据
        
        模拟真实用户行为模式：
        - 80/20法则：20%商品获得80%交互
        - 用户偏好：用户倾向于特定类别
        - 时间模式：周末和晚上更活跃
        
        Args:
            users: 用户DataFrame
            items: 商品DataFrame
            n_interactions: 交互记录数量
            
        Returns:
            交互DataFrame
        """
        interactions = []
        
        # 为每个用户生成偏好类别
        user_preferences = {}
        for _, user in users.iterrows():
            # 每个用户偏好1-3个类别
            n_prefs = random.randint(1, 3)
            prefs = random.sample(list(self.CATEGORIES.keys()), n_prefs)
            user_preferences[user['user_id']] = prefs
        
        # 生成交互
        for _ in range(n_interactions):
            user = users.sample(1).iloc[0]
            user_prefs = user_preferences[user['user_id']]
            
            # 70%概率选择偏好类别，30%随机
            if random.random() < 0.7:
                preferred_items = items[items['category'].isin(user_prefs)]
                if len(preferred_items) > 0:
                    item = preferred_items.sample(1).iloc[0]
                else:
                    item = items.sample(1).iloc[0]
            else:
                # 热门商品更有可能被选中（幂律分布）
                item = items.sample(1, weights=np.power(items.index + 1, -0.5)).iloc[0]
            
            # 交互类型概率
            interaction_type = np.random.choice(
                ['view', 'click', 'cart', 'purchase'],
                p=[0.50, 0.30, 0.12, 0.08]
            )
            
            # 生成时间戳（考虑时间模式）
            base_date = fake.date_time_between(start_date='-6M', end_date='now')
            # 添加时间偏好：晚上8-10点更活跃
            if random.random() < 0.4:
                base_date = base_date.replace(hour=random.randint(20, 22))
            
            interaction = {
                'user_id': user['user_id'],
                'item_id': item['item_id'],
                'interaction_type': interaction_type,
                'rating': np.random.choice([1,2,3,4,5], p=[0.05,0.1,0.2,0.35,0.3]) if interaction_type == 'purchase' else None,
                'timestamp': base_date,
                'device': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.35, 0.05]),
                'session_id': fake.uuid4()
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)

# 使用示例
if __name__ == '__main__':
    generator = DataGenerator(seed=42)
    
    print("=" * 60)
    print("SmartShop 数据生成器")
    print("=" * 60)
    
    # 生成数据
    print("\n[1/3] 生成用户数据...")
    users_df = generator.generate_users(n_users=10000)
    print(f"      ✓ 生成 {len(users_df)} 个用户")
    
    print("\n[2/3] 生成商品数据...")
    items_df = generator.generate_items(n_items=5000)
    print(f"      ✓ 生成 {len(items_df)} 个商品")
    
    print("\n[3/3] 生成交互数据...")
    interactions_df = generator.generate_interactions(
        users_df, items_df, n_interactions=100000
    )
    print(f"      ✓ 生成 {len(interactions_df)} 条交互记录")
    
    # 数据分布统计
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    
    print("\n用户年龄分布:")
    print(users_df['age_group'].value_counts())
    
    print("\n商品类别分布:")
    print(items_df['category'].value_counts())
    
    print("\n交互类型分布:")
    print(interactions_df['interaction_type'].value_counts())
    
    # 保存数据
    users_df.to_csv('users.csv', index=False)
    items_df.to_csv('items.csv', index=False)
    interactions_df.to_csv('interactions.csv', index=False)
    
    print("\n" + "=" * 60)
    print("数据已保存到 CSV 文件")
    print("=" * 60)
```

---

## 60.4 推荐系统实现

### 60.4.1 协同过滤模型

```python
"""
协同过滤推荐算法

费曼法理解：协同过滤就像问朋友"你喜欢什么"。
- 用户协同过滤：找"和你相似的人"，推荐他们喜欢的
- 物品协同过滤：找"和你喜欢的物品相似的"其他物品
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CollaborativeFiltering:
    """
    协同过滤推荐系统
    
    实现用户协同过滤和物品协同过滤两种策略
    """
    
    def __init__(self, n_factors: int = 50):
        """
        初始化协同过滤模型
        
        Args:
            n_factors: SVD降维后的因子数量
        """
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.user_sim_matrix = None
        self.item_sim_matrix = None
        
    def fit(self, interactions_df: pd.DataFrame) -> 'CollaborativeFiltering':
        """
        训练协同过滤模型
        
        Args:
            interactions_df: 交互数据，包含user_id, item_id, rating
            
        Returns:
            self
        """
        logger.info("开始训练协同过滤模型...")
        
        # 创建ID映射
        unique_users = interactions_df['user_id'].unique()
        unique_items = interactions_df['item_id'].unique()
        
        self.user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_mapping = {iid: idx for idx, iid in enumerate(unique_items)}
        
        self.reverse_user_mapping = {v: k for k, v in self.user_mapping.items()}
        self.reverse_item_mapping = {v: k for k, v in self.item_mapping.items()}
        
        # 构建用户-物品评分矩阵
        logger.info("构建评分矩阵...")
        rows = [self.user_mapping[uid] for uid in interactions_df['user_id']]
        cols = [self.item_mapping[iid] for iid in interactions_df['item_id']]
        
        # 根据交互类型分配权重
        weights = interactions_df['interaction_type'].map({
            'view': 1,
            'click': 2,
            'cart': 3,
            'purchase': 5
        }).fillna(1)
        
        if 'rating' in interactions_df.columns:
            ratings = interactions_df['rating'].fillna(0) * weights
        else:
            ratings = weights
        
        # 创建稀疏矩阵
        self.user_item_matrix = csr_matrix(
            (ratings, (rows, cols)),
            shape=(len(unique_users), len(unique_items))
        )
        
        logger.info(f"评分矩阵形状: {self.user_item_matrix.shape}")
        logger.info(f"矩阵稀疏度: {1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.4f}")
        
        # 使用SVD进行矩阵分解
        logger.info(f"执行SVD分解 (n_factors={self.n_factors})...")
        svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        self.user_factors = svd.fit_transform(self.user_item_matrix)
        self.item_factors = svd.components_.T
        
        logger.info(f"用户因子矩阵: {self.user_factors.shape}")
        logger.info(f"物品因子矩阵: {self.item_factors.shape}")
        
        # 计算相似度矩阵（用于基于内存的方法）
        logger.info("计算用户相似度矩阵...")
        self.user_sim_matrix = cosine_similarity(self.user_factors)
        
        logger.info("计算物品相似度矩阵...")
        self.item_sim_matrix = cosine_similarity(self.item_factors)
        
        logger.info("✓ 协同过滤模型训练完成")
        return self
    
    def recommend_user_based(
        self, 
        user_id: str, 
        n_recommendations: int = 10,
        k_neighbors: int = 20
    ) -> List[Tuple[str, float]]:
        """
        基于用户的协同过滤推荐
        
        找与目标用户最相似的k个用户，推荐他们喜欢的物品
        
        Args:
            user_id: 目标用户ID
            n_recommendations: 推荐数量
            k_neighbors: 相似用户数量
            
        Returns:
            推荐物品列表 [(item_id, score), ...]
        """
        if user_id not in self.user_mapping:
            logger.warning(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # 找到k个最相似的用户
        user_sims = self.user_sim_matrix[user_idx]
        similar_users = np.argsort(user_sims)[::-1][1:k_neighbors+1]  # 排除自己
        
        # 获取目标用户已有的物品
        user_items = set(self.user_item_matrix[user_idx].nonzero()[1])
        
        # 计算候选物品的得分
        scores = {}
        for sim_user_idx in similar_users:
            similarity = user_sims[sim_user_idx]
            sim_user_items = self.user_item_matrix[sim_user_idx].nonzero()[1]
            
            for item_idx in sim_user_items:
                if item_idx not in user_items:  # 只推荐新物品
                    if item_idx not in scores:
                        scores[item_idx] = 0
                    scores[item_idx] += similarity * self.user_item_matrix[sim_user_idx, item_idx]
        
        # 排序并返回top-n
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [
            (self.reverse_item_mapping[item_idx], score)
            for item_idx, score in sorted_items[:n_recommendations]
        ]
        
        return recommendations
    
    def recommend_item_based(
        self, 
        user_id: str, 
        n_recommendations: int = 10
    ) -> List[Tuple[str, float]]:
        """
        基于物品的协同过滤推荐
        
        基于用户历史喜欢的物品，推荐相似的物品
        
        Args:
            user_id: 目标用户ID
            n_recommendations: 推荐数量
            
        Returns:
            推荐物品列表 [(item_id, score), ...]
        """
        if user_id not in self.user_mapping:
            logger.warning(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.user_mapping[user_id]
        
        # 获取用户交互过的物品
        user_items = self.user_item_matrix[user_idx].nonzero()[1]
        user_ratings = self.user_item_matrix[user_idx].data
        
        if len(user_items) == 0:
            return []
        
        # 计算候选物品的得分
        scores = {}
        for item_idx, rating in zip(user_items, user_ratings):
            # 找到与当前物品相似的其他物品
            item_sims = self.item_sim_matrix[item_idx]
            
            for candidate_idx, sim in enumerate(item_sims):
                if candidate_idx not in user_items and sim > 0:  # 新物品且相似度>0
                    if candidate_idx not in scores:
                        scores[candidate_idx] = 0
                    scores[candidate_idx] += sim * rating
        
        # 归一化
        if len(user_items) > 0:
            for item_idx in scores:
                scores[item_idx] /= len(user_items)
        
        # 排序并返回
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = [
            (self.reverse_item_mapping[item_idx], score)
            for item_idx, score in sorted_items[:n_recommendations]
        ]
        
        return recommendations
    
    def recommend_matrix_factorization(
        self,
        user_id: str,
        n_recommendations: int = 10
    ) -> List[Tuple[str, float]]:
        """
        基于矩阵分解的推荐
        
        使用学习到的用户和物品隐向量计算推荐
        
        Args:
            user_id: 目标用户ID
            n_recommendations: 推荐数量
            
        Returns:
            推荐物品列表
        """
        if user_id not in self.user_mapping:
            logger.warning(f"用户 {user_id} 不在训练集中")
            return []
        
        user_idx = self.user_mapping[user_id]
        user_vec = self.user_factors[user_idx]
        
        # 获取用户已有的物品
        user_items = set(self.user_item_matrix[user_idx].nonzero()[1])
        
        # 计算所有物品的预测评分
        scores = np.dot(self.item_factors, user_vec)
        
        # 排除已有物品
        candidate_indices = [i for i in range(len(scores)) if i not in user_items]
        candidate_scores = [(i, scores[i]) for i in candidate_indices]
        
        # 排序并返回
        sorted_items = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        recommendations = [
            (self.reverse_item_mapping[item_idx], float(score))
            for item_idx, score in sorted_items[:n_recommendations]
        ]
        
        return recommendations


class NeuralCollaborativeFiltering(nn.Module):
    """
    神经协同过滤 (NCF)
    
    使用深度神经网络学习用户-物品交互的非线性关系
    
    架构:
    - 输入层: 用户ID嵌入 + 物品ID嵌入
    - 隐藏层: 多层全连接，学习复杂交互模式
    - 输出层: 预测评分或交互概率
    """
    
    def __init__(
        self, 
        n_users: int, 
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64, 32]
    ):
        """
        初始化NCF模型
        
        Args:
            n_users: 用户数量
            n_items: 物品数量
            embedding_dim: 嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # 嵌入层
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # 构建MLP层
        layers = []
        input_dim = embedding_dim * 2  # 拼接用户和物品嵌入
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            user_ids: 用户ID张量 [batch_size]
            item_ids: 物品ID张量 [batch_size]
            
        Returns:
            预测评分 [batch_size, 1]
        """
        # 获取嵌入
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, embedding_dim]
        
        # 拼接
        vector = torch.cat([user_emb, item_emb], dim=-1)  # [batch_size, embedding_dim*2]
        
        # MLP
        output = self.mlp(vector)  # [batch_size, hidden_dims[-1]]
        
        # 输出预测
        rating = self.output_layer(output)  # [batch_size, 1]
        
        return torch.sigmoid(rating)  # 归一化到0-1


class RecommendationEnsemble:
    """
    推荐集成器
    
    组合多种推荐算法的结果，提供更准确的推荐
    
    费曼法理解：就像咨询多个购物顾问，然后综合他们的建议
    """
    
    def __init__(
        self,
        cf_model: CollaborativeFiltering,
        ncf_model: NeuralCollaborativeFiltering = None,
        weights: Dict[str, float] = None
    ):
        """
        初始化集成器
        
        Args:
            cf_model: 协同过滤模型
            ncf_model: 神经协同过滤模型（可选）
            weights: 各算法的权重
        """
        self.cf_model = cf_model
        self.ncf_model = ncf_model
        self.weights = weights or {
            'user_cf': 0.3,
            'item_cf': 0.3,
            'matrix_factorization': 0.4
        }
    
    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        context: Dict = None
    ) -> List[Tuple[str, float]]:
        """
        集成推荐
        
        Args:
            user_id: 目标用户ID
            n_recommendations: 推荐数量
            context: 推荐上下文
            
        Returns:
            推荐物品列表
        """
        all_scores = {}
        
        # 1. 用户协同过滤
        if self.weights.get('user_cf', 0) > 0:
            user_cf_recs = self.cf_model.recommend_user_based(
                user_id, n_recommendations=n_recommendations*2
            )
            for item_id, score in user_cf_recs:
                if item_id not in all_scores:
                    all_scores[item_id] = 0
                all_scores[item_id] += score * self.weights['user_cf']
        
        # 2. 物品协同过滤
        if self.weights.get('item_cf', 0) > 0:
            item_cf_recs = self.cf_model.recommend_item_based(
                user_id, n_recommendations=n_recommendations*2
            )
            for item_id, score in item_cf_recs:
                if item_id not in all_scores:
                    all_scores[item_id] = 0
                all_scores[item_id] += score * self.weights['item_cf']
        
        # 3. 矩阵分解
        if self.weights.get('matrix_factorization', 0) > 0:
            mf_recs = self.cf_model.recommend_matrix_factorization(
                user_id, n_recommendations=n_recommendations*2
            )
            for item_id, score in mf_recs:
                if item_id not in all_scores:
                    all_scores[item_id] = 0
                all_scores[item_id] += score * self.weights['matrix_factorization']
        
        # 排序并返回
        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]


# 模型训练脚本
if __name__ == '__main__':
    print("=" * 60)
    print("SmartShop 推荐系统训练")
    print("=" * 60)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    interactions_df = pd.read_csv('interactions.csv')
    print(f"      ✓ 加载 {len(interactions_df)} 条交互记录")
    
    # 训练协同过滤模型
    print("\n[2/4] 训练协同过滤模型...")
    cf_model = CollaborativeFiltering(n_factors=50)
    cf_model.fit(interactions_df)
    
    # 测试推荐
    print("\n[3/4] 测试推荐...")
    test_user = interactions_df['user_id'].iloc[0]
    print(f"\n为用户 {test_user} 生成推荐:")
    
    print("\n用户协同过滤推荐:")
    user_recs = cf_model.recommend_user_based(test_user, n_recommendations=5)
    for item_id, score in user_recs:
        print(f"  - {item_id}: {score:.4f}")
    
    print("\n物品协同过滤推荐:")
    item_recs = cf_model.recommend_item_based(test_user, n_recommendations=5)
    for item_id, score in item_recs:
        print(f"  - {item_id}: {score:.4f}")
    
    print("\n矩阵分解推荐:")
    mf_recs = cf_model.recommend_matrix_factorization(test_user, n_recommendations=5)
    for item_id, score in mf_recs:
        print(f"  - {item_id}: {score:.4f}")
    
    # 保存模型
    print("\n[4/4] 保存模型...")
    import pickle
    with open('cf_model.pkl', 'wb') as f:
        pickle.dump(cf_model, f)
    print("      ✓ 模型已保存到 cf_model.pkl")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
