# 第五十四章 神经符号AI与可解释推理

> **费曼法一句话**：想象你的大脑有两个部分——左脑像严谨的数学家，右脑像直觉敏锐的艺术家。神经符号AI就是让AI同时拥有"数学家的逻辑"和"艺术家的直觉"，既能看懂图片，又能进行严谨推理！

---

## 目录

1. [引言：连接主义与符号主义的融合](#1-引言连接主义与符号主义的融合)
2. [知识图谱与神经网络](#2-知识图谱与神经网络)
3. [可微分编程与神经程序合成](#3-可微分编程与神经程序合成)
4. [视觉推理与组合泛化](#4-视觉推理与组合泛化)
5. [大语言模型的推理能力](#5-大语言模型的推理能力)
6. [可解释性与因果推理](#6-可解释性与因果推理)
7. [实战案例：神经符号问答系统](#7-实战案例神经符号问答系统)
8. [本章小结](#8-本章小结)
9. [练习题](#9-练习题)
10. [参考文献](#10-参考文献)

---

## 1. 引言：连接主义与符号主义的融合

### 1.1 两种AI范式：左右脑的分工

想象你的大脑是一个超级计算机：

- **右脑**（连接主义）🎨：直觉、模式识别、创造力。看到一个陌生人的脸，你立刻认出他是谁，却说不出为什么。
- **左脑**（符号主义）🧮：逻辑、推理、数学证明。解方程时一步步推导，每一步都有明确的规则。

**类比：神经符号AI = 左右脑协同工作**

就像一位优秀的数学家既有直觉又有严谨的逻辑，神经符号AI试图让AI系统同时具备：
- 神经网络的感知和模式识别能力
- 符号系统的逻辑推理和可解释性

### 1.2 为什么需要融合？

#### 深度学习的局限

现代深度学习像是一个"天才但健忘的学生"：

```
❌ 问题1：黑盒决策
用户问："为什么推荐这部电影？"
神经网络："...我也不知道，但数据说你会喜欢。"

❌ 问题2：组合泛化能力差
见过"红立方"和"蓝球"，却认不出"红球"

❌ 问题3：缺乏常识推理
知道"猫会爬树"，但推不出"树上的猫怎么下来"

❌ 问题4：需要海量数据
人类看几张猫的照片就认识猫，AI需要几万张
```

#### 符号系统的局限

传统符号AI像是一个"死板但诚实的图书管理员"：

```
❌ 问题1：脆弱性
输入"喵星人"而不是"猫"，系统完全不理解

❌ 问题2：知识获取瓶颈
需要专家手动编写所有规则，费时费力

❌ 问题3：难以处理不确定性
"可能"、"大概"、"也许"难以编码

❌ 问题4：无法从数据中学习
没有归纳能力，只能演绎推理
```

#### 人类认知的启示

人类大脑是完美的融合体：

| 任务 | 人类怎么做 | AI需要什么 |
|------|------------|------------|
| 认猫 | 看一眼就认出 + 知道猫会抓老鼠 | 神经网络识别 + 知识图谱推理 |
| 数学证明 | 直觉猜方向 + 严谨验证 | 神经启发 + 符号证明 |
| 语言理解 | 听懂字面意思 + 理解言外之意 | 语义嵌入 + 逻辑推理 |

### 1.3 神经符号AI的定义与愿景

**正式定义**：神经符号AI（Neuro-Symbolic AI）是将神经网络的学习能力与符号系统的推理能力相结合的AI范式，旨在创建既具备感知能力又具备推理能力的智能系统。

**核心愿景**：
1. **可解释的AI**：不仅给出答案，还能解释为什么
2. **数据高效的学习**：利用先验知识减少数据需求
3. **组合泛化**：像搭乐高一样组合已知概念解决新问题
4. **常识推理**：像人类一样运用世界知识

**近期突破性进展**：

- **AlphaProof** (2024): DeepMind的系统，结合神经网络与形式化证明，解决了国际数学奥林匹克的几何问题
- **MathVista**: 多模态数学推理基准，测试视觉+符号推理
- **GPT-4 + 符号验证**: 大语言模型生成候选解，符号系统验证正确性

---

## 2. 知识图谱与神经网络

### 2.1 知识图谱：AI的家族族谱

**费曼法比喻**：知识图谱就像一本详细的**家族族谱**。每个人（实体）都有名字，人与人之间的关系（边）清清楚楚——谁是父亲、谁是兄弟、谁嫁给了谁。你可以顺着关系链找到远房亲戚，就像AI可以顺着知识图谱推理出"拿破仑的妻子的故乡"。

#### 什么是知识图谱？

知识图谱是一个**三元组**的集合：(头实体, 关系, 尾实体)，记作 $(h, r, t)$。

```
知识图谱示例：

(爱因斯坦, 发现, 相对论)
(相对论, 属于, 物理学)
(爱因斯坦, 获得, 诺贝尔奖)
(诺贝尔奖, 颁发地, 斯德哥尔摩)

可以推理出：
爱因斯坦 → [获得] → 诺贝尔奖 → [颁发地] → 斯德哥尔摩
∴ 爱因斯坦与斯德哥尔摩有关
```

#### 知识图谱嵌入的挑战

传统符号表示的局限：
- 离散、稀疏、难以计算相似度
- 无法处理"大概相似"的关系
- 难以与神经网络结合

**解决方案**：将实体和关系映射到连续向量空间！

### 2.2 TransE：翻译嵌入模型

**核心思想**：把关系看作实体间的**平移操作**。就像从"北京"到"上海"是向东平移，从"父亲"到"儿子"是下一代的平移。

#### 数学原理

对于三元组 $(h, r, t)$，TransE希望：

$$\mathbf{h} + \mathbf{r} \approx \mathbf{t}$$

即：头实体的向量 + 关系的向量 ≈ 尾实体的向量

**评分函数**：

$$f_r(h, t) = -\|\mathbf{h} + \mathbf{r} - \mathbf{t}\|_{1/2}$$

距离越小，三元组越可能是真实的。

**损失函数**（基于负采样）：

$$\mathcal{L} = \sum_{(h,r,t) \in \mathcal{S}} \sum_{(h',r,t') \in \mathcal{S}'} \max(0, \gamma + f_r(h,t) - f_r(h',t'))$$

其中：
- $\mathcal{S}$ 是真实三元组集合
- $\mathcal{S}'$ 是负采样生成的假三元组
- $\gamma$ 是边界超参数

#### Python实现

```python
"""
TransE: Translating Embeddings for Multi-Relational Data
实现知识图谱嵌入的经典模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import random


class TransE(nn.Module):
    """
    TransE模型：将关系视为向量空间中的平移
    
    核心思想: h + r ≈ t
    
    就像:
    - 北京 + 向东 = 上海
    - 父亲 + 下一代 = 儿子
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        gamma: float = 12.0,
        norm_p: int = 1
    ):
        """
        参数:
            num_entities: 实体数量
            num_relations: 关系数量  
            embedding_dim: 嵌入维度
            gamma: 边界超参数
            norm_p: 范数类型 (1或2)
        """
        super(TransE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.norm_p = norm_p
        
        # 实体嵌入
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        
        # 关系嵌入
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """使用均匀分布初始化"""
        bound = 6 / np.sqrt(self.embedding_dim)
        
        nn.init.uniform_(self.entity_embedding.weight, -bound, bound)
        nn.init.uniform_(self.relation_embedding.weight, -bound, bound)
        
        # 归一化实体嵌入
        self.entity_embedding.weight.data = F.normalize(
            self.entity_embedding.weight.data, p=2, dim=1
        )
        
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        negative_heads: torch.Tensor = None,
        negative_tails: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            heads: 头实体索引 [batch_size]
            relations: 关系索引 [batch_size]
            tails: 尾实体索引 [batch_size]
            negative_heads: 负采样头实体（可选）
            negative_tails: 负采样尾实体（可选）
            
        返回:
            positive_score: 正样本得分
            negative_score: 负样本得分
        """
        # 获取嵌入
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        t = self.entity_embedding(tails)
        
        # 计算正样本得分: -||h + r - t||
        positive_score = self._score(h, r, t)
        
        # 计算负样本得分
        if negative_heads is not None and negative_tails is not None:
            h_neg = self.entity_embedding(negative_heads)
            t_neg = self.entity_embedding(negative_tails)
            negative_score = self._score(h_neg, r, t_neg)
        else:
            negative_score = None
            
        return positive_score, negative_score
    
    def _score(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        计算TransE评分函数
        
        score = -||h + r - t||_p
        
        距离越小，得分越高（越可能是真实三元组）
        """
        score = h + r - t
        score = -torch.norm(score, p=self.norm_p, dim=-1)
        return score
    
    def loss(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor
    ) -> torch.Tensor:
        """
        计算基于边界排名的损失函数
        
        L = Σ max(0, γ + score_neg - score_pos)
        """
        loss = F.relu(self.gamma + negative_score - positive_score)
        return loss.mean()
    
    def predict(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor
    ) -> torch.Tensor:
        """预测三元组的真实性得分"""
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        t = self.entity_embedding(tails)
        return self._score(h, r, t)
    
    def link_prediction(
        self,
        head: int,
        relation: int,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        链接预测：给定头实体和关系，预测最可能的尾实体
        
        这就像: "爱因斯坦 ___ 相对论" -> 预测"发现"
        """
        with torch.no_grad():
            h = self.entity_embedding(torch.tensor([head]))
            r = self.relation_embedding(torch.tensor([relation]))
            
            # 计算所有实体作为尾实体的得分
            all_entities = self.entity_embedding.weight
            scores = -torch.norm(h + r - all_entities, p=self.norm_p, dim=1)
            
            # 获取top-k
            top_k_scores, top_k_indices = torch.topk(scores, k)
            
        return list(zip(top_k_indices.tolist(), top_k_scores.tolist()))


class KnowledgeGraphDataset:
    """知识图谱数据集管理"""
    
    def __init__(self, triples: List[Tuple[int, int, int]], num_entities: int):
        """
        参数:
            triples: 三元组列表 (h, r, t)
            num_entities: 实体总数
        """
        self.triples = triples
        self.num_entities = num_entities
        
        # 构建实体-关系映射
        self.triple_set = set(triples)
        self.hr_to_t = defaultdict(set)
        self.tr_to_h = defaultdict(set)
        
        for h, r, t in triples:
            self.hr_to_t[(h, r)].add(t)
            self.tr_to_h[(t, r)].add(h)
    
    def negative_sampling(
        self,
        batch_triples: List[Tuple[int, int, int]],
        negative_rate: float = 0.5
    ) -> Tuple[List[int], List[int], List[int], List[int], List[int]]:
        """
        生成负样本
        
        策略：随机替换头实体或尾实体
        """
        heads, relations, tails = [], [], []
        neg_heads, neg_tails = [], []
        
        for h, r, t in batch_triples:
            heads.append(h)
            relations.append(r)
            tails.append(t)
            
            # 随机决定是否替换头或尾
            if random.random() < 0.5:
                neg_h = random.randint(0, self.num_entities - 1)
                while neg_h in self.tr_to_h.get((t, r), set()):
                    neg_h = random.randint(0, self.num_entities - 1)
                neg_heads.append(neg_h)
                neg_tails.append(t)
            else:
                neg_t = random.randint(0, self.num_entities - 1)
                while neg_t in self.hr_to_t.get((h, r), set()):
                    neg_t = random.randint(0, self.num_entities - 1)
                neg_heads.append(h)
                neg_tails.append(neg_t)
        
        return heads, relations, tails, neg_heads, neg_tails


def train_transe(
    model: TransE,
    dataset: KnowledgeGraphDataset,
    epochs: int = 1000,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = 'cpu'
):
    """训练TransE模型"""
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    num_batches = len(dataset.triples) // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        
        # 随机打乱
        triples = dataset.triples.copy()
        random.shuffle(triples)
        
        for i in range(num_batches):
            batch = triples[i * batch_size: (i + 1) * batch_size]
            
            # 负采样
            h, r, t, h_neg, t_neg = dataset.negative_sampling(batch)
            
            # 转为tensor
            h = torch.tensor(h, dtype=torch.long, device=device)
            r = torch.tensor(r, dtype=torch.long, device=device)
            t = torch.tensor(t, dtype=torch.long, device=device)
            h_neg = torch.tensor(h_neg, dtype=torch.long, device=device)
            t_neg = torch.tensor(t_neg, dtype=torch.long, device=device)
            
            # 前向传播
            pos_score, neg_score = model(h, r, t, h_neg, t_neg)
            
            # 计算损失
            loss = model.loss(pos_score, neg_score)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 归一化实体嵌入
            with torch.no_grad():
                model.entity_embedding.weight.data = F.normalize(
                    model.entity_embedding.weight.data, p=2, dim=1
                )
            
            total_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


def create_sample_knowledge_graph():
    """创建示例知识图谱"""
    
    triples = [
        (0, 0, 1),  # 中国 -首都-> 北京
        (2, 0, 3),  # 法国 -首都-> 巴黎
        (4, 0, 5),  # 日本 -首都-> 东京
        (6, 0, 7),  # 英国 -首都-> 伦敦
        (1, 1, 0),  # 北京 -位于-> 中国
        (0, 2, 8),  # 中国 -说语言-> 中文
        (2, 2, 9),  # 法国 -说语言-> 法语
        (4, 2, 10), # 日本 -说语言-> 日语
        (6, 2, 11), # 英国 -说语言-> 英语
        (2, 3, 6),  # 法国 -邻国-> 英国
        (0, 3, 4),  # 中国 -邻国-> 日本
    ]
    
    entity_names = {
        0: "中国", 1: "北京", 2: "法国", 3: "巴黎",
        4: "日本", 5: "东京", 6: "英国", 7: "伦敦",
        8: "中文", 9: "法语", 10: "日语", 11: "英语"
    }
    
    relation_names = {
        0: "首都", 1: "位于", 2: "说语言", 3: "邻国"
    }
    
    return triples, entity_names, relation_names
```

### 2.3 RotatE：复数空间中的旋转

**核心思想**：把关系看作复数空间中的**旋转操作**。就像时钟的指针旋转，某些关系更适合用旋转而不是平移来表示。

#### 数学原理

RotatE在**复数空间**中表示实体和关系：

$$\mathbf{h}, \mathbf{t} \in \mathbb{C}^d, \quad \mathbf{r} \in \mathbb{C}^d \text{ 且 } |r_i| = 1$$

关系向量被限制为单位复数：

$$r_i = e^{i\theta_i} = \cos\theta_i + i\sin\theta_i$$

**评分函数**：

$$\mathbf{t} = \mathbf{h} \circ \mathbf{r}$$
$$f_r(h, t) = -\|\mathbf{h} \circ \mathbf{r} - \mathbf{t}\|$$

```python
"""
RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


class RotatE(nn.Module):
    """
    RotatE模型：在复数空间中将关系建模为旋转
    
    核心思想: t = h ∘ r (逐元素复数乘法)
    其中 r 被限制为单位复数 |r| = 1
    """
    
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 250,
        gamma: float = 12.0
    ):
        super(RotatE, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        
        self.embedding_range = (gamma + 2.0) / embedding_dim
        
        # 实体嵌入 [num_entities, embedding_dim * 2]
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim * 2)
        
        # 关系嵌入表示相位（角度）
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.uniform_(
            self.entity_embedding.weight,
            -6 / np.sqrt(self.embedding_dim),
            6 / np.sqrt(self.embedding_dim)
        )
        
        nn.init.uniform_(
            self.relation_embedding.weight,
            -6 / np.sqrt(self.embedding_dim),
            6 / np.sqrt(self.embedding_dim)
        )
        
    def forward(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        negative_heads: torch.Tensor = None,
        negative_tails: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        复数旋转: t = h * r
        其中 r = cos(θ) + i*sin(θ), θ 是关系角度
        """
        # 获取实体嵌入
        h_embed = self.entity_embedding(heads)
        t_embed = self.entity_embedding(tails)
        
        # 分割实部和虚部
        h_re, h_im = torch.chunk(h_embed, 2, dim=-1)
        t_re, t_im = torch.chunk(t_embed, 2, dim=-1)
        
        # 获取关系相位
        r_phase = self.relation_embedding(relations)
        
        # 关系向量限制在单位圆上
        r_re = torch.cos(r_phase)
        r_im = torch.sin(r_phase)
        
        # 复数乘法: h * r
        h_rotate_re = h_re * r_re - h_im * r_im
        h_rotate_im = h_re * r_im + h_im * r_re
        
        # 计算正样本得分
        score_re = h_rotate_re - t_re
        score_im = h_rotate_im - t_im
        
        score = torch.stack([score_re, score_im], dim=0).norm(dim=0).sum(dim=-1)
        positive_score = -score
        
        # 计算负样本得分
        if negative_heads is not None and negative_tails is not None:
            h_neg_embed = self.entity_embedding(negative_heads)
            h_neg_re, h_neg_im = torch.chunk(h_neg_embed, 2, dim=-1)
            
            h_neg_rotate_re = h_neg_re * r_re - h_neg_im * r_im
            h_neg_rotate_im = h_neg_re * r_im + h_neg_im * r_re
            
            score_re_neg = h_neg_rotate_re - t_re
            score_im_neg = h_neg_rotate_im - t_im
            
            score_neg = torch.stack([score_re_neg, score_im_neg], dim=0).norm(dim=0).sum(dim=-1)
            negative_score = -score_neg
        else:
            negative_score = None
        
        return positive_score, negative_score
    
    def loss(
        self,
        positive_score: torch.Tensor,
        negative_score: torch.Tensor
    ) -> torch.Tensor:
        """边界排名损失"""
        return F.relu(self.gamma - positive_score + negative_score).mean()
```

---

## 3. 可微分编程与神经程序合成

### 3.1 神经程序解释器

**费曼法比喻**：想象一个**魔术师猜牌**的表演。观众心里想一个过程（程序），给出一个结果（输出），魔术师要猜出这个过程。神经程序合成就是训练AI成为这个"魔术师"——从输入输出示例中推断出背后的程序。

```python
"""
Neural Turing Machine (NTM) - 简化实现
神经网络 + 外部记忆 = 可学习的图灵机
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class NTMController(nn.Module):
    """
    NTM控制器：生成读写操作的参数
    """
    
    def __init__(
        self,
        input_size: int,
        controller_size: int,
        output_size: int,
        memory_n: int = 128,
        memory_m: int = 20
    ):
        super().__init__()
        
        self.controller_size = controller_size
        self.memory_n = memory_n
        self.memory_m = memory_m
        
        # 控制器（LSTM）
        self.controller = nn.LSTMCell(input_size + memory_m, controller_size)
        
        # 输出层
        self.output_layer = nn.Linear(controller_size + memory_m, output_size)
        
        # 读写头参数生成
        self.read_key_layer = nn.Linear(controller_size, memory_m)
        self.read_strength_layer = nn.Linear(controller_size, 1)
        
        self.write_key_layer = nn.Linear(controller_size, memory_m)
        self.write_strength_layer = nn.Linear(controller_size, 1)
        self.erase_layer = nn.Linear(controller_size, memory_m)
        self.add_layer = nn.Linear(controller_size, memory_m)
        
    def forward(
        self,
        x: torch.Tensor,
        prev_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        前向传播
        
        参数:
            x: 输入 [batch, input_size]
            prev_state: (controller_state, controller_cell, read_weight, memory)
        """
        controller_state, controller_cell, prev_read_weight, memory = prev_state
        
        # 从记忆中读取
        read_vector = torch.matmul(prev_read_weight.unsqueeze(1), memory).squeeze(1)
        
        # 控制器输入 = 输入 + 读取的记忆
        controller_input = torch.cat([x, read_vector], dim=-1)
        
        # 更新控制器状态
        new_state, new_cell = self.controller(controller_input, (controller_state, controller_cell))
        
        # 生成读取参数
        read_key = self.read_key_layer(new_state)
        read_strength = F.softplus(self.read_strength_layer(new_state))
        
        # 计算读取权重
        read_weight = self._content_addressing(memory, read_key, read_strength)
        
        # 读取新内容
        read_vector = torch.matmul(read_weight.unsqueeze(1), memory).squeeze(1)
        
        # 生成输出
        output = self.output_layer(torch.cat([new_state, read_vector], dim=-1))
        
        # 生成写入参数
        write_key = self.write_key_layer(new_state)
        write_strength = F.softplus(self.write_strength_layer(new_state))
        erase_vector = torch.sigmoid(self.erase_layer(new_state))
        add_vector = torch.tanh(self.add_layer(new_state))
        
        # 计算写入权重
        write_weight = self._content_addressing(memory, write_key, write_strength)
        
        # 写入记忆
        new_memory = self._write_memory(memory, write_weight, erase_vector, add_vector)
        
        return output, read_weight, write_weight, (new_state, new_cell, read_weight, new_memory)
    
    def _content_addressing(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        strength: torch.Tensor
    ) -> torch.Tensor:
        """基于内容的寻址"""
        key_norm = F.normalize(key, dim=-1)
        memory_norm = F.normalize(memory, dim=-1)
        
        similarity = torch.matmul(memory_norm, key_norm.unsqueeze(-1)).squeeze(-1)
        weights = F.softmax(similarity * strength.squeeze(-1), dim=-1)
        
        return weights
    
    def _write_memory(
        self,
        memory: torch.Tensor,
        write_weight: torch.Tensor,
        erase_vector: torch.Tensor,
        add_vector: torch.Tensor
    ) -> torch.Tensor:
        """写入记忆: M_t = M_{t-1} ∘ (1 - w_t * e_t) + w_t * a_t"""
        w = write_weight.unsqueeze(-1)
        e = erase_vector.unsqueeze(1)
        a = add_vector.unsqueeze(1)
        
        erase_term = memory * (1 - w * e)
        add_term = w * a
        
        return erase_term + add_term
```

### 3.2 可微分逻辑编程

```python
"""
可微分归纳逻辑编程 (Differentiable ILP)
从示例中学习逻辑规则
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class DifferentiableRuleLearner(nn.Module):
    """
    可微分规则学习器
    
    学习形式如: target(X, Y) :- body1(X, Z), body2(Z, Y)
    的规则
    """
    
    def __init__(
        self,
        num_predicates: int,
        max_body_atoms: int = 2,
        num_rules: int = 10,
        embedding_dim: int = 32
    ):
        super().__init__()
        
        self.num_predicates = num_predicates
        self.max_body_atoms = max_body_atoms
        self.num_rules = num_rules
        
        # 规则头选择
        self.rule_head_logits = nn.Parameter(torch.randn(num_rules, num_predicates))
        
        # 规则体选择
        self.rule_body_logits = nn.Parameter(
            torch.randn(num_rules, max_body_atoms, num_predicates)
        )
        
        # 规则权重
        self.rule_weights = nn.Parameter(torch.ones(num_rules))
        
    def soft_select(self, logits: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """软选择：使用softmax模拟离散选择"""
        return F.softmax(logits / temperature, dim=-1)
    
    def forward(
        self,
        facts: torch.Tensor,
        num_constants: int,
        temperature: float = 0.5
    ) -> torch.Tensor:
        """
        前向推理
        
        参数:
            facts: [num_predicates, num_constants, num_constants] 初始事实
            num_constants: 常量数量
        """
        # 规则头的软选择
        rule_heads = self.soft_select(self.rule_head_logits, temperature)
        
        # 规则体的软选择
        rule_bodies = self.soft_select(
            self.rule_body_logits.view(-1, self.num_predicates),
            temperature
        ).view(self.num_rules, self.max_body_atoms, self.num_predicates)
        
        # 软推理
        inferred = self._soft_forward_chain(facts, rule_heads, rule_bodies, num_constants)
        
        return inferred
    
    def _soft_forward_chain(
        self,
        facts: torch.Tensor,
        rule_heads: torch.Tensor,
        rule_bodies: torch.Tensor,
        num_constants: int
    ) -> torch.Tensor:
        """
        可微分的前向链推理
        
        使用软逻辑:
        - AND: a * b
        - OR: 1 - (1-a)*(1-b)
        """
        inferred = facts.clone()
        
        for rule_idx in range(self.num_rules):
            body_preds = rule_bodies[rule_idx]
            
            body_values = []
            
            for atom_idx in range(self.max_body_atoms):
                pred_weights = body_preds[atom_idx]
                
                weighted_fact = torch.sum(
                    pred_weights.view(-1, 1, 1) * inferred,
                    dim=0
                )
                
                body_values.append(weighted_fact)
            
            # 体部合取
            body_conjunction = body_values[0]
            for i in range(1, len(body_values)):
                body_conjunction = body_conjunction * body_values[i]
            
            # 头部赋值
            head_weights = rule_heads[rule_idx]
            
            for pred_idx in range(self.num_predicates):
                update = head_weights[pred_idx] * body_conjunction
                inferred[pred_idx] = 1 - (1 - inferred[pred_idx]) * (1 - update)
        
        return inferred
```

---

## 4. 视觉推理与组合泛化

### 4.1 神经网络模块（Neural Module Networks）

**费曼法比喻**：想象用**乐高积木**搭建房子。每个积木（模块）有特定功能：有的当墙壁，有的当屋顶。Neural Module Networks就像给AI一套视觉推理的乐高积木——每个模块执行特定的视觉操作（找红色物体、数数量、比较大小），然后根据问题组合这些模块。

```python
"""
Neural Module Networks (NMN) - 简化实现
视觉问答的模块化方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import json


class NMNModule(nn.Module):
    """NMN模块基类"""
    
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim


class SceneModule(NMNModule):
    """Scene模块：返回图像的整体特征"""
    
    def __init__(self, dim: int = 256):
        super().__init__(dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, dim, 3, padding=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        参数:
            image: [batch, 3, H, W]
        返回:
            features: [batch, dim, 7, 7]
        """
        return self.image_encoder(image)


class FindModule(NMNModule):
    """Find模块：在图像中查找特定类型的物体"""
    
    def __init__(self, dim: int = 256, num_classes: int = 10):
        super().__init__(dim)
        
        self.query_encoder = nn.Linear(dim, dim)
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, 1, 1)
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        query_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            image_features: [batch, dim, H, W]
            query_vector: [batch, dim]
        返回:
            attention: [batch, H, W]
        """
        batch, dim, H, W = image_features.shape
        
        query_encoded = self.query_encoder(query_vector)
        query_expanded = query_encoded.view(batch, dim, 1, 1).expand(-1, -1, H, W)
        
        combined = torch.cat([image_features, query_expanded], dim=1)
        
        attention = self.spatial_attention(combined).squeeze(1)
        attention = torch.sigmoid(attention)
        
        return attention


class CountModule(NMNModule):
    """Count模块：计算注意力图中的物体数量"""
    
    def __init__(self, dim: int = 256, max_count: int = 10):
        super().__init__(dim)
        
        self.max_count = max_count
        
        self.count_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_count + 1)
        )
        
        self.feature_extractor = nn.Conv2d(1, dim, 1)
    
    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        """
        参数:
            attention: [batch, H, W]
        返回:
            count_logits: [batch, max_count+1]
        """
        features = self.feature_extractor(attention.unsqueeze(1))
        count_logits = self.count_network(features)
        
        return count_logits


class NeuralModuleNetwork(nn.Module):
    """完整的神经模块网络"""
    
    def __init__(
        self,
        dim: int = 256,
        num_attributes: int = 10,
        num_relations: int = 4,
        max_count: int = 10
    ):
        super().__init__()
        
        self.dim = dim
        
        # 创建模块库
        self.modules = nn.ModuleDict({
            'scene': SceneModule(dim),
            'find': FindModule(dim, num_attributes),
            'count': CountModule(dim, max_count),
        })
        
        # 属性嵌入
        self.attribute_embeddings = nn.Embedding(num_attributes, dim)
        
    def execute_program(
        self,
        image: torch.Tensor,
        program: List[Dict[str, Any]]
    ) -> torch.Tensor:
        """
        执行程序
        
        参数:
            image: [batch, 3, H, W]
            program: 程序列表
        """
        # 初始化场景
        scene_features = self.modules['scene'](image)
        
        # 执行栈
        stack = []
        
        for step in program:
            module_name = step['module']
            params = step.get('params', {})
            
            if module_name == 'find':
                attr_id = params['attribute']
                query_vec = self.attribute_embeddings(torch.tensor([attr_id]))
                attention = self.modules['find'](scene_features, query_vec)
                stack.append(attention)
                
            elif module_name == 'count':
                attention = stack.pop()
                count_logits = self.modules['count'](attention)
                return count_logits
        
        return stack[-1] if stack else None
```

### 4.2 Slot Attention

```python
"""
Slot Attention: 学习对象的表示
实现组合泛化的关键模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Slot Attention模块
    
    将输入特征分解为K个独立槽位（物体）的表示
    
    就像把杂乱房间里的物品分类整理到不同的盒子里
    """
    
    def __init__(
        self,
        num_slots: int = 7,
        slot_dim: int = 64,
        input_dim: int = 64,
        num_iterations: int = 3,
        mlp_hidden_dim: int = 128
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations
        
        self.scale = slot_dim ** -0.5
        
        # 槽位初始化参数
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, slot_dim))
        
        # 输入投影
        self.norm_inputs = nn.LayerNorm(input_dim)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.to_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(input_dim, slot_dim, bias=False)
        
        # MLP更新
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim)
        )
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        参数:
            inputs: [batch, num_inputs, input_dim]
        返回:
            slots: [batch, num_slots, slot_dim]
        """
        batch_size = inputs.shape[0]
        
        # 归一化输入
        inputs = self.norm_inputs(inputs)
        
        # 计算K和V
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        
        # 初始化槽位
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(
            batch_size, self.num_slots, self.slot_dim,
            device=inputs.device, dtype=inputs.dtype
        )
        
        # 多次迭代细化槽位
        for _ in range(self.num_iterations):
            slots_prev = slots
            
            slots = self.norm_slots(slots)
            
            # 计算注意力
            q = self.to_q(slots)
            
            attn_logits = torch.einsum('bnd,bmd->bnm', q, k) * self.scale
            attn = F.softmax(attn_logits, dim=-1)
            
            # 加权平均
            attn_norm = attn / (attn.sum(dim=-2, keepdim=True) + 1e-8)
            
            # 更新槽位
            updates = torch.einsum('bnm,bmd->bnd', attn_norm, v)
            
            # MLP更新
            slots = slots_prev + self.mlp(updates)
        
        return slots
```

---

## 5. 大语言模型的推理能力

### 5.1 Chain-of-Thought Prompting

**费曼法比喻**：想象解数学题时你在**草稿纸**上一步步写下过程。Chain-of-Thought就是让AI也"写草稿"——在给出最终答案前，先生成中间的推理步骤。

```python
"""
Chain-of-Thought Prompting 实现
让大语言模型生成推理链
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import random


class ChainOfThoughtPrompting:
    """
    Chain-of-Thought提示生成器
    
    核心思想: 在答案之前先生成推理步骤
    就像解数学题时先写草稿
    """
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.demonstrations = []
    
    def add_demonstration(self, question: str, reasoning: str, answer: str):
        """添加示例"""
        self.demonstrations.append({
            'question': question,
            'reasoning': reasoning,
            'answer': answer
        })
    
    def build_prompt(
        self,
        question: str,
        use_demonstrations: bool = True,
        trigger_phrase: str = "让我们一步步思考："
    ) -> str:
        """构建Chain-of-Thought提示"""
        prompt_parts = []
        
        prompt_parts.append("请按照以下格式回答问题，展示你的推理过程：\n")
        
        # 添加示例
        if use_demonstrations and self.demonstrations:
            for demo in self.demonstrations:
                prompt_parts.append(f"问题: {demo['question']}")
                prompt_parts.append(f"推理: {demo['reasoning']}")
                prompt_parts.append(f"答案: {demo['answer']}")
                prompt_parts.append("")
        
        # 添加新问题
        prompt_parts.append(f"问题: {question}")
        prompt_parts.append(f"推理: {trigger_phrase}")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        question: str,
        max_length: int = 512,
        temperature: float = 0.7,
        use_demonstrations: bool = True
    ) -> Dict[str, str]:
        """生成带推理链的回答"""
        if self.model is None:
            return self._simulate_generation(question)
        
        prompt = self.build_prompt(question, use_demonstrations)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        result = self._parse_output(generated_text, prompt)
        
        return result
    
    def _simulate_generation(self, question: str) -> Dict[str, str]:
        """模拟生成"""
        
        if "鸡" in question and "蛋" in question:
            reasoning = """首先，我需要确定鸡的数量和每只鸡每天下的蛋数。
从问题中可知：有5只鸡，每只鸡每天下2个蛋。
所以每天的总蛋数是：5 × 2 = 10个蛋
一周有7天，因此一周的蛋数是：10 × 7 = 70个蛋"""
            answer = "70"
        else:
            reasoning = "让我分析这个问题..."
            answer = "需要根据具体计算"
        
        return {
            'question': question,
            'reasoning': reasoning,
            'answer': answer,
            'full_response': f"推理: {reasoning}\n答案: {answer}"
        }
    
    def _parse_output(self, generated: str, prompt: str) -> Dict[str, str]:
        """解析生成的输出"""
        response = generated[len(prompt):].strip()
        
        lines = response.split('\n')
        
        reasoning_lines = []
        answer = ""
        
        for line in lines:
            if line.strip().startswith("答案:") or line.strip().startswith("Answer:"):
                answer = line.split(":", 1)[1].strip()
                break
            else:
                reasoning_lines.append(line)
        
        reasoning = "\n".join(reasoning_lines).strip()
        
        return {
            'reasoning': reasoning,
            'answer': answer,
            'full_response': response
        }
    
    def self_consistency(
        self,
        question: str,
        num_samples: int = 10,
        temperature: float = 0.7
    ) -> Dict:
        """
        Self-Consistency解码
        
        多次采样，选择最一致的答案
        """
        answers = []
        
        for _ in range(num_samples):
            result = self.generate(question, temperature=temperature)
            answers.append(result['answer'])
        
        from collections import Counter
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]
        
        return {
            'answer': most_common[0],
            'confidence': most_common[1] / num_samples,
            'all_answers': answers,
            'answer_distribution': dict(answer_counts)
        }
```

### 5.2 ReAct：推理与行动结合

**费曼法比喻**：想象你在厨房里做菜。你不是想好所有步骤再做，而是**边想边做**——切菜的时候思考下一步，发现没有盐就去拿盐。ReAct就是让AI这种"边想边做"的能力。

```python
"""
ReAct: Synergizing Reasoning and Acting in Language Models
推理与行动的结合
"""

import torch
import torch.nn as nn
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    """动作类型"""
    THINK = "think"
    SEARCH = "search"
    CALCULATE = "calculate"
    LOOKUP = "lookup"
    FINISH = "finish"


@dataclass
class Action:
    """动作定义"""
    action_type: ActionType
    content: str
    result: Any = None


@dataclass
class ReActStep:
    """ReAct的一个步骤"""
    thought: str
    action: Action
    observation: str = ""


class ToolKit:
    """工具箱：ReAct Agent可以使用的工具"""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, str] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str
    ):
        """注册工具"""
        self.tools[name] = func
        self.tool_descriptions[name] = description
    
    def execute(self, tool_name: str, *args, **kwargs) -> Any:
        """执行工具"""
        if tool_name not in self.tools:
            return f"错误: 工具'{tool_name}'不存在"
        
        try:
            result = self.tools[tool_name](*args, **kwargs)
            return result
        except Exception as e:
            return f"执行错误: {str(e)}"
    
    def get_tool_list(self) -> str:
        """获取工具列表"""
        descriptions = []
        for name, desc in self.tool_descriptions.items():
            descriptions.append(f"- {name}: {desc}")
        return "\n".join(descriptions)


class ReActAgent:
    """
    ReAct Agent：结合推理和行动的AI Agent
    
    ReAct循环:
    思考(Thought) -> 行动(Action) -> 观察(Observation) -> ...
    """
    
    def __init__(
        self,
        model=None,
        tokenizer=None,
        toolkit: Optional[ToolKit] = None,
        max_iterations: int = 10
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.toolkit = toolkit or ToolKit()
        self.max_iterations = max_iterations
        self.trajectory: List[ReActStep] = []
    
    def build_prompt(self, question: str) -> str:
        """构建ReAct提示"""
        prompt_parts = []
        
        prompt_parts.append("""你是一个智能助手，需要通过交替思考和行动来解决问题。

可用工具:
""")
        prompt_parts.append(self.toolkit.get_tool_list())
        
        prompt_parts.append("""

格式要求:
思考: [你的推理过程]
行动: [工具名(参数)]
观察: [工具执行结果]
""")
        
        # 添加历史轨迹
        if self.trajectory:
            prompt_parts.append("历史:\n")
            for step in self.trajectory:
                prompt_parts.append(f"思考: {step.thought}")
                prompt_parts.append(f"行动: {step.action.action_type.value}({step.action.content})")
                prompt_parts.append(f"观察: {step.observation}")
                prompt_parts.append("")
        
        prompt_parts.append(f"问题: {question}\n")
        prompt_parts.append("思考:")
        
        return "\n".join(prompt_parts)
    
    def parse_action(self, action_str: str) -> Action:
        """解析行动字符串"""
        try:
            if '(' in action_str and action_str.endswith(')'):
                action_type_str, content = action_str.split('(', 1)
                content = content[:-1]
                
                action_type = ActionType(action_type_str.strip().lower())
                return Action(action_type, content)
            else:
                return Action(ActionType.THINK, action_str)
        except Exception as e:
            return Action(ActionType.THINK, f"解析错误: {action_str}")
    
    def execute_action(self, action: Action) -> str:
        """执行行动"""
        if action.action_type == ActionType.THINK:
            return "[思考完成]"
        
        elif action.action_type == ActionType.SEARCH:
            if 'search' in self.toolkit.tools:
                return str(self.toolkit.execute('search', action.content))
            else:
                return f"[搜索结果: {action.content}]"
        
        elif action.action_type == ActionType.CALCULATE:
            try:
                result = eval(action.content)
                return str(result)
            except:
                return "计算错误"
        
        elif action.action_type == ActionType.LOOKUP:
            return f"[查找结果: {action.content}]"
        
        elif action.action_type == ActionType.FINISH:
            return f"[完成: {action.content}]"
        
        else:
            return f"[未知行动类型: {action.action_type}]"
    
    def run(self, question: str) -> Dict:
        """执行ReAct循环"""
        print("=" * 60)
        print(f"ReAct Agent - 问题: {question}")
        print("=" * 60)
        
        self.trajectory = []
        
        for iteration in range(self.max_iterations):
            print(f"\n--- 迭代 {iteration + 1} ---")
            
            # 模拟生成思考和行动
            if len(self.trajectory) == 0:
                thought = "我需要先分析问题，确定需要使用哪些工具。"
                action = Action(ActionType.SEARCH, "相关信息")
            elif len(self.trajectory) == 1:
                thought = "根据搜索结果，我需要进一步验证和计算。"
                action = Action(ActionType.CALCULATE, "5 * 2 * 7")
            else:
                thought = "我已经收集了足够的信息，可以给出答案。"
                action = Action(ActionType.FINISH, "70个蛋")
            
            print(f"思考: {thought}")
            print(f"行动: {action.action_type.value}({action.content})")
            
            # 执行行动
            observation = self.execute_action(action)
            print(f"观察: {observation}")
            
            # 记录步骤
            step = ReActStep(thought, action, observation)
            self.trajectory.append(step)
            
            # 检查是否结束
            if action.action_type == ActionType.FINISH:
                print("\n" + "=" * 60)
                print("任务完成!")
                print("=" * 60)
                break
        
        return {
            'question': question,
            'trajectory': self.trajectory,
            'num_steps': len(self.trajectory),
            'final_answer': self.trajectory[-1].action.content if self.trajectory else None
        }
```

---

## 6. 可解释性与因果推理

### 6.1 神经概念学习器

```python
"""
神经概念学习器 (Neural Concept Learner)
学习可解释的概念表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class ConceptLayer(nn.Module):
    """
    概念层：学习人类可解释的概念
    
    每个神经元对应一个语义概念（如"红色"、"圆形"、"大的"）
    """
    
    def __init__(
        self,
        input_dim: int,
        num_concepts: int,
        concept_names: List[str] = None
    ):
        super().__init__()
        
        self.num_concepts = num_concepts
        self.concept_names = concept_names or [f"concept_{i}" for i in range(num_concepts)]
        
        # 概念激活网络
        self.concept_activation = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_concepts),
            nn.Sigmoid()  # 每个概念的存在概率 [0, 1]
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回概念激活值和解释
        """
        concepts = self.concept_activation(x)
        
        # 生成解释
        explanations = []
        for i, batch_concepts in enumerate(concepts):
            active = [
                self.concept_names[j] 
                for j, score in enumerate(batch_concepts) 
                if score > 0.5
            ]
            explanations.append(active)
        
        return concepts, explanations


class ExplainableClassifier(nn.Module):
    """
    可解释分类器
    
    分类决策基于明确的概念
    """
    
    def __init__(
        self,
        input_dim: int,
        num_concepts: int,
        num_classes: int,
        concept_names: List[str] = None
    ):
        super().__init__()
        
        # 特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # 概念层
        self.concept_layer = ConceptLayer(256, num_concepts, concept_names)
        
        # 基于概念的分类
        self.classifier = nn.Linear(num_concepts, num_classes)
        
    def forward(self, x: torch.Tensor) -> Dict:
        """
        前向传播，返回分类结果和解释
        """
        # 提取特征
        features = self.feature_extractor(x)
        
        # 获取概念激活
        concepts, explanations = self.concept_layer(features)
        
        # 分类
        logits = self.classifier(concepts)
        probs = F.softmax(logits, dim=-1)
        
        # 生成每个预测的解释
        detailed_explanations = []
        for i, (concept_vals, pred_class) in enumerate(zip(concepts, probs.argmax(dim=-1))):
            class_concepts = self.classifier.weight[pred_class]
            top_concept_indices = torch.argsort(class_concepts.abs(), descending=True)[:3]
            
            detailed_explanations.append({
                'predicted_class': pred_class.item(),
                'confidence': probs[i, pred_class].item(),
                'active_concepts': explanations[i],
                'important_concepts_for_class': [
                    (self.concept_layer.concept_names[idx.item()], 
                     class_concepts[idx].item())
                    for idx in top_concept_indices
                ]
            })
        
        return {
            'logits': logits,
            'probs': probs,
            'concepts': concepts,
            'explanations': detailed_explanations
        }
    
    def explain_prediction(self, x: torch.Tensor) -> str:
        """生成人类可读的预测解释"""
        result = self.forward(x)
        exp = result['explanations'][0]
        
        explanation_text = f"""预测结果:
- 类别: {exp['predicted_class']}
- 置信度: {exp['confidence']:.2%}

激活的概念: {', '.join(exp['active_concepts'])}

对决策最重要的概念:
"""
        for concept, weight in exp['important_concepts_for_class']:
            direction = "支持" if weight > 0 else "反对"
            explanation_text += f"- {concept}: {direction}预测 (权重={weight:.3f})\n"
        
        return explanation_text
```

---

## 7. 实战案例：神经符号问答系统

```python
"""
神经符号问答系统：完整实战案例

结合：
- 知识图谱嵌入（TransE/RotatE）
- 神经模块网络（视觉推理）
- Chain-of-Thought（推理链）
- ReAct（工具使用）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class NeuroSymbolicQASystem:
    """
    神经符号问答系统
    
    集成多个神经符号组件的综合系统
    """
    
    def __init__(
        self,
        kg_embedding_dim: int = 128,
        visual_dim: int = 256,
        num_reasoning_steps: int = 5
    ):
        super().__init__()
        
        self.kg_embedding_dim = kg_embedding_dim
        self.visual_dim = visual_dim
        self.num_reasoning_steps = num_reasoning_steps
        
        # 1. 知识图谱嵌入模块
        self.entity_embeddings = nn.Embedding(1000, kg_embedding_dim)
        self.relation_embeddings = nn.Embedding(100, kg_embedding_dim)
        
        # 2. 视觉理解模块
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, visual_dim, 3, padding=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # 3. 问题理解模块
        self.question_encoder = nn.LSTM(
            input_size=300,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 4. 推理控制器
        self.reasoning_controller = nn.Sequential(
            nn.Linear(512 + visual_dim + kg_embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4种推理类型
        )
        
        # 5. 答案生成器
        self.answer_generator = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1000)
        )
        
        # 6. 推理链追踪
        self.reasoning_chain = []
    
    def answer_question(
        self,
        question: str,
        image: Optional[torch.Tensor] = None,
        kg_context: Optional[List[Tuple[int, int, int]]] = None
    ) -> Dict:
        """
        回答问题的完整流程
        
        参数:
            question: 问题文本
            image: 可选的图像输入
            kg_context: 可选的知识图谱上下文
        
        返回:
            包含答案和推理过程的dict
        """
        results = {
            'question': question,
            'reasoning_steps': [],
            'final_answer': None,
            'confidence': None
        }
        
        # 步骤1：分析问题类型
        reasoning_type = self._classify_reasoning_type(question)
        results['reasoning_steps'].append({
            'step': 1,
            'action': '问题类型分析',
            'result': reasoning_type
        })
        
        # 步骤2：根据类型选择推理路径
        if reasoning_type == 'knowledge_graph':
            answer, confidence = self._kg_reasoning(question, kg_context)
        elif reasoning_type == 'visual':
            answer, confidence = self._visual_reasoning(question, image)
        elif reasoning_type == 'multi_hop':
            answer, confidence = self._multi_hop_reasoning(question, kg_context)
        else:
            answer, confidence = self._direct_answer(question)
        
        results['final_answer'] = answer
        results['confidence'] = confidence
        results['reasoning_steps'].append({
            'step': 2,
            'action': f'{reasoning_type}推理',
            'result': answer
        })
        
        return results
    
    def _classify_reasoning_type(self, question: str) -> str:
        """分类问题所需的推理类型"""
        # 简化的规则分类
        if any(kw in question for kw in ['谁', '哪里', '什么时候', '是什么']):
            return 'knowledge_graph'
        elif any(kw in question for kw in ['多少', '颜色', '形状']):
            return 'visual'
        elif any(kw in question for kw in ['为什么', '怎么', '原因']):
            return 'multi_hop'
        else:
            return 'direct'
    
    def _kg_reasoning(
        self,
        question: str,
        kg_context: List[Tuple[int, int, int]]
    ) -> Tuple[str, float]:
        """知识图谱推理"""
        # 使用TransE风格的嵌入推理
        # 实际实现需要完整的实体链接和路径搜索
        return "知识图谱答案", 0.85
    
    def _visual_reasoning(
        self,
        question: str,
        image: torch.Tensor
    ) -> Tuple[str, float]:
        """视觉推理"""
        # 使用NMN风格的模块推理
        visual_features = self.visual_encoder(image)
        # ... 进一步处理
        return "视觉答案", 0.78
    
    def _multi_hop_reasoning(
        self,
        question: str,
        kg_context: List[Tuple[int, int, int]]
    ) -> Tuple[str, float]:
        """多跳推理"""
        # 使用Chain-of-Thought风格的多步推理
        return "多跳推理答案", 0.72
    
    def _direct_answer(self, question: str) -> Tuple[str, float]:
        """直接回答"""
        return "直接答案", 0.65


# ==================== 系统演示 ====================

def demo_neuro_symbolic_qa():
    """演示神经符号问答系统"""
    
    print("=" * 60)
    print("神经符号问答系统演示")
    print("=" * 60)
    
    # 创建系统
    system = NeuroSymbolicQASystem()
    
    # 测试问题
    questions = [
        "爱因斯坦发现了什么理论？",
        "图片中有多少个红色立方体？",
        "为什么天空是蓝色的？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        result = system.answer_question(question)
        print(f"答案: {result['final_answer']}")
        print(f"置信度: {result['confidence']}")
        print("推理过程:")
        for step in result['reasoning_steps']:
            print(f"  步骤 {step['step']}: {step['action']} -> {step['result']}")


if __name__ == "__main__":
    demo_neuro_symbolic_qa()
```

---

## 8. 本章小结

### 核心概念回顾

1. **神经符号AI**：融合神经网络的感知能力和符号系统的推理能力

2. **知识图谱嵌入**：
   - TransE: 将关系视为平移
   - RotatE: 将关系视为复数空间中的旋转
   - ComplEx: 复数双线性模型

3. **神经程序合成**：从输入-输出示例中学习程序

4. **视觉推理**：
   - Neural Module Networks: 组合式视觉推理
   - Slot Attention: 对象中心表示学习

5. **大语言模型推理**：
   - Chain-of-Thought: 生成推理步骤
   - ReAct: 推理与行动结合

6. **可解释性**：概念学习器和因果推理

### 神经符号AI的未来

- **更紧密的融合**：神经网络和符号系统不再是独立的组件
- **自动知识获取**：从非结构化数据中自动构建知识图谱
- **通用推理引擎**：一个系统处理多种推理任务
- **可信赖的AI**：可解释、可验证、可控制的智能系统

---

## 9. 练习题

### 基础题 (3道)

**练习1**：解释神经符号AI的核心思想。为什么要结合神经网络和符号系统？

**练习2**：比较知识图谱嵌入与GNN节点嵌入的区别。

**练习3**：Chain-of-Thought prompting为什么能提高大语言模型的推理能力？

### 数学推导题 (3道)

**练习4**：推导TransE的评分函数及其损失函数。解释为什么使用边界排名损失。

**练习5**：证明RotatE在复数空间中的旋转性质。说明为什么旋转比平移更适合某些关系。

**练习6**：推导神经定理证明中的可微分前向链规则。解释软逻辑如何实现可微分推理。

### 编程题 (3道)

**练习7**：实现TransE知识图谱嵌入，并在小规模知识图谱上进行链接预测。

```python
# 提示：使用以下三元组数据
triples = [
    (0, 0, 1),  # Alice -friend-> Bob
    (1, 0, 2),  # Bob -friend-> Carol
    (2, 0, 3),  # Carol -friend-> David
    (0, 1, 2),  # Alice -colleague-> Carol
]
# 预测：(Alice, friend, ?) 的答案
```

**练习8**：实现简化版Neural Module Network进行视觉问答。

**练习9**：实现Chain-of-Thought风格的推理链生成器，支持Self-Consistency解码。

---

## 10. 参考文献

1. Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. *Advances in Neural Information Processing Systems*, 26.

2. Sun, Z., Deng, Z. H., Nie, J. Y., & Tang, J. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *International Conference on Learning Representations*.

3. Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

4. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations*.

5. Andreas, J., Rohrbach, M., Darrell, T., & Klein, D. (2016). Neural module networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 39-48.

6. Garcez, A. d., & Lamb, L. C. (2020). Neurosymbolic AI: The 3rd wave. *arXiv preprint arXiv:2012.05876*.

7. Marcus, G. (2020). The next decade in AI: Four steps towards robust artificial intelligence. *arXiv preprint arXiv:2002.06177*.

8. Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex embeddings for simple link prediction. *International Conference on Machine Learning*, 2071-2080.

9. Rocktäschel, T., & Riedel, S. (2017). End-to-end differentiable proving. *Advances in Neural Information Processing Systems*, 30.

10. Lake, B. M., & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. *International Conference on Machine Learning*, 2873-2882.

---

*本章完成于 2026-03-27*
*累计正文字数: ~16,000字*
*代码行数: ~1,800行*
