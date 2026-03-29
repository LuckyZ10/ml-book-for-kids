# 第五十六章 神经架构搜索进阶——AutoML的未来

> *"让AI自己设计AI，这是一场关于创造的革命。"*

## 开篇故事：从手工设计到自动创造

想象一下，你是一名建筑师。

在20年前，设计一座摩天大楼意味着无数张图纸、无数次计算、无数个不眠之夜。每一个支撑结构、每一个承重墙的位置，都需要工程师们反复推敲。

但今天，你有了一个神奇的助手。你告诉它："我想要一座300米高的大楼，能抵御8级地震，能耗要低，还要在预算内。"然后，它生成了100种设计方案，每一种都在不同方面达到了完美的平衡。这就是**神经架构搜索（Neural Architecture Search, NAS）**带给我们的革命。

从2017年Google的AutoML首次展示自动设计的神经网络在图像分类上超越人类专家设计的成果开始，NAS已经从学术研究的前沿走向了工业应用的核心。本章将带你深入这场"让AI自己设计AI"的技术革命最前沿。

---

## 56.1 可微分架构搜索的进化：从DARTS到DARTS+

### 56.1.1 回顾：DARTS的核心思想

在第三十七章中，我们学习了DARTS（Differentiable Architecture Search）如何通过**连续松弛（Continuous Relaxation）**将离散的架构搜索问题转化为可微分的优化问题。让我们快速回顾一下核心公式：

对于一个包含$N$个节点的计算单元（cell），每条边$(i,j)$上的操作$o$被赋予一个架构参数$\alpha_o^{(i,j)}$。通过softmax松弛，操作的输出被表示为所有候选操作的加权和：

$$\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)$$

这个巧妙的设计允许我们使用**梯度下降**同时优化网络权重$w$和架构参数$\alpha$：

$$\min_\alpha \mathcal{L}_{val}(w^*(\alpha), \alpha) \quad \text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{train}(w, \alpha)$$

这就是**双层优化（Bi-level Optimization）**问题——在训练集上优化权重，在验证集上优化架构。

### 56.1.2 DARTS的困境：性能崩溃

然而，DARTS并非完美。研究者们很快发现了一个严重的问题：**性能崩溃（Performance Collapse）**。

想象你在一家自助餐厅选择食物。最初，你尝试了各种菜肴。但渐渐地，你发现自己越来越倾向于选择那些"最容易吃"的食物——比如白米饭，而不是需要剥壳的虾。这不是因为它们最营养，而是因为它们最"省力"。

在DARTS中，这种"省力"的操作就是**跳跃连接（skip connection）**。跳跃连接几乎不增加计算量，却能让梯度顺畅地反向传播。于是，架构参数会逐渐偏向选择跳跃连接，而不是更有意义的卷积操作。结果就是：**搜索出来的架构一堆跳跃连接，性能惨不忍睹**。

数学上，这表现为架构参数的**熵坍缩**——softmax输出的分布变得极度尖锐，几乎所有概率质量都集中在跳跃连接上。

### 56.1.3 DARTS+：早期停止的智慧

2019年，Liang等人提出了**DARTS+**，一个简单的解决方案：**提前停止搜索**。

他们的观察很巧妙：性能崩溃不是突然发生的，而是一个渐进的过程。如果在崩溃发生之前就停止搜索，就能获得一个健康的架构。

DARTS+引入了两个关键规则：

1. **跳跃连接数量限制**：当任何一个边选择的跳跃连接数量超过预设阈值$K$时，强制停止搜索
2. **架构熵监控**：监控架构分布的熵，当熵低于阈值时触发早停

$$\text{Entropy}(\alpha) = -\sum_{o \in \mathcal{O}} p_o \log p_o, \quad \text{其中 } p_o = \frac{\exp(\alpha_o)}{\sum_{o'} \exp(\alpha_{o'})}$$

这就像是在自助餐厅设置一个规则："如果你连续三次选择白米饭，我们就提醒你去尝尝别的菜。"

```python
"""
DARTS+：带早期停止的可微分架构搜索
包含架构熵监控和跳跃连接限制
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
from copy import deepcopy


class DartsPlusArchitect(nn.Module):
    """
    DARTS+架构优化器：在原始DARTS基础上增加早停机制
    """
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        num_skips_threshold: int = 2,
        entropy_threshold: float = 0.5,
        arch_lr: float = 3e-4,
        weight_lr: float = 0.025,
        momentum: float = 0.9,
        weight_decay: float = 3e-4
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        
        # DARTS+早停阈值
        self.num_skips_threshold = num_skips_threshold
        self.entropy_threshold = entropy_threshold
        
        # 架构参数（需要注册为参数才能优化）
        self.arch_parameters = self._build_arch_parameters()
        for name, param in self.arch_parameters.items():
            self.register_parameter(name, param)
        
        # 优化器
        self.arch_optimizer = torch.optim.Adam(
            self.arch_parameters.values(),
            lr=arch_lr,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
        
        self.weight_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=weight_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.weight_optimizer, T_max=50
        )
        
        # 早停状态
        self.early_stop_triggered = False
        self.stop_reason = None
        
    def _build_arch_parameters(self) -> Dict[str, nn.Parameter]:
        """构建架构参数：每个边对应一组操作权重"""
        arch_params = {}
        # 假设有14条边，每条边有8个候选操作
        num_edges = 14
        num_ops = 8
        
        for edge_id in range(num_edges):
            # 随机初始化架构参数
            param = nn.Parameter(
                torch.randn(num_ops) * 0.001
            )
            arch_params[f"arch_{edge_id}"] = param
            
        return arch_params
    
    def compute_arch_entropy(self, alpha: torch.Tensor) -> float:
        """
        计算架构分布的熵
        
        Args:
            alpha: 架构参数向量
            
        Returns:
            熵值（越高表示分布越均匀）
        """
        probs = F.softmax(alpha, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        return entropy.item()
    
    def count_skip_connections(self) -> int:
        """统计当前架构中跳跃连接的数量"""
        num_skips = 0
        skip_op_index = 3  # 假设跳跃连接是第4个操作
        
        for name, alpha in self.arch_parameters.items():
            probs = F.softmax(alpha, dim=-1)
            selected_op = probs.argmax().item()
            if selected_op == skip_op_index:
                num_skips += 1
                
        return num_skips
    
    def check_early_stop(self) -> Tuple[bool, str]:
        """
        检查是否应该提前停止
        
        Returns:
            (是否停止, 停止原因)
        """
        if self.early_stop_triggered:
            return True, self.stop_reason
            
        # 检查1：跳跃连接数量
        num_skips = self.count_skip_connections()
        if num_skips > self.num_skips_threshold * len(self.arch_parameters):
            self.early_stop_triggered = True
            self.stop_reason = f"Too many skip connections: {num_skips}"
            return True, self.stop_reason
        
        # 检查2：平均架构熵
        total_entropy = 0
        for alpha in self.arch_parameters.values():
            total_entropy += self.compute_arch_entropy(alpha)
        avg_entropy = total_entropy / len(self.arch_parameters)
        
        if avg_entropy < self.entropy_threshold:
            self.early_stop_triggered = True
            self.stop_reason = f"Low entropy: {avg_entropy:.3f}"
            return True, self.stop_reason
            
        return False, ""
    
    def step(
        self,
        train_data: torch.Tensor,
        train_target: torch.Tensor,
        val_data: torch.Tensor,
        val_target: torch.Tensor
    ) -> Dict[str, float]:
        """
        执行一步双层优化
        
        Args:
            train_data: 训练数据
            train_target: 训练标签
            val_data: 验证数据
            val_target: 验证标签
            
        Returns:
            训练指标字典
        """
        # 检查早停条件
        should_stop, reason = self.check_early_stop()
        if should_stop:
            print(f"Early stop triggered: {reason}")
            return {"stopped": True, "reason": reason}
        
        # ====== 阶段1：更新网络权重（内循环）======
        self.weight_optimizer.zero_grad()
        
        # 前向传播使用当前架构
        output = self.model(train_data, self.arch_parameters)
        train_loss = self.criterion(output, train_target)
        
        train_loss.backward()
        self.weight_optimizer.step()
        
        # ====== 阶段2：更新架构参数（外循环）======
        self.arch_optimizer.zero_grad()
        
        # 在验证集上评估
        with torch.no_grad():
            # 先复制当前权重（无梯度）
            w_prime = {name: param.clone() 
                      for name, param in self.model.named_parameters()}
        
        # 单步权重更新近似
        output = self.model(val_data, self.arch_parameters)
        val_loss = self.criterion(output, val_target)
        
        val_loss.backward()
        self.arch_optimizer.step()
        self.scheduler.step()
        
        # 计算当前架构统计
        avg_entropy = np.mean([
            self.compute_arch_entropy(alpha)
            for alpha in self.arch_parameters.values()
        ])
        
        return {
            "train_loss": train_loss.item(),
            "val_loss": val_loss.item(),
            "avg_entropy": avg_entropy,
            "num_skips": self.count_skip_connections(),
            "stopped": False
        }


class MixedOperation(nn.Module):
    """
    DARTS混合操作：所有候选操作的加权和
    """
    def __init__(self, primitives: List[nn.Module]):
        super().__init__()
        self.ops = nn.ModuleList(primitives)
        
    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征
            alpha: 架构参数
            
        Returns:
            加权和输出
        """
        weights = F.softmax(alpha, dim=-1)
        output = sum(w * op(x) for w, op in zip(weights, self.ops))
        return output


class SkipConnectionRegularizer:
    """
    跳跃连接正则化器：防止DARTS过度选择跳跃连接
    """
    def __init__(
        self,
        skip_penalty: float = 0.1,
        diversity_weight: float = 0.05
    ):
        self.skip_penalty = skip_penalty
        self.diversity_weight = diversity_weight
        
    def __call__(self, arch_parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算正则化损失
        
        Args:
            arch_parameters: 所有架构参数
            
        Returns:
            正则化损失
        """
        total_loss = 0
        skip_op_index = 3  # 假设跳跃连接是第4个操作
        
        for alpha in arch_parameters.values():
            probs = F.softmax(alpha, dim=-1)
            
            # 1. 跳跃连接惩罚
            skip_prob = probs[skip_op_index]
            total_loss += self.skip_penalty * skip_prob
            
            # 2. 多样性奖励（熵最大化）
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            total_loss -= self.diversity_weight * entropy
            
        return total_loss


# ====== 使用示例 ======
def demo_darts_plus():
    """DARTS+演示"""
    print("=" * 60)
    print("DARTS+：带早期停止的可微分架构搜索")
    print("=" * 60)
    
    # 创建模拟模型
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            
        def forward(self, x, arch_params):
            return self.conv(x)
    
    model = DummyModel()
    criterion = nn.CrossEntropyLoss()
    
    # 创建DARTS+优化器
    architect = DartsPlusArchitect(
        model=model,
        criterion=criterion,
        num_skips_threshold=2,
        entropy_threshold=0.3,
        arch_lr=3e-4,
        weight_lr=0.025
    )
    
    print("\n模拟训练过程...")
    
    # 模拟训练
    for epoch in range(10):
        # 模拟数据
        train_data = torch.randn(4, 3, 32, 32)
        train_target = torch.randint(0, 10, (4,))
        val_data = torch.randn(4, 3, 32, 32)
        val_target = torch.randint(0, 10, (4,))
        
        metrics = architect.step(
            train_data, train_target,
            val_data, val_target
        )
        
        if metrics.get("stopped"):
            print(f"\n⏹️  Epoch {epoch}: {metrics['reason']}")
            break
        else:
            print(f"Epoch {epoch}: train_loss={metrics['train_loss']:.3f}, "
                  f"val_loss={metrics['val_loss']:.3f}, "
                  f"entropy={metrics['avg_entropy']:.3f}, "
                  f"skips={metrics['num_skips']}")
    
    print("\n" + "=" * 60)
    print("DARTS+演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo_darts_plus()
