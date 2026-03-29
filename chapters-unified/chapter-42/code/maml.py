"""
元学习 - MAML实现
第42章：元学习与少样本学习
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict


class MAML:
    """
    Model-Agnostic Meta-Learning
    
    核心思想：学习一个好的初始化参数，使得几步梯度下降就能适应新任务
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, n_inner_steps=5, first_order=False):
        """
        Args:
            model: 基础模型（需要支持functional_forward）
            inner_lr: 内循环学习率 α
            meta_lr: 外循环（元）学习率 β
            n_inner_steps: 内循环梯度下降步数
            first_order: 是否使用一阶近似（更快但可能略差）
        """
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.n_inner_steps = n_inner_steps
        self.first_order = first_order
        
        self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
    
    def inner_loop(self, support_x, support_y, query_x, query_y):
        """
        对一个任务执行内循环适应和外循环评估
        
        内循环：在支持集上梯度下降，得到任务特定的参数
        外循环：在查询集上评估，计算元损失
        
        Args:
            support_x, support_y: 支持集（用于适应）
            query_x, query_y: 查询集（用于评估元损失）
        
        Returns:
            query_loss: 查询集上的损失
            query_acc: 查询集上的准确率
        """
        # 克隆当前模型参数作为快速权重的初始值
        fast_weights = OrderedDict(self.model.named_parameters())
        
        # 内循环：在支持集上执行K步梯度下降
        for step in range(self.n_inner_steps):
            # 使用当前快速权重进行前向传播
            support_pred = self.model.functional_forward(support_x, fast_weights)
            support_loss = nn.CrossEntropyLoss()(support_pred, support_y)
            
            # 计算梯度
            # create_graph=True 允许二阶导数（如果不用一阶近似）
            grads = torch.autograd.grad(
                support_loss, 
                fast_weights.values(),
                create_graph=not self.first_order
            )
            
            # 更新快速权重：θ' = θ - α * ∇L
            fast_weights = OrderedDict(
                (name, param - self.inner_lr * grad)
                for (name, param), grad in zip(fast_weights.items(), grads)
            )
        
        # 外循环：在查询集上评估适应后的模型
        query_pred = self.model.functional_forward(query_x, fast_weights)
        query_loss = nn.CrossEntropyLoss()(query_pred, query_y)
        query_acc = (query_pred.argmax(dim=1) == query_y).float().mean()
        
        return query_loss, query_acc
    
    def train_step(self, batch_tasks):
        """
        一次元训练步骤
        
        Args:
            batch_tasks: 一批任务，每个任务是(support_x, support_y, query_x, query_y)的元组
        
        Returns:
            meta_loss: 平均元损失
            meta_acc: 平均元准确率
        """
        self.meta_optimizer.zero_grad()
        
        meta_loss = 0.0
        meta_acc = 0.0
        
        # 对每个任务执行内循环
        for task in batch_tasks:
            support_x, support_y, query_x, query_y = task
            
            loss, acc = self.inner_loop(support_x, support_y, query_x, query_y)
            meta_loss += loss
            meta_acc += acc
        
        # 平均损失（批次内所有任务）
        meta_loss = meta_loss / len(batch_tasks)
        meta_acc = meta_acc / len(batch_tasks)
        
        # 外循环：用元优化器更新初始参数
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item(), meta_acc.item()


class SimpleCNN(nn.Module):
    """简单的CNN编码器"""
    def __init__(self, in_channels=1, hidden_dim=64, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, hidden_dim),
            self._conv_block(hidden_dim, output_dim),
            nn.AdaptiveAvgPool2d(1)
        )
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
    
    def functional_forward(self, x, params):
        """
        使用给定参数进行前向传播（用于MAML内循环）
        
        Args:
            x: 输入
            params: 有序字典，包含所有参数
        """
        # 简化的实现，实际应用需要完整实现每个层的功能
        # 这里仅作为示例
        for name, module in self.named_children():
            if name == 'encoder':
                for layer_name, layer in module.named_children():
                    # 这里简化处理，实际需要对每一层手动实现
                    pass
        
        # 实际实现中，需要手动用params中的参数替换self.parameters()
        # 这里返回一个占位符
        return torch.randn(x.size(0), 64)  # 示例输出


if __name__ == "__main__":
    print("MAML - Model-Agnostic Meta-Learning")
    print("=" * 50)
    print("\n核心思想：学习一个好的初始化参数")
    print("使得几步梯度下降就能适应新任务")
    print("\n关键公式：")
    print("  内循环: θ' = θ - α∇L(θ)")
    print("  外循环: θ ← θ - β∇L(θ')")
    print("\n使用方法：")
    print("  model = SimpleCNN()")
    print("  maml = MAML(model, inner_lr=0.01, meta_lr=0.001)")
    print("  loss, acc = maml.train_step(batch_tasks)")
