"""
MC-Dropout不确定性估计实现
基于Gal & Ghahramani (2016)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MCDropoutNet(nn.Module):
    """支持MC-Dropout的神经网络"""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        task_type: str = 'regression'
    ):
        super().__init__()
        self.task_type = task_type
        self.dropout_rate = dropout_rate
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)  # 关键：使用Dropout层
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 输出层
        if task_type == 'regression':
            # 回归：输出均值和方差
            self.mean_layer = nn.Linear(prev_dim, output_dim)
            self.var_layer = nn.Linear(prev_dim, output_dim)
        else:
            # 分类：输出logits
            self.logit_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor, dropout: bool = True) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            dropout: 是否启用Dropout（测试时设为True用于MC采样）
        
        Returns:
            预测输出
        """
        features = self.feature_layers(x)
        
        if self.task_type == 'regression':
            mean = self.mean_layer(features)
            # 使用softplus确保方差为正
            var = F.softplus(self.var_layer(features)) + 1e-6
            return torch.cat([mean, var], dim=-1)
        else:
            return self.logit_layer(features)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC-Dropout预测，返回均值、偶然不确定性和认知不确定性
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            n_samples: MC采样次数
        
        Returns:
            mean: 预测均值 [batch_size, output_dim]
            aleatoric_unc: 偶然不确定性 [batch_size, output_dim]
            epistemic_unc: 认知不确定性 [batch_size, output_dim]
        """
        self.train()  # 关键：保持train模式以启用Dropout
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                if self.task_type == 'regression':
                    output = self.forward(x, dropout=True)
                    pred_mean = output[:, :output.shape[1]//2]
                    predictions.append(pred_mean)
                else:
                    logits = self.forward(x, dropout=True)
                    probs = F.softmax(logits, dim=-1)
                    predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, output_dim]
        
        # 计算统计量
        pred_mean = predictions.mean(dim=0)
        pred_var = predictions.var(dim=0)
        
        if self.task_type == 'regression':
            # 偶然不确定性：预测方差的平均值
            aleatoric_unc = predictions.var(dim=0, unbiased=False).mean(dim=0, keepdim=True).T
            # 认知不确定性：预测均值的方差
            epistemic_unc = pred_var
        else:
            # 分类：使用预测熵
            aleatoric_unc = None
            epistemic_unc = pred_var
        
        self.eval()
        return pred_mean, aleatoric_unc, epistemic_unc


class MCDropoutTrainer:
    """MC-Dropout模型训练器"""
    
    def __init__(
        self, 
        model: MCDropoutNet,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """计算损失函数"""
        if self.model.task_type == 'regression':
            # 分离均值和方差
            pred_mean = predictions[:, :predictions.shape[1]//2]
            pred_var = predictions[:, predictions.shape[1]//2:]
            
            # 负对数似然（考虑异方差噪声）
            nll = 0.5 * torch.log(pred_var) + 0.5 * (targets - pred_mean)**2 / pred_var
            return nll.mean()
        else:
            # 交叉熵损失
            return F.cross_entropy(predictions, targets)
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            predictions = self.model(batch_x, dropout=True)
            loss = self.compute_loss(predictions, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)


def demo_mc_dropout_regression():
    """MC-Dropout回归演示"""
    import matplotlib.pyplot as plt
    
    # 生成训练数据
    np.random.seed(42)
    X_train = np.linspace(-3, 3, 20)
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, 20)
    
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    # 创建模型
    model = MCDropoutNet(
        input_dim=1, 
        hidden_dims=[64, 64], 
        output_dim=1,
        dropout_rate=0.1
    )
    
    trainer = MCDropoutTrainer(model)
    
    # 训练
    for epoch in range(500):
        trainer.optimizer.zero_grad()
        pred = model(X_train_tensor)
        pred_mean = pred[:, :1]
        pred_var = pred[:, 1:]
        
        loss = 0.5 * torch.log(pred_var) + 0.5 * (y_train_tensor - pred_mean)**2 / pred_var
        loss = loss.mean()
        
        loss.backward()
        trainer.optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # 测试
    X_test = np.linspace(-5, 5, 200)
    X_test_tensor = torch.FloatTensor(X_test).reshape(-1, 1)
    
    mean, aleatoric, epistemic = model.predict_with_uncertainty(X_test_tensor, n_samples=100)
    
    # 绘制
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, c='red', alpha=0.5, label='Training data')
    plt.plot(X_test, mean.numpy(), 'b-', label='Prediction')
    
    total_unc = np.sqrt(aleatoric.numpy() + epistemic.numpy())
    plt.fill_between(
        X_test.flatten(), 
        (mean.numpy() - 2*total_unc).flatten(), 
        (mean.numpy() + 2*total_unc).flatten(),
        alpha=0.3, label='95% confidence'
    )
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('MC-Dropout Uncertainty Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('mc_dropout_demo.png', dpi=150)
    plt.show()
    
    print("\nDemo completed! See mc_dropout_demo.png")


if __name__ == "__main__":
    demo_mc_dropout_regression()
