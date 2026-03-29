"""
证据深度学习(EDL)实现
基于Sensoy et al. (2018) for Classification
基于Amini et al. (2020) for Regression
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def dirichlet_loss(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算Dirichlet损失函数
    
    Args:
        alpha: Dirichlet参数 [batch_size, num_classes]
        y: 真实标签（one-hot） [batch_size, num_classes]
    
    Returns:
        损失值
    """
    # 总证据强度
    alpha_0 = alpha.sum(dim=1, keepdim=True)
    
    # 负面对数似然
    nll = torch.lgamma(alpha_0) - torch.lgamma(alpha).sum(dim=1, keepdim=True) + \
          ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(alpha_0))).sum(dim=1, keepdim=True)
    
    # KL散度正则化（相对于均匀先验）
    beta = torch.ones_like(alpha)
    kl = torch.lgamma(alpha_0) - torch.lgamma(alpha).sum(dim=1, keepdim=True) + \
         torch.lgamma(beta.sum(dim=1, keepdim=True)) - torch.lgamma(beta).sum(dim=1, keepdim=True) + \
         ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(alpha_0))).sum(dim=1, keepdim=True)
    
    loss = (nll + kl).mean()
    return loss


class EvidentialClassificationNet(nn.Module):
    """证据深度学习分类网络"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list = [128, 64],
        num_classes: int = 10
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.alpha_layer = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出Dirichlet参数
        
        Returns:
            alpha: Dirichlet浓度参数 [batch_size, num_classes]
        """
        features = self.feature_layers(x)
        # 使用softplus确保alpha > 1（有证据）
        alpha = F.softplus(self.alpha_layer(features)) + 1.0
        return alpha
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并计算不确定性
        
        Returns:
            probs: 期望概率 [batch_size, num_classes]
            total_uncertainty: 总不确定性（预测熵）
            vacuity: 认知不确定性（基于证据强度）
        """
        alpha = self.forward(x)
        alpha_0 = alpha.sum(dim=1, keepdim=True)
        
        # 期望概率
        probs = alpha / alpha_0
        
        # 总不确定性（预测熵）
        total_uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        # 认知不确定性：证据越少，不确定性越高
        # vacuity = K / alpha_0，其中K是类别数
        num_classes = alpha.shape[1]
        vacuity = num_classes / alpha_0.squeeze()
        
        return probs, total_uncertainty, vacuity


class NIGLoss(nn.Module):
    """Normal-Inverse-Gamma损失函数（用于回归）"""
    
    def __init__(self, lambda_reg: float = 0.01):
        super().__init__()
        self.lambda_reg = lambda_reg
    
    def forward(
        self, 
        gamma: torch.Tensor, 
        nu: torch.Tensor, 
        alpha: torch.Tensor, 
        beta: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        计算NIG损失
        
        Args:
            gamma: 预测均值 [batch_size, 1]
            nu: 精度参数 [batch_size, 1]
            alpha: 形状参数 [batch_size, 1]
            beta: 尺度参数 [batch_size, 1]
            y: 真实值 [batch_size, 1]
        """
        # 确保参数有效
        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + 1.01  # alpha > 1
        beta = F.softplus(beta) + 1e-6
        
        # NLL损失
        omega = 2 * beta * (1 + nu)
        nll = 0.5 * torch.log(np.pi / nu) - alpha * torch.log(2 * beta) + \
              (alpha + 0.5) * torch.log((y - gamma)**2 * nu + omega) + \
              torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
        
        # 正则化项
        reg = torch.abs(y - gamma)
        
        return (nll + self.lambda_reg * reg).mean()


class EvidentialRegressionNet(nn.Module):
    """证据深度学习回归网络"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list = [128, 64]
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # 输出NIG的四个参数
        self.gamma_layer = nn.Linear(prev_dim, 1)  # 均值
        self.nu_layer = nn.Linear(prev_dim, 1)     # 精度
        self.alpha_layer = nn.Linear(prev_dim, 1)  # 形状
        self.beta_layer = nn.Linear(prev_dim, 1)   # 尺度
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播，输出NIG参数
        
        Returns:
            gamma, nu, alpha, beta: NIG分布参数
        """
        features = self.feature_layers(x)
        
        gamma = self.gamma_layer(features)
        nu = F.softplus(self.nu_layer(features)) + 1e-6
        alpha = F.softplus(self.alpha_layer(features)) + 1.01
        beta = F.softplus(self.beta_layer(features)) + 1e-6
        
        return gamma, nu, alpha, beta
    
    def predict(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测并计算不确定性
        
        Returns:
            pred_mean: 预测均值
            aleatoric: 偶然不确定性
            epistemic: 认知不确定性
        """
        gamma, nu, alpha, beta = self.forward(x)
        
        # 预测均值
        pred_mean = gamma
        
        # 偶然不确定性：数据噪声
        aleatoric = beta / (alpha - 1)
        
        # 认知不确定性：模型不确定性
        epistemic = beta / (nu * (alpha - 1))
        
        return pred_mean, aleatoric, epistemic


class EvidentialTrainer:
    """证据深度学习训练器"""
    
    def __init__(
        self, 
        model: nn.Module,
        task_type: str = 'classification',
        lr: float = 1e-3
    ):
        self.model = model
        self.task_type = task_type
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        if task_type == 'regression':
            self.criterion = NIGLoss()
    
    def train_epoch(self, train_loader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            self.optimizer.zero_grad()
            
            if self.task_type == 'classification':
                alpha = self.model(batch_x)
                # 将标签转为one-hot
                y_onehot = F.one_hot(batch_y, num_classes=alpha.shape[1]).float()
                loss = dirichlet_loss(alpha, y_onehot)
            else:
                gamma, nu, alpha, beta = self.model(batch_x)
                loss = self.criterion(gamma, nu, alpha, beta, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)


def demo_evidential_classification():
    """证据深度学习分类演示"""
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # 生成数据
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=10, n_classes=3, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    model = EvidentialClassificationNet(
        input_dim=20,
        hidden_dims=[64, 32],
        num_classes=3
    )
    
    trainer = EvidentialTrainer(model, task_type='classification')
    
    # 训练
    for epoch in range(50):
        loss = trainer.train_epoch(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 测试
    model.eval()
    with torch.no_grad():
        probs, total_unc, vacuity = model.predict(X_test_tensor)
        pred_classes = probs.argmax(dim=-1)
        accuracy = (pred_classes == y_test_tensor).float().mean()
    
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"平均总不确定性: {total_unc.mean():.4f}")
    print(f"平均认知不确定性(vacuity): {vacuity.mean():.4f}")
    
    # 可视化不确定性分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(total_unc.numpy(), bins=30, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Total Uncertainty (Entropy)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Total Uncertainty')
    
    axes[1].hist(vacuity.numpy(), bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1].set_xlabel('Epistemic Uncertainty (Vacuity)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Epistemic Uncertainty')
    
    plt.tight_layout()
    plt.savefig('evidential_classification_demo.png', dpi=150)
    plt.show()
    
    print("\nDemo completed! See evidential_classification_demo.png")


def demo_evidential_regression():
    """证据深度学习回归演示"""
    import matplotlib.pyplot as plt
    
    # 生成数据
    np.random.seed(42)
    X_train = np.linspace(-3, 3, 50)
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, 50)
    
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 创建模型
    model = EvidentialRegressionNet(input_dim=1, hidden_dims=[64, 64])
    trainer = EvidentialTrainer(model, task_type='regression')
    
    # 训练
    for epoch in range(200):
        loss = trainer.train_epoch(train_loader)
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    # 测试
    X_test = np.linspace(-5, 5, 200)
    X_test_tensor = torch.FloatTensor(X_test).reshape(-1, 1)
    
    model.eval()
    with torch.no_grad():
        pred_mean, aleatoric, epistemic = model.predict(X_test_tensor)
    
    # 绘制
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 预测
    axes[0].scatter(X_train, y_train, c='red', alpha=0.5, label='Training data')
    axes[0].plot(X_test, pred_mean.numpy(), 'b-', label='Prediction')
    total_std = np.sqrt(aleatoric.numpy() + epistemic.numpy())
    axes[0].fill_between(
        X_test.flatten(),
        (pred_mean.numpy() - 2*total_std).flatten(),
        (pred_mean.numpy() + 2*total_std).flatten(),
        alpha=0.3, label='95% confidence'
    )
    axes[0].set_title('Prediction with Total Uncertainty')
    axes[0].legend()
    
    # 偶然不确定性
    axes[1].plot(X_test, aleatoric.numpy(), 'g-', label='Aleatoric')
    axes[1].set_title('Aleatoric Uncertainty')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Variance')
    
    # 认知不确定性
    axes[2].plot(X_test, epistemic.numpy(), 'm-', label='Epistemic')
    axes[2].set_title('Epistemic Uncertainty')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Variance')
    
    plt.tight_layout()
    plt.savefig('evidential_regression_demo.png', dpi=150)
    plt.show()
    
    print("\nDemo completed! See evidential_regression_demo.png")


if __name__ == "__main__":
    print("=== Evidential Deep Learning Classification Demo ===")
    demo_evidential_classification()
    
    print("\n=== Evidential Deep Learning Regression Demo ===")
    demo_evidential_regression()
