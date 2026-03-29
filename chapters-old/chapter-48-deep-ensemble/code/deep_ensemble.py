"""
深度集成不确定性估计实现
基于Lakshminarayanan et al. (2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from copy import deepcopy


class EnsembleNet(nn.Module):
    """单个集成成员网络"""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        task_type: str = 'regression'
    ):
        super().__init__()
        self.task_type = task_type
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        if task_type == 'regression':
            self.mean_layer = nn.Linear(prev_dim, output_dim)
            self.var_layer = nn.Linear(prev_dim, output_dim)
        else:
            self.logit_layer = nn.Linear(prev_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layers(x)
        
        if self.task_type == 'regression':
            mean = self.mean_layer(features)
            var = F.softplus(self.var_layer(features)) + 1e-6
            return torch.cat([mean, var], dim=-1)
        else:
            return self.logit_layer(features)


class DeepEnsemble:
    """深度集成模型"""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dims: list = [128, 64],
        output_dim: int = 1,
        n_models: int = 5,
        task_type: str = 'regression'
    ):
        self.n_models = n_models
        self.task_type = task_type
        
        # 创建多个独立模型
        self.models = nn.ModuleList([
            EnsembleNet(input_dim, hidden_dims, output_dim, task_type)
            for _ in range(n_models)
        ])
    
    def fit(
        self, 
        train_loader, 
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        """训练所有集成成员"""
        for i, model in enumerate(self.models):
            print(f"训练集成成员 {i+1}/{self.n_models}...")
            
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
            
            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    
                    predictions = model(batch_x)
                    
                    if self.task_type == 'regression':
                        pred_mean = predictions[:, :predictions.shape[1]//2]
                        pred_var = predictions[:, predictions.shape[1]//2:]
                        loss = 0.5 * torch.log(pred_var) + \
                               0.5 * (batch_y - pred_mean)**2 / pred_var
                        loss = loss.mean()
                    else:
                        loss = F.cross_entropy(predictions, batch_y)
                    
                    loss.backward()
                    optimizer.step()
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        集成预测与不确定性估计
        
        Returns:
            mean: 集成预测均值
            data_uncertainty: 数据不确定性（偶然不确定性）
            model_uncertainty: 模型不确定性（认知不确定性）
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                output = model(x)
                
                if self.task_type == 'regression':
                    pred_mean = output[:, :output.shape[1]//2]
                    predictions.append(pred_mean)
                else:
                    probs = F.softmax(output, dim=-1)
                    predictions.append(probs)
        
        predictions = torch.stack(predictions)  # [n_models, batch_size, output_dim]
        
        # 集成均值
        ensemble_mean = predictions.mean(dim=0)
        
        # 总不确定性
        total_unc = predictions.var(dim=0)
        
        if self.task_type == 'regression':
            # 对于回归，模型不确定性是预测均值的方差
            model_uncertainty = total_unc
            # 数据不确定性需要额外计算
            data_uncertainty = None
        else:
            # 对于分类，使用熵分解
            mean_pred = ensemble_mean
            total_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=-1)
            
            # 平均单个模型的熵
            individual_entropies = -torch.sum(
                predictions * torch.log(predictions + 1e-10), 
                dim=-1
            ).mean(dim=0)
            
            model_uncertainty = total_entropy - individual_entropies
            data_uncertainty = individual_entropies
        
        return ensemble_mean, data_uncertainty, model_uncertainty


def demo_deep_ensemble_classification():
    """深度集成分类演示"""
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    
    # 生成数据
    X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 转换为tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 创建集成模型
    ensemble = DeepEnsemble(
        input_dim=2,
        hidden_dims=[64, 64],
        output_dim=2,
        n_models=5,
        task_type='classification'
    )
    
    # 训练
    ensemble.fit(train_loader, epochs=100)
    
    # 预测
    mean, data_unc, model_unc = ensemble.predict_with_uncertainty(X_test_tensor)
    
    # 计算准确率
    pred_classes = mean.argmax(dim=-1)
    accuracy = (pred_classes == torch.LongTensor(y_test)).float().mean()
    print(f"\n测试集准确率: {accuracy:.4f}")
    print(f"平均模型不确定性: {model_unc.mean():.4f}")
    
    # 可视化
    xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-2, 2, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    
    grid_mean, _, grid_unc = ensemble.predict_with_uncertainty(grid_tensor)
    grid_pred = grid_mean.argmax(dim=-1).numpy().reshape(xx.shape)
    grid_unc = grid_unc.numpy().reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 决策边界
    axes[0].contourf(xx, yy, grid_pred, alpha=0.3, cmap='RdYlBu')
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', edgecolors='k')
    axes[0].set_title('Deep Ensemble Decision Boundary')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    
    # 不确定性
    contour = axes[1].contourf(xx, yy, grid_unc, alpha=0.6, cmap='hot')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='blue', alpha=0.5, edgecolors='k')
    axes[1].set_title('Model Uncertainty (Epistemic)')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    plt.colorbar(contour, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('deep_ensemble_demo.png', dpi=150)
    plt.show()
    
    print("\nDemo completed! See deep_ensemble_demo.png")


if __name__ == "__main__":
    demo_deep_ensemble_classification()
