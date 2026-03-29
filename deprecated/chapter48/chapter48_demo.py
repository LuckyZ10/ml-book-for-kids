"""
不确定性量化完整示例：对比MC-Dropout、深度集成和EDL
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# 导入我们的实现
from mc_dropout import MCDropoutNet, MCDropoutTrainer
from deep_ensemble import DeepEnsemble
from evidential_deep_learning import (
    EvidentialClassificationNet, 
    EvidentialRegressionNet,
    EvidentialTrainer
)


def compare_uncertainty_methods_regression():
    """对比三种不确定性估计方法在回归任务上的表现"""
    
    print("=" * 60)
    print("回归任务：不确定性方法对比")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    X_train = np.linspace(-3, 3, 30)
    y_train = np.sin(X_train) + np.random.normal(0, 0.1, 30)
    
    # 测试点（包括训练区域内和外）
    X_test = np.linspace(-5, 5, 200)
    
    X_train_tensor = torch.FloatTensor(X_train).reshape(-1, 1)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test).reshape(-1, 1)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 1. MC-Dropout
    print("\n[1/3] 训练 MC-Dropout...")
    mc_model = MCDropoutNet(1, [64, 64], 1, dropout_rate=0.1)
    mc_trainer = MCDropoutTrainer(mc_model)
    
    for epoch in range(300):
        mc_trainer.optimizer.zero_grad()
        pred = mc_model(X_train_tensor)
        pred_mean = pred[:, :1]
        pred_var = pred[:, 1:]
        loss = (0.5 * torch.log(pred_var) + 0.5 * (y_train_tensor - pred_mean)**2 / pred_var).mean()
        loss.backward()
        mc_trainer.optimizer.step()
    
    mc_mean, mc_aleatoric, mc_epistemic = mc_model.predict_with_uncertainty(X_test_tensor, n_samples=100)
    mc_mean = mc_mean.numpy().flatten()
    mc_total_std = np.sqrt(mc_aleatoric.numpy() + mc_epistemic.numpy()).flatten()
    
    # 2. Deep Ensemble
    print("[2/3] 训练 Deep Ensemble...")
    ensemble = DeepEnsemble(1, [64, 64], 1, n_models=5)
    ensemble.fit(train_loader, epochs=200)
    ens_mean, _, ens_model_unc = ensemble.predict_with_uncertainty(X_test_tensor)
    ens_mean = ens_mean.numpy().flatten()
    ens_std = np.sqrt(ens_model_unc.numpy()).flatten() if ens_model_unc is not None else np.zeros_like(ens_mean)
    
    # 3. Evidential Deep Learning
    print("[3/3] 训练 Evidential Deep Learning...")
    edl_model = EvidentialRegressionNet(1, [64, 64])
    edl_trainer = EvidentialTrainer(edl_model, task_type='regression')
    
    for epoch in range(300):
        edl_trainer.train_epoch(train_loader)
    
    edl_model.eval()
    with torch.no_grad():
        edl_mean, edl_aleatoric, edl_epistemic = edl_model.predict(X_test_tensor)
    
    edl_mean = edl_mean.numpy().flatten()
    edl_total_std = np.sqrt(edl_aleatoric.numpy() + edl_epistemic.numpy()).flatten()
    
    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    methods = [
        ('MC-Dropout', mc_mean, mc_total_std),
        ('Deep Ensemble', ens_mean, ens_std),
        ('EDL', edl_mean, edl_total_std)
    ]
    
    for idx, (name, mean, std) in enumerate(methods):
        # 上行：预测
        axes[0, idx].scatter(X_train, y_train, c='red', alpha=0.5, s=30, label='Training data')
        axes[0, idx].plot(X_test, np.sin(X_test), 'g--', alpha=0.5, label='True function')
        axes[0, idx].plot(X_test, mean, 'b-', label='Prediction')
        axes[0, idx].fill_between(
            X_test, mean - 2*std, mean + 2*std, alpha=0.3, label='95% CI'
        )
        axes[0, idx].axvline(x=-3, color='gray', linestyle=':', alpha=0.5)
        axes[0, idx].axvline(x=3, color='gray', linestyle=':', alpha=0.5)
        axes[0, idx].set_title(f'{name} - Prediction')
        axes[0, idx].set_ylim(-2, 2)
        if idx == 0:
            axes[0, idx].legend(fontsize=8)
        
        # 下行：不确定性
        axes[1, idx].plot(X_test, std, 'purple', linewidth=2)
        axes[1, idx].axvline(x=-3, color='gray', linestyle=':', alpha=0.5)
        axes[1, idx].axvline(x=3, color='gray', linestyle=':', alpha=0.5)
        axes[1, idx].fill_between([-3, 3], [0, 0], [std.max(), std.max()], alpha=0.1, color='green')
        axes[1, idx].set_title(f'{name} - Uncertainty')
        axes[1, idx].set_xlabel('x')
        axes[1, idx].set_ylabel('Std Dev')
    
    plt.suptitle('Uncertainty Quantification Methods Comparison (Regression)', fontsize=14)
    plt.tight_layout()
    plt.savefig('uncertainty_comparison_regression.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ 对比图已保存: uncertainty_comparison_regression.png")


def compare_uncertainty_methods_classification():
    """对比三种不确定性估计方法在分类任务上的表现（包括OOD检测）"""
    
    print("\n" + "=" * 60)
    print("分类任务：不确定性方法对比 + OOD检测")
    print("=" * 60)
    
    # 生成训练数据（2D二分类）
    np.random.seed(42)
    X_train, y_train = make_moons(n_samples=400, noise=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
    # OOD数据（来自不同分布）
    X_ood = np.random.uniform(-2, 3, (100, 2))
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_ood_tensor = torch.FloatTensor(X_ood)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    results = {}
    
    # 1. MC-Dropout
    print("\n[1/3] MC-Dropout...")
    mc_model = MCDropoutNet(2, [64, 64], 2, dropout_rate=0.2, task_type='classification')
    mc_trainer = MCDropoutTrainer(mc_model)
    
    for epoch in range(100):
        for batch_x, batch_y in train_loader:
            mc_trainer.optimizer.zero_grad()
            logits = mc_model(batch_x)
            loss = nn.CrossEntropyLoss()(logits, batch_y)
            loss.backward()
            mc_trainer.optimizer.step()
    
    _, _, mc_val_unc = mc_model.predict_with_uncertainty(X_val_tensor, n_samples=50)
    _, _, mc_ood_unc = mc_model.predict_with_uncertainty(X_ood_tensor, n_samples=50)
    
    results['MC-Dropout'] = {
        'in': mc_val_unc.numpy(),
        'ood': mc_ood_unc.numpy()
    }
    
    # 2. Deep Ensemble
    print("[2/3] Deep Ensemble...")
    ensemble = DeepEnsemble(2, [64, 64], 2, n_models=5, task_type='classification')
    ensemble.fit(train_loader, epochs=100)
    
    _, ens_data_unc, ens_model_unc = ensemble.predict_with_uncertainty(X_val_tensor)
    _, _, ens_ood_unc = ensemble.predict_with_uncertainty(X_ood_tensor)
    
    results['Deep Ensemble'] = {
        'in': ens_model_unc.numpy() if ens_model_unc is not None else np.zeros(len(X_val)),
        'ood': ens_ood_unc.numpy() if ens_ood_unc is not None else np.zeros(len(X_ood))
    }
    
    # 3. Evidential Deep Learning
    print("[3/3] Evidential Deep Learning...")
    edl_model = EvidentialClassificationNet(2, [64, 64], 2)
    edl_trainer = EvidentialTrainer(edl_model, task_type='classification')
    
    for epoch in range(100):
        edl_trainer.train_epoch(train_loader)
    
    edl_model.eval()
    with torch.no_grad():
        _, _, edl_val_unc = edl_model.predict(X_val_tensor)
        _, _, edl_ood_unc = edl_model.predict(X_ood_tensor)
    
    results['EDL'] = {
        'in': edl_val_unc.numpy(),
        'ood': edl_ood_unc.numpy()
    }
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, data) in enumerate(results.items()):
        axes[idx].hist(data['in'], bins=20, alpha=0.7, label='In-distribution', color='blue')
        axes[idx].hist(data['ood'], bins=20, alpha=0.7, label='OOD', color='red')
        axes[idx].set_xlabel('Uncertainty')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(name)
        axes[idx].legend()
        
        # 计算简单的分离度
        separation = np.mean(data['ood']) - np.mean(data['in'])
        axes[idx].text(0.5, 0.95, f'Separation: {separation:.3f}',
                      transform=axes[idx].transAxes, ha='center', va='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('OOD Detection: Uncertainty Distribution Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('uncertainty_comparison_ood.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ 对比图已保存: uncertainty_comparison_ood.png")


def demo_active_learning():
    """演示基于不确定性的主动学习"""
    
    print("\n" + "=" * 60)
    print("主动学习演示")
    print("=" * 60)
    
    from mc_dropout import MCDropoutNet
    
    # 生成数据
    np.random.seed(42)
    X_full = np.random.uniform(-3, 3, (500, 2))
    y_full = (X_full[:, 0]**2 + X_full[:, 1]**2 < 4).astype(int)
    
    X_full_tensor = torch.FloatTensor(X_full)
    y_full_tensor = torch.LongTensor(y_full)
    
    # 初始随机标注50个样本
    labeled_idx = np.random.choice(500, 50, replace=False)
    unlabeled_idx = list(set(range(500)) - set(labeled_idx))
    
    # 主动学习循环
    test_acc_history = []
    n_labeled_history = []
    
    for iteration in range(5):
        print(f"\n迭代 {iteration+1}/5:")
        print(f"  已标注样本: {len(labeled_idx)}")
        
        # 在已标注数据上训练
        X_labeled = X_full_tensor[labeled_idx]
        y_labeled = y_full_tensor[labeled_idx]
        
        model = MCDropoutNet(2, [32, 32], 2, task_type='classification')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(100):
            optimizer.zero_grad()
            logits = model(X_labeled)
            loss = nn.CrossEntropyLoss()(logits, y_labeled)
            loss.backward()
            optimizer.step()
        
        # 评估（用所有数据作为测试集）
        model.eval()
        with torch.no_grad():
            logits = model(X_full_tensor)
            pred = logits.argmax(dim=-1)
            acc = (pred == y_full_tensor).float().mean().item()
        
        test_acc_history.append(acc)
        n_labeled_history.append(len(labeled_idx))
        print(f"  测试准确率: {acc:.4f}")
        
        # 主动采样：选择不确定性最高的50个样本
        if len(unlabeled_idx) > 0:
            X_unlabeled = X_full_tensor[unlabeled_idx]
            
            # 计算不确定性
            model.train()
            predictions = []
            with torch.no_grad():
                for _ in range(20):
                    logits = model(X_unlabeled)
                    probs = torch.softmax(logits, dim=-1)
                    predictions.append(probs)
            
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=-1)
            
            # 选择不确定性最高的样本
            top_uncertain_idx = entropy.argsort(descending=True)[:50]
            selected_idx = [unlabeled_idx[i] for i in top_uncertain_idx.tolist()]
            
            labeled_idx = np.concatenate([labeled_idx, selected_idx])
            unlabeled_idx = list(set(unlabeled_idx) - set(selected_idx))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 学习曲线
    axes[0].plot(n_labeled_history, test_acc_history, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Labeled Samples')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_title('Active Learning: Performance vs Labeled Data')
    axes[0].grid(True, alpha=0.3)
    
    # 最终决策边界
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid)
    
    model.eval()
    with torch.no_grad():
        logits = model(grid_tensor)
        grid_pred = logits.argmax(dim=-1).numpy().reshape(xx.shape)
    
    axes[1].contourf(xx, yy, grid_pred, alpha=0.3, cmap='RdYlBu')
    axes[1].scatter(X_full[labeled_idx[:50], 0], X_full[labeled_idx[:50], 1], 
                    c='red', marker='o', s=50, edgecolors='k', label='Initial 50')
    axes[1].scatter(X_full[labeled_idx[50:], 0], X_full[labeled_idx[50:], 1], 
                    c='green', marker='^', s=50, edgecolors='k', label='Active selected')
    axes[1].set_title('Final Decision Boundary & Selected Samples')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('active_learning_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ 主动学习演示图已保存: active_learning_demo.png")


if __name__ == "__main__":
    # 运行所有演示
    compare_uncertainty_methods_regression()
    compare_uncertainty_methods_classification()
    demo_active_learning()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("生成的文件:")
    print("  - uncertainty_comparison_regression.png")
    print("  - uncertainty_comparison_ood.png")
    print("  - active_learning_demo.png")
    print("=" * 60)
