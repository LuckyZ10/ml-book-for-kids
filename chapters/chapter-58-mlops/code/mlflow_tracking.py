"""
MLflow 实验追踪系统完整实现
包含：参数记录、指标追踪、模型保存、可视化对比
"""

import os
import json
import hashlib
import pickle
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    log_loss, mean_squared_error
)
from scipy import stats

# 尝试导入MLflow，如果没有则使用本地模拟
# 实际使用时，请安装: pip install mlflow
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed. Using local simulation mode. "
                  "Install with: pip install mlflow")


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class ExperimentConfig:
    """实验配置类"""
    experiment_name: str
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelConfig:
    """模型配置类"""
    model_type: str  # 'random_forest', 'logistic_regression', 'svm', 'xgboost'
    hyperparameters: Dict[str, Any]
    random_state: int = 42
    
    def get_model(self):
        """根据配置创建模型实例"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.hyperparameters.get('n_estimators', 100),
                max_depth=self.hyperparameters.get('max_depth', None),
                min_samples_split=self.hyperparameters.get('min_samples_split', 2),
                min_samples_leaf=self.hyperparameters.get('min_samples_leaf', 1),
                random_state=self.random_state
            )
        elif self.model_type == 'logistic_regression':
            return LogisticRegression(
                C=self.hyperparameters.get('C', 1.0),
                penalty=self.hyperparameters.get('penalty', 'l2'),
                solver=self.hyperparameters.get('solver', 'lbfgs'),
                max_iter=self.hyperparameters.get('max_iter', 1000),
                random_state=self.random_state
            )
        elif self.model_type == 'svm':
            return SVC(
                C=self.hyperparameters.get('C', 1.0),
                kernel=self.hyperparameters.get('kernel', 'rbf'),
                gamma=self.hyperparameters.get('gamma', 'scale'),
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.hyperparameters.get('n_estimators', 100),
                learning_rate=self.hyperparameters.get('learning_rate', 0.1),
                max_depth=self.hyperparameters.get('max_depth', 3),
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


# =============================================================================
# 本地实验追踪器（MLflow不可用时的备用方案）
# =============================================================================

class LocalExperimentTracker:
    """本地实验追踪器，当MLflow不可用时使用"""
    
    def __init__(self, tracking_dir: str = "./mlruns"):
        self.tracking_dir = tracking_dir
        self.current_run = None
        self.experiments = {}
        self._ensure_dir_exists()
    
    def _ensure_dir_exists(self):
        os.makedirs(self.tracking_dir, exist_ok=True)
    
    def create_experiment(self, name: str) -> str:
        """创建实验，返回实验ID"""
        exp_id = hashlib.md5(name.encode()).hexdigest()[:8]
        exp_path = os.path.join(self.tracking_dir, exp_id)
        os.makedirs(exp_path, exist_ok=True)
        
        self.experiments[exp_id] = {
            'name': name,
            'path': exp_path,
            'created_at': datetime.now().isoformat()
        }
        
        # 保存实验元数据
        with open(os.path.join(exp_path, 'meta.json'), 'w') as f:
            json.dump(self.experiments[exp_id], f, indent=2)
        
        return exp_id
    
    def start_run(self, experiment_id: str, run_name: str = None):
        """开始一个新的运行"""
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        run_path = os.path.join(self.tracking_dir, experiment_id, run_id)
        os.makedirs(run_path, exist_ok=True)
        
        self.current_run = {
            'run_id': run_id,
            'experiment_id': experiment_id,
            'run_name': run_name or run_id,
            'path': run_path,
            'params': {},
            'metrics': {},
            'tags': {},
            'artifacts': [],
            'start_time': datetime.now().isoformat()
        }
        
        print(f"[LocalTracker] Started run: {run_id}")
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_run:
            self.current_run['end_time'] = datetime.now().isoformat()
            # 保存运行数据
            run_file = os.path.join(self.current_run['path'], 'run.json')
            with open(run_file, 'w') as f:
                # 转换为可JSON序列化的格式
                run_data = self.current_run.copy()
                run_data['params'] = {k: str(v) for k, v in run_data['params'].items()}
                run_data['metrics'] = {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                      for k, v in run_data['metrics'].items()}
                json.dump(run_data, f, indent=2)
            print(f"[LocalTracker] Saved run to: {run_file}")
            self.current_run = None
    
    def log_param(self, key: str, value: Any):
        """记录参数"""
        if self.current_run:
            self.current_run['params'][key] = value
    
    def log_params(self, params: Dict[str, Any]):
        """批量记录参数"""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """记录指标"""
        if self.current_run:
            metric_key = f"{key}_step_{step}" if step is not None else key
            self.current_run['metrics'][metric_key] = value
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """批量记录指标"""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_artifact(self, local_path: str):
        """记录 artifact"""
        if self.current_run:
            import shutil
            dst = os.path.join(self.current_run['path'], 'artifacts')
            os.makedirs(dst, exist_ok=True)
            shutil.copy(local_path, dst)
            self.current_run['artifacts'].append(os.path.basename(local_path))
    
    def log_figure(self, figure, artifact_file: str):
        """记录图表"""
        if self.current_run:
            dst = os.path.join(self.current_run['path'], 'artifacts')
            os.makedirs(dst, exist_ok=True)
            filepath = os.path.join(dst, artifact_file)
            figure.savefig(filepath, dpi=150, bbox_inches='tight')
            self.current_run['artifacts'].append(artifact_file)
    
    def log_dict(self, dictionary: Dict, artifact_file: str):
        """记录字典"""
        if self.current_run:
            dst = os.path.join(self.current_run['path'], 'artifacts')
            os.makedirs(dst, exist_ok=True)
            filepath = os.path.join(dst, artifact_file)
            with open(filepath, 'w') as f:
                json.dump(dictionary, f, indent=2)
            self.current_run['artifacts'].append(artifact_file)
    
    def set_tag(self, key: str, value: str):
        """设置标签"""
        if self.current_run:
            self.current_run['tags'][key] = value


# =============================================================================
# 主要实验管理类
# =============================================================================

class MLExperimentManager:
    """
    ML实验管理器
    
    提供统一的接口来管理实验，支持MLflow和本地追踪
    """
    
    def __init__(self, tracking_uri: str = None, use_mlflow: bool = True):
        """
        初始化实验管理器
        
        Args:
            tracking_uri: MLflow追踪URI（如 'http://localhost:5000'）
            use_mlflow: 是否使用MLflow（如果可用）
        """
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        
        if self.use_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient()
            print(f"[ExperimentManager] Using MLflow at: {mlflow.get_tracking_uri()}")
        else:
            self.local_tracker = LocalExperimentTracker()
            print("[ExperimentManager] Using local experiment tracker")
    
    def create_experiment(self, name: str, artifact_location: str = None) -> str:
        """创建实验"""
        if self.use_mlflow:
            try:
                exp_id = mlflow.create_experiment(name, artifact_location=artifact_location)
            except mlflow.exceptions.MlflowException:
                # 实验已存在
                exp = mlflow.get_experiment_by_name(name)
                exp_id = exp.experiment_id
            return exp_id
        else:
            return self.local_tracker.create_experiment(name)
    
    def start_run(self, experiment_id: str = None, run_name: str = None, 
                  nested: bool = False, tags: Dict[str, str] = None):
        """开始运行"""
        if self.use_mlflow:
            return mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                nested=nested,
                tags=tags
            )
        else:
            return self.local_tracker.start_run(experiment_id, run_name)
    
    def end_run(self):
        """结束运行"""
        if self.use_mlflow:
            mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """记录参数"""
        if self.use_mlflow:
            mlflow.log_params(params)
        else:
            self.local_tracker.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """记录指标"""
        if self.use_mlflow:
            mlflow.log_metrics(metrics, step=step)
        else:
            self.local_tracker.log_metrics(metrics, step)
    
    def log_model(self, model, artifact_path: str = "model", 
                  registered_model_name: str = None):
        """记录模型"""
        if self.use_mlflow:
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )
        else:
            # 本地保存模型
            if self.local_tracker.current_run:
                model_path = os.path.join(
                    self.local_tracker.current_run['path'], 
                    artifact_path
                )
                os.makedirs(model_path, exist_ok=True)
                with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
                    pickle.dump(model, f)
    
    def log_artifact(self, local_path: str):
        """记录 artifact"""
        if self.use_mlflow:
            mlflow.log_artifact(local_path)
        else:
            self.local_tracker.log_artifact(local_path)
    
    def log_figure(self, figure, artifact_file: str):
        """记录图表"""
        if self.use_mlflow:
            mlflow.log_figure(figure, artifact_file)
        else:
            self.local_tracker.log_figure(figure, artifact_file)
    
    def set_tag(self, key: str, value: str):
        """设置标签"""
        if self.use_mlflow:
            mlflow.set_tag(key, value)
        else:
            self.local_tracker.set_tag(key, value)


# =============================================================================
# 模型训练与评估类
# =============================================================================

class ModelTrainer:
    """
    模型训练器，集成实验追踪
    """
    
    def __init__(self, experiment_manager: MLExperimentManager):
        self.exp_manager = experiment_manager
        self.results = []
    
    def compute_confidence_interval(self, scores: np.ndarray, confidence: float = 0.95) -> tuple:
        """
        计算置信区间
        
        数学原理：
        对于样本均值，置信区间为：
        CI = mean ± z * (std / sqrt(n))
        
        其中z是标准正态分布的分位数：
        - 95% CI: z = 1.96
        - 99% CI: z = 2.576
        """
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)
        n = len(scores)
        
        # 计算z值
        alpha = 1 - confidence
        z = stats.norm.ppf(1 - alpha/2)
        
        margin = z * (std / np.sqrt(n))
        return mean - margin, mean + margin
    
    def perform_mcnemar_test(self, y_true, y_pred_a, y_pred_b) -> Dict[str, Any]:
        """
        McNemar检验：比较两个分类器的统计显著性
        
        数学原理：
        χ² = (|n01 - n10| - 1)² / (n01 + n10)
        
        其中：
        - n01: 模型A正确，模型B错误的数量
        - n10: 模型A错误，模型B正确的数量
        """
        n01 = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
        n10 = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
        
        if n01 + n10 == 0:
            return {
                'chi2': 0.0,
                'p_value': 1.0,
                'significant': False,
                'n01': n01,
                'n10': n10
            }
        
        # McNemar检验统计量（连续性校正）
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'n01': int(n01),
            'n10': int(n10)
        }
    
    def train_and_evaluate(self, config: ModelConfig, X_train, X_test, 
                           y_train, y_test, experiment_id: str = None,
                           cv_folds: int = 5) -> Dict[str, Any]:
        """
        训练并评估模型，自动记录到实验追踪系统
        
        Args:
            config: 模型配置
            X_train, X_test, y_train, y_test: 训练测试数据
            experiment_id: 实验ID
            cv_folds: 交叉验证折数
        
        Returns:
            包含训练结果的字典
        """
        # 创建运行名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.model_type}_{timestamp}"
        
        # 开始实验记录
        with self.exp_manager.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={'model_type': config.model_type}
        ):
            print(f"\n{'='*60}")
            print(f"训练模型: {config.model_type}")
            print(f"超参数: {config.hyperparameters}")
            print(f"{'='*60}")
            
            # 1. 记录所有参数
            params = {
                'model_type': config.model_type,
                'random_state': config.random_state,
                'cv_folds': cv_folds,
                **{f'hp_{k}': v for k, v in config.hyperparameters.items()}
            }
            self.exp_manager.log_params(params)
            
            # 2. 创建并训练模型
            model = config.get_model()
            
            # 交叉验证
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            cv_ci_low, cv_ci_high = self.compute_confidence_interval(cv_scores)
            
            print(f"\n交叉验证结果 ({cv_folds}折):")
            print(f"  准确率: {cv_mean:.4f} ± {cv_std:.4f}")
            print(f"  95% CI: [{cv_ci_low:.4f}, {cv_ci_high:.4f}]")
            
            # 记录交叉验证指标
            self.exp_manager.log_metrics({
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'cv_accuracy_min': np.min(cv_scores),
                'cv_accuracy_max': np.max(cv_scores),
                'cv_ci_lower': cv_ci_low,
                'cv_ci_upper': cv_ci_high
            })
            
            # 3. 在完整训练集上训练最终模型
            model.fit(X_train, y_train)
            
            # 4. 测试集评估
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # 计算各项指标
            metrics = {
                'test_accuracy': accuracy_score(y_test, y_pred),
                'test_precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'test_recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'test_f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            }
            
            # 多分类的AUC
            if y_prob is not None and len(np.unique(y_test)) > 2:
                try:
                    from sklearn.preprocessing import label_binarize
                    classes = np.unique(y_test)
                    y_test_bin = label_binarize(y_test, classes=classes)
                    metrics['test_roc_auc_ovr'] = roc_auc_score(
                        y_test_bin, y_prob, multi_class='ovr', average='macro'
                    )
                except:
                    pass
            elif y_prob is not None:
                metrics['test_roc_auc'] = roc_auc_score(y_test, y_prob[:, 1])
            
            print(f"\n测试集性能:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            
            self.exp_manager.log_metrics(metrics)
            
            # 5. 生成并保存混淆矩阵图
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xlabel='Predicted label',
                ylabel='True label',
                title=f'Confusion Matrix - {config.model_type}'
            )
            # 添加数值标注
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            self.exp_manager.log_figure(fig, "confusion_matrix.png")
            plt.close()
            
            # 6. 保存分类报告
            report = classification_report(y_test, y_pred, output_dict=True)
            self.exp_manager.log_dict(report, "classification_report.json")
            
            # 7. 保存模型
            self.exp_manager.log_model(model, artifact_path="model")
            
            # 8. 设置标签
            self.exp_manager.set_tag("status", "completed")
            self.exp_manager.set_tag("model_family", config.model_type.split('_')[0])
            
            print(f"\n实验完成！")
            
            return {
                'model': model,
                'config': config,
                'cv_scores': cv_scores,
                'test_metrics': metrics,
                'predictions': y_pred,
                'probabilities': y_prob
            }


# =============================================================================
# 实验对比分析
# =============================================================================

class ExperimentAnalyzer:
    """
    实验分析器：用于比较多个实验结果
    """
    
    def __init__(self, experiment_manager: MLExperimentManager):
        self.exp_manager = experiment_manager
    
    def compare_models(self, results: List[Dict], metric: str = 'test_accuracy') -> pd.DataFrame:
        """
        比较多个模型的性能
        
        Args:
            results: ModelTrainer.train_and_evaluate返回的结果列表
            metric: 用于比较的主要指标
        
        Returns:
            比较DataFrame
        """
        comparison = []
        for result in results:
            comparison.append({
                'model_type': result['config'].model_type,
                'cv_mean': np.mean(result['cv_scores']),
                'cv_std': np.std(result['cv_scores']),
                'test_accuracy': result['test_metrics'].get('test_accuracy', 0),
                'test_f1': result['test_metrics'].get('test_f1_macro', 0),
                'hyperparameters': str(result['config'].hyperparameters)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values(metric, ascending=False)
        
        print(f"\n{'='*80}")
        print("模型性能对比")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        return df
    
    def plot_model_comparison(self, results: List[Dict], save_path: str = None):
        """可视化模型比较"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 交叉验证分数箱线图
        cv_data = [r['cv_scores'] for r in results]
        labels = [r['config'].model_type for r in results]
        
        bp = axes[0].boxplot(cv_data, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        axes[0].set_ylabel('CV Accuracy')
        axes[0].set_title('Cross-Validation Accuracy Comparison')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 2. 测试集指标柱状图
        metrics_to_plot = ['test_accuracy', 'test_precision_macro', 
                          'test_recall_macro', 'test_f1_macro']
        x = np.arange(len(results))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            values = [r['test_metrics'].get(metric, 0) for r in results]
            axes[1].bar(x + i*width, values, width, label=metric.replace('test_', ''))
        
        axes[1].set_ylabel('Score')
        axes[1].set_title('Test Set Metrics Comparison')
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels(labels, rotation=45)
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


# =============================================================================
# 主程序：完整的实验工作流
# =============================================================================

def run_complete_experiment_demo():
    """
    运行完整的实验演示
    
    这个函数展示了如何使用MLExperimentManager进行：
    1. 实验创建和管理
    2. 多模型超参数搜索
    3. 自动实验追踪
    4. 结果对比分析
    """
    print("="*80)
    print("MLOps 实验管理完整演示")
    print("="*80)
    
    # 1. 初始化实验管理器
    # 如果你有MLflow服务器，可以设置: tracking_uri="http://localhost:5000"
    exp_manager = MLExperimentManager(use_mlflow=False)
    
    # 2. 创建实验
    experiment_name = "classification_benchmark_v1"
    experiment_id = exp_manager.create_experiment(experiment_name)
    print(f"\n✓ 创建实验: {experiment_name} (ID: {experiment_id})")
    
    # 3. 准备数据
    print("\n" + "-"*40)
    print("准备数据集...")
    
    # 使用合成数据集（也可以用真实数据集）
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"  训练集: {X_train.shape}")
    print(f"  测试集: {X_test.shape}")
    print(f"  类别分布: {np.bincount(y)}")
    
    # 4. 定义要尝试的模型配置
    model_configs = [
        # 随机森林配置
        ModelConfig(
            model_type='random_forest',
            hyperparameters={'n_estimators': 100, 'max_depth': 10}
        ),
        ModelConfig(
            model_type='random_forest',
            hyperparameters={'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5}
        ),
        # 逻辑回归配置
        ModelConfig(
            model_type='logistic_regression',
            hyperparameters={'C': 1.0, 'solver': 'lbfgs'}
        ),
        ModelConfig(
            model_type='logistic_regression',
            hyperparameters={'C': 0.1, 'penalty': 'l2', 'max_iter': 2000}
        ),
        # SVM配置
        ModelConfig(
            model_type='svm',
            hyperparameters={'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'}
        ),
        # 梯度提升配置
        ModelConfig(
            model_type='gradient_boosting',
            hyperparameters={'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
        ),
    ]
    
    # 5. 训练所有模型
    print("\n" + "-"*40)
    print("开始训练模型...")
    
    trainer = ModelTrainer(exp_manager)
    results = []
    
    for config in model_configs:
        try:
            result = trainer.train_and_evaluate(
                config=config,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                experiment_id=experiment_id,
                cv_folds=5
            )
            results.append(result)
        except Exception as e:
            print(f"✗ 训练失败 {config.model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. 对比分析
    print("\n" + "-"*40)
    print("实验结果分析...")
    
    analyzer = ExperimentAnalyzer(exp_manager)
    comparison_df = analyzer.compare_models(results, metric='test_accuracy')
    
    # 保存对比图
    fig = analyzer.plot_model_comparison(results, save_path='model_comparison.png')
    
    # 7. 统计显著性检验（比较前两个最佳模型）
    if len(results) >= 2:
        print("\n" + "-"*40)
        print("McNemar统计显著性检验:")
        
        best_model = results[0]
        second_best = results[1]
        
        mcnemar_result = trainer.perform_mcnemar_test(
            y_test,
            best_model['predictions'],
            second_best['predictions']
        )
        
        print(f"\n比较: {best_model['config'].model_type} vs {second_best['config'].model_type}")
        print(f"  模型A正确/B错误: {mcnemar_result['n01']}")
        print(f"  模型A错误/B正确: {mcnemar_result['n10']}")
        print(f"  χ²统计量: {mcnemar_result['chi2']:.4f}")
        print(f"  p-value: {mcnemar_result['p_value']:.4f}")
        print(f"  统计显著: {'是' if mcnemar_result['significant'] else '否'} (α=0.05)")
    
    # 8. 保存最终报告
    final_report = {
        'experiment_name': experiment_name,
        'experiment_id': experiment_id,
        'num_models_tested': len(model_configs),
        'num_successful': len(results),
        'best_model': results[0]['config'].model_type if results else None,
        'best_accuracy': float(results[0]['test_metrics']['test_accuracy']) if results else 0,
        'comparison': comparison_df.to_dict('records')
    }
    
    with open('experiment_report.json', 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("实验演示完成！")
    print(f"  实验名称: {experiment_name}")
    print(f"  实验ID: {experiment_id}")
    print(f"  最佳模型: {final_report['best_model']} (准确率: {final_report['best_accuracy']:.4f})")
    print(f"  报告已保存: experiment_report.json")
    print(f"{'='*80}\n")
    
    return results, comparison_df


# =============================================================================
# 如果直接运行此文件
# =============================================================================

if __name__ == "__main__":
    results, comparison = run_complete_experiment_demo()