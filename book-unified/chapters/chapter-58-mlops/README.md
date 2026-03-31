# 第五十八章 MLOps——机器学习工程化

> *"将模型从实验室带到生产环境，就像把一架纸飞机变成一架能够载客穿越大洋的波音747。"*

## 本章学习目标

学完本章，你将能够：
- 🔬 使用MLflow追踪实验，确保可复现性
- 🏪 理解特征存储的价值，实现特征工程的标准化
- 📦 掌握模型版本管理和注册中心的架构设计
- 🚀 实施蓝绿部署、金丝雀部署和A/B测试策略
- 📊 使用统计检验（KS检验、卡方检验）检测数据漂移
- 🔄 构建CI/CD流水线，实现ML模型的自动化部署
- ✅ 建立数据验证框架，确保输入数据质量

---

## 59.1 引言：从实验室到工厂

### 59.1.1 为什么90%的模型从未投入生产？

想象你是一位才华横溢的厨师，在后厨精心研发了一道惊世骇俗的新菜品。你邀请了美食评论家品尝，获得了五星好评。但当你试图在连锁餐厅每天供应1000份这道菜时，灾难发生了：

- 食材供应不稳定，今天的番茄和昨天的味道不同
- 厨房设备不同，有的灶台火力不够
- 厨师们按照各自的"理解"做菜，味道千差万别
- 没有记录配方，没人知道最初的版本是什么样的
- 顾客投诉变多，但你不知道问题出在哪里

**这就是机器学习模型的现状**。

2015年，Google的研究团队在NeurIPS发表了一篇震撼业界的论文《Hidden Technical Debt in Machine Learning Systems》。他们指出了一个惊人的事实：**在一个典型的生产ML系统中，真正的ML代码只占整个代码库的一小部分**。就像冰山一角，海面上你看到的是模型训练代码，而海面下是庞大的基础设施：数据收集、特征工程、配置管理、监控系统等。

### 59.1.2 MLOps的诞生

**MLOps（Machine Learning Operations）**是将DevOps实践与机器学习相结合的一门工程学科。它的核心目标是：

1. **可复现性**：任何人、任何时间都能重现模型训练过程
2. **自动化**：从数据到部署的端到端自动化
3. **监控**：持续监控模型性能，及时发现问题
4. **协作**：打通数据科学家、工程师和运维团队之间的壁垒

**费曼比喻**：想象MLOps是汽车制造业的"精益生产"系统。在福特发明流水线之前，汽车是工匠手工打造的，每辆车都不一样。流水线让汽车生产标准化、可重复、高质量。MLOps就是为机器学习打造的"流水线"。

### 59.1.3 ML系统 vs 传统软件系统

| 特性 | 传统软件 | 机器学习系统 |
|------|----------|--------------|
| **失败模式** | 崩溃、报错（明显） | 静默降级（难以察觉） |
| **版本控制** | 代码版本 | 代码+数据+模型版本 |
| **测试** | 单元测试、集成测试 | 模型性能测试、数据验证 |
| **依赖** | 代码库依赖 | 数据依赖+代码依赖 |
| **环境** | 相对静态 | 持续变化的数据分布 |

### 59.1.4 本章结构

我们将按照ML模型生命周期的顺序，深入探讨MLOps的各个方面：

```
┌─────────────────────────────────────────────────────────────┐
│                     MLOps 生命周期                           │
├─────────────────────────────────────────────────────────────┤
│  1. 实验管理 → 2. 特征工程 → 3. 模型训练 → 4. 模型注册       │
│       ↓              ↓              ↓              ↓         │
│  5. 部署策略 → 6. 监控与漂移检测 → 7. 自动化流水线           │
└─────────────────────────────────────────────────────────────┘
```

---

## 59.2 实验管理与可复现性

### 59.2.1 科学家的笔记本：为什么需要实验追踪？

想象你是一位19世纪的化学家，在实验室里进行着突破性的研究。你的实验记录本上应该有什么？
- 实验日期和时间
- 使用的化学品批量和用量
- 实验步骤的详细记录
- 观察到的现象和测量数据
- 当时的室温、湿度等环境因素
- 你的假设和结论

没有这些记录，科学就是不可复现的巫术。

**在机器学习中，情况更糟**：
- 你运行了50次实验，哪个版本效果最好？
- 你修改了学习率，但还有其他参数变了吗？
- 训练数据的版本是什么？预处理步骤呢？
- 你同事想复现你的结果，需要手动配置所有参数？

**费曼比喻**：实验追踪工具（如MLflow）就是科学家的实验室笔记本，但它会自动记录一切——你用了什么"试剂"（数据）、什么"温度"（超参数）、得到了什么"产物"（模型）。而且它是全组共享的，所有人都能看到彼此的实验。

### 59.2.2 MLflow架构详解

MLflow是一个开源的机器学习生命周期管理平台，由Databricks在2018年开源。它包含四个核心组件：

```
┌──────────────────────────────────────────────────────────────┐
│                        MLflow                                 │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   Tracking  │    Projects │    Models   │    Registry      │
│   (追踪)    │   (项目)    │   (模型)    │    (注册中心)     │
├─────────────┴─────────────┴─────────────┴──────────────────┤
│                    后端存储                                   │
│         (文件系统 / 数据库 / S3 / Azure Blob等)              │
└──────────────────────────────────────────────────────────────┘
```

**1. Tracking（实验追踪）**
- 记录参数、指标、模型、代码版本、数据版本
- 支持多种后端存储：本地文件、数据库、云存储
- 可视化对比不同实验

**2. Projects（项目打包）**
- 标准化项目结构
- 依赖管理和环境复现
- 可重复执行

**3. Models（模型格式）**
- 标准化模型打包格式
- 支持多种框架（scikit-learn、PyTorch、TensorFlow等）
- 统一的部署接口

**4. Model Registry（模型注册中心）**
- 模型版本管理
- 模型阶段转换（Staging → Production → Archived）
- 权限控制和审批流程

### 59.2.3 数学基础：实验设计的统计原理

在深入代码之前，让我们理解实验追踪背后的数学原理。

**假设检验框架**：
当我们比较两个模型的性能时，本质上是在进行统计假设检验。

设模型A的准确率为 $p_A$，模型B的准确率为 $p_B$。

零假设 $H_0$：$p_A = p_B$（两个模型性能相同）
备择假设 $H_1$：$p_A \neq p_B$（两个模型性能不同）

使用McNemar检验来比较两个分类器：

$$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$

其中：
- $n_{01}$：模型A正确，模型B错误的样本数
- $n_{10}$：模型A错误，模型B正确的样本数

**置信区间计算**：
对于准确率 $p$，在样本量 $n$ 下的95%置信区间为：

$$p \pm 1.96 \sqrt{\frac{p(1-p)}{n}}$$

这就是为什么我们需要多次实验和交叉验证——单一指标可能具有随机性。

### 59.2.4 完整代码实现：MLflow实验追踪系统

现在让我们实现一个完整的MLflow实验追踪系统：

```python
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
```

---

## 59.3 特征存储与特征工程自动化

### 59.3.1 中央食材库：什么是特征存储？

想象你是一家连锁餐厅的主厨。每天，你需要为不同的菜品准备食材：
- 切洋葱、胡萝卜丁
- 熬制高汤
- 调制秘制酱料

如果没有中央厨房，每家分店都要从头开始准备这些基础食材，既浪费时间，又难以保证味道一致。

**特征存储（Feature Store）**就是机器学习的"中央食材库"。它是一个集中化的存储和管理系统，用于：
1. **存储特征**：保存经过计算的特征值
2. **服务特征**：为模型训练和在线推理提供一致的特征数据
3. **管理特征**：版本控制、血缘追踪、权限管理

**费曼比喻**：特征存储就像图书馆的卡片目录系统。以前，每个研究员（数据科学家）都要自己整理文献卡片（特征）。现在有了统一的卡片目录，所有人都可以查找、借阅（复用）已有的资料，还能贡献自己的发现。

### 59.3.2 训练-服务偏差（Training-Serving Skew）

这是不使用特征存储时最常见的问题。

**场景**：
- 训练时，你用Python计算用户过去30天的平均消费
- 生产环境，Java工程师用不同的逻辑计算同样的特征
- 两个实现有细微差别（边界条件、空值处理等）
- 模型在生产环境表现很差，但你找不到原因

**训练-服务偏差**指的是训练阶段和推理阶段使用的特征计算逻辑不一致，导致模型性能下降。

### 59.3.3 特征存储架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           特征存储架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │   特征定义    │───▶│  特征计算引擎 │───▶│  离线存储    │              │
│  │  (Feature    │    │ (Spark/Flink)│    │ (Data Lake)  │              │
│  │   Views)     │    └──────────────┘    └──────┬───────┘              │
│  └──────────────┘                               │                       │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │                      特征注册中心                            │        │
│  │    (Feature Registry: 元数据、版本、血缘、权限)               │        │
│  └────────────────────────────────────────────────────────────┘        │
│         │                                       │                       │
│         ▼                                       ▼                       │
│  ┌──────────────┐                        ┌──────────────┐              │
│  │   在线存储    │◀───────────────────────│  实时流处理   │              │
│  │  (Redis/     │    特征同步/物化        │  (Kafka/     │              │
│  │   DynamoDB)  │                        │   Flink)     │              │
│  └──────┬───────┘                        └──────────────┘              │
│         │                                                              │
│         ▼                                                              │
│  ┌────────────────────────────────────────────────────────────┐        │
│  │                     特征服务 API                             │        │
│  │  get_historical_features()  ← 训练/批处理                    │        │
│  │  get_online_features()      ← 在线推理                      │        │
│  └────────────────────────────────────────────────────────────┘        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**关键组件**：

1. **离线存储（Offline Store）**
   - 存储历史特征数据
   - 支持时间旅行查询（point-in-time correct join）
   - 技术：Parquet文件、Delta Lake、Snowflake等

2. **在线存储（Online Store）**
   - 低延迟特征查询（<10ms）
   - 存储预计算的实时特征
   - 技术：Redis、DynamoDB、Cassandra等

3. **特征注册中心（Feature Registry）**
   - 特征定义和元数据
   - 血缘追踪（数据来源、依赖关系）
   - 权限控制和治理

4. **特征计算引擎**
   - 批处理：Spark、Hive
   - 流处理：Flink、Spark Streaming

### 59.3.4 Feast开源特征存储实战

Feast是业界最流行的开源特征存储，由Gojek和Tecton开源。让我们实现一个完整的特征存储系统：

```python
"""
特征存储完整实现
模拟Feast的核心功能：特征定义、物化、服务
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Callable, Any
from enum import Enum
import pickle
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# 数据模型
# =============================================================================

class FeatureType(Enum):
    """特征类型"""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOL = "bool"
    TIMESTAMP = "timestamp"


@dataclass
class Entity:
    """
    实体定义
    
    实体是特征的主体，如用户、商品、订单等
    """
    name: str  # 如 "user", "product"
    join_key: str  # 如 "user_id", "product_id"
    description: str = ""
    
    def __hash__(self):
        return hash(self.name)


@dataclass  
class Feature:
    """
    特征定义
    
    描述一个特征的元数据
    """
    name: str
    dtype: FeatureType
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class FeatureView:
    """
    特征视图
    
    一组相关的特征，通常来自同一个数据源
    类似于数据库的"视图"
    """
    name: str
    entities: List[Entity]
    features: List[Feature]
    source: str  # 数据源标识
    ttl: Optional[timedelta] = None  # 特征有效期（Time-To-Live）
    online: bool = True  # 是否启用在线服务
    description: str = ""
    
    def get_feature_names(self) -> List[str]:
        return [f.name for f in self.features]
    
    def get_entity_names(self) -> List[str]:
        return [e.name for e in self.entities]


# =============================================================================
# 离线存储实现
# =============================================================================

class OfflineStore:
    """
    离线特征存储
    
    存储历史特征数据，支持时间旅行查询
    """
    
    def __init__(self, storage_path: str = "./feast_offline"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._data: Dict[str, pd.DataFrame] = {}
    
    def write(self, feature_view: FeatureView, df: pd.DataFrame):
        """
        写入特征数据
        
        Args:
            feature_view: 特征视图
            df: 包含特征的数据框，必须包含：
                - 实体列（如 user_id）
                - 时间戳列（event_timestamp）
                - 特征列
        """
        # 确保必要列存在
        required_cols = [e.join_key for e in feature_view.entities] + ['event_timestamp']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 确保时间戳列是datetime类型
        df = df.copy()
        df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
        
        # 存储数据
        self._data[feature_view.name] = df
        
        # 持久化到Parquet（实际生产环境）
        file_path = os.path.join(self.storage_path, f"{feature_view.name}.parquet")
        df.to_parquet(file_path, index=False)
        
        print(f"[OfflineStore] Written {len(df)} rows to {feature_view.name}")
    
    def get_historical_features(
        self,
        feature_views: List[FeatureView],
        entity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        获取历史特征（Point-in-Time Correct Join）
        
        这是特征存储的核心功能！确保不会出现数据泄漏。
        
        数学原理：
        对于每个(entity, timestamp)对，找到特征视图中：
        - entity匹配
        - event_timestamp ≤ timestamp（训练时间）
        - 最新的记录
        
        Args:
            feature_views: 需要的特征视图列表
            entity_df: 实体数据框，包含 entity_id 和 timestamp 列
        
        Returns:
            合并后的特征数据框
        """
        result = entity_df.copy()
        result['request_timestamp'] = pd.to_datetime(result['timestamp'])
        
        for fv in feature_views:
            if fv.name not in self._data:
                raise ValueError(f"Feature view {fv.name} not found")
            
            feature_df = self._data[fv.name].copy()
            
            # Point-in-Time Correct Join
            # 对于每个entity和request_timestamp，找到最新的有效特征
            merged_rows = []
            
            for _, row in result.iterrows():
                entity_filter = True
                for entity in fv.entities:
                    entity_filter = entity_filter & (feature_df[entity.join_key] == row[entity.join_key])
                
                # 筛选：特征时间戳 ≤ 请求时间戳（防止数据泄漏！）
                time_filter = feature_df['event_timestamp'] <= row['request_timestamp']
                
                valid_features = feature_df[entity_filter & time_filter]
                
                if len(valid_features) > 0:
                    # 取最新的记录
                    latest = valid_features.loc[valid_features['event_timestamp'].idxmax()]
                    merged_rows.append(latest[fv.get_feature_names()].to_dict())
                else:
                    # 无有效特征，填充NaN
                    merged_rows.append({f: np.nan for f in fv.get_feature_names()})
            
            # 合并到结果
            feature_cols = pd.DataFrame(merged_rows)
            result = pd.concat([result.reset_index(drop=True), feature_cols], axis=1)
        
        return result


# =============================================================================
# 在线存储实现
# =============================================================================

class OnlineStore:
    """
    在线特征存储
    
    低延迟的特征查询，使用内存数据库（如Redis）
    这里使用SQLite模拟
    """
    
    def __init__(self, db_path: str = "./feast_online.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()
    
    def _init_tables(self):
        """初始化表结构"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_values (
                feature_view TEXT,
                entity_key TEXT,
                feature_name TEXT,
                feature_value TEXT,
                event_timestamp TEXT,
                PRIMARY KEY (feature_view, entity_key, feature_name)
            )
        """)
        self.conn.commit()
    
    def materialize(self, feature_view: FeatureView, df: pd.DataFrame):
        """
        物化特征到在线存储
        
        将离线计算好的特征同步到在线存储，供实时推理使用
        """
        cursor = self.conn.cursor()
        
        # 构建实体键（多个实体用#连接）
        df = df.copy()
        df['entity_key'] = df[[e.join_key for e in feature_view.entities]].apply(
            lambda x: '#'.join(map(str, x)), axis=1
        )
        
        # 只保存最新的特征值
        df = df.sort_values('event_timestamp').groupby('entity_key').last().reset_index()
        
        for _, row in df.iterrows():
            for feature in feature_view.features:
                value = row.get(feature.name)
                if pd.notna(value):
                    cursor.execute("""
                        INSERT OR REPLACE INTO feature_values
                        (feature_view, entity_key, feature_name, feature_value, event_timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        feature_view.name,
                        row['entity_key'],
                        feature.name,
                        str(value),
                        row['event_timestamp']
                    ))
        
        self.conn.commit()
        print(f"[OnlineStore] Materialized {len(df)} entities for {feature_view.name}")
    
    def get_online_features(
        self,
        feature_views: List[FeatureView],
        entity_rows: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        获取在线特征
        
        用于实时推理，必须保证低延迟（<10ms）
        
        Args:
            feature_views: 特征视图列表
            entity_rows: 实体列表，如 [{'user_id': 'user_123'}]
        
        Returns:
            特征值数据框
        """
        cursor = self.conn.cursor()
        results = []
        
        for row in entity_rows:
            row_result = row.copy()
            
            for fv in feature_views:
                # 构建实体键
                entity_key = '#'.join(str(row.get(e.join_key, '')) for e in fv.entities)
                
                # 查询所有特征
                feature_names = fv.get_feature_names()
                placeholders = ','.join(['?'] * len(feature_names))
                
                cursor.execute(f"""
                    SELECT feature_name, feature_value
                    FROM feature_values
                    WHERE feature_view = ? AND entity_key = ? AND feature_name IN ({placeholders})
                """, [fv.name, entity_key] + feature_names)
                
                features = {f: None for f in feature_names}  # 默认值
                for fname, fvalue in cursor.fetchall():
                    # 类型转换
                    try:
                        features[fname] = float(fvalue)
                    except:
                        features[fname] = fvalue
                
                row_result.update(features)
            
            results.append(row_result)
        
        return pd.DataFrame(results)
    
    def close(self):
        self.conn.close()


# =============================================================================
# 特征注册中心
# =============================================================================

class FeatureRegistry:
    """
    特征注册中心
    
    管理特征视图的元数据、版本和血缘
    """
    
    def __init__(self, registry_path: str = "./feast_registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        
        self._feature_views: Dict[str, FeatureView] = {}
        self._entities: Dict[str, Entity] = {}
        self._load_registry()
    
    def _load_registry(self):
        """从磁盘加载注册信息"""
        registry_file = os.path.join(self.registry_path, 'registry.json')
        if os.path.exists(registry_file):
            with open(registry_file, 'r') as f:
                data = json.load(f)
            # 这里简化处理，实际应该完整反序列化
            print(f"[Registry] Loaded registry from {registry_file}")
    
    def _save_registry(self):
        """保存注册信息到磁盘"""
        registry_file = os.path.join(self.registry_path, 'registry.json')
        # 简化序列化
        data = {
            'feature_views': list(self._feature_views.keys()),
            'entities': list(self._entities.keys()),
            'updated_at': datetime.now().isoformat()
        }
        with open(registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def apply_entity(self, entity: Entity):
        """注册实体"""
        self._entities[entity.name] = entity
        self._save_registry()
        print(f"[Registry] Registered entity: {entity.name}")
    
    def apply_feature_view(self, feature_view: FeatureView):
        """注册特征视图"""
        # 验证实体已注册
        for entity in feature_view.entities:
            if entity.name not in self._entities:
                raise ValueError(f"Entity {entity.name} not registered")
        
        self._feature_views[feature_view.name] = feature_view
        self._save_registry()
        print(f"[Registry] Registered feature view: {feature_view.name}")
    
    def get_feature_view(self, name: str) -> Optional[FeatureView]:
        """获取特征视图"""
        return self._feature_views.get(name)
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """获取实体"""
        return self._entities.get(name)
    
    def list_feature_views(self) -> List[str]:
        """列出所有特征视图"""
        return list(self._feature_views.keys())


# =============================================================================
# 特征存储主类
# =============================================================================

class FeatureStore:
    """
    特征存储主类
    
    整合离线存储、在线存储和注册中心的完整特征存储实现
    """
    
    def __init__(self, base_path: str = "./feature_store"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        self.registry = FeatureRegistry(os.path.join(base_path, 'registry'))
        self.offline_store = OfflineStore(os.path.join(base_path, 'offline'))
        self.online_store = OnlineStore(os.path.join(base_path, 'online.db'))
    
    def apply(self, obj: Union[Entity, FeatureView]):
        """
        应用实体或特征视图定义
        
        类似于Terraform的apply操作，声明式配置
        """
        if isinstance(obj, Entity):
            self.registry.apply_entity(obj)
        elif isinstance(obj, FeatureView):
            self.registry.apply_feature_view(obj)
        else:
            raise ValueError(f"Unknown object type: {type(obj)}")
    
    def ingest(self, feature_view_name: str, df: pd.DataFrame):
        """
        摄取特征数据
        
        将特征数据写入离线存储，并可选地物化到在线存储
        """
        fv = self.registry.get_feature_view(feature_view_name)
        if fv is None:
            raise ValueError(f"Feature view {feature_view_name} not found")
        
        # 写入离线存储
        self.offline_store.write(fv, df)
        
        # 如果启用在线服务，物化到在线存储
        if fv.online:
            self.online_store.materialize(fv, df)
    
    def get_historical_features(
        self,
        feature_refs: List[str],
        entity_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        获取历史特征
        
        Args:
            feature_refs: 特征引用列表，如 ['user_stats:num_purchases', 'user_stats:avg_order_value']
            entity_df: 实体数据框
        """
        # 解析特征引用，按特征视图分组
        fv_names = set()
        for ref in feature_refs:
            if ':' in ref:
                fv_names.add(ref.split(':')[0])
            else:
                fv_names.add(ref)
        
        feature_views = [self.registry.get_feature_view(n) for n in fv_names]
        feature_views = [fv for fv in feature_views if fv is not None]
        
        return self.offline_store.get_historical_features(feature_views, entity_df)
    
    def get_online_features(
        self,
        feature_refs: List[str],
        entity_rows: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        获取在线特征
        
        用于实时推理场景
        """
        fv_names = set(ref.split(':')[0] if ':' in ref else ref for ref in feature_refs)
        feature_views = [self.registry.get_feature_view(n) for n in fv_names]
        feature_views = [fv for fv in feature_views if fv is not None]
        
        return self.online_store.get_online_features(feature_views, entity_rows)


# =============================================================================
# 演示：完整的特征存储工作流
# =============================================================================

def demo_feature_store():
    """
    演示特征存储的完整工作流
    """
    print("="*80)
    print("特征存储演示")
    print("="*80)
    
    # 1. 初始化特征存储
    store = FeatureStore("./demo_feature_store")
    
    # 2. 定义实体
    user_entity = Entity(
        name="user",
        join_key="user_id",
        description="应用用户"
    )
    
    product_entity = Entity(
        name="product",
        join_key="product_id",
        description="商品"
    )
    
    store.apply(user_entity)
    store.apply(product_entity)
    
    # 3. 定义特征视图
    user_stats_fv = FeatureView(
        name="user_stats",
        entities=[user_entity],
        features=[
            Feature(name="num_purchases", dtype=FeatureType.INT, description="购买次数"),
            Feature(name="avg_order_value", dtype=FeatureType.FLOAT, description="平均订单金额"),
            Feature(name="days_since_last_purchase", dtype=FeatureType.INT, description="距上次购买天数"),
            Feature(name="favorite_category", dtype=FeatureType.STRING, description="偏好类目"),
        ],
        source="user_transactions_table",
        ttl=timedelta(days=7),
        description="用户统计特征"
    )
    
    product_stats_fv = FeatureView(
        name="product_stats",
        entities=[product_entity],
        features=[
            Feature(name="num_views", dtype=FeatureType.INT, description="浏览次数"),
            Feature(name="num_purchases", dtype=FeatureType.INT, description="购买次数"),
            Feature(name="conversion_rate", dtype=FeatureType.FLOAT, description="转化率"),
        ],
        source="product_events_table",
        ttl=timedelta(days=1),
        description="商品统计特征"
    )
    
    store.apply(user_stats_fv)
    store.apply(product_stats_fv)
    
    print("\n" + "-"*40)
    print("注册的实体和特征视图:")
    print(f"  实体: {store.registry.list_feature_views()}")
    
    # 4. 生成示例数据
    print("\n" + "-"*40)
    print("生成示例特征数据...")
    
    np.random.seed(42)
    
    # 用户特征数据
    user_data = []
    for user_id in range(100):
        for days_ago in [0, 7, 14, 30]:  # 多个时间点
            user_data.append({
                'user_id': f'user_{user_id}',
                'event_timestamp': datetime.now() - timedelta(days=days_ago),
                'num_purchases': np.random.poisson(5 + days_ago * 0.1),
                'avg_order_value': np.random.normal(100, 30),
                'days_since_last_purchase': days_ago,
                'favorite_category': np.random.choice(['electronics', 'clothing', 'food'])
            })
    user_df = pd.DataFrame(user_data)
    
    # 商品特征数据
    product_data = []
    for product_id in range(50):
        views = np.random.poisson(1000)
        purchases = np.random.poisson(views * 0.05)
        for days_ago in [0, 1, 2]:  # 商品特征更新更频繁
            product_data.append({
                'product_id': f'product_{product_id}',
                'event_timestamp': datetime.now() - timedelta(days=days_ago),
                'num_views': views + np.random.poisson(10),
                'num_purchases': purchases,
                'conversion_rate': purchases / max(views, 1)
            })
    product_df = pd.DataFrame(product_data)
    
    # 5. 摄取数据
    print("\n摄取用户特征...")
    store.ingest("user_stats", user_df)
    
    print("\n摄取商品特征...")
    store.ingest("product_stats", product_df)
    
    # 6. 获取历史特征（用于训练）
    print("\n" + "-"*40)
    print("获取历史特征（用于训练）...")
    
    # 构造训练样本：假设我们要预测用户是否会购买某商品
    training_entities = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(10)],
        'product_id': [f'product_{i % 50}' for i in range(10)],
        'timestamp': [datetime.now() - timedelta(days=1) for _ in range(10)]
    })
    
    historical_features = store.get_historical_features(
        feature_refs=['user_stats:num_purchases', 'user_stats:avg_order_value', 
                     'product_stats:conversion_rate'],
        entity_df=training_entities
    )
    
    print("\n历史特征数据（前5行）:")
    print(historical_features.head().to_string())
    
    # 7. 获取在线特征（用于推理）
    print("\n" + "-"*40)
    print("获取在线特征（用于推理）...")
    
    online_entities = [
        {'user_id': 'user_0', 'product_id': 'product_1'},
        {'user_id': 'user_5', 'product_id': 'product_10'},
        {'user_id': 'user_10', 'product_id': 'product_20'},
    ]
    
    online_features = store.get_online_features(
        feature_refs=['user_stats:num_purchases', 'user_stats:avg_order_value',
                     'product_stats:conversion_rate'],
        entity_rows=online_entities
    )
    
    print("\n在线特征数据:")
    print(online_features.to_string())
    
    # 8. 展示Point-in-Time Correctness的重要性
    print("\n" + "-"*40)
    print("Point-in-Time Correctness 验证...")
    
    # 假设我们在3天前进行预测
    entity_past = pd.DataFrame({
        'user_id': ['user_0'],
        'timestamp': [datetime.now() - timedelta(days=3)]
    })
    
    features_past = store.get_historical_features(
        feature_refs=['user_stats:num_purchases'],
        entity_df=entity_past
    )
    
    print(f"\n3天前的预测，使用的特征:")
    print(features_past[['user_id', 'timestamp', 'num_purchases']].to_string())
    print("\n注意：这是3天前的用户购买次数，不包含之后的数据！")
    print("这就是Point-in-Time Correctness，防止数据泄漏的关键。")
    
    print(f"\n{'='*80}")
    print("特征存储演示完成！")
    print(f"{'='*80}\n")
    
    return store


if __name__ == "__main__":
    store = demo_feature_store()
```

**费曼解释**：特征存储就像时间机器。当你训练模型时，你可以回到过去任何时刻，看看当时的特征是什么样子。这确保了你不会"偷看未来"（数据泄漏），同时又能为实时推理提供最新特征。

---

## 59.4 模型版本管理与注册中心

### 59.4.1 为什么需要模型注册中心？

想象你是一家制药公司的研发主管。你们研发了50种候选药物，每一种都有多个版本（不同的配方、剂量）。没有中央管理系统：
- 测试部门不知道用哪个版本做实验
- 生产车间拿错了配方
- 监管审计时找不到历史记录

**模型注册中心（Model Registry）**就是机器学习的"药物管理系统"：
1. **版本控制**：每个模型都有唯一的版本号
2. **阶段管理**：Development → Staging → Production → Archived
3. **血缘追踪**：数据来源、训练参数、性能指标
4. **审批流程**：谁可以把模型提升到生产环境？

### 59.4.2 模型注册中心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       Model Registry                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │ Model V1 │───▶│ Model V2 │───▶│ Model V3 │                 │
│   │ v1.0.0   │    │ v1.1.0   │    │ v2.0.0   │                 │
│   │          │    │          │    │          │                 │
│   │ Accuracy │    │ Accuracy │    │ Accuracy │                 │
│   │ 0.85     │    │ 0.87     │    │ 0.89     │                 │
│   │          │    │          │    │          │                 │
│   │ Status:  │    │ Status:  │    │ Status:  │                 │
│   │ Archived │    │ Staging  │    │Production│                 │
│   └──────────┘    └──────────┘    └──────────┘                 │
│                                                                  │
│   元数据：                                                       │
│   - 训练数据版本：data_v2.3.1                                   │
│   - 代码版本：git_commit_abc123                                 │
│   - 超参数：{learning_rate: 0.01, batch_size: 32}              │
│   - 训练时间：2024-03-15 08:30:00                               │
│   - 训练者：data_scientist_a@company.com                        │
│   - 审批者：ml_engineer_b@company.com                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 59.4.3 完整实现：模型注册中心

```python
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
```

**费曼解释**：模型注册中心就像一个精密的"图书馆卡片系统"。每本书（模型）都有唯一的编号（版本），放在特定的书架上（阶段）。你可以查看书籍的借阅历史（血缘），知道是谁推荐的（审批人），还能对比不同版本的区别。最重要的是，当读者（服务）需要一本书时，总能找到当前最权威的版本（Production模型）。


