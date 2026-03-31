#!/usr/bin/env python3
"""
第六十章 完整项目——端到端的AI应用
代码实现部分

本章涵盖：
- 数据工程：ETL管道设计
- 特征工程：RFM模型、行为特征
- 模型开发：PyTorch神经网络
- MLOps：实验追踪、模型注册、部署
- 服务部署：REST API接口
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== 60.1 配置管理 ==========

@dataclass
class PipelineConfig:
    """管道配置"""
    source_path: str
    target_path: str
    batch_size: int = 10000
    max_workers: int = 4


@dataclass
class Feature:
    """特征定义"""
    name: str
    description: str
    feature_type: str
    data_type: str
    is_pii: bool = False
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class SuccessMetrics:
    """项目成功指标容器"""
    
    model_metrics: Dict[str, float] = None
    business_metrics: Dict[str, float] = None
    engineering_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        self.model_metrics = self.model_metrics or {
            'roc_auc': 0.85,
            'precision_at_k': 0.70,
            'recall_at_k': 0.75,
            'f1_score': 0.72,
        }
        self.business_metrics = self.business_metrics or {
            'churn_reduction_rate': 0.20,
            'intervention_roi': 3.0,
            'coverage_rate': 0.80,
        }
        self.engineering_metrics = self.engineering_metrics or {
            'latency_p99': 100,
            'availability': 0.999,
            'prediction_throughput': 1000,
        }


# ========== 60.2 数据工程：ETL管道 ==========

class DataPipeline:
    """
    数据ETL管道
    
    费曼比喻：这就像工厂的生产线——原材料从一端进入，
    经过清洗、加工、质检，最后包装成成品从另一端输出。
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.transforms: List = []
        self.quality_checks: List = []
        
    def add_transform(self, func):
        """添加转换步骤"""
        self.transforms.append(func)
        return self
    
    def add_quality_check(self, func):
        """添加质量检查"""
        self.quality_checks.append(func)
        return self
    
    def extract(self, source: str, **kwargs):
        """数据抽取"""
        logger.info(f"Extracting from {source}...")
        
        if 'customer' in source:
            df = self._generate_synthetic_customer_data(**kwargs)
        elif 'billing' in source:
            df = self._generate_synthetic_billing_data(**kwargs)
        elif 'usage' in source:
            df = self._generate_synthetic_usage_data(**kwargs)
        else:
            df = self._generate_synthetic_customer_data(**kwargs)
        
        logger.info(f"Extracted: {len(df)} rows")
        return df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据转换"""
        logger.info("Transforming data...")
        
        for i, transform in enumerate(self.transforms):
            logger.info(f"  Step {i+1}/{len(self.transforms)}: {transform.__name__}")
            df = transform(df)
            
        logger.info("Transform complete")
        return df
    
    def validate(self, df: pd.DataFrame) -> bool:
        """数据质量验证"""
        logger.info("Running quality checks...")
        
        all_passed = True
        for check in self.quality_checks:
            result = check(df)
            status = "PASS" if result else "FAIL"
            logger.info(f"  {status}: {check.__name__}")
            all_passed = all_passed and result
            
        return all_passed
    
    def load(self, df: pd.DataFrame, target: str):
        """数据加载"""
        logger.info(f"Loading to {target}: {len(df)} rows")
        return True
    
    def run(self, source: str, target: str, **kwargs):
        """执行完整管道"""
        logger.info("="*60)
        logger.info("Starting Data Pipeline")
        logger.info("="*60)
        
        df = self.extract(source, **kwargs)
        df = self.transform(df)
        
        if not self.validate(df):
            logger.error("Quality check failed")
            return False
        
        self.load(df, target)
        
        logger.info("Pipeline completed successfully!")
        return True
    
    # 合成数据生成器
    def _generate_synthetic_customer_data(self, n_samples=10000, random_state=42):
        np.random.seed(random_state)
        return pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(n_samples)],
            'age': np.random.randint(18, 80, n_samples),
            'gender': np.random.choice(['M', 'F', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
            'tenure_months': np.random.exponential(24, n_samples).clip(1, 120).astype(int),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
            'plan_type': np.random.choice(['Basic', 'Standard', 'Premium'], n_samples, p=[0.3, 0.5, 0.2]),
            'monthly_charge': np.random.normal(50, 20, n_samples).clip(20, 150).round(2),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                             n_samples, p=[0.55, 0.25, 0.2]),
            'paperless_billing': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
            'payment_method': np.random.choice(['Bank transfer', 'Credit card', 'Electronic check', 'Mailed check'],
                                              n_samples),
        })
    
    def _generate_synthetic_billing_data(self, n_samples=100000, random_state=42):
        np.random.seed(random_state)
        n_customers = 10000
        records = []
        for month_offset in range(6):
            month_date = datetime.now() - timedelta(days=30*month_offset)
            for i in range(n_customers):
                is_churn_risk = np.random.random() < 0.2
                records.append({
                    'customer_id': f'CUST_{i:06d}',
                    'billing_month': month_date.strftime('%Y-%m'),
                    'total_charges': np.random.normal(65, 25) * (0.8 if is_churn_risk else 1.0),
                    'payment_delay_days': np.random.poisson(5) if is_churn_risk else max(0, np.random.poisson(1)),
                    'is_paid': np.random.random() > (0.3 if is_churn_risk else 0.05),
                    'support_calls': np.random.poisson(3) if is_churn_risk else np.random.poisson(0.5),
                })
        return pd.DataFrame(records)
    
    def _generate_synthetic_usage_data(self, n_samples=50000, random_state=42):
        np.random.seed(random_state)
        n_customers = 10000
        records = []
        for day_offset in range(30):
            date = datetime.now() - timedelta(days=day_offset)
            daily_customers = np.random.choice(n_customers, size=n_customers//3, replace=False)
            for i in daily_customers:
                is_churn_risk = np.random.random() < 0.2
                records.append({
                    'customer_id': f'CUST_{i:06d}',
                    'date': date.strftime('%Y-%m-%d'),
                    'call_minutes': np.random.exponential(100) * (0.7 if is_churn_risk else 1.0),
                    'data_gb': np.random.exponential(5) * (0.8 if is_churn_risk else 1.0),
                    'sms_count': np.random.poisson(50) * (0.6 if is_churn_risk else 1.0),
                    'roaming_minutes': np.random.poisson(2),
                    'international_calls': np.random.poisson(1),
                })
        return pd.DataFrame(records)


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['age'] = df['age'].fillna(df['age'].median())
    df['gender'] = df['gender'].fillna('Unknown')
    df['signup_date'] = pd.to_datetime(df.get('signup_date', datetime.now()))
    df['age'] = df['age'].clip(18, 100)
    df['tenure_months'] = df['tenure_months'].clip(0, 200)
    return df


def check_no_duplicate_ids(df: pd.DataFrame) -> bool:
    if 'customer_id' not in df.columns:
        return True
    return df['customer_id'].nunique() == len(df)


def check_no_missing_critical_fields(df: pd.DataFrame) -> bool:
    critical_fields = ['customer_id']
    for field in critical_fields:
        if field in df.columns and df[field].isnull().sum() > 0:
            return False
    return True


# ========== 60.3 特征工程 ==========

class FeatureStore:
    """
    特征仓库 - 统一管理所有特征
    
    就像餐厅的中央厨房，所有食材在这里统一处理、储存、分发。
    """
    
    def __init__(self):
        self.features: Dict[str, Feature] = {}
        self.feature_values: Dict[str, pd.DataFrame] = {}
        
    def register_feature(self, feature: Feature):
        self.features[feature.name] = feature
        print(f"Registered feature: {feature.name} ({feature.feature_type})")
        
    def compute_features(self, 
                        customer_df: pd.DataFrame,
                        billing_df: pd.DataFrame,
                        usage_df: pd.DataFrame,
                        as_of_date: datetime = None) -> pd.DataFrame:
        if as_of_date is None:
            as_of_date = datetime.now()
        
        print(f"Computing features (as of {as_of_date.date()})...")
        
        base_features = self._compute_base_features(customer_df)
        rfm_features = self._compute_rfm_features(billing_df, as_of_date)
        behavior_features = self._compute_behavior_features(usage_df, as_of_date)
        agg_features = self._compute_aggregation_features(billing_df, usage_df)
        temporal_features = self._compute_temporal_features(billing_df, usage_df)
        
        features = base_features.copy()
        for df in [rfm_features, behavior_features, agg_features, temporal_features]:
            features = features.merge(df, on='customer_id', how='left')
        
        features = self._fill_missing_values(features)
        
        print(f"Features computed: {len(features)} customers x {len(features.columns)} features")
        return features
    
    def _compute_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df[['customer_id']].copy()
        result['age_group'] = pd.cut(df['age'], 
                                     bins=[0, 25, 35, 50, 65, 100],
                                     labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        result['tenure_group'] = pd.cut(df['tenure_months'],
                                        bins=[0, 12, 24, 48, 120],
                                        labels=['0-1y', '1-2y', '2-4y', '4y+'])
        result['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)
        result['is_loyal_customer'] = (df['tenure_months'] >= 48).astype(int)
        
        plan_mapping = {'Basic': 1, 'Standard': 2, 'Premium': 3}
        result['plan_level'] = df['plan_type'].map(plan_mapping)
        
        result['has_contract'] = (df['contract_type'] != 'Month-to-month').astype(int)
        
        contract_months = {'Month-to-month': 0, 'One year': 12, 'Two year': 24}
        result['contract_duration'] = df['contract_type'].map(contract_months)
        result['is_paperless'] = df['paperless_billing'].astype(int)
        result['payment_risk'] = (df['payment_method'] == 'Electronic check').astype(int)
        
        return result
    
    def _compute_rfm_features(self, billing_df: pd.DataFrame, as_of_date: datetime) -> pd.DataFrame:
        result = billing_df.groupby('customer_id').agg({
            'billing_month': 'max',
            'total_charges': ['sum', 'mean', 'std'],
            'payment_delay_days': ['mean', 'max'],
            'support_calls': ['sum', 'mean'],
        }).reset_index()
        
        result.columns = ['customer_id', 'last_billing_month', 
                         'total_revenue', 'avg_monthly_charge', 'charge_std',
                         'avg_payment_delay', 'max_payment_delay',
                         'total_support_calls', 'avg_support_calls']
        
        result['last_billing_month'] = pd.to_datetime(result['last_billing_month'])
        result['recency_days'] = (as_of_date - result['last_billing_month']).dt.days
        result['payment_behavior_score'] = 100 - (result['avg_payment_delay'] * 5).clip(0, 100)
        
        return result[['customer_id', 'recency_days', 'total_revenue', 'avg_monthly_charge',
                      'charge_std', 'payment_behavior_score', 'total_support_calls']]
    
    def _compute_behavior_features(self, usage_df: pd.DataFrame, as_of_date: datetime) -> pd.DataFrame:
        if len(usage_df) == 0:
            return pd.DataFrame({'customer_id': []})
        
        result = usage_df.groupby('customer_id').agg({
            'call_minutes': ['sum', 'mean', 'std'],
            'data_gb': ['sum', 'mean', 'std'],
            'sms_count': ['sum', 'mean'],
            'roaming_minutes': 'sum',
            'international_calls': 'sum',
            'date': ['count', 'max'],
        }).reset_index()
        
        result.columns = ['customer_id',
                         'total_call_minutes', 'avg_daily_calls', 'call_volatility',
                         'total_data_gb', 'avg_daily_data', 'data_volatility',
                         'total_sms', 'avg_daily_sms',
                         'total_roaming_minutes', 'total_intl_calls',
                         'active_days', 'last_active_date']
        
        result['activity_rate'] = result['active_days'] / 30
        result['last_active_date'] = pd.to_datetime(result['last_active_date'])
        result['days_since_active'] = (as_of_date - result['last_active_date']).dt.days
        result['service_diversity'] = (
            (result['total_call_minutes'] > 0).astype(int) +
            (result['total_data_gb'] > 0).astype(int) +
            (result['total_sms'] > 0).astype(int)
        )
        result['is_international_user'] = (
            (result['total_roaming_minutes'] > 0) | 
            (result['total_intl_calls'] > 0)
        ).astype(int)
        
        return result
    
    def _compute_aggregation_features(self, billing_df: pd.DataFrame, usage_df: pd.DataFrame) -> pd.DataFrame:
        billing_trends = billing_df.groupby('customer_id').apply(
            lambda x: self._calculate_trend(x, 'total_charges')
        ).reset_index()
        billing_trends.columns = ['customer_id', 'charge_trend']
        
        risk_signals = billing_df.groupby('customer_id').agg({
            'is_paid': lambda x: (x == False).sum(),
            'support_calls': 'sum',
        }).reset_index()
        risk_signals.columns = ['customer_id', 'unpaid_count', 'support_call_total']
        
        result = billing_trends.merge(risk_signals, on='customer_id', how='outer')
        return result.fillna(0)
    
    def _calculate_trend(self, df: pd.DataFrame, value_col: str) -> float:
        if len(df) < 2:
            return 0
        df = df.sort_values('billing_month')
        x = np.arange(len(df))
        y = df[value_col].values
        return np.polyfit(x, y, 1)[0]
    
    def _compute_temporal_features(self, billing_df: pd.DataFrame, usage_df: pd.DataFrame) -> pd.DataFrame:
        recent = billing_df[billing_df['billing_month'] >= billing_df['billing_month'].max() - pd.Timedelta(days=30)]
        older = billing_df[billing_df['billing_month'] < billing_df['billing_month'].max() - pd.Timedelta(days=30)]
        
        recent_avg = recent.groupby('customer_id')['total_charges'].mean().reset_index()
        recent_avg.columns = ['customer_id', 'recent_avg_charge']
        
        older_avg = older.groupby('customer_id')['total_charges'].mean().reset_index()
        older_avg.columns = ['customer_id', 'older_avg_charge']
        
        trend = recent_avg.merge(older_avg, on='customer_id', how='outer').fillna(0)
        trend['charge_change_pct'] = ((trend['recent_avg_charge'] - trend['older_avg_charge']) / 
                                      (trend['older_avg_charge'] + 1)) * 100
        
        return trend[['customer_id', 'charge_change_pct']]
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in numeric_cols:
            if col != 'customer_id':
                df[col] = df[col].fillna(0)
        
        for col in categorical_cols:
            if col != 'customer_id':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df


# ========== 60.4 模型开发 ==========

class ChurnPredictor:
    """
    客户流失预测模型
    
    费曼比喻：就像医生根据体检指标判断疾病风险，
    我们根据客户行为特征判断流失概率。
    """
    
    def __init__(self, model_type='ensemble'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.metrics = {}
        
    def prepare_data(self, features_df: pd.DataFrame, labels_df: pd.DataFrame = None):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        print("Preparing data...")
        
        df = features_df.copy()
        customer_ids = df['customer_id'].values
        df = df.drop('customer_id', axis=1)
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        if labels_df is not None:
            labels_map = dict(zip(labels_df['customer_id'], labels_df['churned']))
            y = np.array([labels_map.get(cid, 0) for cid in customer_ids])
            
            X_train, X_test, y_train, y_test = train_test_split(
                df, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            print(f"Train: {len(X_train)}, Test: {len(X_test)}, Positive: {y.mean():.2%}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test, df.columns.tolist()
        
        else:
            X_scaled = self.scaler.transform(df)
            return X_scaled, customer_ids
    
    def build_simple_model(self, input_dim: int):
        """构建简化版模型（无需PyTorch）"""
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        print(f"Model built: Gradient Boosting, input dim {input_dim}")
        return self.model
    
    def train_simple(self, X_train, y_train):
        """训练简化版模型"""
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("Training complete")
    
    def evaluate_simple(self, X_test, y_test, feature_names=None):
        from sklearn.metrics import (roc_auc_score, classification_report, 
                                     confusion_matrix, accuracy_score, 
                                     precision_score, recall_score, f1_score)
        
        print("Evaluating model...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        print("\n" + "="*50)
        print("Model Evaluation Results")
        print("="*50)
        for metric, value in self.metrics.items():
            print(f"{metric.upper():12s}: {value:.4f}")
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        if feature_names:
            print("\nFeature Importance (Top 10):")
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            for i in range(min(10, len(feature_names))):
                print(f"  {i+1:2d}. {feature_names[indices[i]]:25s}: {importances[indices[i]]:.4f}")
        
        return self.metrics
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


# ========== 60.5 MLOps管道 ==========

class MLOpsPipeline:
    """
    MLOps流水线 - 端到端自动化
    
    费曼比喻：就像餐厅的中央厨房系统——从采购、备菜、烹饪到上菜，
    全流程标准化、自动化、可监控。
    """
    
    def __init__(self, project_name: str = "churn_prediction"):
        self.project_name = project_name
        self.run_id = self._generate_run_id()
        self.artifacts = {}
        
        print(f"MLOps Pipeline started")
        print(f"  Project: {project_name}")
        print(f"  Run ID: {self.run_id}")
    
    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(np.random.random()).encode()).hexdigest()[:6]
        return f"{timestamp}_{random_suffix}"
    
    def log_params(self, params: dict):
        self.artifacts['params'] = params
        print(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: dict):
        self.artifacts['metrics'] = metrics
        print("Logged metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    def register_model(self, model_name: str, metrics: dict, threshold: float = 0.80):
        if metrics.get('roc_auc', 0) >= threshold:
            print(f"Model '{model_name}' registered (AUC={metrics['roc_auc']:.4f})")
            return True
        else:
            print(f"Model '{model_name}' below threshold ({metrics['roc_auc']:.4f} < {threshold})")
            return False
    
    def deploy_model(self, model, deployment_type='shadow'):
        deployments = {
            'shadow': 'Shadow mode (logging only)',
            'canary': 'Canary (1% traffic)',
            'blue_green': 'Blue-Green (50% traffic)',
            'full': 'Full rollout (100% traffic)'
        }
        print(f"Deploying: {deployments.get(deployment_type, deployment_type)}")
        return True
    
    def monitor_model(self, predictions_log: pd.DataFrame):
        print("Monitoring model...")
        recent_preds = predictions_log.tail(1000)['prediction']
        drift_score = abs(recent_preds.mean() - 0.5) * 2
        
        if drift_score > 0.3:
            print(f"WARNING: Drift detected: {drift_score:.3f}")
        else:
            print(f"Model healthy: drift score {drift_score:.3f}")
        
        return drift_score


# ========== 60.6 演示运行 ==========

def run_demo():
    """运行完整演示"""
    print("\n" + "="*70)
    print("Chapter 60: End-to-End AI Project Demo")
    print("="*70)
    
    # Step 1: Data Engineering
    print("\n" + "-"*70)
    print("Step 1: Data Engineering")
    print("-"*70)
    
    config = PipelineConfig(
        source_path="/data/raw",
        target_path="/data/curated"
    )
    
    pipeline = DataPipeline(config)
    pipeline.add_transform(clean_customer_data)
    pipeline.add_quality_check(check_no_duplicate_ids)
    pipeline.add_quality_check(check_no_missing_critical_fields)
    
    pipeline.run("customer", "curated_customers", n_samples=1000)
    
    # Step 2: Feature Engineering
    print("\n" + "-"*70)
    print("Step 2: Feature Engineering")
    print("-"*70)
    
    np.random.seed(42)
    n_customers = 1000
    
    customer_df = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
        'age': np.random.randint(18, 80, n_customers),
        'gender': np.random.choice(['M', 'F'], n_customers),
        'tenure_months': np.random.exponential(24, n_customers).clip(1, 120).astype(int),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_customers),
        'plan_type': np.random.choice(['Basic', 'Standard', 'Premium'], n_customers),
        'monthly_charge': np.random.normal(50, 20, n_customers).clip(20, 150).round(2),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'paperless_billing': np.random.choice([True, False], n_customers),
        'payment_method': np.random.choice(['Bank transfer', 'Credit card', 'Electronic check'], n_customers),
    })
    
    billing_records = []
    for i in range(n_customers):
        is_churn = np.random.random() < 0.2
        for month in range(6):
            billing_records.append({
                'customer_id': f'CUST_{i:06d}',
                'billing_month': pd.Timestamp('2024-01-01') + pd.DateOffset(months=month),
                'total_charges': np.random.normal(65, 25) * (0.7 if is_churn else 1.0),
                'payment_delay_days': np.random.poisson(5) if is_churn else np.random.poisson(1),
                'is_paid': np.random.random() > (0.3 if is_churn else 0.05),
                'support_calls': np.random.poisson(3) if is_churn else np.random.poisson(0.5),
            })
    billing_df = pd.DataFrame(billing_records)
    
    usage_records = []
    for i in range(n_customers):
        is_churn = np.random.random() < 0.2
        for day in range(30):
            if np.random.random() < 0.7:
                usage_records.append({
                    'customer_id': f'CUST_{i:06d}',
                    'date': pd.Timestamp('2024-06-01') + pd.Timedelta(days=day),
                    'call_minutes': np.random.exponential(100) * (0.7 if is_churn else 1.0),
                    'data_gb': np.random.exponential(5) * (0.8 if is_churn else 1.0),
                    'sms_count': np.random.poisson(50) * (0.6 if is_churn else 1.0),
                    'roaming_minutes': np.random.poisson(2),
                    'international_calls': np.random.poisson(1),
                })
    usage_df = pd.DataFrame(usage_records)
    
    store = FeatureStore()
    features = store.compute_features(customer_df, billing_df, usage_df)
    print(f"Features shape: {features.shape}")
    
    # Step 3: Model Development
    print("\n" + "-"*70)
    print("Step 3: Model Development")
    print("-"*70)
    
    np.random.seed(42)
    n_customers = 5000
    
    features_df = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
        'tenure_months': np.random.exponential(24, n_customers).clip(1, 120),
        'monthly_charge': np.random.normal(65, 25, n_customers).clip(20, 150),
        'total_calls': np.random.exponential(3000, n_customers),
        'support_calls': np.random.poisson(2, n_customers),
        'payment_delays': np.random.poisson(1, n_customers),
        'contract_months': np.random.choice([0, 12, 24], n_customers),
        'plan_level': np.random.choice([1, 2, 3], n_customers),
        'is_paperless': np.random.choice([0, 1], n_customers),
    })
    
    churn_prob = (
        0.3 * (features_df['tenure_months'] < 6) +
        0.2 * (features_df['monthly_charge'] > 80) +
        0.25 * (features_df['support_calls'] > 3) +
        0.15 * (features_df['contract_months'] == 0) +
        0.1 * np.random.random(n_customers)
    )
    labels_df = pd.DataFrame({
        'customer_id': features_df['customer_id'],
        'churned': (churn_prob > 0.5).astype(int)
    })
    
    print(f"Customers: {n_customers}, Churn rate: {labels_df['churned'].mean():.2%}")
    
    predictor = ChurnPredictor()
    X_train, X_test, y_train, y_test, feature_names = predictor.prepare_data(
        features_df, labels_df
    )
    
    predictor.build_simple_model(input_dim=X_train.shape[1])
    predictor.train_simple(X_train, y_train)
    metrics = predictor.evaluate_simple(X_test, y_test, feature_names)
    
    # Step 4: MLOps
    print("\n" + "-"*70)
    print("Step 4: MLOps Pipeline")
    print("-"*70)
    
    mlops = MLOpsPipeline(project_name="customer_churn_v1")
    mlops.log_params({
        'model_type': 'GradientBoosting',
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
    })
    mlops.log_metrics(metrics)
    mlops.register_model("churn_model_v1", metrics, threshold=0.85)
    mlops.deploy_model(None, deployment_type='canary')
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("Project Complete!")
    print("="*70)
    print(f"""
    Summary:
    - Data Engineering: ETL pipeline with quality checks
    - Feature Engineering: RFM + behavioral features
    - Model: Gradient Boosting (AUC={metrics['roc_auc']:.4f})
    - MLOps: Experiment tracking, model registry, deployment
    
    Book Complete: 60/60 Chapters!
    """)


if __name__ == "__main__":
    run_demo()
