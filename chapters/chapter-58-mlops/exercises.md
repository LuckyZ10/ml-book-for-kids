# 第五十九章 MLOps——机器学习工程化 练习题

## 练习题 1: MLflow基础 (⭐)

**目标**: 理解实验追踪的基本概念

**题目**: 
运行本章的MLflow实验追踪代码，完成以下任务：
1. 启动MLflow UI，查看实验记录
2. 修改`train_model()`函数中的学习率（改为0.001、0.01、0.1），分别运行3次实验
3. 在UI中比较这3次实验的准确率、损失曲线

**思考问题**: 
- 为什么学习率=0.01时效果最好？
- 如果没有实验追踪，你如何记录这些结果？

---

## 练习题 2: 特征存储设计 (⭐⭐)

**目标**: 理解特征存储的价值

**题目**: 
假设你是一家电商公司的数据工程师，需要设计一个特征存储系统来支持推荐系统。

1. 列出至少5个用户特征和5个商品特征
2. 区分哪些是在线特征（需要实时计算），哪些是离线特征（可以预计算）
3. 设计一个特征版本控制方案，当特征计算逻辑改变时如何管理？

**思考问题**: 
- 特征存储和普通数据库有什么区别？
- 为什么特征存储能提高模型训练和推理的一致性？

---

## 练习题 3: 蓝绿部署模拟 (⭐⭐)

**目标**: 理解零停机部署策略

**题目**: 
使用Python和Flask实现一个简单的模型服务，然后模拟蓝绿部署过程：

```python
# 提示：实现一个简单的模型版本切换系统
# 要求：
# 1. 同时加载"蓝色"和"绿色"两个模型版本
# 2. 通过API参数或请求头控制流量切换
# 3. 实现健康检查接口
```

**思考问题**: 
- 蓝绿部署和滚动更新有什么区别？
- 什么时候应该选择金丝雀发布而不是蓝绿部署？

---

## 练习题 4: 数据漂移检测 (⭐⭐⭐)

**目标**: 理解模型监控的核心概念

**题目**: 
实现一个PSI（Population Stability Index）计算函数：

```python
import numpy as np

def calculate_psi(expected, actual, buckets=10):
    """
    计算PSI值
    
    参数:
    - expected: 基准分布（训练数据）
    - actual: 当前分布（生产数据）
    - buckets: 分箱数
    
    返回:
    - psi: PSI值
    - interpretation: 解释（如"稳定"、"需要关注"等）
    """
    # 你的代码实现
    pass

# 测试数据
train_scores = np.random.beta(7, 3, 1000)  # 训练时的高分分布
current_scores = np.random.beta(5, 5, 1000)  # 现在的分布（变化了）

psi, interpretation = calculate_psi(train_scores, current_scores)
print(f"PSI: {psi:.4f}, 解释: {interpretation}")
```

**要求**: 
- 实现分箱逻辑
- 计算每个箱的占比差异
- 根据PSI值给出解释（<0.1稳定，0.1-0.25需关注，>0.25不稳定）

**思考问题**: 
- PSI和KL散度有什么区别？
- 为什么PSI在风控领域特别常用？

---

## 练习题 5: CI/CD流水线设计 (⭐⭐⭐)

**目标**: 理解自动化部署流程

**题目**: 
设计一个机器学习模型的CI/CD流水线，使用伪代码或流程图描述以下步骤：

1. **代码提交阶段**: 触发自动化测试
2. **数据验证阶段**: 检查数据质量和漂移
3. **模型训练阶段**: 自动训练并记录实验
4. **模型评估阶段**: 与生产模型对比性能
5. **部署阶段**: 自动或人工审批后部署

**要求**: 
- 每个阶段列出检查点（gate）
- 说明失败时的回滚策略
- 考虑A/B测试的集成

**思考问题**: 
- 为什么ML的CI/CD比传统软件复杂？
- 自动化部署中，哪些决策应该保留人工审批？

---

## 练习题 6: 模型版本管理 (⭐⭐)

**目标**: 理解模型注册中心的概念

**题目**: 
使用本章的MLflow模型注册代码，完成以下操作：

1. 注册3个不同版本的模型（改变超参数或训练数据）
2. 将性能最好的版本设置为"Production"阶段
3. 实现一个模型加载函数，自动获取当前生产版本

**思考问题**: 
- 模型版本和代码版本有什么区别？
- 为什么需要模型阶段（Staging/Production/Archived）？

---

## 练习题 7: 监控仪表板设计 (⭐⭐⭐)

**目标**: 设计可观测性系统

**题目**: 
设计一个模型监控仪表板，需要展示以下指标：

**模型性能指标**:
- 准确率、精确率、召回率随时间变化
- 预测分数分布直方图

**数据质量指标**:
- 特征缺失率
- 特征分布变化（PSI）
- 数据延迟时间

**业务指标**:
- 每日预测量
- 平均推理延迟
- 错误率

**要求**: 
- 用Markdown表格或ASCII图表展示仪表板布局
- 为每个指标设定告警阈值
- 说明不同颜色（绿/黄/红）的判断标准

---

## 练习题 8: 特征工程流水线 (⭐⭐⭐⭐)

**目标**: 构建端到端特征处理流程

**题目**: 
实现一个特征工程管道，包含以下步骤：

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeaturePipeline:
    """
    特征工程流水线
    
    要求：
    1. 数值特征：标准化 + 异常值处理
    2. 类别特征：One-Hot编码 + 低频类别合并
    3. 时间特征：提取年月日、星期、是否节假日
    4. 特征选择：移除高相关性特征
    """
    
    def fit(self, X, y=None):
        # 学习特征统计信息
        pass
    
    def transform(self, X):
        # 转换特征
        pass
    
    def save(self, path):
        # 保存特征转换器，确保训练/推理一致性
        pass
    
    @classmethod
    def load(cls, path):
        # 加载特征转换器
        pass
```

**思考问题**: 
- 为什么必须保存特征转换器，而不能在推理时重新计算？
- 如何处理训练时没见过的新类别？

---

## 练习题 9: 端到端项目实践 (⭐⭐⭐⭐⭐)

**目标**: 整合本章所有知识点

**题目**: 
完成一个完整的MLOps项目：房价预测模型的生命周期管理

**要求**: 

1. **数据准备**: 使用sklearn的Boston Housing或California Housing数据集

2. **实验追踪**: 
   - 使用MLflow记录至少5组不同超参数的实验
   - 比较不同特征组合的效果

3. **模型注册**: 
   - 将最佳模型注册到Model Registry
   - 版本设为"v1.0"

4. **部署服务**: 
   - 使用Flask或FastAPI创建预测服务
   - 实现健康检查接口

5. **监控**: 
   - 记录每次预测的输入特征分布
   - 实现简单的漂移检测（特征均值变化>10%时告警）

6. **文档**: 
   - 编写部署文档，说明如何启动服务
   - 记录模型性能基线

**交付物**: 
- 完整的Python代码
- README.md说明文档
- 截图：MLflow UI、API测试结果

**思考问题**: 
- 如果你要把这个项目交给同事维护，还需要补充什么？
- 生产环境中，还有哪些本章没覆盖的问题？

---

## 参考答案与提示

<details>
<summary>练习题4 PSI计算提示</summary>

```python
def calculate_psi(expected, actual, buckets=10):
    """计算Population Stability Index"""
    # 1. 创建分箱边界
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # 2. 计算每个箱的占比
    expected_counts, _ = np.histogram(expected, breakpoints)
    actual_counts, _ = np.histogram(actual, breakpoints)
    
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)
    
    # 3. 处理0值，添加小epsilon
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # 4. 计算PSI
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    # 5. 解释
    if psi < 0.1:
        interpretation = "稳定 - 分布没有显著变化"
    elif psi < 0.25:
        interpretation = "需要关注 - 有轻微变化"
    else:
        interpretation = "不稳定 - 需要调查原因"
    
    return psi, interpretation
```

</details>

---

**学习建议**: 
- 练习题1-3是基础，建议必做
- 练习题4-6是进阶，做完可深入理解MLOps
- 练习题7-9是实战，完成后具备生产环境部署能力

**参考资源**:
- MLflow官方文档: https://mlflow.org/docs/latest/index.html
- Google的ML测试评分卡: https://research.google/pubs/pub46555/
- MLOps社区最佳实践: https://ml-ops.org/
