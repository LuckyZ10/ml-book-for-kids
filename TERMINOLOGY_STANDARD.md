# 机器学习教材术语对照表

> **版本**: v1.0  
> **更新日期**: 2026-03-27  
> **适用范围**: 全书60章

---

## 一、核心概念术语（必须统一）

| 序号 | 中文术语（标准） | 英文术语 | 禁用变体 | 说明 |
|:---:|:---|:---|:---|:---|
| 1 | 机器学习 | Machine Learning | 机械学习、机学 | 标准译名 |
| 2 | 深度学习 | Deep Learning | 深层学习 | 标准译名 |
| 3 | 神经网络 | Neural Network | 神经网路、神经网络 | "网络"非"网路" |
| 4 | 人工神经网络 | Artificial Neural Network | 人造神经网络 | ANN全称 |
| 5 | 神经元 | Neuron | 神经单元 | 生物/计算统一 |
| 6 | 层 | Layer | 层级 | 网络结构组件 |
| 7 | 输入层 | Input Layer | 输入层级 |  |
| 8 | 隐藏层 | Hidden Layer | 隐含层、隐层 | 统一用"隐藏" |
| 9 | 输出层 | Output Layer | 输出层级 |  |
| 10 | 权重 | Weight | 权值、参数 | 矩阵/向量元素 |
| 11 | 偏置 | Bias | 偏移量、截距 | 统一用"偏置" |
| 12 | 激活函数 | Activation Function | 激励函数 | 统一用"激活" |
| 13 | 损失函数 | Loss Function | 损失函数、损失函数 | 训练阶段优化目标 |
| 14 | 损失函数 | Cost Function | 损失函数 | 注意：与Loss区分时用 |
| 15 | 损失函数 | Objective Function | 优化目标 | 最广泛概念 |
| 16 | 梯度下降 | Gradient Descent | 梯度下將 | "下降"非"下將" |
| 17 | 学习率 | Learning Rate | 学习速率、步长 | 超参数 |
| 18 | 批量 | Batch | 批量 | 数据分组 |
| 19 | 轮次 | Epoch | 迭代轮数 | 完整遍历数据集 |
| 20 | 迭代 | Iteration | 步骤 | 一次参数更新 |
| 21 | 反向传播 | Backpropagation | 反向传递 | 梯度计算算法 |
| 22 | 前向传播 | Forward Propagation | 前向传递 | 预测计算过程 |
| 23 | 优化器 | Optimizer | 优化算法、求解器 | 参数更新策略 |
| 24 | 正则化 | Regularization | 规则化 | 防止过拟合 |
| 25 | 过拟合 | Overfitting | 过度拟合 | 训练集表现好，测试集差 |
| 26 | 欠拟合 | Underfitting | 拟合不足 | 训练不足 |
| 27 | 泛化 | Generalization | 推广能力 | 对新数据的表现 |
| 28 | 超参数 | Hyperparameter | 超级参数 | 需手动设置的参数 |
| 29 | 参数 | Parameter | 参量 | 模型学习的值 |
| 30 | 特征 | Feature | 属性、变量 | 输入数据维度 |
| 31 | 标签 | Label | 目标、标记 | 监督学习的目标 |
| 32 | 样本 | Sample | 实例、例子 | 一条数据 |
| 33 | 数据集 | Dataset | 数据集合 |  |
| 34 | 训练集 | Training Set | 训练数据集 |  |
| 35 | 验证集 | Validation Set | 校验集 | 调参用 |
| 36 | 测试集 | Test Set | 测试数据集 | 最终评估用 |
| 37 | 推理 | Inference | 推断、预测 | 模型使用阶段 |
| 38 | 部署 | Deployment | 发布、上线 | 生产环境使用 |
| 39 | 嵌入 | Embedding | 嵌入向量、词向量 | 稠密向量表示 |
| 40 | 注意力机制 | Attention Mechanism | 注意力模块 | Transformer核心 |

---

## 二、算法与模型术语

| 序号 | 中文术语（标准） | 英文术语 | 禁用变体 |
|:---:|:---|:---|:---|
| 41 | 卷积神经网络 | Convolutional Neural Network, CNN | 卷积网络（简称可用）|
| 42 | 循环神经网络 | Recurrent Neural Network, RNN | 递归神经网络 |
| 43 | 长短期记忆网络 | Long Short-Term Memory, LSTM | 长短时记忆 |
| 44 | 门控循环单元 | Gated Recurrent Unit, GRU | 门限循环单元 |
| 45 | Transformer | Transformer | 变换器、转换器 | 保留英文或全称 |
| 46 | 生成对抗网络 | Generative Adversarial Network, GAN | 对抗生成网络 |
| 47 | 变分自编码器 | Variational Autoencoder, VAE | 变分自动编码器 |
| 48 | 扩散模型 | Diffusion Model | 扩散式模型 |
| 49 | 强化学习 | Reinforcement Learning, RL | 增强学习 |
| 50 | 图神经网络 | Graph Neural Network, GNN | 图网络 |
| 51 | 自编码器 | Autoencoder | 自动编码器 |
| 52 | 多层感知机 | Multilayer Perceptron, MLP | 多层感知机 |
| 53 | 支持向量机 | Support Vector Machine, SVM | 支撑向量机 |
| 54 | 决策树 | Decision Tree | 判定树 |
| 55 | 随机森林 | Random Forest | 随机树林 |
| 56 | 梯度提升树 | Gradient Boosting Decision Tree, GBDT | 梯度提升决策树 |
| 57 | K近邻 | K-Nearest Neighbors, KNN | K最近邻 |
| 58 | K均值聚类 | K-Means Clustering | KMeans、K-平均 |
| 59 | 主成分分析 | Principal Component Analysis, PCA | 主分量分析 |
| 60 | 朴素贝叶斯 | Naive Bayes | 简单贝叶斯 |
| 61 | 逻辑回归 | Logistic Regression | 逻辑斯特回归 |
| 62 | 线性回归 | Linear Regression | 线性回归分析 |
| 63 | 岭回归 | Ridge Regression | 山脊回归 |
| 64 | Lasso回归 | Lasso Regression | L1回归 |
| 65 | 嵌入 | Word Embedding | 词向量 |
| 66 | 位置编码 | Positional Encoding | 位置嵌入 |
| 67 | 多头注意力 | Multi-Head Attention | 多头部注意力 |
| 68 | 自注意力 | Self-Attention | 自身注意力 |
| 69 | 残差连接 | Residual Connection | 跳跃连接、Shortcut |
| 70 | 层归一化 | Layer Normalization | 层标准化 |
| 71 | 批量归一化 | Batch Normalization | 批归一化 |
| 72 | Dropout | Dropout | 随机失活 | 保留英文 |
| 73 | 学习率调度 | Learning Rate Scheduling | 学习率衰减策略 |
| 74 | 早停 | Early Stopping | 提前停止 |
| 75 | 数据增强 | Data Augmentation | 数据扩充 |

---

## 三、优化与训练术语

| 序号 | 中文术语（标准） | 英文术语 | 禁用变体 |
|:---:|:---|:---|:---|
| 76 | 随机梯度下降 | Stochastic Gradient Descent, SGD |  |
| 77 | 动量 | Momentum | 冲量 |
| 78 | AdaGrad | AdaGrad | 自适应梯度 | 保留英文 |
| 79 | RMSprop | RMSprop | 均方根传播 | 保留英文 |
| 80 | Adam | Adam | 自适应矩估计 | 保留英文 |
| 81 | AdamW | AdamW | 解耦权重衰减Adam | 保留英文 |
| 82 | 学习率衰减 | Learning Rate Decay | 学习率下降 |
| 83 | 预热 | Warmup | 热启动 |
| 84 | 余弦退火 | Cosine Annealing | 余弦衰减 |
| 85 | 梯度裁剪 | Gradient Clipping | 梯度截断 |
| 86 | 梯度累积 | Gradient Accumulation | 梯度累加 |
| 87 | 迁移学习 | Transfer Learning | 转移学习 |
| 88 | 微调 | Fine-tuning | 精细调整 |
| 89 | 预训练 | Pre-training | 预先训练 |
| 90 | 零样本学习 | Zero-shot Learning | 零次学习 |
| 91 | 少样本学习 | Few-shot Learning | 小样本学习 |

---

## 四、评估与指标术语

| 序号 | 中文术语（标准） | 英文术语 | 禁用变体 |
|:---:|:---|:---|:---|
| 92 | 准确率 | Accuracy | 准确度 |
| 93 | 精确率 | Precision | 精度、查准率 |
| 94 | 召回率 | Recall | 查全率、敏感度 |
| 95 | F1分数 | F1 Score | F1值 |
| 96 | 混淆矩阵 | Confusion Matrix | 误差矩阵 |
| 97 | ROC曲线 | ROC Curve | 受试者工作特征曲线 |
| 98 | AUC | Area Under Curve | 曲线下面积 | 保留英文缩写 |
| 99 | 均方误差 | Mean Squared Error, MSE | 平均平方误差 |
| 100 | 平均绝对误差 | Mean Absolute Error, MAE |  |
| 101 | 交叉熵 | Cross-Entropy | 交叉熵损失 |
| 102 | 对数似然 | Log-Likelihood | 对数概率 |
| 103 | 困惑度 | Perplexity | 混乱度 |
| 104 | 准确率-召回率曲线 | Precision-Recall Curve | P-R曲线 |
| 105 | 置信度 | Confidence | 置信水平 |
| 106 | 置信区间 | Confidence Interval | 信赖区间 |

---

## 五、概率与统计术语

| 序号 | 中文术语（标准） | 英文术语 | 禁用变体 |
|:---:|:---|:---|:---|
| 107 | 概率分布 | Probability Distribution | 几率分布 |
| 108 | 期望 | Expectation / Expected Value | 期望值 |
| 109 | 方差 | Variance | 变异数 |
| 110 | 标准差 | Standard Deviation | 标准偏差 |
| 111 | 协方差 | Covariance | 共变异数 |
| 112 | 相关系数 | Correlation Coefficient |  |
| 113 | 正态分布 | Normal Distribution | 常态分布、高斯分布 |
| 114 | 高斯分布 | Gaussian Distribution | 正态分布 | 与Normal同义 |
| 115 | 贝叶斯定理 | Bayes' Theorem | 贝叶斯公式 |
| 116 | 先验概率 | Prior Probability | 事前概率 |
| 117 | 后验概率 | Posterior Probability | 事后概率 |
| 118 | 似然 | Likelihood | 似然函数 |
| 119 | 最大似然估计 | Maximum Likelihood Estimation, MLE | 最大似然 |
| 120 | 最大后验估计 | Maximum A Posteriori, MAP |  |
| 121 | 蒙特卡洛方法 | Monte Carlo Method | 蒙特卡罗 |
| 122 | 马尔可夫链 | Markov Chain | 马氏链 |
| 123 | 马尔可夫决策过程 | Markov Decision Process, MDP |  |
| 124 | 隐马尔可夫模型 | Hidden Markov Model, HMM |  |
| 125 | 贝叶斯网络 | Bayesian Network | 贝叶斯信念网络 |
| 126 | 高斯过程 | Gaussian Process, GP |  |
| 127 | 采样 | Sampling | 抽样 |

---

## 六、线性代数与微积分术语

| 序号 | 中文术语（标准） | 英文术语 | 禁用变体 |
|:---:|:---|:---|:---|
| 128 | 向量 | Vector | 矢量 |
| 129 | 矩阵 | Matrix |  |
| 130 | 张量 | Tensor | 张量（深度学习专用）|
| 131 | 标量 | Scalar | 纯量 |
| 132 | 矩阵乘法 | Matrix Multiplication | 矩阵相乘 |
| 133 | 点积 | Dot Product | 内积、点乘 |
| 134 | 外积 | Outer Product | 外乘 |
| 135 | 转置 | Transpose | 转置矩阵 |
| 136 | 逆矩阵 | Inverse Matrix | 反矩阵 |
| 137 | 行列式 | Determinant |  |
| 138 | 特征值 | Eigenvalue | 本征值 |
| 139 | 特征向量 | Eigenvector | 本征向量 |
| 140 | 奇异值分解 | Singular Value Decomposition, SVD |  |
| 141 | 迹 | Trace | 矩阵的迹 |
| 142 | 范数 | Norm | 模长 |
| 143 | 导数 | Derivative | 微商 |
| 144 | 偏导数 | Partial Derivative | 偏微商 |
| 145 | 梯度 | Gradient |  |
| 146 |  Hessian矩阵 | Hessian Matrix | 海森矩阵 |
| 147 |  Jacobian矩阵 | Jacobian Matrix | 雅可比矩阵 |
| 148 | 链式法则 | Chain Rule | 连锁律 |
| 149 | 泰勒展开 | Taylor Expansion | 泰勒级数 |
| 150 | 拉格朗日乘子 | Lagrange Multiplier | 拉氏乘子 |

---

## 七、工程实践术语

| 序号 | 中文术语（标准） | 英文术语 | 禁用变体 |
|:---:|:---|:---|:---|
| 151 | 模型量化 | Quantization | 量化压缩 |
| 152 | 模型剪枝 | Pruning | 剪枝压缩 |
| 153 | 知识蒸馏 | Knowledge Distillation | 知识提取 |
| 154 | 神经架构搜索 | Neural Architecture Search, NAS | 神经网络架构搜索 |
| 155 | 超参数优化 | Hyperparameter Optimization, HPO | 超参数调优 |
| 156 | MLOps | MLOps | 机器学习运维 | 保留英文 |
| 157 | 特征工程 | Feature Engineering | 特征提取 |
| 158 | 特征选择 | Feature Selection | 特征筛选 |
| 159 | 特征提取 | Feature Extraction | 特征抽取 |
| 160 | 降维 | Dimensionality Reduction | 维度降低 |
| 161 | 主成分 | Principal Component | 主分量 |
| 162 | 流水线 | Pipeline | 管道 |
| 163 | 交叉验证 | Cross-Validation | 交叉确认 |
| 164 | K折交叉验证 | K-Fold Cross-Validation |  |
| 165 | 网格搜索 | Grid Search |  |
| 166 | 随机搜索 | Random Search |  |
| 167 | 贝叶斯优化 | Bayesian Optimization |  |
| 168 | A/B测试 | A/B Testing | 对照实验 |
| 169 | 蓝绿部署 | Blue-Green Deployment |  |
| 170 | 金丝雀发布 | Canary Release | 灰度发布 |
| 171 | 漂移检测 | Drift Detection | 分布漂移检测 |
| 172 | 概念漂移 | Concept Drift | 概念变化 |
| 173 | 数据漂移 | Data Drift | 数据分布变化 |
| 174 | 模型监控 | Model Monitoring | 模型观测 |
| 175 | 可解释性 | Explainability | 可解释AI、XAI |
| 176 | 公平性 | Fairness | 公正性 |
| 177 | 鲁棒性 | Robustness | 健壮性、稳健性 |
| 178 | 隐私保护 | Privacy Preservation | 隐私计算 |
| 179 | 差分隐私 | Differential Privacy | 差分隐私保护 |
| 180 | 联邦学习 | Federated Learning | 联合学习 |

---

## 八、前沿技术术语

| 序号 | 中文术语（标准） | 英文术语 | 禁用变体 |
|:---:|:---|:---|:---|
| 181 | 大语言模型 | Large Language Model, LLM | 大型语言模型 |
| 182 | 基础模型 | Foundation Model | 基石模型 |
| 183 | 提示工程 | Prompt Engineering | 提示词工程 |
| 184 | 上下文学习 | In-Context Learning | 情境学习 |
| 185 | 思维链 | Chain of Thought, CoT | 思考链 |
| 186 | 检索增强生成 | Retrieval-Augmented Generation, RAG |  |
| 187 | 指令微调 | Instruction Tuning | 指令调优 |
| 188 | 人类反馈强化学习 | RLHF | 来自人类反馈的RL |
| 189 | 对齐 | Alignment | 价值对齐 |
| 190 | 涌现能力 | Emergent Ability | 突现能力 |
| 191 | 缩放定律 | Scaling Law | 规模定律 |
| 192 | 零样本 | Zero-shot | 零示例 |
| 193 | 少样本 | Few-shot | 少示例 |
| 194 | 思维树 | Tree of Thoughts, ToT |  |
| 195 | 自监督学习 | Self-Supervised Learning | 自监督 |
| 196 | 对比学习 | Contrastive Learning | 对比式学习 |
| 197 | 掩码语言模型 | Masked Language Model, MLM |  |
| 198 | 因果推断 | Causal Inference | 因果推理 |
| 199 | 潜在结果框架 | Potential Outcomes Framework | Rubin因果模型 |
| 200 | 结构因果模型 | Structural Causal Model, SCM | Pearl因果模型 |

---

## 九、常见术语错误对照（需重点检查）

| 错误写法 | 正确写法 | 出现位置检查 |
|:---|:---|:---:|
| 神经网路 | 神经网络 | 全文搜索 |
| 梯度下將 | 梯度下降 | 全文搜索 |
| 代价函數 | 损失函数 | 全文搜索 |
| 優化器 | 优化器 | 全文搜索 |
| 參數 | 参数 | 全文搜索 |
| 學习率 | 学习率 | 全文搜索 |
| 過拟合 | 过拟合 | 全文搜索 |
| 層 | 层 | 全文搜索 |
| 權重 | 权重 | 全文搜索 |
| 損失 | 损失 | 全文搜索 |

**注意**: 以上繁体字或错别字需要全文排查替换。

---

## 十、术语使用规范

### 10.1 首次出现规范
- 首次出现时：**中文术语（英文术语，缩写）**
- 示例："卷积神经网络（Convolutional Neural Network，CNN）"

### 10.2 后续使用规范
- 优先使用中文术语
- 代码中保留英文
- 公式中保留英文符号

### 10.3 代码注释规范
- 代码注释统一使用中文
- 关键术语保留英文对照
- 示例：`# 计算损失函数（loss function）`

### 10.4 公式书写规范
- 公式中使用英文符号
- 公式后紧跟中文解释
- 关键变量首次出现时解释

---

## 附录：术语索引（按拼音排序）

A-D: 激活函数、注意力机制、贝叶斯定理、变量、残差连接、层归一化  
E-H: 泛化、非监督学习、梯度下降、概率分布、过拟合、核函数  
I-L: 激活、监督学习、交叉熵、梯度、聚类、卷积  
M-P: 马尔可夫链、模型、批量归一化、偏导数、迁移学习、前向传播  
Q-T: 欠拟合、特征、梯度下降、推理、微调、Transformer  
U-Z: 验证集、维度、无监督学习、先验概率、训练集、正则化

---

*本术语表作为全书校对标准，确保术语一致性。*
