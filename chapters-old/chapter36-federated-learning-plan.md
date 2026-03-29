# 第三十六章：联邦学习 (Federated Learning)

## 章节规划

### 36.1 什么是联邦学习？
- 联邦学习的诞生背景与动机
- 三大学习范式对比：集中式 vs 分布式 vs 联邦式
- 联邦学习的核心挑战（统计异质性、系统异质性、隐私安全）

### 36.2 FedAvg算法
- 基础FedAvg算法原理
- 本地训练与全局聚合的数学推导
- 收敛性分析

### 36.3 联邦优化算法进阶
- FedProx：处理非独立同分布数据
- SCAFFOLD：方差缩减与本地状态
- FedNova：标准化本地更新

### 36.4 差分隐私
- 差分隐私基础概念
- DP-SGD在联邦学习中的应用
- 隐私预算(ε, δ)分析

### 36.5 安全聚合
- 安全多方计算基础
- 秘密共享技术
- 安全聚合协议

### 36.6 个性化联邦学习
- 本地个性化技术
- Meta-Learning for FL
- 聚类联邦学习

### 36.7 应用场景与前沿
- 移动键盘预测
- 医疗健康
- 金融风控
- 跨设备/跨机构联邦

### 36.8 练习题
- 3基础 + 3进阶 + 3挑战

## 预期产出
- 字数: ~15,000字
- 代码: ~900行
- 参考文献: ~12篇

## 费曼法比喻构思
- 联合考试：学生各自在家做题，只提交答案统计
- 联邦制国家：各州保持自治但遵循联邦宪法
- 合唱团：各声部独立练习，最后合成

## 核心论文清单
- [ ] McMahan et al. 2017 - FedAvg (AISTATS)
- [ ] Kairouz et al. 2019 - FL Survey
- [ ] Dwork & Roth 2014 - 差分隐私教材
- [ ] Sattler et al. 2019 - Sparse Ternary Compression
- [ ] Li et al. 2020 - FedProx
- [ ] Karimireddy et al. 2020 - SCAFFOLD
- [ ] Wang et al. 2020 - FedNova
- [ ] Bonawitz et al. 2017 - Secure Aggregation
- [ ] Duchi et al. 2014 - 隐私优化
- [ ] Hsu et al. 2019 - FedAvg收敛性分析
