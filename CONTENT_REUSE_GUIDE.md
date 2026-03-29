# 章节内容复用指南

> 重写章节前必读！先查本指南，避免重复劳动。

---

## 高价值内容 (可直接作为基础)

| 章节 | 来源文件 | 行数 | 内容质量 | 建议 |
|:---|:---|:---:|:---:|:---|
| **chapter-14** | chapters-old/chapter-14-hierarchical/manuscript.md | 1104 | ⭐⭐⭐⭐⭐ | 高质量，建议直接在其基础上扩展 |
| **chapter-57** | chapters-old/chapter-57-meta-learning/chapter57.md | 3383 | ⭐⭐⭐⭐ | 内容充实，需要规范化 |
| **chapter-31** | chapters-old/chapter-31-deep-rl/README.md | 3179 | ⭐⭐⭐⭐ | 深度RL内容完整 |
| **chapter-58** | chapters-old/chapter-58-few-shot-learning/chapter58.md | 2703 | ⭐⭐⭐⭐ | 少样本学习，内容充实 |
| **chapter-26** | chapters-old/chapter-26-llm-prompting/README.md | 2636 | ⭐⭐⭐⭐ | LLM提示工程，较新内容 |
| **chapter-30** | chapters-old/chapter-30-reinforcement-learning/README.md | 2513 | ⭐⭐⭐⭐ | RL基础完整 |
| **chapter-59** | chapters-old/chapter-59-mlops/README.md | 2484 | ⭐⭐⭐⭐ | MLOps内容详细 |
| **chapter-23** | chapters-old/chapter-23-rnn/README.md | 2422 | ⭐⭐⭐⭐ | RNN/LSTM/GRU完整 |
| **chapter-35** | chapters-old/chapter-35-self-supervised/README.md | 2388 | ⭐⭐⭐⭐ | 自监督学习，内容新 |
| **chapter-17** | chapters-old/chapter-17-neural-network/README.md | 2383 | ⭐⭐⭐⭐ | 神经网络基础完整 |

---

## 中等价值内容 (可作为参考)

| 章节 | 来源文件 | 行数 | 建议 |
|:---|:---|:---:|:---|
| chapter-12 | chapters-old/chapter-12-ensemble/chapter12.md | 259 | 内容较少，需大幅扩展 |
| chapter-13 | chapters-old/chapter-13-kmeans/manuscript.md | 325 | 内容较少，需大幅扩展 |
| chapter-20 | chapters-old/chapter-20-optimizers/README.md | 2255 | 优化器内容较完整 |
| chapter-21 | chapters-old/chapter-21-regularization/README.md | 1968 | 正则化内容完整 |
| chapter-18 | chapters-old/chapter-18-backpropagation/README.md | 1950 | 反向传播完整 |
| chapter-22 | chapters-old/chapter-22-cnn/README.md | 1867 | CNN内容完整 |
| chapter-19 | chapters-old/chapter-19-activation/chapter-19.md | 1779 | 激活函数内容完整 |
| chapter-27 | chapters-old/chapter-27-rag/chapter27_rag.md | 1770 | RAG内容较新 |
| chapter-48 | chapters-old/chapter-48-deep-ensemble/chapter48.md | 1764 | 深度集成/不确定性 |
| chapter-36 | chapters-old/chapter-36-federated/README.md | 1716 | 联邦学习内容完整 |
| chapter-53 | chapters-old/chapter-53-gnn-geometric/README.md | 1635 | GNN几何深度学习 |
| chapter-33 | chapters-old/chapter-33-temporal-forecasting/chapter-33.md | 1585 | 时序预测内容完整 |
| chapter-43 | chapters-old/chapter-43-multimodal-frontier/README.md | 1575 | 多模态前沿内容 |

---

## 需要重点重写的章节 (内容缺失或不足)

| 章节 | 当前状态 | 建议 |
|:---|:---|:---|
| chapter-12 | 只有259行manuscript.md | 需重写，参考Random Forest/XGBoost经典论文 |
| chapter-13 | 只有325行manuscript.md | 需重写，参考K-Means经典论文 |
| chapter-29 | 无内容 | 需从零写，参考GAN经典论文 |
| chapter-32 | 无内容 | 需从零写，参考GNN经典论文 |
| chapter-34 | 只有代码 | 需写完整内容，参考NAS综述 |
| chapter-44 | 无内容 | 需从零写，参考AI Agent最新论文 |
| chapter-45 | 无内容 | 需从零写，参考不确定性量化 |
| chapter-46 | 无内容 | 需从零写，参考神经符号AI |
| chapter-47 | 有1716行但可能是贝叶斯优化 | 需确认主题是否符合"线性代数" |
| chapter-49 | 有内容但可能是物理信息ML | 需确认主题是否符合"概率论" |
| chapter-51 | 有因果推断内容 | 需规范化 |
| chapter-52 | 有生成模型内容 | 需规范化 |
| chapter-54 | 有神经符号内容 | 需规范化 |
| chapter-55 | 有持续学习内容 | 需规范化 |
| chapter-56 | 有HPO内容 | 需规范化 |

---

## 按优先级的内容获取策略

### P0 - 核心基础章节

**chapter-12 集成学习**
- 可用: chapters-old/chapter-12-ensemble/chapter12.md (259行)
- 策略: 在此基础上大幅扩展，补充Bagging/Boosting数学推导
- 参考: Breiman (2001) Random Forests, Chen & Guestrin (2016) XGBoost

**chapter-13 K-Means聚类**
- 可用: chapters-old/chapter-13-kmeans/manuscript.md (325行)
- 策略: 重写，现有内容太少
- 参考: Lloyd (1982), Arthur & Vassilvitskii (2007) k-means++

**chapter-14 层次聚类与DBSCAN**
- 可用: ⭐ **chapters-old/chapter-14-hierarchical/manuscript.md (1104行)**
- 策略: 直接使用，补充DBSCAN内容即可
- 这是高质量内容！

**chapter-15 PCA降维**
- 可用: chapters-old/chapter-15-dimensionality-reduction/chapter-15.md
- 策略: 重写，现有内容不完整
- 参考: Pearson (1901), Hotelling (1933)

**chapter-19 激活函数**
- 可用: chapters-old/chapter-19-activation/chapter-19.md (1779行)
- 策略: 直接使用，补充Swish/GELU等现代激活函数

### P1 - 热门实用章节

**chapter-27 RAG**
- 可用: chapters-old/chapter-27-rag/chapter27_rag.md (1770行)
- 策略: 直接使用，这是较新内容

**chapter-29 GAN**
- 可用: 无
- 策略: 从零写，参考Goodfellow (2014), Radford (2016) DCGAN, Karras (2019) StyleGAN

**chapter-30 RL基础**
- 可用: chapters-old/chapter-30-reinforcement-learning/README.md (2513行)
- 策略: 直接使用，内容完整

**chapter-31 深度RL**
- 可用: chapters-old/chapter-31-deep-rl/README.md (3179行)
- 策略: 直接使用，内容完整

**chapter-32 GNN**
- 可用: 无
- 策略: 从零写，参考Kipf & Welling (2017) GCN, Veličković (2018) GAT

**chapter-33 时序预测**
- 可用: chapters-old/chapter-33-temporal-forecasting/chapter-33.md (1585行)
- 策略: 直接使用，补充Informer/Autoformer

### P2 - 前沿专题

**chapter-43 多模态前沿**
- 可用: chapters-old/chapter-43-multimodal-frontier/README.md (1575行)
- 策略: 直接使用

**chapter-44 AI Agent**
- 可用: 无
- 策略: 从零写，参考ReAct, AutoGPT, Multi-Agent最新论文

**chapter-45 不确定性**
- 可用: chapters-old/chapter-48-deep-ensemble/chapter48.md (1764行)
- 策略: 使用深度集成内容，补充MC Dropout/BNN

**chapter-46 神经符号AI**
- 可用: chapters-old/chapter-54-neuro-symbolic-advanced/README.md (2030行)
- 策略: 使用第54章内容

---

## 重写前检查清单

在重写每个章节前：

1. [ ] 查本指南，看是否有可用内容
2. [ ] 检查 chapters-old/ 中是否有对应文件
3. [ ] 检查 deprecated/ 中是否有对应文件
4. [ ] 如果有可用内容，评估是否需要重写还是直接规范化
5. [ ] 确认参考文献是否需要更新

---

## 快速查找命令

```bash
# 查找某章节的所有相关文件
find chapters-old/ deprecated/ -type f -name "*chapter-XX*" 2>/dev/null

# 查看文件行数
wc -l chapters-old/chapter-XX-主题/*.md

# 搜索特定主题
grep -r "主题关键词" chapters-old/ --include="*.md"
```

---

**记住**: 先复用，再创新。节省的时间用来打磨质量！

*最后更新: 2026-03-30*
