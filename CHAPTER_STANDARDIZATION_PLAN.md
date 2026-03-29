# 章节目录规范化计划

## 标准结构

每个章节必须包含：
```
chapter-XX-主题/
├── README.md           # 主章节内容 (必须)
├── code/               # 代码子目录 (必须)
│   ├── xxx_numpy.py   # NumPy手写实现
│   └── xxx_torch.py   # PyTorch框架实现
├── exercises.md        # 练习题 (9道: 3基础+3进阶+3挑战)
└── references.bib      # 参考文献 (APA格式, 10+篇)
```

## 当前状态统计

| 项目 | 数量 | 占比 |
|:---|:---:|:---:|
| 总章节 | 60 | 100% |
| 有 README.md | ~40 | 67% |
| 有 code/ | ~40 | 67% |
| 有 exercises.md | 1 | 2% |
| 有 references.bib | 1 | 2% |
| **完全规范** | **1** | **2%** |

## 规范化策略

### 策略1: 新章节直接按规范写 ✅
- chapter-12-ensemble 及之后的新写章节
- 严格执行4件套标准

### 策略2: 已有章节逐步补充
按优先级分批处理：

#### P0 - 核心章节 (先补充练习和文献)
- chapter-01 到 chapter-10 (基础部分)
- chapter-16 到 chapter-21 (深度学习基础)

#### P1 - 重要章节
- chapter-22 到 chapter-28 (深度学习进阶)
- chapter-35 到 chapter-42 (前沿专题)
- chapter-50, chapter-60 (数学/项目)

#### P2 - 待补全内容章节
- chapter-12 到 chapter-15 (缺失README)
- chapter-19, chapter-27 (缺失README)
- chapter-29 到 chapter-34 (缺失README)
- chapter-36 (缺失README)

#### P3 - 空壳章节 (待写作)
- chapter-43 到 chapter-59 (目前只有目录)

## 执行计划

### 阶段1: 新章节规范化 (进行中)
- [x] chapter-11-naive-bayes ✅ 完全规范
- [ ] chapter-12-ensemble (写作中)
- [ ] chapter-13-kmeans
- [ ] chapter-14-hierarchical-dbscan
- [ ] chapter-15-pca
- [ ] chapter-19-activation

### 阶段2: 核心章节补充练习和文献
当阶段1完成后，回头补充：
- [ ] chapter-01 到 chapter-10 的 exercises.md + references.bib
- [ ] chapter-16 到 chapter-21 的 exercises.md + references.bib

### 阶段3: 重要章节补充
- [ ] chapter-22 到 chapter-28
- [ ] chapter-35 到 chapter-42
- [ ] chapter-50, chapter-60

### 阶段4: 缺失README的章节重写
这些章节目前内容不完整或只有代码，需要重写：
- [ ] chapter-12 集成学习 (重写)
- [ ] chapter-13 K-Means (重写)
- [ ] chapter-14 层次聚类 (重写)
- [ ] chapter-15 PCA (重写)
- [ ] chapter-19 激活函数 (重写)
- [ ] chapter-27 RAG (重写)
- [ ] chapter-29 GAN (重写)
- [ ] chapter-30 强化学习 (重写)
- [ ] chapter-31 深度RL (重写)
- [ ] chapter-32 GNN (重写)
- [ ] chapter-33 时序预测 (重写)
- [ ] chapter-34 NAS基础 (重写)
- [ ] chapter-36 联邦学习 (重写)

### 阶段5: 空壳章节写作
- [ ] chapter-43 到 chapter-59

## 时间估算

| 阶段 | 章节数 | 时间/章 | 总时间 |
|:---|:---:|:---:|:---:|
| 阶段1 | 6章 | 3-4小时 | 3-4天 |
| 阶段2 | 16章 | 1-2小时 | 3-5天 |
| 阶段3 | 10章 | 1-2小时 | 2-3天 |
| 阶段4 | 13章 | 3-4小时 | 5-7天 |
| 阶段5 | 17章 | 3-4小时 | 7-10天 |
| **总计** | **62章** | - | **20-30天** |

## 当前决策

**按用户建议**: "按你自己的节奏来"

**我的节奏**:
1. **先完成阶段1** (新章节规范化) - 保证新写的都是完美的
2. **穿插阶段2** (核心章节补充) - 在写新章节的间隙补充旧章节
3. **每完成一个章节，更新本计划**

**不追求速度，追求质量** - 每个章节都要经得起100遍打磨！

---

*计划创建时间: 2026-03-30*  
*最后更新: 2026-03-30*


## 有价值但遗漏的内容 (已整合)

### 已整合到 chapters/ 目录的内容:
- chapters-old/chapter-09-decision-tree/CONTENT.md (655行) → chapter-09-decision-tree/
- chapters-old/chapter-12-ensemble/manuscript.md (259行) → chapter-12-ensemble/
- chapters-old/chapter-13-kmeans/manuscript.md (325行) → chapter-13-kmeans/
- chapters-old/chapter-14-hierarchical/manuscript.md (1104行) → chapter-14-hierarchical-dbscan/
- book/chapter21_regularization.md (506行) → chapter-21-regularization/
- book/chapter21_regularization_part1.md (352行) → chapter-21-regularization/
- chapters-unified/chapter-37/code/*.py → chapter-37-nas-advanced/code/

**说明**: 这些内容可作为重写时的参考材料，特别是 chapter-14-hierarchical 的 1104 行 manuscript.md 是高质量内容。

## 建议

在重写每个章节前，先检查 chapters-old/ 中是否有可用内容，避免重复劳动。
