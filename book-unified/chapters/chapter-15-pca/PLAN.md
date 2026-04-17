# Chapter-15-PCA 补充任务方案

## 📊 现状诊断

| 指标 | 当前 | 目标 | 差距 |
|------|------|------|------|
| README | 833行 | 1600+行 | -767行 |
| 代码文件 | 1个 | 4-5个 | -3个 |
| 代码总行 | 1216行 | 1500+行 | -284行 |
| 练习题 | 47行/6题 | 已达标 | ✅ |
| 参考文献 | 7篇 | 10+篇 | -3篇 |

## 🎯 补充计划

### Phase 1: README扩充 (+800行)
需要增加的内容：
1. **更多生活化比喻** (3-4个)
   - 照片压缩比喻 → 已存在
   - 需要增加：地图投影、影子游戏、调味比喻
   
2. **数学推导细节** (+300行)
   - 协方差矩阵的完整推导
   - 特征值分解的几何意义
   - t-SNE的KL散度推导
   
3. **算法变体** (+200行)
   - 核PCA (Kernel PCA)
   - 概率PCA (Probabilistic PCA)
   - 增量PCA (Incremental PCA)
   
4. **应用案例** (+200行)
   - 人脸识别 (Eigenfaces)
   - 基因数据分析
   - 推荐系统降维

5. **常见问题FAQ** (+100行)
   - PCA vs LDA
   - 如何选择主成分数量
   - 缺失值处理

### Phase 2: 代码重构 (1个文件 → 5个文件)

当前：`pca_tsne.py` (1216行)

目标结构：
```
code/
├── pca_numpy.py          # NumPy实现PCA (400行)
├── pca_torch.py          # PyTorch实现PCA (350行)
├── tsne_numpy.py         # NumPy实现t-SNE (450行)
├── pca_advanced.py       # 核PCA、概率PCA (400行)
└── visualization.py      # 可视化工具 (300行)
```

总计：1900行，超过1500行目标

### Phase 3: 参考文献补充 (+3篇)
需要增加的论文：
1. Kernel PCA论文 (Schölkopf et al., 1998)
2. Probabilistic PCA论文 (Tipping & Bishop, 1999)
3. Autoencoder与PCA关系论文

## ⏱️ 时间估算
- README扩充：2-3小时
- 代码重构：3-4小时
- 参考文献：30分钟
- 总计：6-8小时

## ⚠️ 注意
- 无需"删除源文件"步骤（chapters-old/不存在）
- 需要保留现有高质量内容
- 保持与chapter-14一致的写作风格
