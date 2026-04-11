# Source Content Tracking

## 融合来源记录

本章节内容基于以下高质量源文件融合整理：

### 主要来源
1. **chapters-old/chapter-14-hierarchical-clustering/manuscript.md** (1104行)
   - 层次聚类与DBSCAN的儿童向教程
   - 包含完整的费曼比喻和数学推导
   - 内容质量：⭐⭐⭐⭐⭐

2. **chapters-old/chapter-14-hierarchical-clustering/chapter-14.md** (27KB)
   - 补充技术细节和数学公式

### 代码来源
3. **chapters-old/chapter-14-hierarchical-clustering/code/hierarchical_dbscan.py** (514行)
   - Lance-Williams递推公式完整实现
   - 纯Python教学实现

4. **chapters-old/chapter-14-hierarchical-clustering/code/hierarchical_clustering.py** (396行)
   - AGNES和DIANA实现

5. **chapters-old/chapter-14-hierarchical-clustering/code/dbscan_clustering.py** (467行)
   - DBSCAN完整实现

### 融合操作记录
- **融合时间**: 2026-03-30
- **融合方式**: 整合manuscript.md内容 + 规范化代码结构
- **删除的源文件**: 
  - chapters-old/chapter-14-hierarchical-clustering/manuscript.md
  - chapters-old/chapter-14-hierarchical-clustering/chapter-14.md

## 内容升级记录

### 2026-04-12 大幅扩展与整理（本次工作）
1. **规范化文件结构**：将根目录下的3个Python文件（clustering_validation.py、dbscan_clustering.py、hierarchical_clustering.py）移动到code/目录
2. **大幅扩展README.md**：从约5300字扩展至16000+字，1587行
   - 新增"算法背后的历史故事"章节（林奈、Ward、Lance-Williams、Ester等）
   - 扩展费曼比喻从3个到8个
   - 详细展开6种链接方法（单链接、全链接、平均链接、Ward法、重心法、中位线法）
   - 新增OPTICS和HDBSCAN的介绍
   - 新增三大实战案例（客户分群、POI聚类、图像分割）
   - 新增算法选择决策指南和常见错误避坑指南
   - 新增附录D（费曼学习笔记与进阶问答）和附录E（跨章节联系图）
   - 补充大量Python实现详解和调试备忘录
3. **更新参考文献**：从12篇扩展至14篇，新增Ankerst (1999) OPTICS和Campello (2013) HDBSCAN

### 往期升级内容
1. 增加了PyTorch版本的层次聚类和DBSCAN实现
2. 扩展了参考文献至10+篇
3. 增加了3个新的费曼比喻
4. 创建了9道练习题
5. 补充了实战案例

### 规范化改进
1. 统一代码风格
2. 增加详细注释
3. 完善数学推导
4. 增加可视化示例
