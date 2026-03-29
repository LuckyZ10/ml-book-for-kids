#!/bin/bash
# 整合遗漏的章节内容到新结构中

cd /root/.openclaw/workspace/ml-book-for-kids

echo "=== 开始整合遗漏的章节内容 ==="

# 1. 整合 chapter-09-decision-tree 的 CONTENT.md
if [ -f "chapters-old/chapter-09-decision-tree/CONTENT.md" ]; then
    echo "整合 chapter-09-decision-tree CONTENT.md (655行)..."
    cp chapters-old/chapter-09-decision-tree/CONTENT.md chapters/chapter-09-decision-tree/CONTENT.md
fi

# 2. 整合 chapter-12-ensemble 的 manuscript.md
if [ -f "chapters-old/chapter-12-ensemble-learning/manuscript.md" ]; then
    echo "整合 chapter-12-ensemble manuscript.md (259行)..."
    cp chapters-old/chapter-12-ensemble-learning/manuscript.md chapters/chapter-12-ensemble/manuscript.md
fi

# 3. 整合 chapter-13-kmeans 的 manuscript.md
if [ -f "chapters-old/chapter-13-kmeans-clustering/manuscript.md" ]; then
    echo "整合 chapter-13-kmeans manuscript.md (325行)..."
    cp chapters-old/chapter-13-kmeans-clustering/manuscript.md chapters/chapter-13-kmeans/manuscript.md
fi

# 4. 整合 chapter-14-hierarchical-dbscan 的 manuscript.md (1104行，高质量内容)
if [ -f "chapters-old/chapter-14-hierarchical-clustering/manuscript.md" ]; then
    echo "整合 chapter-14-hierarchical manuscript.md (1104行)..."
    cp chapters-old/chapter-14-hierarchical-clustering/manuscript.md chapters/chapter-14-hierarchical-dbscan/manuscript.md
fi

# 5. 整合 book/ 中的 chapter21 内容
if [ -f "book/chapter21_regularization.md" ]; then
    echo "整合 book/chapter21_regularization.md (506行)..."
    cp book/chapter21_regularization.md chapters/chapter-21-regularization/book_content.md
fi

if [ -f "book/chapter21_regularization_part1.md" ]; then
    echo "整合 book/chapter21_regularization_part1.md (352行)..."
    cp book/chapter21_regularization_part1.md chapters/chapter-21-regularization/book_content_part1.md
fi

# 6. 整合 chapters-unified/ 中的代码
if [ -d "chapters-unified/chapter-37" ]; then
    echo "整合 chapters-unified/chapter-37 代码..."
    cp chapters-unified/chapter-37/code/*.py chapters/chapter-37-nas-advanced/code/ 2>/dev/null || true
fi

echo ""
echo "=== 整合完成 ==="
echo "已整合的内容:"
echo "- chapter-09-decision-tree/CONTENT.md"
echo "- chapter-12-ensemble/manuscript.md"
echo "- chapter-13-kmeans/manuscript.md"
echo "- chapter-14-hierarchical-dbscan/manuscript.md (1104行高质量内容)"
echo "- chapter-21-regularization/book_content.md"
echo "- chapter-21-regularization/book_content_part1.md"
echo "- chapter-37-nas-advanced/code/ (来自chapters-unified)"
