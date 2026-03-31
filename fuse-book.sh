#!/bin/bash
# 融合脚本：将3个Part合并为统一的60章书籍

cd /root/.openclaw/workspace/ml-book-for-kids

# 清空目标目录
rm -rf book-unified/chapters/*
mkdir -p book-unified/chapters

echo "开始融合60章..."

# Part A: 1-20章
echo "融合 Part A (1-20章)..."
cp -r chapters/part-a-foundations/chapters/* book-unified/chapters/ 2>/dev/null

# Part B: 21-40章
echo "融合 Part B (21-40章)..."
cp -r chapters/part-b-deep-learning/chapters/* book-unified/chapters/ 2>/dev/null

# Part C: 41-60章
echo "融合 Part C (41-60章)..."
cp -r chapters/part-c-advanced/chapters/* book-unified/chapters/ 2>/dev/null

# 统计
count=$(ls book-unified/chapters/ | grep -c "chapter-")
echo "融合完成：共 ${count} 章"

# 创建完整目录清单
echo "生成目录清单..."
ls book-unified/chapters/ | sort -V > book-unified/TOC.txt

echo "书籍融合完成！"
