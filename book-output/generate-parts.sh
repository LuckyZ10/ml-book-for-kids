#!/bin/bash
# 生成ML教材分卷PDF

cd /root/.openclaw/workspace/ml-book-for-kids/book-output

HEADER="latex-header.tex"

echo "正在生成 Part1 PDF..."
pandoc "Part1-基础概念与Python热身.md" -o "ml-book-part1.pdf" \
  --pdf-engine=xelatex \
  -V CJKmainfont="Noto Serif CJK SC" \
  -V geometry:margin=2.5cm \
  -V colorlinks=true \
  --toc \
  --number-sections \
  --listings \
  -H "$HEADER" 2>&1 | tail -5

echo "正在生成 Part2 PDF..."
pandoc "Part2-经典机器学习算法.md" -o "ml-book-part2.pdf" \
  --pdf-engine=xelatex \
  -V CJKmainfont="Noto Serif CJK SC" \
  -V geometry:margin=2.5cm \
  -V colorlinks=true \
  --toc \
  --number-sections \
  --listings \
  -H "$HEADER" 2>&1 | tail -5

echo "正在生成 Part3 PDF..."
pandoc "Part3-神经网络与深度学习基础.md" -o "ml-book-part3.pdf" \
  --pdf-engine=xelatex \
  -V CJKmainfont="Noto Serif CJK SC" \
  -V geometry:margin=2.5cm \
  -V colorlinks=true \
  --toc \
  --number-sections \
  --listings \
  -H "$HEADER" 2>&1 | tail -5

echo "正在生成 Part4 PDF..."
pandoc "Part4-深度学习进阶专题.md" -o "ml-book-part4.pdf" \
  --pdf-engine=xelatex \
  -V CJKmainfont="Noto Serif CJK SC" \
  -V geometry:margin=2.5cm \
  -V colorlinks=true \
  --toc \
  --number-sections \
  --listings \
  -H "$HEADER" 2>&1 | tail -5

echo "正在生成 Part5 PDF..."
pandoc "Part5-数学武器库.md" -o "ml-book-part5.pdf" \
  --pdf-engine=xelatex \
  -V CJKmainfont="Noto Serif CJK SC" \
  -V geometry:margin=2.5cm \
  -V colorlinks=true \
  --toc \
  --number-sections \
  --listings \
  -H "$HEADER" 2>&1 | tail -5

echo "正在生成 Part6 PDF..."
pandoc "Part6-工程实践与完整项目.md" -o "ml-book-part6.pdf" \
  --pdf-engine=xelatex \
  -V CJKmainfont="Noto Serif CJK SC" \
  -V geometry:margin=2.5cm \
  -V colorlinks=true \
  --toc \
  --number-sections \
  --listings \
  -H "$HEADER" 2>&1 | tail -5

echo "所有分卷PDF生成完成！"
