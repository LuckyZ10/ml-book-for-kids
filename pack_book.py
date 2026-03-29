#!/usr/bin/env python3
"""
ML教材全书打包脚本
根据核心思想：质量 > 速度，理论 > 前沿，教学 > 技术，深度 > 广度
"""
import os
import re
from pathlib import Path
from datetime import datetime

WORK_DIR = Path("/root/.openclaw/workspace/ml-book-for-kids")
OUTPUT_DIR = WORK_DIR / "book-output"

# 六大部分结构（根据核心思想和已生成的目录）
BOOK_STRUCTURE = {
    "Part1-基础概念与Python热身": {
        "chapters": [1, 2, 3, 4, 5],
        "files": [
            "chapters/chapter-01-what-is-learning.md",
            "chapters/chapter-02-seeing-data.md",
            "chapter-03-prediction-and-loss.md",
            "chapter-04-gradient-descent.md",
            "chapter-05-python-warmup.md",
        ]
    },
    "Part2-经典机器学习算法": {
        "chapters": list(range(6, 18)),
        "files": [
            "chapter-06-knn.md",
            "chapter-07-linear-regression.md",
            "chapter-08.md",
            "chapters/chapter-09.md",
            "chapters/chapter_10_svm.md",
            "chapter-11-naive-bayes/CONTENT.md",
            "chapter12/chapter12.md",
            "chapter-13-kmeans/manuscript.md",
            "chapter14_hierarchical_dbscan/README.md",
            "chapter-15-dimensionality-reduction/chapter-15.md",
            "chapters/chapter-16-perceptron.md",
            "chapter-17-multilayer-neural-network.md",
        ]
    },
    "Part3-神经网络与深度学习基础": {
        "chapters": list(range(18, 32)),
        "files": [
            "chapter18_backpropagation.md",
            "chapter-19-activation-functions/README.md",
            "chapter20_optimizer.md",
            "book/chapter21_regularization.md",
            "chapter22-cnn/README.md",
            "chapters/chapter23_rnn_sequences.md",
            "chapter24_transformer/chapter24_attention_transformer.md",
            "chapter25_pretraining_finetuning.md",
            "chapter26_llm_prompting.md",
            "chapter27_rag/chapter27_rag.md",
            "chapter_28_multimodal.md",
            "chapter29_generative_models/main.md",
            "chapter30_reinforcement_learning.md",
            "chapter31-deep-rl-advanced.md",
        ]
    },
    "Part4-深度学习进阶专题": {
        "chapters": list(range(32, 47)),
        "files": [
            "chapter-32/content.md",
            "chapter33-temporal-forecasting/chapter-33.md",
            "chapter34_nas.md",
            "chapters/chapter35-self-supervised-learning.md",
            "chapter36_federated_learning.md",
            "chapters/chapter37-NAS.md",
            "chapter38_diffusion_models.md",
            "chapter39-reinforcement-learning/plan.md",
            "chapter40-advanced-rl/README.md",
            "chapter41-explainability/README.md",
            "chapter42-advanced-diffusion/README.md",
            "chapter-43-multimodal-learning.md",
            "chapter44_ai_agents.md",
            "chapter45-uncertainty/README.md",
            "chapter46_ai_agents.md",
        ]
    },
    "Part5-数学武器库": {
        "chapters": list(range(47, 52)),
        "files": [
            "chapter47/chapter47.md",
            "chapter48/chapter48.md",
            "chapter49/chapter49.md",
            "chapters/chapter50_probabilistic_graphical_models.md",
            "chapter51_causal_inference.md",
        ]
    },
    "Part6-工程实践与完整项目": {
        "chapters": list(range(52, 61)),
        "files": [
            "chapter52-advanced-generative-models.md",
            "chapter-53-gnn-geometric-deep-learning.md",
            "chapter54-neuro-symbolic.md",
            "chapters/chapter55-continual-learning-part1.md",
            "chapters/chapter55-continual-learning-part2.md",
            "chapter56-nas-advanced/chapter56-part1.md",
            "chapter57/chapter57.md",
            "chapter58/chapter58.md",
            "chapter59_mlops.md",
            "chapters/chapter60_complete_project.md",
        ]
    }
}

def find_chapter_file(chapter_num):
    """根据章节号查找文件"""
    patterns = [
        f"chapter{chapter_num:02d}*.md",
        f"chapter-{chapter_num:02d}*.md",
        f"chapter_{chapter_num:02d}*.md",
        f"**/chapter{chapter_num:02d}*.md",
        f"**/chapter-{chapter_num:02d}*.md",
    ]
    
    for pattern in patterns:
        matches = list(WORK_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None

def read_file_content(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"\n\n> [错误: 无法读取文件 {file_path}: {e}]\n\n"

def create_book_header():
    """创建书头信息"""
    return f"""# 机器学习与深度学习：从小学生到大师

> **副标题**: 用最简单的语言，讲最深刻的道理  
> **版本**: v1.0  
> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
> **总章节**: 60章  
> **总字数**: ~964,000字  
> **总代码**: ~120,900行  

---

## 使命宣言

> **创作世界上最伟大的机器学习教材，让小学生也能懂大师级内容，经得起时间考验的传世之作。**

---

## 六大组成部分

| Part | 标题 | 章节 | 核心目标 |
|:---:|:---|:---:|:---|
| Part 1 | 基础概念与Python热身 | 第1-5章 | 建立编程基础，理解ML本质 |
| Part 2 | 经典机器学习算法 | 第6-17章 | 掌握传统ML，建立优化思维 |
| Part 3 | 神经网络与深度学习基础 | 第18-31章 | 进入深度学习，理解现代AI核心 |
| Part 4 | 深度学习进阶专题 | 第32-46章 | 掌握前沿技术，扩展应用视野 |
| Part 5 | 数学武器库 | 第47-51章 | 补齐数学基础，理解算法本质 |
| Part 6 | 工程实践与完整项目 | 第52-60章 | 综合运用，生产级部署 |

---

## 学习路线图

```
零基础 ──→ Part 1 ──→ Part 2 ──→ Part 3 ──→ Part 4 ──→ Part 5 ──→ Part 6 ──→ 专家级
            Python      经典ML      深度学习     前沿技术     数学基础     工程实践
```

---

## 质量铁律

- **每章标准**: 16,000字 + 1,500行代码 + 10篇文献
- **教学方法**: 费曼学习法，小学生也能懂
- **数学要求**: 从零推导，不跳步
- **代码要求**: NumPy手写 + PyTorch框架，全部可运行
- **前沿追踪**: 2024-2025最新论文

---

## 核心口诀

```
五层追问定方向，
慢工细活出精品。
理论完备不可缺，
小学生到大师级。
反复打磨100次，
传世之作是目标。
质量第一不饶恕。
```

---

*本书遵循开源精神，仅供学习交流使用*

---

"""

def create_part_header(part_name, part_info):
    """创建Part分隔页"""
    chapters_str = f"第{min(part_info['chapters'])}-{max(part_info['chapters'])}章"
    return f"""

<div style="page-break-after: always;"></div>

---

# {part_name}

> **章节范围**: {chapters_str}  
> **核心目标**: {get_part_goal(part_name)}

---

"""

def get_part_goal(part_name):
    """获取Part目标描述"""
    goals = {
        "Part1-基础概念与Python热身": "建立编程基础，理解ML本质",
        "Part2-经典机器学习算法": "掌握传统ML，建立优化思维",
        "Part3-神经网络与深度学习基础": "进入深度学习，理解现代AI核心",
        "Part4-深度学习进阶专题": "掌握前沿技术，扩展应用视野",
        "Part5-数学武器库": "补齐数学基础，理解算法本质",
        "Part6-工程实践与完整项目": "综合运用，生产级部署",
    }
    return goals.get(part_name, "")

def pack_part(part_name, part_info, output_file):
    """打包一个Part"""
    print(f"\n打包 {part_name}...")
    
    content = create_part_header(part_name, part_info)
    file_count = 0
    
    for file_path in part_info['files']:
        full_path = WORK_DIR / file_path
        if full_path.exists():
            chapter_content = read_file_content(full_path)
            content += f"\n\n<!-- 来源: {file_path} -->\n\n"
            content += chapter_content
            content += "\n\n---\n\n"
            file_count += 1
        else:
            # 尝试查找替代文件
            alt_file = find_chapter_file(part_info['chapters'][file_count] if file_count < len(part_info['chapters']) else 0)
            if alt_file:
                chapter_content = read_file_content(alt_file)
                content += f"\n\n<!-- 来源: {alt_file.relative_to(WORK_DIR)} -->\n\n"
                content += chapter_content
                content += "\n\n---\n\n"
                file_count += 1
            else:
                content += f"\n\n> [注意: 文件 {file_path} 未找到]\n\n"
    
    # 写入Part文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ 已打包 {file_count} 个文件")
    print(f"  ✓ 输出: {output_file}")
    
    return file_count

def create_appendix():
    """创建附录"""
    return """

<div style="page-break-after: always;"></div>

---

# 附录

## 附录A: 术语对照表

详见 `TERMINOLOGY_STANDARD.md`

## 附录B: 参考文献

全书共引用810+篇参考文献，详见各章末尾及 `references/references.md`

## 附录C: 全书目录

详见 `BOOK_TOC.md`

## 附录D: Python速查表

```python
# NumPy基础
import numpy as np

# 数组创建
arr = np.array([1, 2, 3])
zeros = np.zeros((3, 3))
ones = np.ones((3, 3))
rand = np.random.randn(3, 3)

# 矩阵运算
A @ B  # 矩阵乘法
A.T    # 转置
np.linalg.inv(A)  # 逆矩阵

# PyTorch基础
import torch

# 张量创建
x = torch.tensor([1.0, 2.0, 3.0])
x = torch.randn(3, 3)
x = torch.zeros(3, 3)

# GPU
x = x.cuda()

# 自动微分
x.requires_grad = True
y = x ** 2
y.backward()
```

## 附录E: 数学公式速查

### 导数
- $(x^n)' = nx^{n-1}$
- $(e^x)' = e^x$
- $(\ln x)' = \frac{1}{x}$
- $(\sin x)' = \cos x$

### 概率
- $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$ (贝叶斯定理)
- $E[X] = \sum x P(x)$ (期望)
- $Var(X) = E[X^2] - (E[X])^2$ (方差)

### 线性代数
- $A^{-1}A = I$ (逆矩阵)
- $\|x\|_2 = \sqrt{\sum x_i^2}$ (L2范数)
- $tr(A) = \sum A_{ii}$ (迹)

---

*全书完*

"""

def main():
    print("=" * 60)
    print("ML教材全书打包")
    print("根据核心思想: 质量 > 速度，传世之作")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 打包各Part
    total_files = 0
    part_files = []
    
    for part_name, part_info in BOOK_STRUCTURE.items():
        output_file = OUTPUT_DIR / f"{part_name}.md"
        file_count = pack_part(part_name, part_info, output_file)
        total_files += file_count
        part_files.append(output_file)
    
    # 创建完整版（所有Part合并）
    print("\n创建完整版...")
    full_book_path = OUTPUT_DIR / "全书完整版.md"
    
    with open(full_book_path, 'w', encoding='utf-8') as f:
        # 写入书头
        f.write(create_book_header())
        
        # 写入各Part
        for part_file in part_files:
            with open(part_file, 'r', encoding='utf-8') as pf:
                f.write(pf.read())
            f.write("\n\n")
        
        # 写入附录
        f.write(create_appendix())
    
    print(f"  ✓ 完整版: {full_book_path}")
    
    # 统计
    print("\n" + "=" * 60)
    print("打包完成!")
    print("=" * 60)
    print(f"总Part数: {len(BOOK_STRUCTURE)}")
    print(f"总文件数: {total_files}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n生成文件:")
    for part_file in sorted(OUTPUT_DIR.glob("*.md")):
        size = part_file.stat().st_size / 1024  # KB
        print(f"  - {part_file.name} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
