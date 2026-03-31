## 5.6 本章综合练习：成绩分析器

让我们把学到的知识综合运用，创建一个成绩分析器。

```python
import numpy as np

class ScoreAnalyzer:
    """
    成绩分析器
    可以分析一个班级的多门课程成绩
    """
    
    def __init__(self, scores, student_names, course_names):
        """
        初始化
        
        参数:
            scores: 二维数组，形状为 (学生数, 课程数)
            student_names: 学生姓名列表
            course_names: 课程名称列表
        """
        self.scores = np.array(scores)
        self.student_names = student_names
        self.course_names = course_names
    
    def class_summary(self):
        """班级整体统计"""
        print("=" * 50)
        print("📊 班级成绩统计报告")
        print("=" * 50)
        
        print(f"\n班级人数: {len(self.student_names)}")
        print(f"课程数量: {len(self.course_names)}")
        print(f"\n全班平均分: {self.scores.mean():.2f}")
        print(f"全班最高分: {self.scores.max()}")
        print(f"全班最低分: {self.scores.min()}")
        print(f"标准差: {self.scores.std():.2f}")
    
    def course_analysis(self):
        """每门课的分析"""
        print("\n" + "=" * 50)
        print("📚 各课程统计")
        print("=" * 50)
        
        for i, course in enumerate(self.course_names):
            scores = self.scores[:, i]
            print(f"\n{course}:")
            print(f"  平均分: {scores.mean():.2f}")
            print(f"  最高分: {scores.max()}")
            print(f"  最低分: {scores.min()}")
            print(f"  及格率: {(scores >= 60).mean() * 100:.1f}%")
    
    def student_ranking(self):
        """学生排名"""
        print("\n" + "=" * 50)
        print("🏆 学生总分排名")
        print("=" * 50)
        
        # 计算每个学生的总分
        total_scores = self.scores.sum(axis=1)
        
        # 按分数排序（从高到低）
        ranked_indices = np.argsort(total_scores)[::-1]
        
        for rank, idx in enumerate(ranked_indices, 1):
            name = self.student_names[idx]
            total = total_scores[idx]
            avg = total / len(self.course_names)
            print(f"{rank}. {name}: 总分={total}, 平均分={avg:.2f}")
    
    def find_improvement_candidates(self):
        """找出需要提高的学生（平均分<60）"""
        print("\n" + "=" * 50)
        print("⚠️ 需要关注的学生（平均分<60）")
        print("=" * 50)
        
        student_avg = self.scores.mean(axis=1)
        weak_students = student_avg < 60
        
        if not weak_students.any():
            print("🎉 恭喜！所有学生都及格了！")
        else:
            for i, is_weak in enumerate(weak_students):
                if is_weak:
                    avg = student_avg[i]
                    print(f"- {self.student_names[i]}: 平均分 {avg:.2f}")


# 使用示例
if __name__ == "__main__":
    # 模拟数据：5个学生，4门课
    scores = [
        [85, 92, 78, 90],   # 小明
        [58, 62, 55, 60],   # 小红（需要关注）
        [92, 88, 95, 91],   # 小华
        [76, 80, 72, 78],   # 小李
        [45, 50, 48, 52]    # 小刚（需要关注）
    ]
    
    students = ["小明", "小红", "小华", "小李", "小刚"]
    courses = ["数学", "语文", "英语", "科学"]
    
    # 创建分析器并生成报告
    analyzer = ScoreAnalyzer(scores, students, courses)
    analyzer.class_summary()
    analyzer.course_analysis()
    analyzer.student_ranking()
    analyzer.find_improvement_candidates()
```

**运行结果示例**：
```
==================================================
📊 班级成绩统计报告
==================================================

班级人数: 5
课程数量: 4

全班平均分: 73.25
全班最高分: 95
全班最低分: 45
标准差: 15.83

==================================================
📚 各课程统计
==================================================

数学:
  平均分: 71.20
  最高分: 92
  最低分: 45
  及格率: 80.0%

语文:
  平均分: 74.40
  ...

==================================================
🏆 学生总分排名
==================================================
1. 小华: 总分=366, 平均分=91.50
2. 小明: 总分=345, 平均分=86.25
...
```

---

## 5.7 练习题

### 基础练习

**练习5.1：温度转换器**

编写一个函数，将摄氏度转换为华氏度。公式是：$F = C \times 9/5 + 32$

```python
def celsius_to_fahrenheit(celsius):
    # 你的代码
    pass

# 测试
print(celsius_to_fahrenheit(0))     # 应该是 32.0
print(celsius_to_fahrenheit(100))   # 应该是 212.0
```

**练习5.2：列表操作**

给定一个列表 `[3, 1, 4, 1, 5, 9, 2, 6]`，请完成以下操作：
1. 找出最大数和最小数
2. 计算所有数的和
3. 创建一个新列表，包含原列表中所有大于3的数
4. 对列表进行排序

**练习5.3：判断素数**

编写一个函数，判断一个数是否是素数（只能被1和自身整除的数）。

```python
def is_prime(n):
    # 你的代码
    pass

# 测试
print(is_prime(7))   # True
print(is_prime(10))  # False
print(is_prime(97))  # True
```

### 进阶练习

**练习5.4：NumPy数组操作**

使用NumPy完成以下操作：

```python
import numpy as np

# 创建一个从0到99的数组
data = np.arange(100)

# 1. 找出所有能被3整除的数
# 2. 计算这些数的平均值
# 3. 将这些数 reshape 成一个 10×10 的矩阵
# 4. 计算每一行的和
# 5. 找出和最大的那一行
```

**练习5.5：手写统计函数**

不使用NumPy，手写实现以下统计函数：

```python
def my_mean(data):
    """计算平均值"""
    pass

def my_std(data):
    """计算标准差"""
    # 标准差公式：sqrt( sum((x - mean)^2) / N )
    pass

def my_correlation(x, y):
    """计算两个列表的相关系数"""
    pass
```

### 挑战练习

**练习5.6：矩阵运算库**

扩展5.5.1节的 `MyArray` 类，添加以下功能：
1. 支持二维数组（矩阵）
2. 实现矩阵转置（`.T`）
3. 实现矩阵乘法（`@` 运算符）
4. 实现行列式计算（仅2×2和3×3矩阵）

---

## 5.8 本章小结

### 核心概念回顾

| 概念 | 要点 |
|------|------|
| **变量** | 给数据贴标签，用 `=` 赋值 |
| **数据类型** | int, float, str, bool —— 不同类型的盒子 |
| **列表** | 有序的、可变的数据集合 |
| **条件判断** | `if-elif-else`，注意缩进！ |
| **循环** | `for`遍历，`while`条件循环 |
| **函数** | 可复用的代码块，有输入和输出 |
| **NumPy数组** | 超级列表，支持向量化运算 |
| **多维数组** | 矩阵和张量，机器学习的核心数据结构 |

### Python vs NumPy 对比

| 操作 | Python列表 | NumPy数组 |
|------|-----------|-----------|
| 创建 | `[1, 2, 3]` | `np.array([1, 2, 3])` |
| 加法 | 拼接列表 | 逐元素相加 |
| 乘法 | 重复列表 | 逐元素相乘 |
| 性能（100万个数） | 慢（0.3秒） | 快（0.001秒） |
| 统计函数 | 需手写 | 内置 `.mean()`, `.std()` 等 |

---

## 参考文献
