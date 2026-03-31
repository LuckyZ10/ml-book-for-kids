"""
第五章配套代码：Python热身
================================
包含：
1. Python基础语法示例
2. NumPy基础操作
3. 手搓MyArray实现
4. 手搓矩阵乘法
5. 成绩分析器完整实现

运行方式：
    python chapter-05-code.py
"""

# =============================================================================
# 第一部分：Python基础语法
# =============================================================================

print("=" * 60)
print("第一部分：Python基础语法")
print("=" * 60)

# 1.1 变量和数据类型
print("\n--- 1.1 变量和数据类型 ---")

age = 25                    # 整数
height = 1.75               # 浮点数
name = "Alice"              # 字符串
is_student = True           # 布尔值

print(f"age = {age}, 类型: {type(age).__name__}")
print(f"height = {height}, 类型: {type(height).__name__}")
print(f"name = '{name}', 类型: {type(name).__name__}")
print(f"is_student = {is_student}, 类型: {type(is_student).__name__}")

# 1.2 列表操作
print("\n--- 1.2 列表操作 ---")

scores = [85, 92, 78, 90, 88]
print(f"原始列表: {scores}")
print(f"第一个元素: {scores[0]}")
print(f"最后一个元素: {scores[-1]}")

scores[2] = 80              # 修改元素
print(f"修改后: {scores}")

scores.append(95)           # 添加元素
print(f"添加后: {scores}")

print(f"列表长度: {len(scores)}")

# 1.3 条件判断
print("\n--- 1.3 条件判断 ---")

def grade_level(score):
    """根据分数返回等级"""
    if score >= 90:
        return "优秀"
    elif score >= 80:
        return "良好"
    elif score >= 60:
        return "及格"
    else:
        return "不及格"

test_scores = [95, 85, 75, 55]
for score in test_scores:
    print(f"分数 {score} -> {grade_level(score)}")

# 1.4 循环
print("\n--- 1.4 循环 ---")

# for循环遍历列表
print("遍历列表:")
for i, score in enumerate(scores):
    print(f"  第{i+1}个分数: {score}")

# range函数
print("\nrange(5):")
for i in range(5):
    print(f"  {i}", end=" ")
print()

# while循环
print("\nwhile循环:")
count = 0
while count < 3:
    print(f"  计数: {count}")
    count += 1

# 1.5 函数定义
print("\n--- 1.5 函数定义 ---")

def calculate_average(scores):
    """
    计算列表中数字的平均值
    
    参数:
        scores: 一个包含数字的列表
    
    返回:
        平均值（浮点数）
    """
    if len(scores) == 0:
        return 0
    total = sum(scores)
    count = len(scores)
    return total / count

my_scores = [85, 92, 78, 90, 88]
avg = calculate_average(my_scores)
print(f"分数: {my_scores}")
print(f"平均分: {avg:.2f}")


# =============================================================================
# 第二部分：NumPy基础
# =============================================================================

print("\n" + "=" * 60)
print("第二部分：NumPy基础")
print("=" * 60)

import numpy as np

# 2.1 创建数组
print("\n--- 2.1 创建数组 ---")

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"np.array([1,2,3,4,5]): {arr1}")

# 创建特定数组
zeros = np.zeros(5)
print(f"np.zeros(5): {zeros}")

ones = np.ones(5)
print(f"np.ones(5): {ones}")

arange = np.arange(0, 10, 2)
print(f"np.arange(0, 10, 2): {arange}")

linspace = np.linspace(0, 1, 5)
print(f"np.linspace(0, 1, 5): {linspace}")

# 2.2 向量化运算
print("\n--- 2.2 向量化运算 ---")

scores = np.array([85, 92, 78, 90, 88])
print(f"原始分数: {scores}")
print(f"加5分后: {scores + 5}")
print(f"乘以0.9: {scores * 0.9}")
print(f"平方: {scores ** 2}")

# 统计函数
print(f"\n平均分: {scores.mean():.2f}")
print(f"最高分: {scores.max()}")
print(f"最低分: {scores.min()}")
print(f"标准差: {scores.std():.2f}")

# 2.3 多维数组（矩阵）
print("\n--- 2.3 多维数组（矩阵） ---")

grades = np.array([
    [85, 92, 78, 90],
    [88, 76, 95, 89],
    [92, 88, 85, 91]
])

print(f"成绩矩阵:\n{grades}")
print(f"形状 (shape): {grades.shape}")
print(f"维度 (ndim): {grades.ndim}")
print(f"元素总数 (size): {grades.size}")

# 索引访问
print(f"\n第一行第一列: {grades[0, 0]}")
print(f"第二行第三列: {grades[1, 2]}")
print(f"第一行所有列: {grades[0, :]}")
print(f"所有行第二列: {grades[:, 1]}")

# 按轴计算
print(f"\n每个学生平均分: {grades.mean(axis=1)}")
print(f"每门课平均分: {grades.mean(axis=0)}")


# =============================================================================
# 第三部分：手搓实现
# =============================================================================

print("\n" + "=" * 60)
print("第三部分：手搓实现")
print("=" * 60)

# 3.1 手搓MyArray类
print("\n--- 3.1 手搓MyArray类 ---")

class MyArray:
    """
    简化版的NumPy数组实现
    只支持一维数组和基本操作
    """
    
    def __init__(self, data):
        """初始化数组"""
        self.data = list(data)
        self.size = len(self.data)
    
    def __getitem__(self, index):
        """获取元素"""
        return self.data[index]
    
    def __setitem__(self, index, value):
        """设置元素"""
        self.data[index] = value
    
    def __add__(self, other):
        """数组加法"""
        if isinstance(other, (int, float)):
            return MyArray([x + other for x in self.data])
        elif isinstance(other, MyArray):
            if self.size != other.size:
                raise ValueError("数组长度必须相同！")
            return MyArray([a + b for a, b in zip(self.data, other.data)])
        else:
            raise TypeError("不支持的操作类型")
    
    def __mul__(self, other):
        """数组乘法"""
        if isinstance(other, (int, float)):
            return MyArray([x * other for x in self.data])
        raise TypeError("只支持数字乘法")
    
    def sum(self):
        """求和"""
        total = 0
        for x in self.data:
            total += x
        return total
    
    def mean(self):
        """求平均值"""
        return self.sum() / self.size
    
    def __repr__(self):
        """字符串表示"""
        return f"MyArray({self.data})"


# 测试MyArray
arr1 = MyArray([1, 2, 3, 4, 5])
arr2 = MyArray([10, 20, 30, 40, 50])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"arr1 + 10: {arr1 + 10}")
print(f"arr1 + arr2: {arr1 + arr2}")
print(f"arr1 * 2: {arr1 * 2}")
print(f"arr1的平均值: {arr1.mean()}")

# 3.2 手搓矩阵乘法
print("\n--- 3.2 手搓矩阵乘法 ---")

def matrix_multiply(A, B):
    """
    手搓矩阵乘法
    
    参数:
        A: m×n 的二维列表
        B: n×p 的二维列表
    
    返回:
        m×p 的结果矩阵
    """
    m = len(A)
    n = len(A[0])
    p = len(B[0])
    
    if len(B) != n:
        raise ValueError("矩阵维度不匹配！")
    
    # 初始化结果矩阵
    C = [[0 for _ in range(p)] for _ in range(m)]
    
    # 三重循环计算矩阵乘法
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    
    return C


# 测试矩阵乘法
A = [[1, 2], [3, 4], [5, 6]]       # 3×2
B = [[7, 8, 9], [10, 11, 12]]      # 2×3

C = matrix_multiply(A, B)

print("矩阵 A (3×2):")
for row in A:
    print(f"  {row}")

print("\n矩阵 B (2×3):")
for row in B:
    print(f"  {row}")

print("\n矩阵 C = A × B (3×3):")
for row in C:
    print(f"  {row}")

print(f"\n验证: C[0][0] = 1×7 + 2×10 = {1*7 + 2*10}")


# =============================================================================
# 第四部分：成绩分析器
# =============================================================================

print("\n" + "=" * 60)
print("第四部分：成绩分析器")
print("=" * 60)


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
        
        total_scores = self.scores.sum(axis=1)
        ranked_indices = np.argsort(total_scores)[::-1]
        
        for rank, idx in enumerate(ranked_indices, 1):
            name = self.student_names[idx]
            total = total_scores[idx]
            avg = total / len(self.course_names)
            print(f"{rank}. {name}: 总分={total}, 平均分={avg:.2f}")
    
    def find_improvement_candidates(self):
        """找出需要提高的学生"""
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


# 运行成绩分析器
scores = [
    [85, 92, 78, 90],
    [58, 62, 55, 60],
    [92, 88, 95, 91],
    [76, 80, 72, 78],
    [45, 50, 48, 52]
]

students = ["小明", "小红", "小华", "小李", "小刚"]
courses = ["数学", "语文", "英语", "科学"]

analyzer = ScoreAnalyzer(scores, students, courses)
analyzer.class_summary()
analyzer.course_analysis()
analyzer.student_ranking()
analyzer.find_improvement_candidates()


# =============================================================================
# 第五部分：练习题解答
# =============================================================================

print("\n" + "=" * 60)
print("第五部分：练习题解答")
print("=" * 60)

# 练习5.1：温度转换器
print("\n--- 练习5.1：温度转换器 ---")

def celsius_to_fahrenheit(celsius):
    """摄氏度转华氏度"""
    return celsius * 9/5 + 32

print(f"0°C = {celsius_to_fahrenheit(0)}°F")
print(f"100°C = {celsius_to_fahrenheit(100)}°F")

# 练习5.2：列表操作
print("\n--- 练习5.2：列表操作 ---")

data = [3, 1, 4, 1, 5, 9, 2, 6]
print(f"原始列表: {data}")
print(f"最大数: {max(data)}")
print(f"最小数: {min(data)}")
print(f"总和: {sum(data)}")
print(f"大于3的数: {[x for x in data if x > 3]}")
print(f"排序后: {sorted(data)}")

# 练习5.3：判断素数
print("\n--- 练习5.3：判断素数 ---")

def is_prime(n):
    """判断n是否是素数"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

test_numbers = [7, 10, 97, 1, 0]
for n in test_numbers:
    print(f"{n} 是素数吗? {is_prime(n)}")

# 练习5.4：NumPy数组操作
print("\n--- 练习5.4：NumPy数组操作 ---")

data = np.arange(100)
print(f"原始数据: 0到99的数组")

# 1. 找出所有能被3整除的数
divisible_by_3 = data[data % 3 == 0]
print(f"能被3整除的数: {divisible_by_3[:10]}... (共{len(divisible_by_3)}个)")

# 2. 计算平均值
print(f"这些数的平均值: {divisible_by_3.mean():.2f}")

# 3. reshape成10×10矩阵
matrix = divisible_by_3.reshape(10, -1)
print(f"reshape后的形状: {matrix.shape}")

# 4. 计算每行的和
row_sums = matrix.sum(axis=1)
print(f"每行的和: {row_sums}")

# 5. 找出和最大的那一行
max_row_idx = np.argmax(row_sums)
print(f"和最大的行索引: {max_row_idx}, 和为: {row_sums[max_row_idx]}")


# =============================================================================
# 运行结束
# =============================================================================

print("\n" + "=" * 60)
print("🎉 所有代码运行完成！")
print("=" * 60)
