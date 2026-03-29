"""
第八章配套代码：逻辑回归——分类的艺术

本章代码包含：
1. Sigmoid函数实现与可视化
2. 从零实现逻辑回归
3. 考试通过预测实战
4. 垃圾邮件分类器
5. 多分类逻辑回归
6. L2正则化实现

作者：Kimi Claw
日期：2026-03-24
"""

import math
import random


# =============================================================================
# 第一部分：Sigmoid函数
# =============================================================================

def sigmoid(z):
    """
    Sigmoid函数：将任意实数映射到(0,1)区间
    
    数学公式：σ(z) = 1 / (1 + e^(-z))
    
    参数：
        z: 输入值（可以是任意实数）
    
    返回：
        0到1之间的概率值
    """
    # 防止数值溢出
    if z < -500:
        return 0.0
    if z > 500:
        return 1.0
    
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_derivative(z):
    """
    Sigmoid函数的导数
    
    数学公式：σ'(z) = σ(z) × (1 - σ(z))
    
    这个性质在反向传播中非常有用！
    """
    s = sigmoid(z)
    return s * (1 - s)


def test_sigmoid():
    """测试Sigmoid函数"""
    print("=" * 70)
    print("Sigmoid函数测试")
    print("=" * 70)
    
    test_values = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
    
    print("\n  x     |  σ(x)   |  1-σ(x)  |  导数    | 含义")
    print("-" * 70)
    
    for x in test_values:
        s = sigmoid(x)
        ds = sigmoid_derivative(x)
        
        if s < 0.2:
            meaning = "非常不可能是"
        elif s < 0.4:
            meaning = "不太可能是"
        elif s < 0.6:
            meaning = "可能是"
        elif s < 0.8:
            meaning = "很可能是"
        else:
            meaning = "非常可能是"
        
        print(f" {x:6.1f} | {s:7.4f} | {1-s:8.4f} | {ds:8.4f} | {meaning}")
    
    print("\n  注意：σ(x) + (1-σ(x)) = 1，且导数在x=0时最大（0.25）")


# =============================================================================
# 第二部分：逻辑回归实现
# =============================================================================

class LogisticRegression:
    """
    逻辑回归分类器（从零实现）
    
    这是一个完整的逻辑回归实现，包括：
    - Sigmoid激活函数
    - 梯度下降训练
    - 预测与分类
    - L2正则化
    
    不使用任何外部机器学习库！
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, random_state=None):
        """
        初始化逻辑回归模型
        
        参数：
            learning_rate: 学习率，控制每一步更新的大小
            max_iterations: 最大迭代次数
            random_state: 随机种子（用于可复现性）
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = []  # 权重
        self.bias = 0      # 偏置项
        self.loss_history = []  # 记录训练过程中的损失
        self.random_state = random_state
        
        if random_state is not None:
            random.seed(random_state)
    
    def _sigmoid(self, z):
        """Sigmoid激活函数（内部使用）"""
        if z < -500:
            return 0.0
        if z > 500:
            return 1.0
        return 1.0 / (1.0 + math.exp(-z))
    
    def fit(self, X, y, verbose=True):
        """
        训练模型（使用随机梯度下降）
        
        参数：
            X: 训练数据，列表的列表，每个内列表是一个样本的特征
            y: 标签，0或1的列表
            verbose: 是否打印训练进度
        
        返回：
            self
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        # 初始化权重和偏置为小的随机值
        if self.random_state is not None:
            self.weights = [random.uniform(-0.1, 0.1) for _ in range(n_features)]
        else:
            self.weights = [0.0] * n_features
        self.bias = 0.0
        
        if verbose:
            print(f"开始训练逻辑回归...")
            print(f"  样本数: {n_samples}")
            print(f"  特征数: {n_features}")
            print(f"  学习率: {self.learning_rate}")
            print(f"  最大迭代: {self.max_iterations}")
            print()
        
        # 梯度下降
        for iteration in range(self.max_iterations):
            total_loss = 0.0
            
            # 创建随机索引（随机梯度下降）
            indices = list(range(n_samples))
            if self.random_state is not None:
                random.shuffle(indices)
            
            # 对每个样本进行随机梯度下降
            for i in indices:
                # 前向传播：计算预测值
                linear = self.bias
                for j in range(n_features):
                    linear += self.weights[j] * X[i][j]
                
                predicted = self._sigmoid(linear)
                
                # 计算损失（二元交叉熵）
                epsilon = 1e-15
                p = max(epsilon, min(1 - epsilon, predicted))
                loss = -(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))
                total_loss += loss
                
                # 计算梯度
                error = predicted - y[i]
                
                # 更新偏置
                self.bias -= self.learning_rate * error
                
                # 更新权重
                for j in range(n_features):
                    gradient = error * X[i][j]
                    self.weights[j] -= self.learning_rate * gradient
            
            # 记录平均损失
            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)
            
            # 打印进度
            if verbose and ((iteration + 1) % 100 == 0 or iteration == 0):
                print(f"迭代 {iteration + 1:4d}/{self.max_iterations}: 损失 = {avg_loss:.6f}")
        
        if verbose:
            print(f"\n训练完成！最终损失: {self.loss_history[-1]:.6f}")
        
        return self
    
    def fit_with_regularization(self, X, y, lambda_reg=0.01, verbose=True):
        """
        带L2正则化的训练
        
        参数：
            X: 训练数据
            y: 标签
            lambda_reg: 正则化强度
            verbose: 是否打印进度
        """
        n_samples = len(X)
        n_features = len(X[0])
        
        self.weights = [0.0] * n_features
        self.bias = 0.0
        
        if verbose:
            print(f"开始训练（带L2正则化，λ={lambda_reg}）...")
        
        for iteration in range(self.max_iterations):
            total_loss = 0.0
            
            for i in range(n_samples):
                linear = self.bias
                for j in range(n_features):
                    linear += self.weights[j] * X[i][j]
                
                predicted = self._sigmoid(linear)
                
                epsilon = 1e-15
                p = max(epsilon, min(1 - epsilon, predicted))
                loss = -(y[i] * math.log(p) + (1 - y[i]) * math.log(1 - p))
                
                # 添加L2正则化损失（不包括偏置）
                reg_loss = lambda_reg * sum(w * w for w in self.weights)
                total_loss += loss + reg_loss
                
                error = predicted - y[i]
                
                # 更新偏置（不加正则化）
                self.bias -= self.learning_rate * error
                
                # 更新权重（加L2正则化）
                for j in range(n_features):
                    gradient = error * X[i][j] + 2 * lambda_reg * self.weights[j]
                    self.weights[j] -= self.learning_rate * gradient
            
            avg_loss = total_loss / n_samples
            self.loss_history.append(avg_loss)
            
            if verbose and (iteration + 1) % 100 == 0:
                print(f"迭代 {iteration + 1:4d}: 损失 = {avg_loss:.6f}")
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数：
            X: 输入特征
        
        返回：
            预测为类别1的概率列表
        """
        result = []
        for sample in X:
            linear = self.bias
            for j in range(len(sample)):
                linear += self.weights[j] * sample[j]
            result.append(self._sigmoid(linear))
        return result
    
    def predict(self, X, threshold=0.5):
        """
        预测类别
        
        参数：
            X: 输入特征
            threshold: 决策阈值
        
        返回：
            预测的类别（0或1）
        """
        probabilities = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probabilities]
    
    def score(self, X, y):
        """
        计算准确率
        
        参数：
            X: 测试数据
            y: 真实标签
        
        返回：
            准确率（0到1之间）
        """
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)
    
    def get_params(self):
        """获取模型参数"""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'max_iterations': self.max_iterations
        }
    
    def print_model(self):
        """打印模型信息"""
        print("\n" + "=" * 60)
        print("逻辑回归模型")
        print("=" * 60)
        print(f"偏置 (bias): {self.bias:.4f}")
        print("权重:")
        for i, w in enumerate(self.weights):
            print(f"  w{i}: {w:.4f}")
        print()
        print("决策边界方程:")
        terms = [f"{self.bias:.4f}"]
        for i, w in enumerate(self.weights):
            sign = "+" if w >= 0 else "-"
            terms.append(f"{sign} {abs(w):.4f}*x{i}")
        equation = " ".join(terms)
        print(f"  z = {equation}")
        print(f"  如果 z >= 0，预测为类别1")
        print(f"  如果 z < 0，预测为类别0")


# =============================================================================
# 第三部分：实战示例
# =============================================================================

def demo_exam_prediction():
    """考试通过预测演示"""
    print("\n" + "=" * 70)
    print("实战：考试通过预测")
    print("=" * 70)
    
    # 数据集：学习小时数 vs 是否通过
    X = [
        [1], [1.5], [2], [2.5], [3],
        [3.5], [4], [4.5], [5], [5.5], [6]
    ]
    y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    
    print("\n训练数据：")
    print("学习小时 | 结果")
    print("-" * 25)
    for x, label in zip(X, y):
        result = "通过" if label == 1 else "失败"
        print(f"  {x[0]:5.1f}   | {result}")
    
    # 训练模型
    model = LogisticRegression(learning_rate=0.3, max_iterations=300, random_state=42)
    model.fit(X, y)
    model.print_model()
    
    # 预测
    print("\n" + "=" * 60)
    print("预测不同学习时长的通过概率")
    print("=" * 60)
    
    test_hours = [[0], [1], [2], [2.5], [3], [3.5], [4], [5], [6], [7]]
    
    print("\n学习小时 | 通过概率 | 预测 | 可视化")
    print("-" * 60)
    
    for hours in test_hours:
        prob = model.predict_proba([hours])[0]
        pred = model.predict([hours])[0]
        result = "通过" if pred == 1 else "失败"
        bar = "█" * int(prob * 30)
        print(f"  {hours[0]:5.1f}    |  {prob:6.1%}  | {result} | {bar}")
    
    # 找到临界点
    print(f"\n临界点分析：")
    print(f"   当 z = 0 时，需要学习 {(-model.bias / model.weights[0]):.1f} 小时")
    print(f"   这是从失败到通过的分界线")
    
    return model


def demo_spam_classification():
    """垃圾邮件分类演示"""
    print("\n" + "=" * 70)
    print("实战：垃圾邮件分类")
    print("=" * 70)
    
    # 特征提取函数
    def extract_features(email_text):
        spam_keywords = ["免费", "优惠", "点击", "立即", "赚钱", "发财", "中奖",
                        "恭喜", "限量", "抢购", "特价", "赠送", "机会", "秘密"]
        
        text = email_text.lower()
        
        # 特征1：垃圾关键词数量
        spam_word_count = sum(1 for word in spam_keywords if word in text)
        
        # 特征2：感叹号数量
        exclamation_count = text.count("！") + text.count("!")
        
        # 特征3：大写字母比例
        upper_count = sum(1 for c in email_text if c.isupper())
        upper_ratio = upper_count / max(len(email_text), 1) * 100
        
        # 特征4：数字数量
        digit_count = sum(1 for c in email_text if c.isdigit())
        
        # 特征5：链接数量
        link_count = text.count("http") + text.count("www") + text.count("点击")
        
        return [spam_word_count, exclamation_count, upper_ratio, digit_count, link_count]
    
    # 训练数据
    emails = [
        # 正常邮件 (0)
        "你好，请问明天的会议是几点？",
        "附件是本周的工作报告，请查收。",
        "感谢你的帮助，这个问题解决了。",
        "周末一起去吃饭吧？",
        "发票已经寄出，请注意查收。",
        "项目进度正常，按计划进行。",
        "明天下午3点有部门会议，请准时参加。",
        "请确认一下这个方案是否可行，谢谢。",
        
        # 垃圾邮件 (1)
        "恭喜您中奖了！免费领取iPhone！点击领取！",
        "限时优惠！立即点击领取百万大奖！发财机会！",
        "赚钱发财的秘密！点击了解！免费赠送！",
        "免费赠送！限量抢购！机会难得！马上点击！",
        "恭喜您被选中！立即领取8888元红包！",
        "特价优惠！马上点击！发财致富！限量！",
        "中奖通知！免费机会！立即查看！点击！",
        "限量赠送！点击赚钱！优惠特价！机会！"
    ]
    
    labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    
    # 提取特征
    X = [extract_features(email) for email in emails]
    
    # 训练
    model = LogisticRegression(learning_rate=0.2, max_iterations=400, random_state=42)
    model.fit(X, labels)
    
    print("\n特征权重分析：")
    feature_names = ["垃圾关键词", "感叹号", "大写比例", "数字数量", "链接"]
    for name, weight in zip(feature_names, model.weights):
        importance = "★" * int(abs(weight) * 5)
        print(f"  {name:10s}: {weight:7.3f} {importance}")
    
    # 测试
    test_emails = [
        "你好，请问明天有空吗？一起去喝咖啡。",
        "恭喜你！免费中奖机会！立即点击领取大奖！免费！",
        "工作报告已提交，请审核。项目进展顺利。",
        "限时特价！免费赠送！赚钱机会！限量抢购！点击！",
        "请问这个月的报销什么时候能下来？"
    ]
    
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    
    for email in test_emails:
        features = extract_features(email)
        prob = model.predict_proba([features])[0]
        pred = model.predict([features])[0]
        result = "垃圾邮件" if pred == 1 else "正常邮件"
        
        display = email[:35] + "..." if len(email) > 35 else email
        confidence = "高" if prob > 0.8 or prob < 0.2 else "中"
        
        print(f"\n{display}")
        print(f"  预测: {result} (置信度: {confidence})")
        print(f"  垃圾概率: {prob:.1%}")
    
    # 准确率
    train_acc = model.score(X, labels)
    print(f"\n训练集准确率: {train_acc:.1%}")


# =============================================================================
# 第四部分：多分类逻辑回归
# =============================================================================

class MulticlassLogisticRegression:
    """
    多分类逻辑回归（One-vs-Rest策略）
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, random_state=None):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.classifiers = {}
        self.classes = []
    
    def fit(self, X, y, verbose=False):
        """训练多分类模型"""
        self.classes = sorted(list(set(y)))
        
        if verbose:
            print(f"\n训练多分类模型（{len(self.classes)}个类别）...")
        
        for cls in self.classes:
            # 为当前类别创建二分类标签
            binary_y = [1 if label == cls else 0 for label in y]
            
            # 训练一个二分类器
            clf = LogisticRegression(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations,
                random_state=self.random_state
            )
            clf.fit(X, binary_y, verbose=False)
            
            self.classifiers[cls] = clf
            
            if verbose:
                print(f"  类别 '{cls}' 的分类器训练完成")
        
        return self
    
    def predict_proba(self, X):
        """预测每个类别的概率"""
        result = []
        for sample in X:
            probs = {}
            for cls, clf in self.classifiers.items():
                probs[cls] = clf.predict_proba([sample])[0]
            result.append(probs)
        return result
    
    def predict(self, X):
        """预测类别"""
        predictions = []
        
        for probs in self.predict_proba(X):
            best_class = max(probs.keys(), key=lambda k: probs[k])
            predictions.append(best_class)
        
        return predictions
    
    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)


def demo_multiclass():
    """多分类演示"""
    print("\n" + "=" * 70)
    print("多分类演示：水果分类")
    print("=" * 70)
    
    # 特征：[重量, 甜度, 直径]
    X = [
        # 苹果
        [150, 12, 7], [160, 11, 7.5], [140, 13, 6.5],
        [155, 12, 7.2], [145, 14, 6.8],
        # 橙子
        [180, 10, 7.5], [190, 9, 8], [175, 11, 7.8],
        [185, 10, 7.6], [195, 8, 8.2],
        # 香蕉
        [120, 15, 4], [130, 16, 4.2], [115, 14, 3.8],
        [125, 15, 4.1], [135, 17, 4.3]
    ]
    
    y = ["苹果", "苹果", "苹果", "苹果", "苹果",
         "橙子", "橙子", "橙子", "橙子", "橙子",
         "香蕉", "香蕉", "香蕉", "香蕉", "香蕉"]
    
    # 标准化特征
    n_features = len(X[0])
    means = [sum(X[i][j] for i in range(len(X))) / len(X) for j in range(n_features)]
    stds = [math.sqrt(sum((X[i][j] - means[j])**2 for i in range(len(X))) / len(X)) 
            for j in range(n_features)]
    
    X_normalized = [[(X[i][j] - means[j]) / stds[j] for j in range(n_features)] 
                    for i in range(len(X))]
    
    model = MulticlassLogisticRegression(learning_rate=0.5, max_iterations=300, random_state=42)
    model.fit(X_normalized, y, verbose=True)
    
    # 测试
    test_fruits = [
        ([148, 12, 7], "苹果"),
        ([182, 10, 7.8], "橙子"),
        ([122, 15, 4], "香蕉"),
        ([170, 11, 7.4], "橙子"),
        ([152, 13, 7.1], "苹果")
    ]
    
    print("\n" + "=" * 60)
    print("预测测试")
    print("=" * 60)
    print(f"{'样本':15s} | {'真实':8s} | {'预测':8s} | 各类别概率")
    print("-" * 70)
    
    for features, true_label in test_fruits:
        normalized = [(features[j] - means[j]) / stds[j] for j in range(n_features)]
        probs = model.predict_proba([normalized])[0]
        pred = model.predict([normalized])[0]
        
        prob_str = ", ".join([f"{cls}:{prob:.1%}" for cls, prob in probs.items()])
        correct = "对" if pred == true_label else "错"
        
        print(f"{str(features):15s} | {true_label:8s} | {pred:8s} | {prob_str} {correct}")


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":
    print("=" * 68)
    print("第八章：逻辑回归")
    print("Classification with Logistic Regression")
    print("=" * 68)
    
    # 1. Sigmoid测试
    test_sigmoid()
    
    # 2. 考试预测
    input("\n按Enter继续到下一个演示...")
    demo_exam_prediction()
    
    # 3. 垃圾邮件分类
    input("\n按Enter继续到下一个演示...")
    demo_spam_classification()
    
    # 4. 多分类
    input("\n按Enter继续到下一个演示...")
    demo_multiclass()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)
