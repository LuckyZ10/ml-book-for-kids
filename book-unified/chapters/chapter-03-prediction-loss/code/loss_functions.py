# chapter-03-loss.py
# 第三章：预测与误差——损失函数从零实现
# 
# 《机器学习与深度学习：从小学生到大师》
# 配套代码：第三章
#
# 学习目标：
# 1. 理解损失函数的数学原理
# 2. 从零实现MSE、MAE、RMSE
# 3. 可视化损失函数
# 4. 对比不同损失函数的特性

class LossFunctions:
    """
    损失函数集合 - 纯Python实现，无外部依赖
    
    历史背景：
    - 1801年：高斯用最小二乘法预测谷神星轨道
    - 1805年：勒让德首次发表最小二乘法
    - 1847年：柯西提出梯度下降法
    
    作者：ml-book-for-kids
    日期：2026-03-24
    """
    
    @staticmethod
    def mse(y_true, y_pred):
        """
        均方误差 (Mean Squared Error)
        
        公式：MSE = (1/n) * Σ(y_true - y_pred)²
        
        历史：源自高斯的最小二乘法(1801)
        特点：对大误差惩罚更重（平方放大效应）
        
        参数:
            y_true: 真实值列表
            y_pred: 预测值列表
        
        返回:
            MSE值 (float)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("真实值和预测值长度必须相同！")
        
        if len(y_true) == 0:
            return 0.0
        
        # 计算每个点的误差平方
        squared_errors = []
        for yt, yp in zip(y_true, y_pred):
            error = yt - yp
            squared_errors.append(error * error)
        
        # 求平均
        mse_value = sum(squared_errors) / len(squared_errors)
        return mse_value
    
    @staticmethod
    def mae(y_true, y_pred):
        """
        平均绝对误差 (Mean Absolute Error)
        
        公式：MAE = (1/n) * Σ|y_true - y_pred|
        
        特点：对异常值更稳健
        
        参数:
            y_true: 真实值列表
            y_pred: 预测值列表
        
        返回:
            MAE值 (float)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("真实值和预测值长度必须相同！")
        
        if len(y_true) == 0:
            return 0.0
        
        # 计算每个点的误差绝对值
        abs_errors = []
        for yt, yp in zip(y_true, y_pred):
            error = yt - yp
            # 绝对值：如果为负，取相反数
            if error < 0:
                error = -error
            abs_errors.append(error)
        
        # 求平均
        mae_value = sum(abs_errors) / len(abs_errors)
        return mae_value
    
    @staticmethod
    def rmse(y_true, y_pred):
        """
        均方根误差 (Root Mean Squared Error)
        
        公式：RMSE = √MSE
        
        和MSE相比，RMSE的单位与原始数据相同，更直观
        """
        mse_value = LossFunctions.mse(y_true, y_pred)
        
        # 牛顿迭代法求平方根
        return LossFunctions._sqrt(mse_value)
    
    @staticmethod
    def _sqrt(x, epsilon=1e-10):
        """
        牛顿迭代法计算平方根
        
        原理：不断逼近 √x
        公式：next = 0.5 * (guess + x/guess)
        """
        if x < 0:
            raise ValueError("不能对负数求平方根！")
        if x == 0:
            return 0.0
        
        # 初始猜测
        guess = x
        
        # 迭代改进
        for _ in range(100):  # 最多100次迭代
            next_guess = 0.5 * (guess + x / guess)
            if abs(next_guess - guess) < epsilon:
                return next_guess
            guess = next_guess
        
        return guess
    
    @staticmethod
    def huber_loss(y_true, y_pred, delta=1.0):
        """
        Huber损失 - MSE和MAE的结合
        
        公式：
        - 当 |误差| <= delta: 0.5 * 误差²
        - 当 |误差| > delta: delta * (|误差| - 0.5 * delta)
        
        特点：小误差用MSE，大误差用MAE，兼顾光滑性和稳健性
        
        参数:
            delta: 切换阈值，默认1.0
        """
        if len(y_true) != len(y_pred):
            raise ValueError("真实值和预测值长度必须相同！")
        
        total_loss = 0.0
        for yt, yp in zip(y_true, y_pred):
            error = yt - yp
            if error < 0:
                error = -error
            
            if error <= delta:
                # 小误差：使用MSE
                total_loss += 0.5 * error * error
            else:
                # 大误差：使用MAE风格
                total_loss += delta * (error - 0.5 * delta)
        
        return total_loss / len(y_true)
    
    @staticmethod
    def visualize_loss_1d():
        """
        可视化一维损失函数
        
        展示当预测值偏离真实值时，不同损失函数的变化
        """
        true_value = 5.0  # 真实值固定为5
        
        # 生成预测值从0到10
        predictions = [i * 0.5 for i in range(21)]
        
        # 计算各种损失
        mse_losses = []
        mae_losses = []
        
        for pred in predictions:
            mse_losses.append((true_value - pred) ** 2)
            mae_losses.append(abs(true_value - pred))
        
        # ASCII可视化
        print("=" * 60)
        print("损失函数可视化 (真实值 = 5.0)")
        print("=" * 60)
        print(f"{'预测值':>8} | {'MSE':>8} | {'MAE':>8} | MSE图 | MAE图")
        print("-" * 60)
        
        max_mse = max(mse_losses)
        max_mae = max(mae_losses)
        
        for i, pred in enumerate(predictions):
            mse_bar = "█" * int(mse_losses[i] / max_mse * 15)
            mae_bar = "▓" * int(mae_losses[i] / max_mae * 15)
            
            marker = " ←★" if pred == true_value else ""
            
            print(f"{pred:>8.1f} | {mse_losses[i]:>8.2f} | {mae_losses[i]:>8.2f} | {mse_bar:<15} | {mae_bar:<15}{marker}")
        
        print("-" * 60)
        print("★ 标记表示预测完全正确 (预测值=真实值)")
        print("注意：MSE在偏离真实值时增长得更快！")
    
    @staticmethod
    def visualize_errors(true_value=5.0):
        """
        可视化误差分布
        """
        print("\n" + "=" * 60)
        print("误差分布可视化")
        print("=" * 60)
        print(f"假设真实值 = {true_value}")
        print()
        
        # 不同预测值
        predictions = [3, 4, 5, 6, 7]
        
        print(f"{'预测值':>8} | {'误差':>8} | {'误差²':>8} | {'|误差|':>8} | 可视化")
        print("-" * 60)
        
        for pred in predictions:
            error = pred - true_value
            sq_error = error ** 2
            abs_error = abs(error)
            
            # 误差条形图
            if error >= 0:
                bar = "█" * abs_error
                bar_str = f"+{bar}"
            else:
                bar = "█" * abs(error)
                bar_str = f"-{bar}"
            
            marker = " ←★" if pred == true_value else ""
            
            print(f"{pred:>8.0f} | {error:>+8.0f} | {sq_error:>8.0f} | {abs_error:>8.0f} | {bar_str:<15}{marker}")
        
        print("-" * 60)


# ==================== 测试代码 ====================

def test_loss_functions():
    """测试所有损失函数"""
    
    print("=" * 60)
    print("🧪 损失函数测试")
    print("=" * 60)
    
    # 测试数据：天气预报场景
    # 预测温度 vs 实际温度
    true_temps = [22, 25, 28, 30, 24, 26, 29]  # 实际温度
    pred_temps = [21, 24, 30, 28, 25, 26, 31]  # 预测温度
    
    print("\n📊 测试数据：一周温度预测")
    print("-" * 40)
    print(f"{'日期':>6} | {'实际':>6} | {'预测':>6} | {'误差':>6}")
    print("-" * 40)
    
    for i, (t, p) in enumerate(zip(true_temps, pred_temps)):
        error = t - p
        print(f"Day {i+1:>2} | {t:>6.1f}°C | {p:>6.1f}°C | {error:>+6.1f}")
    
    print("-" * 40)
    
    # 计算损失
    mse = LossFunctions.mse(true_temps, pred_temps)
    mae = LossFunctions.mae(true_temps, pred_temps)
    rmse = LossFunctions.rmse(true_temps, pred_temps)
    
    print(f"\n📈 评估结果：")
    print(f"   MSE  = {mse:.4f}")
    print(f"   MAE  = {mae:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    
    print(f"\n📝 解释：")
    print(f"   - 平均每次预测误差约 {mae:.2f}°C")
    print(f"   - 误差的平方平均为 {mse:.2f}")
    print(f"   - RMSE约 {rmse:.2f}°C，与温度同单位")
    
    # 可视化
    print("\n")
    LossFunctions.visualize_loss_1d()
    print("\n")
    LossFunctions.visualize_errors()


def compare_mse_mae():
    """对比MSE和MAE对异常值的处理"""
    
    print("\n" + "=" * 60)
    print("🔍 MSE vs MAE：异常值敏感度对比")
    print("=" * 60)
    
    # 正常数据
    normal_true = [10, 20, 30, 40, 50]
    normal_pred = [11, 19, 31, 39, 51]
    
    # 加入一个异常值
    outlier_true = [10, 20, 30, 40, 50]
    outlier_pred = [11, 19, 100, 39, 51]  # 第三个预测严重偏离
    
    print("\n场景A：正常预测误差")
    print(f"真实值: {normal_true}")
    print(f"预测值: {normal_pred}")
    print(f"MSE = {LossFunctions.mse(normal_true, normal_pred):.2f}")
    print(f"MAE = {LossFunctions.mae(normal_true, normal_pred):.2f}")
    
    print("\n场景B：有一个严重错误（预测100，实际30）")
    print(f"真实值: {outlier_true}")
    print(f"预测值: {outlier_pred}")
    
    mse_normal = LossFunctions.mse(normal_true, normal_pred)
    mse_outlier = LossFunctions.mse(outlier_true, outlier_pred)
    mae_normal = LossFunctions.mae(normal_true, normal_pred)
    mae_outlier = LossFunctions.mae(outlier_true, outlier_pred)
    
    print(f"MSE = {mse_outlier:.2f} (增加了 {mse_outlier/mse_normal:.1f} 倍)")
    print(f"MAE = {mae_outlier:.2f} (增加了 {mae_outlier/mae_normal:.1f} 倍)")
    
    print("\n💡 结论：MSE对异常值更敏感，会放大严重错误的影响！")


def test_huber_loss():
    """测试Huber损失"""
    
    print("\n" + "=" * 60)
    print("🔧 Huber损失测试")
    print("=" * 60)
    
    y_true = [10, 20, 30, 40, 50]
    y_pred = [11, 25, 30, 38, 60]
    
    print(f"真实值: {y_true}")
    print(f"预测值: {y_pred}")
    print()
    print(f"MSE        = {LossFunctions.mse(y_true, y_pred):.2f}")
    print(f"MAE        = {LossFunctions.mae(y_true, y_pred):.2f}")
    print(f"Huber(δ=1) = {LossFunctions.huber_loss(y_true, y_pred, delta=1.0):.2f}")
    print(f"Huber(δ=5) = {LossFunctions.huber_loss(y_true, y_pred, delta=5.0):.2f}")
    
    print("\n💡 Huber损失结合了MSE的光滑性和MAE的稳健性！")


def gradient_descent_demo():
    """
    梯度下降简单演示
    
    使用MSE损失，演示如何迭代优化预测值
    """
    print("\n" + "=" * 60)
    print("🎯 梯度下降演示")
    print("=" * 60)
    
    # 假设真实值是10
    true_value = 10.0
    
    # 初始预测值是0
    prediction = 0.0
    
    # 学习率
    learning_rate = 0.1
    
    # 迭代优化
    print(f"真实值: {true_value}")
    print(f"初始预测: {prediction}")
    print()
    print(f"{'迭代':>4} | {'预测':>8} | {'损失(MSE)':>10} | {'梯度':>8}")
    print("-" * 45)
    
    for iteration in range(10):
        # 计算损失
        loss = (prediction - true_value) ** 2
        
        # 计算梯度 (d(MSE)/d(pred) = 2*(pred - true))
        gradient = 2 * (prediction - true_value)
        
        print(f"{iteration:>4} | {prediction:>8.4f} | {loss:>10.4f} | {gradient:>8.4f}")
        
        # 更新预测值（沿着梯度反方向走）
        prediction = prediction - learning_rate * gradient
    
    print("-" * 45)
    print(f"最终预测: {prediction:.4f} (接近真实值 {true_value})")
    print("\n💡 梯度下降让预测值逐步逼近真实值！")


def chapter_exercise_solution():
    """第三章练习题解答"""
    
    print("\n" + "=" * 60)
    print("📝 第三章练习题解答")
    print("=" * 60)
    
    # 练习1答案
    print("\n练习1：手动计算验证")
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    
    print(f"真实值: {y_true}")
    print(f"预测值: {y_pred}")
    
    # 手动计算
    errors = [t - p for t, p in zip(y_true, y_pred)]
    squared_errors = [e ** 2 for e in errors]
    abs_errors = [abs(e) for e in errors]
    
    print(f"\n误差: {errors}")
    print(f"误差平方: {squared_errors}")
    print(f"误差绝对值: {abs_errors}")
    
    mse_manual = sum(squared_errors) / len(squared_errors)
    mae_manual = sum(abs_errors) / len(abs_errors)
    
    print(f"\n手动计算 MSE = {mse_manual:.4f}")
    print(f"手动计算 MAE = {mae_manual:.4f}")
    
    # 用函数验证
    mse_func = LossFunctions.mse(y_true, y_pred)
    mae_func = LossFunctions.mae(y_true, y_pred)
    
    print(f"\n函数计算 MSE = {mse_func:.4f}")
    print(f"函数计算 MAE = {mae_func:.4f}")
    
    if abs(mse_manual - mse_func) < 1e-10 and abs(mae_manual - mae_func) < 1e-10:
        print("\n✅ 验证通过！手动计算与函数结果一致。")
    else:
        print("\n❌ 验证失败！请检查计算。")


if __name__ == "__main__":
    # 运行所有测试
    test_loss_functions()
    compare_mse_mae()
    test_huber_loss()
    gradient_descent_demo()
    chapter_exercise_solution()
    
    print("\n" + "=" * 60)
    print("🎉 第三章测试完成！")
    print("=" * 60)
    print("你已经掌握了：")
    print("  ✅ 损失函数的数学原理")
    print("  ✅ MSE、MAE、RMSE的实现")
    print("  ✅ 异常值敏感度分析")
    print("  ✅ 梯度下降的基本概念")
    print("=" * 60)
