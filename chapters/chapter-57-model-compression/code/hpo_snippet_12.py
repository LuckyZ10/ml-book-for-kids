"""
57.5.2 TPOT遗传编程AutoML概念
"""

import numpy as np


def tpot_concept():
    """
    TPOT的核心概念演示
    
    TPOT使用遗传编程进化机器学习管道
    """
    
    print("TPOT工作原理:")
    print("=" * 60)
    
    print("""
# TPOT使用示例
from tpot import TPOTClassifier

tpot = TPOTClassifier(
    generations=5,        # 进化5代
    population_size=20,   # 每代20个个体
    offspring_size=20,    # 产生20个后代
    mutation_rate=0.9,    # 变异率
    crossover_rate=0.1,   # 交叉率
    scoring='accuracy',   # 评估指标
    cv=5,                 # 5折交叉验证
    verbosity=2
)

# 自动进化最优管道
tpot.fit(X_train, y_train)

# 导出最优管道代码
tpot.export('best_pipeline.py')
""")
    
    print("\n遗传编程操作:")
    print("  1. 交叉(Crossover): 两个父代管道交换子树")
    print("     父代1: PCA → RandomForest")
    print("     父代2: StandardScaler → SVM")
    print("     子代:  PCA → SVM")
    
    print("\n  2. 变异(Mutation): 随机修改管道")
    print("     原管道: PCA → RandomForest")
    print("     变异后: SelectKBest → RandomForest")
    
    print("\nTPOT优势:")
    print("  - 可以探索非常复杂的管道组合")
    print("  - 最终输出可读的Python代码")
    print("  - 不限于固定结构，可以发现创新组合")
    
    print("\nTPOT局限:")
    print("  - 计算成本高（需要评估大量个体）")
    print("  - 没有贝叶斯优化的样本效率高")
    print("  - 可能过拟合验证集")


# 模拟一个简单的遗传进化过程
def simulate_evolution():
    """模拟TPOT的进化过程"""
    
    print("\n模拟进化过程:")
    print("-" * 50)
    
    np.random.seed(42)
    
    # 初始种群
    population = [
        {'pipeline': 'Scaler → RF', 'fitness': 0.75},
        {'pipeline': 'PCA → SVM', 'fitness': 0.78},
        {'pipeline': 'None → KNN', 'fitness': 0.72},
        {'pipeline': 'Scaler → GB', 'fitness': 0.80},
    ]
    
    for gen in range(3):
        print(f"\n第{gen+1}代:")
        
        # 按适应度排序
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        for i, ind in enumerate(population):
            print(f"  排名{i+1}: {ind['pipeline']}, 适应度={ind['fitness']:.3f}")
        
        # 选择最优的繁殖
        survivors = population[:2]
        
        # 产生后代（简化模拟）
        offspring = []
        for parent in survivors:
            # 变异
            new_pipeline = parent['pipeline'].replace('RF', 'RF+SVM')
            new_fitness = min(0.95, parent['fitness'] + np.random.uniform(-0.05, 0.08))
            offspring.append({'pipeline': new_pipeline, 'fitness': new_fitness})
        
        # 新一代
        population = survivors + offspring
    
    print(f"\n最终最优: {population[0]['pipeline']}, 适应度={population[0]['fitness']:.3f}")


if __name__ == "__main__":
    tpot_concept()
    simulate_evolution()