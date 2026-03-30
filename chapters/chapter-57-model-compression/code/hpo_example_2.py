import numpy as np

# 随机搜索示例
for trial in range(n_trials):
    config = {
        'learning_rate': 10 ** np.random.uniform(-5, -1),  # 对数均匀
        'batch_size': np.random.choice([16, 32, 64, 128]),
        'dropout': np.random.uniform(0.1, 0.5)
    }
    evaluate(config)