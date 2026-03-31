# Bagging 伪代码
def bagging_train(data, T):
    models = []
    for t in range(T):
        # 第1步：Bootstrap抽样
        bootstrap_data = bootstrap_sample(data)
        
        # 第2步：训练基学习器
        model = train_base_learner(bootstrap_data)
        models.append(model)
    
    return models

def bagging_predict(models, x):
    predictions = [model.predict(x) for model in models]
    
    # 第3步：投票或平均
    return majority_vote(predictions)  # 分类
    # 或 return mean(predictions)      # 回归