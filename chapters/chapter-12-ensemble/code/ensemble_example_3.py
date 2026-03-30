# Boosting 通用框架
def boosting_train(data, T):
    models = []
    weights = []  # 每个基学习器的权重
    
    for t in range(T):
        # 根据当前表现调整样本权重
        weighted_data = adjust_sample_weights(data, t)
        
        # 训练基学习器（通常用决策树桩）
        model = train_weak_learner(weighted_data)
        
        # 计算这个学习器的权重
        alpha = calculate_model_weight(model, weighted_data)
        
        models.append(model)
        weights.append(alpha)
    
    return models, weights

def boosting_predict(models, weights, x):
    # 加权投票
    prediction = sum(alpha * model.predict(x) for alpha, model in zip(weights, models))
    return sign(prediction)