def calculate_feature_importance(forest, data):
    """通过置换法计算特征重要性"""
    baseline_accuracy = evaluate(forest, data)
    
    importances = []
    for feature in all_features:
        # 打乱这个特征的值
        permuted_data = permute_feature(data, feature)
        
        # 看准确率下降多少
        permuted_accuracy = evaluate(forest, permuted_data)
        
        importance = baseline_accuracy - permuted_accuracy
        importances.append(importance)
    
    return importances