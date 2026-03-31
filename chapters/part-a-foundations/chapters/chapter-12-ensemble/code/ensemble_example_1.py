# 随机森林伪代码
def random_forest_train(data, T, m_try):
    trees = []
    for t in range(T):
        # 第一层随机性：Bootstrap抽样
        bootstrap_data = bootstrap_sample(data)
        
        # 训练一棵树
        tree = build_tree_with_random_features(bootstrap_data, m_try)
        trees.append(tree)
    
    return trees

def build_tree_with_random_features(data, m_try):
    """构建一棵树，在每个节点只考虑m_try个随机特征"""
    if stopping_criterion_met(data):
        return create_leaf(data)
    
    # 关键：从所有特征中随机选择m_try个
    all_features = get_all_features(data)
    selected_features = random_sample(all_features, m_try)
    
    # 只在选中的特征中寻找最佳分裂
    best_feature, best_threshold = find_best_split(data, selected_features)
    
    left_data, right_data = split(data, best_feature, best_threshold)
    
    left_child = build_tree_with_random_features(left_data, m_try)
    right_child = build_tree_with_random_features(right_data, m_try)
    
    return create_node(best_feature, best_threshold, left_child, right_child)