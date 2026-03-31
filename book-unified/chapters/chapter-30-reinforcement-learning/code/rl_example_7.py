# Double DQN
if double_dqn:
    # 用行为网络选择动作
    next_q_behavior = self.q_network.forward(next_state_vecs)
    best_actions = np.argmax(next_q_behavior, axis=1)
    
    # 用目标网络评估
    next_q_target = self.target_network.forward(next_state_vecs)
    max_next_q = next_q_target[np.arange(batch_size), best_actions]
else:
    # 原始DQN
    next_q = self.target_network.forward(next_state_vecs)
    max_next_q = np.max(next_q, axis=1)