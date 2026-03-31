def potential(state, goal):
    return -(abs(state[0] - goal[0]) + abs(state[1] - goal[1]))

def shaped_reward(r, s, s_next, done, gamma, goal):
    if done:
        return r  # 终止状态不加塑造
    return r + gamma * potential(s_next, goal) - potential(s, goal)