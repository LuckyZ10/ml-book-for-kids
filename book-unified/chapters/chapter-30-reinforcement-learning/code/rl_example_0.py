"""
Grid World环境：强化学习的"Hello World"
一个简单的网格世界，智能体需要找到从起点到终点的最短路径

网格布局：
S . . .
. X . G
. . . .

S: 起点 (0,0)
G: 终点/目标 (3,1) - 奖励 +10
X: 陷阱 (1,1) - 奖励 -10
.: 普通格子 - 每步奖励 -0.1
"""

import numpy as np
import random
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Transition:
    """存储状态转移"""
    state: Tuple[int, int]
    action: int
    reward: float
    next_state: Tuple[int, int]
    done: bool


class GridWorld:
    """
    网格世界环境
    
    状态：智能体的位置 (row, col)
    动作：0=上, 1=右, 2=下, 3=左
    """
    
    # 动作映射
    ACTIONS = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    ACTION_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    
    def __init__(self, size: int = 4, seed: Optional[int] = None):
        """
        初始化Grid World环境
        
        Args:
            size: 网格大小 (size x size)
            seed: 随机种子
        """
        self.size = size
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        
        # 设置陷阱位置
        self.traps = set()
        if size >= 4:
            self.traps.add((size // 2, size // 2))
            self.traps.add((size // 2 - 1, size // 2))
        
        # 奖励设置
        self.goal_reward = 10.0
        self.trap_reward = -10.0
        self.step_reward = -0.1
        
        # 状态
        self.agent_pos = self.start_pos
        self.steps = 0
        self.max_steps = size * size * 4
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def reset(self) -> Tuple[int, int]:
        """重置环境，返回初始状态"""
        self.agent_pos = self.start_pos
        self.steps = 0
        return self.agent_pos
    
    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, dict]:
        """
        执行动作，返回 (下一状态, 奖励, 是否结束, 额外信息)
        
        有10%的概率会随机执行一个动作（模拟环境的不确定性）
        """
        self.steps += 1
        
        # 10%概率随机动作（环境随机性）
        if random.random() < 0.1:
            action = random.randint(0, 3)
        
        # 计算新位置
        dr, dc = self.ACTION_DELTA[action]
        new_row = max(0, min(self.size - 1, self.agent_pos[0] + dr))
        new_col = max(0, min(self.size - 1, self.agent_pos[1] + dc))
        new_pos = (new_row, new_col)
        
        self.agent_pos = new_pos
        
        # 计算奖励
        if new_pos == self.goal_pos:
            reward = self.goal_reward
            done = True
        elif new_pos in self.traps:
            reward = self.trap_reward
            done = True
        else:
            reward = self.step_reward
            done = self.steps >= self.max_steps
        
        info = {'steps': self.steps, 'action_taken': action}
        return new_pos, reward, done, info
    
    def get_valid_actions(self, state: Optional[Tuple[int, int]] = None) -> List[int]:
        """获取在指定状态下的有效动作"""
        if state is None:
            state = self.agent_pos
        return [0, 1, 2, 3]  # 在这个简单环境中所有动作都有效
    
    def render(self) -> str:
        """渲染当前环境状态"""
        lines = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                pos = (r, c)
                if pos == self.agent_pos:
                    row.append(' A ')  # 智能体
                elif pos == self.start_pos:
                    row.append(' S ')  # 起点
                elif pos == self.goal_pos:
                    row.append(' G ')  # 终点
                elif pos in self.traps:
                    row.append(' X ')  # 陷阱
                else:
                    row.append(' . ')  # 普通格子
            lines.append(''.join(row))
        return '\n'.join(lines)
    
    def get_state_index(self, state: Tuple[int, int]) -> int:
        """将2D状态转换为1D索引"""
        return state[0] * self.size + state[1]
    
    def get_state_from_index(self, index: int) -> Tuple[int, int]:
        """将1D索引转换为2D状态"""
        return (index // self.size, index % self.size)
    
    @property
    def num_states(self) -> int:
        """状态数量"""
        return self.size * self.size
    
    @property
    def num_actions(self) -> int:
        """动作数量"""
        return 4


def demo_gridworld():
    """演示Grid World环境"""
    print("=" * 50)
    print("Grid World 环境演示")
    print("=" * 50)
    
    env = GridWorld(size=4, seed=42)
    print(f"\n网格大小: {env.size}x{env.size}")
    print(f"起点: {env.start_pos}")
    print(f"终点: {env.goal_pos}")
    print(f"陷阱: {env.traps}")
    
    print("\n初始状态:")
    print(env.render())
    
    # 随机运行几步
    print("\n随机执行动作:")
    state = env.reset()
    total_reward = 0
    
    for i in range(10):
        action = random.randint(0, 3)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"\n第 {i+1} 步:")
        print(f"  动作: {GridWorld.ACTIONS[action]}")
        print(f"  位置: {state} -> {next_state}")
        print(f"  奖励: {reward:.2f}")
        print(env.render())
        
        if done:
            print(f"\n回合结束！总奖励: {total_reward:.2f}")
            break
        
        state = next_state


if __name__ == "__main__":
    demo_gridworld()