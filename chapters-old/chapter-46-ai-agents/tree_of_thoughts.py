"""
第四十六章: Tree of Thoughts (思维树) 实现
==========================================

费曼比喻：Tree of Thoughts就像下棋时的思考过程 ♟️
- 普通LLM像即兴下棋——走一步看一步
- CoT像背棋谱——按固定顺序思考
- ToT像职业棋手——考虑多种走法，评估每种可能，选择最优路径
- 可以"悔棋"（回溯），尝试其他可能性

本章实现Yao et al. (2023)的Tree of Thoughts框架，包含：
- TreeNode: 思维树节点
- TreeOfThoughts: 思维树搜索主类
- BFS和DFS搜索策略
- 状态评估函数
- 数学问题求解示例

参考论文：
Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., & Narasimhan, K. (2023). 
Tree of Thoughts: Deliberate Problem Solving with Large Language Models. 
arXiv preprint arXiv:2305.10601.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import random
from collections import deque


class SearchStrategy(Enum):
    """搜索策略枚举"""
    BFS = "bfs"  # 广度优先搜索
    DFS = "dfs"  # 深度优先搜索
    BEAM = "beam"  # 束搜索


@dataclass
class TreeNode:
    """思维树节点
    
    费曼比喻：TreeNode就像决策树中的一个"决策点"
    - state: 当前状态（已经思考了什么）
    - thought: 这一步的具体思考内容
    - value: 这个思考的价值评估（0-10分）
    - children: 下一步可能的思考方向
    - parent: 上一个思考点（为了回溯）
    - depth: 思考深度（走了多少步）
    
    就像下棋时的局面评估：
    - 当前棋盘局面（state）
    - 准备走的这一步（thought）
    - 这步棋的好坏（value）
    - 对方可能的应对（children）
    """
    
    thought: str  # 这一步的思考内容
    state: str  # 当前完整状态（累积的思考）
    depth: int = 0  # 节点深度
    parent: Optional['TreeNode'] = None  # 父节点
    children: List['TreeNode'] = field(default_factory=list)  # 子节点
    value: Optional[float] = None  # 评估值
    visits: int = 0  # 访问次数（用于MCTS等）
    
    def add_child(self, thought: str) -> 'TreeNode':
        """添加子节点"""
        # 新状态 = 父状态 + 新思考
        new_state = f"{self.state}\n- {thought}" if self.state else thought
        
        child = TreeNode(
            thought=thought,
            state=new_state,
            depth=self.depth + 1,
            parent=self
        )
        self.children.append(child)
        return child
    
    def get_path(self) -> List['TreeNode']:
        """获取从根到当前节点的路径"""
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_path_thoughts(self) -> List[str]:
        """获取路径上的所有思考"""
        return [node.thought for node in self.get_path()]
    
    def is_leaf(self) -> bool:
        """是否是叶子节点"""
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """是否是根节点"""
        return self.parent is None
    
    def get_best_child(self) -> Optional['TreeNode']:
        """获取最佳子节点（基于value）"""
        if not self.children:
            return None
        
        # 过滤出有评估值的子节点
        evaluated = [c for c in self.children if c.value is not None]
        if not evaluated:
            return None
        
        return max(evaluated, key=lambda x: x.value)
    
    def __repr__(self) -> str:
        value_str = f"v={self.value:.2f}" if self.value else "v=?"
        return f"Node(d={self.depth}, {value_str}): {self.thought[:40]}..."


class StateEvaluator:
    """状态评估器
    
    费曼比喻：状态评估器就像考试阅卷老师
    - 给学生的答案打分（evaluate）
    - 判断答案是否合理（is_valid）
    - 确定是否是最终答案（is_solution）
    - 给出改进建议（get_feedback）
    """
    
    def __init__(self, llm_fn: Optional[Callable] = None):
        self.llm = llm_fn
    
    def evaluate(self, state: str, problem: str) -> float:
        """评估状态价值（0-10分）
        
        评分标准：
        - 0-3分：错误的思路或完全无关
        - 4-6分：方向正确但不够具体
        - 7-8分：具体且合理的思路
        - 9-10分：非常可能导向正确答案
        """
        if self.llm:
            # 使用LLM评估
            prompt = f"""请评估以下解题思路的质量（0-10分）：

问题: {problem}

当前思路:
{state}

请只输出一个0-10的数字，表示这个思路的质量（10分=非常优秀，0分=完全错误）："""
            try:
                response = self.llm(prompt)
                # 提取数字
                import re
                numbers = re.findall(r'\d+\.?\d*', response)
                if numbers:
                    score = float(numbers[0])
                    return max(0, min(10, score))  # 限制在0-10
            except:
                pass
        
        # 默认评估：启发式方法
        return self._heuristic_evaluate(state, problem)
    
    def _heuristic_evaluate(self, state: str, problem: str) -> float:
        """启发式评估（不依赖LLM）"""
        score = 5.0  # 基础分
        
        # 检查是否包含数字（数学题通常需要数字）
        import re
        if re.search(r'\d+', state):
            score += 1.0
        
        # 检查是否包含等号（可能有等式推导）
        if '=' in state:
            score += 0.5
        
        # 检查思路长度（太短的思路可能不够详细）
        words = len(state.split())
        if words > 10:
            score += 0.5
        if words > 30:
            score += 0.5
        
        # 检查是否包含常见错误关键词
        error_keywords = ['错误', '不对', 'impossible', 'wrong']
        for kw in error_keywords:
            if kw.lower() in state.lower():
                score -= 2.0
        
        return max(0, min(10, score))
    
    def is_valid(self, state: str, problem: str) -> bool:
        """检查状态是否有效（不是明显错误的）"""
        # 检查是否包含矛盾
        # 这里简化处理，实际可以用LLM判断
        return len(state) > 0 and '矛盾' not in state
    
    def is_solution(self, state: str, problem: str) -> bool:
        """检查是否是完整解决方案"""
        # 简单启发式：包含"答案"、"所以"等词，且包含数字
        import re
        has_answer_keyword = any(kw in state for kw in ['答案', '所以', '因此', 'answer', 'solution'])
        has_number = bool(re.search(r'\d+', state))
        return has_answer_keyword and has_number


class TreeOfThoughts:
    """Tree of Thoughts (思维树) 搜索
    
    费曼比喻：ToT就像一个探险家在迷宫中寻找出口
    - generate_thoughts: 在每个岔路口，思考可能的行进方向
    - evaluate_states: 评估每个方向的希望程度
    - search: 系统地探索，优先走看起来最有希望的路
    - 如果发现死胡同，可以回溯尝试其他路径
    
    与CoT的区别：
    - CoT: 一条路走到黑（贪心）
    - ToT: 探索多条路，选择最优（搜索）
    
    参考算法（BFS版本）：
    1. 从初始问题开始（根节点）
    2. 生成k个候选思考
    3. 评估每个候选，保留top-b个
    4. 对每个保留的候选，继续生成k个子思考
    5. 重复直到找到解决方案或达到最大深度
    """
    
    def __init__(self, 
                 llm_fn: Callable,
                 evaluator: Optional[StateEvaluator] = None,
                 strategy: SearchStrategy = SearchStrategy.BFS,
                 num_thoughts: int = 3,  # 每步生成几个候选思考
                 max_depth: int = 5,     # 最大思考深度
                 beam_width: int = 2,    # 束宽度（保留几个最佳候选）
                 pruning_threshold: float = 3.0):  # 剪枝阈值
        
        self.llm = llm_fn
        self.evaluator = evaluator or StateEvaluator(llm_fn)
        self.strategy = strategy
        self.num_thoughts = num_thoughts
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.pruning_threshold = pruning_threshold
        
        # 统计
        self.nodes_generated = 0
        self.nodes_evaluated = 0
    
    def solve(self, problem: str, verbose: bool = True) -> Tuple[str, List[str]]:
        """解决问题，返回答案和思考路径
        
        Args:
            problem: 要解决的问题
            verbose: 是否打印详细过程
        
        Returns:
            (答案, 思考路径)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🌳 Tree of Thoughts 求解")
            print(f"问题: {problem}")
            print(f"策略: {self.strategy.value.upper()}")
            print(f"{'='*60}\n")
        
        # 重置统计
        self.nodes_generated = 0
        self.nodes_evaluated = 0
        
        # 创建根节点
        root = TreeNode(thought="开始解决问题", state="")
        self.nodes_generated += 1
        
        # 根据策略搜索
        if self.strategy == SearchStrategy.BFS:
            solution = self._bfs_search(root, problem, verbose)
        elif self.strategy == SearchStrategy.DFS:
            solution = self._dfs_search(root, problem, verbose)
        elif self.strategy == SearchStrategy.BEAM:
            solution = self._beam_search(root, problem, verbose)
        else:
            raise ValueError(f"未知策略: {self.strategy}")
        
        if solution:
            path = solution.get_path_thoughts()
            answer = solution.state
            if verbose:
                print(f"\n✅ 找到解决方案!")
                print(f"评估值: {solution.value:.2f}")
                print(f"思考深度: {solution.depth}")
                print(f"生成节点数: {self.nodes_generated}")
                print(f"评估节点数: {self.nodes_evaluated}")
            return answer, path
        else:
            if verbose:
                print(f"\n❌ 未找到解决方案")
            return "", []
    
    def _generate_thoughts(self, node: TreeNode, problem: str, k: int) -> List[str]:
        """生成k个候选思考
        
        费曼比喻：就像头脑风暴时提出多个想法
        - 不评判好坏，先列出所有可能性
        - "这个问题可以从这几个角度思考..."
        """
        prompt = f"""问题: {problem}

当前思考进度:
{node.state if node.state else "[刚刚开始]"}

请提供{k}个不同的下一步思考方向。
每个思考应该简洁（1-2句话），是具体的推理步骤而非泛泛而谈。

格式:
1. [第一个思考方向]
2. [第二个思考方向]
3. [第三个思考方向]

思考:"""
        
        try:
            response = self.llm(prompt)
            # 解析响应，提取k个思考
            thoughts = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # 移除编号（如"1. "、"- "等）
                if line and (line[0].isdigit() or line.startswith('-')):
                    # 提取内容
                    content = line
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*']:
                        if content.startswith(prefix):
                            content = content[len(prefix):].strip()
                    if content:
                        thoughts.append(content)
            
            # 确保至少有k个思考
            while len(thoughts) < k:
                thoughts.append(f"尝试另一种方法 #{len(thoughts)+1}")
            
            return thoughts[:k]
        except Exception as e:
            # 如果LLM调用失败，返回默认思考
            return [f"思考方向 {i+1}" for i in range(k)]
    
    def _evaluate_node(self, node: TreeNode, problem: str) -> float:
        """评估节点价值"""
        if node.value is None:
            node.value = self.evaluator.evaluate(node.state, problem)
            self.nodes_evaluated += 1
        return node.value
    
    def _bfs_search(self, root: TreeNode, problem: str, verbose: bool) -> Optional[TreeNode]:
        """广度优先搜索
        
        费曼比喻：BFS就像层序遍历迷宫
        - 先探索所有第一步可能
        - 然后探索所有第二步可能
        - 一层一层向外扩展
        - 保证找到最短路径（最少思考步数）
        """
        if verbose:
            print(f"🔍 使用BFS搜索...")
        
        # 当前层的节点
        current_level = [root]
        
        for depth in range(self.max_depth):
            if verbose:
                print(f"\n  深度 {depth}: {len(current_level)} 个候选")
            
            next_level = []
            
            for node in current_level:
                # 生成候选子节点
                thoughts = self._generate_thoughts(node, problem, self.num_thoughts)
                
                for thought in thoughts:
                    child = node.add_child(thought)
                    self.nodes_generated += 1
                    
                    # 评估
                    score = self._evaluate_node(child, problem)
                    
                    # 检查是否是解决方案
                    if self.evaluator.is_solution(child.state, problem):
                        return child
                    
                    # 剪枝：太差的不要
                    if score >= self.pruning_threshold:
                        next_level.append(child)
            
            # 选择top-b保留到下一层
            if len(next_level) > self.beam_width:
                next_level.sort(key=lambda x: x.value or 0, reverse=True)
                next_level = next_level[:self.beam_width]
                if verbose:
                    print(f"    剪枝后保留: {len(next_level)} 个")
            
            current_level = next_level
            
            if not current_level:
                break
        
        # 返回最佳候选
        if current_level:
            return max(current_level, key=lambda x: x.value or 0)
        return None
    
    def _dfs_search(self, root: TreeNode, problem: str, verbose: bool,
                    max_branches: int = 2) -> Optional[TreeNode]:
        """深度优先搜索（带剪枝）
        
        费曼比喻：DFS就像"一条道走到黑"
        - 选择一条路径深入探索
        - 如果发现走不通，回溯
        - 尝试其他分支
        """
        if verbose:
            print(f"🔍 使用DFS搜索...")
        
        best_solution = None
        best_score = 0
        
        def dfs(node: TreeNode, depth: int) -> Optional[TreeNode]:
            nonlocal best_solution, best_score
            
            if depth >= self.max_depth:
                return None
            
            # 生成候选
            thoughts = self._generate_thoughts(node, problem, self.num_thoughts)
            
            # 创建子节点并评估
            candidates = []
            for thought in thoughts:
                child = node.add_child(thought)
                self.nodes_generated += 1
                score = self._evaluate_node(child, problem)
                candidates.append((child, score))
            
            # 按分数排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # 只探索top分支（剪枝）
            for child, score in candidates[:max_branches]:
                if verbose and depth < 2:
                    print(f"  {'  '*depth}探索: {child.thought[:40]}... (得分: {score:.2f})")
                
                # 检查是否是解决方案
                if self.evaluator.is_solution(child.state, problem):
                    if score > best_score:
                        best_solution = child
                        best_score = score
                    return child
                
                # 剪枝
                if score < self.pruning_threshold:
                    continue
                
                # 递归探索
                result = dfs(child, depth + 1)
                if result and (result.value or 0) > best_score:
                    best_solution = result
                    best_score = result.value or 0
            
            return best_solution
        
        return dfs(root, 0)
    
    def _beam_search(self, root: TreeNode, problem: str, verbose: bool) -> Optional[TreeNode]:
        """束搜索（BFS的变体，每层只保留beam_width个最佳）
        
        费曼比喻：束搜索就像带着有限的探照灯探索
        - 你的资源有限，只能同时跟踪几个最有希望的线索
        - 每层评估后，只保留最有希望的继续探索
        - 在广度和深度之间取得平衡
        """
        if verbose:
            print(f"🔍 使用Beam Search（束宽={self.beam_width}）...")
        
        # 当前层的候选
        beam = [root]
        
        for depth in range(self.max_depth):
            if verbose:
                print(f"\n  深度 {depth}: 束大小 {len(beam)}")
            
            all_candidates = []
            
            # 扩展每个候选
            for node in beam:
                thoughts = self._generate_thoughts(node, problem, self.num_thoughts)
                
                for thought in thoughts:
                    child = node.add_child(thought)
                    self.nodes_generated += 1
                    score = self._evaluate_node(child, problem)
                    
                    # 检查是否是解决方案
                    if self.evaluator.is_solution(child.state, problem):
                        return child
                    
                    all_candidates.append((child, score))
            
            # 选择top beam_width个
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = [c for c, _ in all_candidates[:self.beam_width]]
            
            if verbose:
                print(f"    选择最佳 {len(beam)} 个继续")
                for i, node in enumerate(beam[:3]):
                    print(f"      #{i+1}: {node.thought[:35]}... (v={node.value:.2f})")
            
            if not beam:
                break
        
        # 返回最佳候选
        if beam:
            return max(beam, key=lambda x: x.value or 0)
        return None
    
    def visualize_tree(self, root: TreeNode, max_depth: int = 3) -> str:
        """可视化思维树"""
        lines = ["思维树结构:", ""]
        
        def draw_node(node: TreeNode, prefix: str = "", is_last: bool = True):
            # 当前节点
            connector = "└── " if is_last else "├── "
            value_str = f"[{node.value:.1f}]" if node.value else "[?]"
            thought_short = node.thought[:30] + "..." if len(node.thought) > 30 else node.thought
            lines.append(f"{prefix}{connector}{value_str} {thought_short}")
            
            # 子节点
            new_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                if node.depth < max_depth:
                    is_last_child = (i == len(node.children) - 1)
                    draw_node(child, new_prefix, is_last_child)
        
        draw_node(root)
        return "\n".join(lines)


# ==================== 数学问题求解器 ====================

class MathProblemSolver:
    """数学问题求解器 - ToT应用示例
    
    解决24点游戏、数学推理等问题
    """
    
    def __init__(self, llm_fn: Optional[Callable] = None):
        # 默认使用模拟LLM
        self.llm = llm_fn or self._mock_llm
        self.tot = TreeOfThoughts(
            llm_fn=self.llm,
            strategy=SearchStrategy.BEAM,
            num_thoughts=3,
            max_depth=4,
            beam_width=2
        )
    
    def _mock_llm(self, prompt: str) -> str:
        """模拟LLM响应（用于演示）"""
        if "生成" in prompt or "思考" in prompt.lower() or "thought" in prompt.lower():
            # 生成思考
            if "24点" in prompt or "24 game" in prompt.lower():
                return self._mock_24game_thoughts(prompt)
            elif "鸡蛋" in prompt or "egg" in prompt.lower():
                return self._mock_egg_thoughts(prompt)
            else:
                return """1. 分析问题的已知条件和要求
2. 列出可能的解题公式或方法
3. 尝试代入数值进行计算验证"""
        
        elif "评估" in prompt or "evaluate" in prompt.lower():
            # 评估
            return "7"
        
        return "这是一个不错的思路。"
    
    def _mock_24game_thoughts(self, prompt: str) -> str:
        """模拟24点游戏的思考"""
        if "4, 6, 8, 2" in prompt:
            return """1. 尝试 (8 - 6) * 4 * 2 = 16，不等于24
2. 尝试 8 * 6 / (4 - 2) = 48/2 = 24 ✓
3. 尝试 (6 + 2) * 4 - 8 = 32 - 8 = 24 ✓"""
        elif "3, 3, 8, 8" in prompt:
            return """1. 尝试 8 + 8 + 3 + 3 = 22，不够
2. 尝试 8 / (3 - 8/3) = 8 / (1/3) = 24 ✓
3. 尝试 (8 - 3) * 3 + 8 = 15 + 8 = 23，接近"""
        else:
            return """1. 尝试用乘法得到接近24的数
2. 尝试用除法简化计算
3. 尝试加减法微调结果"""
    
    def _mock_egg_thoughts(self, prompt: str) -> str:
        """模拟鸡蛋掉落问题的思考"""
        return """1. 使用二分查找策略，从50楼开始
2. 采用线性扫描，从1楼逐层尝试
3. 使用动态规划，平衡最坏情况"""
    
    def solve_24game(self, numbers: List[int]) -> str:
        """解决24点游戏
        
        给定4个数字，通过加减乘除得到24
        """
        problem = f"使用数字 {numbers}，通过加减乘除得到24。每个数字必须使用且只能使用一次。"
        
        print(f"\n{'='*60}")
        print(f"🎮 24点游戏: {numbers}")
        print(f"{'='*60}")
        
        answer, path = self.tot.solve(problem, verbose=True)
        
        print(f"\n💡 解题路径:")
        for i, step in enumerate(path, 1):
            print(f"  Step {i}: {step}")
        
        return answer
    
    def solve_math_word_problem(self, problem: str) -> str:
        """解决数学应用题"""
        print(f"\n{'='*60}")
        print(f"🧮 数学问题求解")
        print(f"{'='*60}")
        
        answer, path = self.tot.solve(problem, verbose=True)
        
        print(f"\n💡 思考过程:")
        for i, step in enumerate(path, 1):
            print(f"  Step {i}: {step}")
        
        print(f"\n📋 最终答案:")
        print(f"  {answer}")
        
        return answer


# ==================== 演示 ====================

def demo_tree_node():
    """演示TreeNode的基本操作"""
    print("\n" + "="*60)
    print("演示: TreeNode基本操作")
    print("="*60)
    
    # 创建根节点
    root = TreeNode(thought="开始", state="开始")
    
    # 添加子节点
    child1 = root.add_child("方向A: 用加法")
    child2 = root.add_child("方向B: 用乘法")
    
    # 继续展开
    grandchild1 = child1.add_child("尝试 2+3")
    grandchild2 = child1.add_child("尝试 4+5")
    
    # 评估
    grandchild1.value = 6.0
    grandchild2.value = 8.5
    child1.value = 7.0
    child2.value = 5.0
    
    print(f"根节点: {root}")
    print(f"子节点: {child1}, {child2}")
    print(f"最佳子节点: {root.get_best_child()}")
    
    # 显示路径
    print(f"\n到最佳叶子节点的路径:")
    for node in grandchild2.get_path():
        print(f"  -> {node.thought}")


def demo_tot_search():
    """演示ToT搜索"""
    print("\n" + "="*60)
    print("演示: Tree of Thoughts搜索")
    print("="*60)
    
    # 模拟LLM
    def mock_llm(prompt: str) -> str:
        if "评估" in prompt:
            return "8"
        elif "生成" in prompt or "思考" in prompt:
            return """1. 首先分析问题中的已知条件
2. 尝试建立变量之间的关系方程
3. 使用代数方法求解未知数
4. 验证答案是否符合题意"""
        return "继续思考..."
    
    # 创建ToT
    tot = TreeOfThoughts(
        llm_fn=mock_llm,
        strategy=SearchStrategy.BFS,
        num_thoughts=3,
        max_depth=3,
        beam_width=2
    )
    
    # 求解问题
    problem = "小明有10个苹果，给了小红3个，又买了5个，现在有多少个？"
    answer, path = tot.solve(problem, verbose=True)
    
    print(f"\n答案: {answer}")


def demo_math_solver():
    """演示数学求解器"""
    solver = MathProblemSolver()
    
    # 24点游戏
    solver.solve_24game([4, 6, 8, 2])
    solver.solve_24game([3, 3, 8, 8])
    
    # 数学应用题
    solver.solve_math_word_problem(
        "一个水池有两个进水管，甲管单独注满需要6小时，乙管单独注满需要4小时。"
        "如果两个水管同时打开，需要多少小时注满水池？"
    )


def demo_compare_strategies():
    """比较不同搜索策略"""
    print("\n" + "="*60)
    print("演示: 比较搜索策略")
    print("="*60)
    
    problem = "计算 2 + 3 * 4 - 5 / 5 的值"
    
    def mock_llm(prompt: str) -> str:
        if "评估" in prompt:
            return "7"
        return """1. 按运算优先级，先乘除后加减
2. 从左到右依次计算
3. 先计算括号内的表达式"""
    
    strategies = [
        (SearchStrategy.BFS, "广度优先"),
        (SearchStrategy.DFS, "深度优先"),
        (SearchStrategy.BEAM, "束搜索")
    ]
    
    for strategy, name in strategies:
        print(f"\n{'─'*40}")
        print(f"策略: {name}")
        print(f"{'─'*40}")
        
        tot = TreeOfThoughts(
            llm_fn=mock_llm,
            strategy=strategy,
            num_thoughts=3,
            max_depth=3,
            beam_width=2
        )
        
        answer, path = tot.solve(problem, verbose=False)
        print(f"生成节点: {tot.nodes_generated}")
        print(f"评估节点: {tot.nodes_evaluated}")
        print(f"找到路径长度: {len(path)}")


if __name__ == "__main__":
    demo_tree_node()
    demo_tot_search()
    demo_compare_strategies()
    demo_math_solver()
