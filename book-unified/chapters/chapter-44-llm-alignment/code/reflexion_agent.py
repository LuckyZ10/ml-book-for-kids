"""
第四十六章: Reflexion Agent 实现
===============================

费曼比喻：Reflexion Agent就像一位运动员回看比赛录像 🏆
- Actor（运动员）：上场比赛，执行任务
- Evaluator（教练）：观看表现，给出评分和反馈
- Self-Reflection（运动员反思）："刚才那个传球选择不好，下次要先观察"
- Memory（训练笔记）：把经验教训记下来，下次改进
- 通过不断"实践-评估-反思-改进"的循环，持续提升表现

本章实现Shinn et al. (2023)的Reflexion框架，包含：
- ReflexionAgent: 主类
- Actor: 执行组件
- Evaluator: 评估组件
- SelfReflection: 反思组件
- 失败记忆存储和检索

参考论文：
Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). 
Reflexion: Self-Reflective Agents. 
arXiv preprint arXiv:2303.11366.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Experience:
    """经验教训记录
    
    费曼比喻：Experience就像运动员的训练笔记
    - task_description: 这次训练的项目
    - failed_attempt: 哪里出了问题
    - reflection: 为什么会出问题
    - lesson_learned: 总结的经验教训
    - improvement_suggestion: 下次如何改进
    """
    task_description: str
    failed_attempt: str
    reflection: str
    lesson_learned: str
    improvement_suggestion: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success_count: int = 0  # 应用此经验后成功的次数
    
    def to_prompt(self) -> str:
        """转换为提示文本"""
        return f"""经验教训 ({self.timestamp}):
任务: {self.task_description}
失败尝试: {self.failed_attempt}
反思: {self.reflection}
学到的教训: {self.lesson_learned}
改进建议: {self.improvement_suggestion}
"""


@dataclass
class TaskResult:
    """任务执行结果"""
    task: str
    output: str
    success: bool
    score: float  # 0-10分
    feedback: str
    attempts: int = 1
    reflections_used: List[str] = field(default_factory=list)


class MemoryStore:
    """经验记忆存储
    
    费曼比喻：MemoryStore就像运动员的笔记本
    - 记录每次训练和比赛的得失
    - 按项目分类整理（标签系统）
    - 经常回顾，避免重复犯错
    - 赛前翻阅，提醒自己注意事项
    """
    
    def __init__(self):
        self.experiences: List[Experience] = []
        self.tag_index: Dict[str, List[int]] = {}  # 标签索引
    
    def add_experience(self, experience: Experience, tags: List[str] = None):
        """添加经验"""
        idx = len(self.experiences)
        self.experiences.append(experience)
        
        # 更新索引
        if tags:
            for tag in tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                self.tag_index[tag].append(idx)
    
    def retrieve_relevant(self, task_description: str, top_k: int = 3) -> List[Experience]:
        """检索相关经验
        
        简单实现：基于关键词匹配
        实际可用：向量相似度搜索
        """
        task_words = set(task_description.lower().split())
        
        scored_experiences = []
        for exp in self.experiences:
            # 计算关键词重叠
            exp_words = set(exp.task_description.lower().split())
            overlap = len(task_words & exp_words)
            
            # 考虑成功率
            success_bonus = exp.success_count * 0.1
            
            score = overlap + success_bonus
            scored_experiences.append((exp, score))
        
        # 排序并返回top_k
        scored_experiences.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in scored_experiences[:top_k]]
    
    def get_all_lessons(self) -> List[str]:
        """获取所有教训摘要"""
        return [exp.lesson_learned for exp in self.experiences]
    
    def update_success_count(self, experience_idx: int):
        """更新经验的成功计数（应用后成功）"""
        if 0 <= experience_idx < len(self.experiences):
            self.experiences[experience_idx].success_count += 1
    
    def clear(self):
        """清空记忆"""
        self.experiences.clear()
        self.tag_index.clear()


class Actor:
    """执行者组件
    
    费曼比喻：Actor就像上场比赛的运动员
    - 接收任务（教练布置的战术）
    - 结合自身经验（训练笔记）
    - 执行任务（比赛）
    - 返回结果（比赛成绩）
    """
    
    def __init__(self, llm_fn: Callable, memory: MemoryStore):
        self.llm = llm_fn
        self.memory = memory
    
    def execute(self, task: str, max_retries: int = 3) -> Tuple[str, List[str]]:
        """执行任务
        
        Args:
            task: 任务描述
            max_retries: 最大重试次数
        
        Returns:
            (执行结果, 使用的经验列表)
        """
        # 检索相关经验
        relevant_exps = self.memory.retrieve_relevant(task)
        lessons = [exp.to_prompt() for exp in relevant_exps]
        
        # 构建提示
        prompt = self._build_execution_prompt(task, lessons)
        
        # 调用LLM执行
        try:
            output = self.llm(prompt)
        except Exception as e:
            output = f"执行出错: {str(e)}"
        
        return output, lessons
    
    def _build_execution_prompt(self, task: str, lessons: List[str]) -> str:
        """构建执行提示"""
        prompt = f"""你是一位AI助手，请完成以下任务。

任务:
{task}
"""
        
        if lessons:
            prompt += "\n\n基于以往的经验教训，请注意:\n"
            for i, lesson in enumerate(lessons, 1):
                prompt += f"\n--- 经验 {i} ---\n{lesson}\n"
        
        prompt += "\n\n请完成任务，并给出详细过程和最终答案。"
        
        return prompt


class Evaluator:
    """评估者组件
    
    费曼比喻：Evaluator就像场边的教练
    - 观看运动员的表现
    - 给出客观评分
    - 指出具体问题和改进点
    - 判断是否达到训练目标
    """
    
    def __init__(self, llm_fn: Callable):
        self.llm = llm_fn
    
    def evaluate(self, task: str, output: str) -> Tuple[float, str]:
        """评估执行结果
        
        Returns:
            (分数0-10, 详细反馈)
        """
        prompt = f"""请评估以下任务执行的质量。

任务:
{task}

执行结果:
{output}

请从以下维度评估:
1. 正确性：结果是否正确？
2. 完整性：是否完成了所有要求？
3. 清晰度：表达是否清晰易懂？

请给出:
- 分数（0-10分，10分=完美）
- 详细反馈（包括优点和需要改进的地方）

格式:
分数: [数字]
反馈: [详细评价]"""
        
        try:
            response = self.llm(prompt)
            score, feedback = self._parse_evaluation(response)
            return score, feedback
        except Exception as e:
            return 0.0, f"评估出错: {str(e)}"
    
    def _parse_evaluation(self, response: str) -> Tuple[float, str]:
        """解析评估响应"""
        # 提取分数
        score_match = re.search(r'分数[:：]\s*(\d+(?:\.\d+)?)', response)
        score = float(score_match.group(1)) if score_match else 5.0
        score = max(0, min(10, score))
        
        # 提取反馈
        feedback_match = re.search(r'反馈[:：]\s*(.+)', response, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else response
        
        return score, feedback
    
    def is_success(self, score: float, threshold: float = 7.0) -> bool:
        """判断是否成功"""
        return score >= threshold


class SelfReflection:
    """自我反思组件
    
    费曼比喻：Self-Reflection就像运动员回看录像时的思考
    - "刚才那个球我处理得太急了"
    - "我应该先观察队友位置"
    - "下次遇到这种情况，我要..."
    - 把领悟记录下来，形成经验
    """
    
    def __init__(self, llm_fn: Callable):
        self.llm = llm_fn
    
    def reflect(self, task: str, output: str, feedback: str) -> Experience:
        """生成反思经验
        
        分析失败原因，提炼经验教训
        """
        prompt = f"""请对以下任务执行进行深刻反思。

任务:
{task}

执行结果:
{output}

评估反馈:
{feedback}

请回答:
1. 失败的根本原因是什么？
2. 我学到了什么教训？
3. 下次如何改进才能成功？

格式:
失败原因: [分析]
教训: [学到的经验]
改进建议: [具体可行的改进方法]"""
        
        try:
            response = self.llm(prompt)
            return self._parse_reflection(task, output, response)
        except Exception as e:
            # 返回默认反思
            return Experience(
                task_description=task[:100],
                failed_attempt=output[:200],
                reflection=f"反思生成失败: {str(e)}",
                lesson_learned="需要更仔细的分析",
                improvement_suggestion="逐步验证每个步骤"
            )
    
    def _parse_reflection(self, task: str, output: str, response: str) -> Experience:
        """解析反思响应"""
        # 提取各部分
        def extract_section(text: str, keywords: List[str]) -> str:
            for keyword in keywords:
                pattern = rf'{keyword}[:：]\s*(.+?)(?=\n\w+[:：]|$)'
                match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return ""
        
        failed_reason = extract_section(response, ["失败原因", "根本原因", "原因"])
        lesson = extract_section(response, ["教训", "学到了", "经验"])
        improvement = extract_section(response, ["改进建议", "改进方法", "如何改进"])
        
        # 如果没有提取到，使用整个响应
        if not failed_reason:
            failed_reason = response[:200]
        if not lesson:
            lesson = "需要更系统的分析方法"
        if not improvement:
            improvement = "逐步验证，及时检查"
        
        return Experience(
            task_description=task[:200],
            failed_attempt=output[:300],
            reflection=failed_reason,
            lesson_learned=lesson,
            improvement_suggestion=improvement
        )


class ReflexionAgent:
    """Reflexion Agent - 自反思智能体
    
    三组件架构：
    1. Actor：执行任务
    2. Evaluator：评估结果
    3. Self-Reflection：生成改进经验
    
    工作流程：
    1. 接收任务
    2. Actor执行任务（结合历史经验）
    3. Evaluator评估结果
    4. 如果成功，返回结果
    5. 如果失败，Self-Reflection生成经验教训
    6. 存储经验，使用新经验重试
    7. 重复直到成功或达到最大尝试次数
    
    费曼比喻：完整的训练-比赛-反思-提升循环
    """
    
    def __init__(self, 
                 llm_fn: Callable,
                 max_iterations: int = 3,
                 success_threshold: float = 7.0):
        
        self.llm = llm_fn
        self.max_iterations = max_iterations
        self.success_threshold = success_threshold
        
        # 初始化组件
        self.memory = MemoryStore()
        self.actor = Actor(llm_fn, self.memory)
        self.evaluator = Evaluator(llm_fn)
        self.reflector = SelfReflection(llm_fn)
        
        # 统计
        self.total_tasks = 0
        self.successful_tasks = 0
        self.total_reflections = 0
    
    def run(self, task: str, verbose: bool = True) -> TaskResult:
        """运行Reflexion循环完成任务
        
        Args:
            task: 任务描述
            verbose: 是否打印详细过程
        
        Returns:
            任务执行结果
        """
        self.total_tasks += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔄 Reflexion Agent 开始任务")
            print(f"任务: {task[:60]}...")
            print(f"{'='*60}\n")
        
        all_reflections = []
        last_output = ""
        last_score = 0.0
        last_feedback = ""
        
        for iteration in range(1, self.max_iterations + 1):
            if verbose:
                print(f"\n{'─'*50}")
                print(f"📝 尝试 #{iteration}")
                print(f"{'─'*50}")
            
            # Step 1: Actor执行任务
            output, reflections_used = self.actor.execute(task)
            last_output = output
            
            if verbose:
                print(f"\n🎭 Actor输出:")
                print(f"{output[:300]}...")
                if reflections_used:
                    print(f"\n📚 应用了 {len(reflections_used)} 条历史经验")
            
            # Step 2: Evaluator评估
            score, feedback = self.evaluator.evaluate(task, output)
            last_score = score
            last_feedback = feedback
            
            if verbose:
                print(f"\n📊 Evaluator评估:")
                print(f"  分数: {score:.1f}/10")
                print(f"  反馈: {feedback[:150]}...")
            
            # Step 3: 检查是否成功
            if self.evaluator.is_success(score, self.success_threshold):
                self.successful_tasks += 1
                
                if verbose:
                    print(f"\n✅ 任务成功完成！")
                    print(f"   最终分数: {score:.1f}/10")
                    print(f"   尝试次数: {iteration}")
                    print(f"   反思次数: {len(all_reflections)}")
                
                return TaskResult(
                    task=task,
                    output=output,
                    success=True,
                    score=score,
                    feedback=feedback,
                    attempts=iteration,
                    reflections_used=[r.lesson_learned for r in all_reflections]
                )
            
            # Step 4: 如果失败且还有尝试次数，进行反思
            if iteration < self.max_iterations:
                if verbose:
                    print(f"\n🤔 执行未达标，进入反思...")
                
                experience = self.reflector.reflect(task, output, feedback)
                all_reflections.append(experience)
                self.total_reflections += 1
                
                # 存储经验
                self.memory.add_experience(experience, tags=["failed_task"])
                
                if verbose:
                    print(f"\n💡 生成的经验教训:")
                    print(f"  失败原因: {experience.reflection[:100]}...")
                    print(f"  学到的教训: {experience.lesson_learned[:100]}...")
                    print(f"  改进建议: {experience.improvement_suggestion[:100]}...")
            else:
                if verbose:
                    print(f"\n⚠️ 达到最大尝试次数，任务未能成功完成")
        
        # 返回最后一次尝试的结果
        return TaskResult(
            task=task,
            output=last_output,
            success=False,
            score=last_score,
            feedback=last_feedback,
            attempts=self.max_iterations,
            reflections_used=[r.lesson_learned for r in all_reflections]
        )
    
    def batch_run(self, tasks: List[str], verbose: bool = False) -> List[TaskResult]:
        """批量处理多个任务"""
        results = []
        for task in tasks:
            result = self.run(task, verbose=verbose)
            results.append(result)
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "success_rate": f"{self.successful_tasks/max(1,self.total_tasks)*100:.1f}%",
            "total_reflections": self.total_reflections,
            "experiences_in_memory": len(self.memory.experiences)
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print(f"📈 Reflexion Agent 统计")
        print(f"{'='*60}")
        print(f"总任务数: {stats['total_tasks']}")
        print(f"成功任务: {stats['successful_tasks']}")
        print(f"成功率: {stats['success_rate']}")
        print(f"总反思次数: {stats['total_reflections']}")
        print(f"记忆中的经验: {stats['experiences_in_memory']}")
    
    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        if not self.memory.experiences:
            return "记忆中暂无经验"
        
        summary = f"记忆中的经验教训 ({len(self.memory.experiences)}条):\n\n"
        for i, exp in enumerate(self.memory.experiences[-5:], 1):  # 最近5条
            summary += f"{i}. {exp.lesson_learned[:80]}...\n"
        return summary


# ==================== 应用场景 ====================

class CodeReflexionAgent(ReflexionAgent):
    """代码生成专用的Reflexion Agent"""
    
    def __init__(self, llm_fn: Callable, max_iterations: int = 3):
        super().__init__(llm_fn, max_iterations, success_threshold=8.0)
    
    def run_code_task(self, code_task: str, test_cases: List[Dict], verbose: bool = True) -> TaskResult:
        """运行代码生成任务，用测试用例验证"""
        
        # 包装原始评估器，添加代码测试
        original_evaluate = self.evaluator.evaluate
        
        def code_evaluate(task: str, output: str) -> Tuple[float, str]:
            # 基础评估
            base_score, base_feedback = original_evaluate(task, output)
            
            # 代码特定评估
            code_score = self._evaluate_code(output, test_cases)
            
            # 综合分数
            final_score = (base_score + code_score) / 2
            
            return final_score, base_feedback
        
        self.evaluator.evaluate = code_evaluate
        
        return self.run(code_task, verbose)
    
    def _evaluate_code(self, code: str, test_cases: List[Dict]) -> float:
        """评估代码质量"""
        score = 10.0
        
        # 检查是否包含代码
        if '```' not in code and 'def ' not in code:
            score -= 5
        
        # 检查是否有语法错误指示
        if 'error' in code.lower() or 'exception' in code.lower():
            score -= 3
        
        # 检查是否有注释
        if '#' not in code:
            score -= 1
        
        return max(0, score)


class MathReflexionAgent(ReflexionAgent):
    """数学问题专用的Reflexion Agent"""
    
    def __init__(self, llm_fn: Callable, max_iterations: int = 3):
        super().__init__(llm_fn, max_iterations, success_threshold=9.0)  # 数学要求高
    
    def verify_answer(self, output: str, correct_answer: Optional[str] = None) -> bool:
        """验证答案正确性"""
        if correct_answer and correct_answer in output:
            return True
        
        # 检查是否包含计算过程
        has_calculation = any(op in output for op in ['=', '+', '-', '*', '/'])
        has_number = bool(re.search(r'\d+', output))
        
        return has_calculation and has_number


# ==================== 演示 ====================

def mock_llm(prompt: str) -> str:
    """模拟LLM响应"""
    prompt_lower = prompt.lower()
    
    # 执行任务
    if "完成任务" in prompt or "请完成" in prompt:
        if "排序" in prompt:
            return """我会使用快速排序算法来完成这个任务。

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# 测试
result = quicksort([3, 1, 4, 1, 5, 9, 2, 6])
print(result)  # [1, 1, 2, 3, 4, 5, 6, 9]
```

最终答案: [1, 1, 2, 3, 4, 5, 6, 9]"""
        elif "计算" in prompt or "求" in prompt:
            return """让我逐步计算：

1. 首先识别已知条件
2. 应用公式进行计算
3. 验证结果

计算过程：...
最终答案: 42"""
        else:
            return "这是我对任务的理解和解决方案...最终答案是..."
    
    # 评估任务
    elif "评估" in prompt or "evaluate" in prompt_lower:
        if "排序" in prompt:
            return """分数: 8.5
反馈: 代码实现正确，使用了经典的快速排序算法。代码结构清晰，包含注释和测试用例。可以改进的地方是添加对边界情况的处理。"""
        return """分数: 7.0
反馈: 回答基本正确，但还可以更加详细和准确。"""
    
    # 反思任务
    elif "反思" in prompt or "reflect" in prompt_lower:
        return """失败原因: 之前的解法过于复杂，没有考虑到更简单的方法
教训: 应该先分析问题的最简单解法，而不是直接使用复杂算法
改进建议: 下次遇到类似问题，先尝试暴力解法，再根据复杂度要求优化"""
    
    return "收到，我将继续处理。"


def demo_reflexion_basic():
    """演示基本Reflexion功能"""
    print("\n" + "="*60)
    print("演示: Reflexion Agent基本功能")
    print("="*60)
    
    agent = ReflexionAgent(llm_fn=mock_llm, max_iterations=3)
    
    # 运行任务
    task = "请实现一个快速排序算法，对数组 [3, 1, 4, 1, 5, 9, 2, 6] 进行排序"
    result = agent.run(task, verbose=True)
    
    print(f"\n📋 最终结果:")
    print(f"  成功: {'✓' if result.success else '✗'}")
    print(f"  分数: {result.score:.1f}/10")
    print(f"  尝试次数: {result.attempts}")


def demo_reflexion_with_failure():
    """演示失败-反思-重试的完整循环"""
    print("\n" + "="*60)
    print("演示: 失败-反思-改进循环")
    print("="*60)
    
    attempt_count = [0]
    
    def failing_then_success_llm(prompt: str) -> str:
        attempt_count[0] += 1
        
        # 执行任务
        if "完成任务" in prompt:
            if "经验教训" not in prompt:  # 第一次，没有经验
                return "我觉得答案是25..."  # 错误答案
            else:
                return """让我重新仔细计算：
1. 小明有10个苹果
2. 给了小红3个，剩余 10 - 3 = 7个
3. 又买了5个，现在有 7 + 5 = 12个
最终答案: 12"""
        
        # 评估
        elif "评估" in prompt:
            if "25" in prompt:
                return """分数: 3.0
反馈: 答案错误。计算过程有误，没有正确跟踪苹果的增减。"""
            else:
                return """分数: 9.0
反馈: 答案正确！计算过程清晰，步骤完整。"""
        
        # 反思
        elif "反思" in prompt:
            return """失败原因: 没有仔细读题，忽略了"给了小红"是减少苹果
教训: 数学应用题要仔细识别关键词，"给"表示减少，"买"表示增加
改进建议: 列出每一步的数量变化，避免心算出错"""
        
        return "继续..."
    
    agent = ReflexionAgent(llm_fn=failing_then_success_llm, max_iterations=3)
    
    task = "小明有10个苹果，给了小红3个，又买了5个，现在有多少个苹果？"
    result = agent.run(task, verbose=True)
    
    print(f"\n🎯 任务结果:")
    print(f"  成功: {'✓' if result.success else '✗'}")
    print(f"  最终分数: {result.score:.1f}/10")
    print(f"  尝试次数: {result.attempts}")


def demo_memory_accumulation():
    """演示经验积累"""
    print("\n" + "="*60)
    print("演示: 经验积累与复用")
    print("="*60)
    
    agent = ReflexionAgent(llm_fn=mock_llm, max_iterations=2)
    
    # 任务1：产生经验
    task1 = "任务类型A：排序问题"
    result1 = agent.run(task1, verbose=False)
    
    # 任务2：产生经验
    task2 = "任务类型B：搜索问题"
    result2 = agent.run(task2, verbose=False)
    
    # 打印记忆状态
    print(agent.get_memory_summary())
    
    # 任务3：复用经验
    print(f"\n现在处理类似任务，会复用之前的经验...")
    task3 = "任务类型A的变体：另一个排序问题"
    result3 = agent.run(task3, verbose=True)
    
    # 最终统计
    agent.print_statistics()


if __name__ == "__main__":
    demo_reflexion_basic()
    demo_reflexion_with_failure()
    demo_memory_accumulation()
