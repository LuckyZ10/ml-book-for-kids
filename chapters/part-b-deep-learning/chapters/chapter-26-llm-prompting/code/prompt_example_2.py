"""
大语言模型与提示工程 - 动手实践
本章实现完整的提示工程框架，包括：
- PromptTemplate: 提示模板管理
- FewShotPromptBuilder: 少样本提示构建
- ChainOfThought: 思维链实现
- SelfConsistency: 自一致性推理
"""

import json
import re
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import Counter


# ==================== 基础数据类 ====================

@dataclass
class Example:
    """少样本示例类"""
    input: str
    output: str
    reasoning: Optional[str] = None  # 思维链推理过程
    
    def to_string(self, include_reasoning: bool = False) -> str:
        """转换为字符串格式"""
        if include_reasoning and self.reasoning:
            return f"输入：{self.input}\n思考：{self.reasoning}\n输出：{self.output}"
        return f"输入：{self.input}\n输出：{self.output}"


@dataclass
class PromptConfig:
    """提示配置类"""
    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)


# ==================== 模拟LLM接口 ====================

class MockLLM:
    """
    模拟大语言模型接口
    实际使用时，请替换为真实的API调用（如OpenAI、Anthropic等）
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.call_count = 0
        
    def generate(self, prompt: str, config: PromptConfig = None) -> str:
        """
        模拟生成文本
        实际实现中，这里应该调用真实的LLM API
        """
        self.call_count += 1
        config = config or PromptConfig()
        
        # 模拟简单的响应逻辑（仅用于演示）
        return self._simulate_response(prompt, config)
    
    def generate_multiple(self, prompt: str, config: PromptConfig = None, 
                         n: int = 5) -> List[str]:
        """生成多个候选答案"""
        responses = []
        for _ in range(n):
            # 通过调整随机性模拟不同答案
            cfg = PromptConfig(
                temperature=config.temperature if config else 0.8,
                max_tokens=config.max_tokens if config else 512
            )
            responses.append(self.generate(prompt, cfg))
        return responses
    
    def _simulate_response(self, prompt: str, config: PromptConfig) -> str:
        """模拟响应生成（仅用于演示）"""
        # 根据提示内容返回不同的模拟响应
        if "计算" in prompt or "多少" in prompt or "=" in prompt:
            return self._simulate_math_response(prompt)
        elif "分类" in prompt or "情感" in prompt:
            return self._simulate_classification_response(prompt)
        elif "翻译" in prompt:
            return self._simulate_translation_response(prompt)
        else:
            return self._simulate_generic_response(prompt)
    
    def _simulate_math_response(self, prompt: str) -> str:
        """模拟数学问题响应"""
        # 提取数字并模拟计算过程
        numbers = re.findall(r'\d+', prompt)
        
        # 鸡兔同笼问题
        if "鸡" in prompt or "兔" in prompt or "头" in prompt and "脚" in prompt:
            return """让我一步步思考：
设鸡有x只，兔子有y只。
根据题意：
x + y = 35（头的数量）
2x + 4y = 94（脚的数量）

解方程：
从第一个方程：x = 35 - y
代入第二个方程：
2(35 - y) + 4y = 94
70 - 2y + 4y = 94
2y = 24
y = 12

所以兔子12只，鸡23只。
答案：鸡23只，兔子12只。"""
        
        # 简单算术
        if len(numbers) >= 2 and "买了" in prompt:
            return f"""让我一步步计算：
初始有{numbers[0]}个。
购买了{numbers[1]}罐，每罐{numbers[2] if len(numbers) > 2 else '若干'}个。
总共是{numbers[0]} + {numbers[1]} × {numbers[2] if len(numbers) > 2 else '3'} = {int(numbers[0]) + int(numbers[1]) * (int(numbers[2]) if len(numbers) > 2 else 3)}个。
答案是{int(numbers[0]) + int(numbers[1]) * (int(numbers[2]) if len(numbers) > 2 else 3)}。"""
        
        return "答案是42。"  # 默认答案
    
    def _simulate_classification_response(self, prompt: str) -> str:
        """模拟分类任务响应"""
        if "好" in prompt or "棒" in prompt or "精彩" in prompt:
            return "正面"
        elif "差" in prompt or "糟" in prompt or "烂" in prompt or "浪费" in prompt:
            return "负面"
        return "中性"
    
    def _simulate_translation_response(self, prompt: str) -> str:
        """模拟翻译响应"""
        # 简单的中英互译模拟
        if "hello" in prompt.lower():
            return "你好"
        elif "world" in prompt.lower():
            return "世界"
        return "翻译结果"
    
    def _simulate_generic_response(self, prompt: str) -> str:
        """模拟通用响应"""
        return "这是一个模拟的AI响应。在实际应用中，这里会返回真实的LLM输出。"


# ==================== PromptTemplate类 ====================

class PromptTemplate:
    """
    提示模板管理类
    
    功能：
    1. 支持变量插值的模板定义
    2. 支持系统提示词和用户提示词分离
    3. 支持模板验证和预览
    4. 支持少样本示例的动态插入
    
    使用示例：
        template = PromptTemplate(
            system_prompt="你是一个{role}。",
            user_template="请{action}以下内容：\n{content}",
            input_variables=["role", "action", "content"]
        )
        prompt = template.format(role="翻译官", action="翻译", content="Hello")
    """
    
    def __init__(self, 
                 template: str = "",
                 system_prompt: str = "",
                 user_template: str = "",
                 input_variables: List[str] = None,
                 partial_variables: Dict[str, str] = None):
        """
        初始化提示模板
        
        Args:
            template: 完整的模板字符串（如果提供，将覆盖system+user组合）
            system_prompt: 系统提示词（设定AI角色和行为）
            user_template: 用户提示词模板
            input_variables: 需要填充的变量列表
            partial_variables: 预填充的部分变量
        """
        self.template = template
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.input_variables = input_variables or []
        self.partial_variables = partial_variables or {}
        
        # 如果提供了完整模板，解析其中的变量
        if template and not input_variables:
            self.input_variables = self._extract_variables(template)
    
    def _extract_variables(self, text: str) -> List[str]:
        """从模板中提取变量名（格式：{variable_name}）"""
        pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        return list(set(re.findall(pattern, text)))
    
    def format(self, **kwargs) -> str:
        """
        填充模板变量
        
        Args:
            **kwargs: 变量名和值的映射
            
        Returns:
            填充后的完整提示
        """
        # 合并预填充变量和传入变量
        variables = {**self.partial_variables, **kwargs}
        
        # 验证必需变量
        missing = set(self.input_variables) - set(variables.keys())
        if missing:
            raise ValueError(f"缺少必需变量: {missing}")
        
        # 构建完整提示
        if self.template:
            return self.template.format(**variables)
        
        parts = []
        if self.system_prompt:
            system_filled = self.system_prompt.format(**variables)
            parts.append(f"[系统指令]\n{system_filled}")
        
        if self.user_template:
            user_filled = self.user_template.format(**variables)
            parts.append(f"[用户输入]\n{user_filled}")
        
        return "\n\n".join(parts)
    
    def format_chat(self, **kwargs) -> List[Dict[str, str]]:
        """
        格式化为聊天格式的消息列表
        
        Returns:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        variables = {**self.partial_variables, **kwargs}
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt.format(**variables)
            })
        
        if self.user_template:
            messages.append({
                "role": "user", 
                "content": self.user_template.format(**variables)
            })
        elif self.template:
            messages.append({
                "role": "user",
                "content": self.template.format(**variables)
            })
            
        return messages
    
    def partial(self, **kwargs) -> 'PromptTemplate':
        """
        创建部分填充的新模板
        
        使用场景：固定某些变量，创建新的模板实例
        """
        new_partial = {**self.partial_variables, **kwargs}
        remaining_vars = [v for v in self.input_variables if v not in new_partial]
        
        return PromptTemplate(
            template=self.template,
            system_prompt=self.system_prompt,
            user_template=self.user_template,
            input_variables=remaining_vars,
            partial_variables=new_partial
        )
    
    def preview(self) -> str:
        """预览模板结构"""
        lines = ["=== 模板预览 ==="]
        lines.append(f"输入变量: {self.input_variables}")
        lines.append(f"预填充变量: {list(self.partial_variables.keys())}")
        if self.system_prompt:
            lines.append(f"\n系统提示:\n{self.system_prompt}")
        if self.user_template:
            lines.append(f"\n用户模板:\n{self.user_template}")
        if self.template:
            lines.append(f"\n完整模板:\n{self.template}")
        return "\n".join(lines)
    
    @classmethod
    def from_examples(cls, 
                     task_description: str,
                     examples: List[Example],
                     input_variables: List[str],
                     suffix: str = "输入：{input}\n输出：",
                     prefix: str = "",
                     example_separator: str = "\n\n") -> 'PromptTemplate':
        """
        从示例创建Few-shot模板
        
        Args:
            task_description: 任务描述
            examples: 示例列表
            input_variables: 输入变量
            suffix: 查询部分模板
            prefix: 前缀
            example_separator: 示例分隔符
        """
        example_texts = [ex.to_string() for ex in examples]
        example_block = example_separator.join(example_texts)
        
        template_parts = []
        if prefix:
            template_parts.append(prefix)
        if task_description:
            template_parts.append(task_description)
        if examples:
            template_parts.append(example_block)
        template_parts.append(suffix)
        
        template = example_separator.join(template_parts)
        
        return cls(
            template=template,
            input_variables=input_variables
        )


# ==================== FewShotPromptBuilder类 ====================

class FewShotPromptBuilder:
    """
    少样本提示构建器
    
    功能：
    1. 管理和选择示例
    2. 支持多种示例选择策略
    3. 支持动态示例加载
    4. 支持Chain of Thought示例
    
    使用示例：
        builder = FewShotPromptBuilder()
        builder.add_example(Example("猫", "动物"))
        builder.add_example(Example("玫瑰", "植物"))
        prompt = builder.build("太阳", task="分类")
    """
    
    # 示例选择策略
    SEQUENTIAL = "sequential"      # 按顺序选择
    RANDOM = "random"              # 随机选择
    SIMILARITY = "similarity"      # 基于相似度（需要实现）
    DIVERSE = "diverse"            # 多样化选择（需要实现）
    
    def __init__(self, 
                 example_separator: str = "\n\n",
                 prefix: str = "",
                 suffix: str = "输入：{input}\n输出："):
        """
        初始化构建器
        
        Args:
            example_separator: 示例之间的分隔符
            prefix: 提示前缀（任务描述等）
            suffix: 查询模板后缀
        """
        self.examples: List[Example] = []
        self.example_separator = example_separator
        self.prefix = prefix
        self.suffix = suffix
        self.max_length = 2000  # 最大提示长度限制
    
    def add_example(self, example: Example) -> 'FewShotPromptBuilder':
        """添加示例（链式调用支持）"""
        self.examples.append(example)
        return self
    
    def add_examples(self, examples: List[Example]) -> 'FewShotPromptBuilder':
        """批量添加示例"""
        self.examples.extend(examples)
        return self
    
    def set_task_description(self, description: str) -> 'FewShotPromptBuilder':
        """设置任务描述"""
        self.prefix = description
        return self
    
    def select_examples(self, 
                       query: str = "", 
                       n: int = 3,
                       strategy: str = SEQUENTIAL) -> List[Example]:
        """
        选择示例
        
        Args:
            query: 查询内容（用于相似度策略）
            n: 选择的示例数量
            strategy: 选择策略
            
        Returns:
            选中的示例列表
        """
        if not self.examples:
            return []
        
        if strategy == self.SEQUENTIAL:
            return self.examples[-n:] if n < len(self.examples) else self.examples
        
        elif strategy == self.RANDOM:
            n = min(n, len(self.examples))
            return random.sample(self.examples, n)
        
        elif strategy == self.SIMILARITY:
            # 简化实现：基于字符串相似度
            return self._select_by_similarity(query, n)
        
        else:
            return self.examples[:n]
    
    def _select_by_similarity(self, query: str, n: int) -> List[Example]:
        """基于简单词重叠的相似度选择"""
        query_words = set(query.lower().split())
        
        def similarity(ex: Example) -> float:
            ex_words = set(ex.input.lower().split())
            if not ex_words:
                return 0.0
            return len(query_words & ex_words) / len(query_words | ex_words)
        
        sorted_examples = sorted(self.examples, key=similarity, reverse=True)
        return sorted_examples[:n]
    
    def build(self, 
             query_input: str,
             include_reasoning: bool = False,
             n_examples: int = 3,
             strategy: str = SEQUENTIAL) -> str:
        """
        构建少样本提示
        
        Args:
            query_input: 当前查询输入
            include_reasoning: 是否包含推理过程
            n_examples: 使用的示例数量
            strategy: 示例选择策略
            
        Returns:
            完整的提示字符串
        """
        # 选择示例
        selected = self.select_examples(query_input, n_examples, strategy)
        
        # 构建示例部分
        example_parts = []
        for ex in selected:
            example_parts.append(ex.to_string(include_reasoning))
        
        # 构建查询部分
        query_part = self.suffix.format(input=query_input)
        
        # 组合所有部分
        parts = []
        if self.prefix:
            parts.append(self.prefix)
        parts.extend(example_parts)
        parts.append(query_part)
        
        return self.example_separator.join(parts)
    
    def build_chat_messages(self,
                           query_input: str,
                           system_prompt: str = "",
                           include_reasoning: bool = False,
                           n_examples: int = 3) -> List[Dict[str, str]]:
        """
        构建聊天格式的消息
        
        Returns:
            符合OpenAI格式的消息列表
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if self.prefix:
            messages.append({"role": "system", "content": self.prefix})
        
        # 添加示例作为few-shot
        selected = self.select_examples(query_input, n_examples)
        for ex in selected:
            messages.append({"role": "user", "content": f"输入：{ex.input}"})
            content = ex.output
            if include_reasoning and ex.reasoning:
                content = f"思考过程：{ex.reasoning}\n输出：{ex.output}"
            messages.append({"role": "assistant", "content": content})
        
        # 添加当前查询
        messages.append({"role": "user", "content": f"输入：{query_input}"})
        
        return messages
    
    def to_template(self) -> PromptTemplate:
        """转换为PromptTemplate对象"""
        example_texts = [ex.to_string() for ex in self.examples]
        example_block = self.example_separator.join(example_texts)
        
        parts = []
        if self.prefix:
            parts.append(self.prefix)
        if self.examples:
            parts.append(example_block)
        parts.append(self.suffix)
        
        template = self.example_separator.join(parts)
        
        return PromptTemplate(
            template=template,
            input_variables=["input"]
        )
    
    def save_examples(self, filepath: str):
        """保存示例到JSON文件"""
        data = [
            {"input": ex.input, "output": ex.output, "reasoning": ex.reasoning}
            for ex in self.examples
        ]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_examples(self, filepath: str):
        """从JSON文件加载示例"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.examples = [
            Example(d["input"], d["output"], d.get("reasoning"))
            for d in data
        ]
        return self


# ==================== ChainOfThought类 ====================

class ChainOfThought:
    """
    思维链推理实现
    
    实现了Wei等人(2022)论文中的Chain of Thought技术，
    通过让模型展示中间推理步骤来解决复杂问题。
    
    支持两种模式：
    1. Few-shot CoT: 提供带推理过程的示例
    2. Zero-shot CoT: 通过触发词引导模型生成推理
    
    使用示例：
        cot = ChainOfThought(llm)
        result = cot.solve("罗杰有5个网球，又买了2罐...")
    """
    
    # 触发词（来自Kojima等人论文）
    DEFAULT_TRIGGER = "让我们一步步思考。"
    ENGLISH_TRIGGER = "Let's think step by step."
    
    def __init__(self, 
                 llm: MockLLM,
                 trigger: str = None,
                 example_pool: List[Example] = None):
        """
        初始化思维链推理器
        
        Args:
            llm: 大语言模型实例
            trigger: Zero-shot CoT的触发词
            example_pool: Few-shot CoT的示例池
        """
        self.llm = llm
        self.trigger = trigger or self.DEFAULT_TRIGGER
        self.example_pool = example_pool or []
        self.builder = FewShotPromptBuilder()
        
        # 初始化默认的CoT示例（数学问题）
        if not self.example_pool:
            self._init_default_examples()
    
    def _init_default_examples(self):
        """初始化默认的思维链示例"""
        self.example_pool = [
            Example(
                input="罗杰有5个网球，他又买了2罐，每罐有3个网球。他现在有多少个网球？",
                output="11",
                reasoning="""罗杰一开始有5个网球。
他买了2罐，每罐3个，所以是2 × 3 = 6个网球。
总共是5 + 6 = 11个网球。"""
            ),
            Example(
                input="食堂有23个苹果，用了20个做午餐，又买了6个。现在有多少个苹果？",
                output="9",
                reasoning="""食堂原来有23个苹果。
用了20个，剩下23 - 20 = 3个。
又买了6个，3 + 6 = 9个。"""
            ),
            Example(
                input="一个农场有若干鸡和兔子，共有35个头和94只脚。问鸡和兔子各有多少只？",
                output="鸡23只，兔子12只",
                reasoning="""设鸡有x只，兔子有y只。
根据头的数量：x + y = 35
根据脚的数量：2x + 4y = 94

从第一个方程：x = 35 - y
代入第二个方程：
2(35 - y) + 4y = 94
70 - 2y + 4y = 94
2y = 24
y = 12

所以兔子有12只，鸡有35 - 12 = 23只。"""
            )
        ]
        self.builder.add_examples(self.example_pool)
    
    def solve(self, 
             problem: str,
             mode: str = "zero-shot",
             n_examples: int = 2) -> Dict[str, Any]:
        """
        使用思维链解决问题
        
        Args:
            problem: 待解决的问题
            mode: "zero-shot" 或 "few-shot"
            n_examples: Few-shot模式下使用的示例数
            
        Returns:
            包含推理过程和最终答案的字典
        """
        if mode == "zero-shot":
            return self._zero_shot_cot(problem)
        elif mode == "few-shot":
            return self._few_shot_cot(problem, n_examples)
        else:
            raise ValueError(f"不支持的模式: {mode}")
    
    def _zero_shot_cot(self, problem: str) -> Dict[str, Any]:
        """
        Zero-shot Chain of Thought
        在问题后添加触发词，引导模型生成推理
        """
        prompt = f"{problem}\n\n{self.trigger}"
        
        config = PromptConfig(temperature=0.3, max_tokens=512)
        response = self.llm.generate(prompt, config)
        
        # 解析响应，提取推理和答案
        reasoning, answer = self._parse_response(response)
        
        return {
            "mode": "zero-shot-cot",
            "problem": problem,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "raw_response": response
        }
    
    def _few_shot_cot(self, problem: str, n_examples: int) -> Dict[str, Any]:
        """
        Few-shot Chain of Thought
        提供带推理过程的示例，让模型模仿
        """
        # 构建Few-shot提示
        prompt = self.builder.build(
            query_input=problem,
            include_reasoning=True,
            n_examples=n_examples,
            strategy=FewShotPromptBuilder.SEQUENTIAL
        )
        
        config = PromptConfig(temperature=0.3, max_tokens=512)
        response = self.llm.generate(prompt, config)
        
        # 解析响应
        reasoning, answer = self._parse_response(response)
        
        return {
            "mode": "few-shot-cot",
            "problem": problem,
            "prompt": prompt,
            "reasoning": reasoning,
            "answer": answer,
            "raw_response": response
        }
    
    def _parse_response(self, response: str) -> tuple:
        """
        解析模型响应，提取推理过程和答案
        
        Returns:
            (reasoning, answer)
        """
        lines = response.strip().split('\n')
        
        # 尝试找到答案行
        answer = response
        reasoning = response
        
        for line in reversed(lines):
            if "答案" in line or "answer" in line.lower() or "结果是" in line:
                answer = line.split("：")[-1] if "：" in line else line
                reasoning = '\n'.join(lines[:lines.index(line)])
                break
        
        return reasoning.strip(), answer.strip()
    
    def add_custom_example(self, problem: str, reasoning: str, answer: str):
        """添加自定义示例"""
        example = Example(
            input=problem,
            output=answer,
            reasoning=reasoning
        )
        self.example_pool.append(example)
        self.builder.add_example(example)
        return self
    
    def evaluate(self, test_cases: List[Dict[str, str]], mode: str = "few-shot") -> Dict:
        """
        在测试集上评估思维链效果
        
        Args:
            test_cases: 测试用例列表，每个包含problem和expected_answer
            mode: 使用的CoT模式
            
        Returns:
            评估结果统计
        """
        results = []
        correct = 0
        
        for case in test_cases:
            result = self.solve(case["problem"], mode=mode)
            is_correct = result["answer"] == case["expected_answer"]
            if is_correct:
                correct += 1
            
            results.append({
                **result,
                "expected": case["expected_answer"],
                "correct": is_correct
            })
        
        accuracy = correct / len(test_cases) if test_cases else 0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_cases),
            "results": results
        }


# ==================== SelfConsistency类 ====================

class SelfConsistency:
    """
    自一致性推理实现
    
    实现了Wang等人(2022)论文中的Self-Consistency技术：
    1. 对同一个问题采样多个思维链
    2. 提取每个链的最终答案
    3. 通过投票选择最一致的答案
    
    这就像考试时的"多检查几遍"，通过多次独立推理来提高准确性。
    
    使用示例：
        sc = SelfConsistency(llm, cot)
        result = sc.solve_with_voting("15 - 4 × 2 = ?", n_paths=5)
    """
    
    def __init__(self, 
                 llm: MockLLM,
                 cot: ChainOfThought = None,
                 n_paths: int = 5,
                 temperature: float = 0.7):
        """
        初始化自一致性推理器
        
        Args:
            llm: 大语言模型实例
            cot: 思维链实例（可选，用于生成推理路径）
            n_paths: 默认采样的推理路径数
            temperature: 采样的温度参数（越高多样性越大）
        """
        self.llm = llm
        self.cot = cot or ChainOfThought(llm)
        self.n_paths = n_paths
        self.temperature = temperature
    
    def solve_with_voting(self,
                         problem: str,
                         n_paths: int = None,
                         extract_answer_fn: Callable = None) -> Dict[str, Any]:
        """
        使用自一致性投票解决问题
        
        Args:
            problem: 待解决的问题
            n_paths: 推理路径数（覆盖默认值）
            extract_answer_fn: 自定义答案提取函数
            
        Returns:
            包含投票结果的字典
        """
        n_paths = n_paths or self.n_paths
        
        # 生成多个推理路径
        paths = self._generate_diverse_paths(problem, n_paths)
        
        # 提取每个路径的答案
        answers = []
        for path in paths:
            if extract_answer_fn:
                ans = extract_answer_fn(path["answer"])
            else:
                ans = self._extract_final_answer(path["answer"])
            answers.append(ans)
        
        # 投票
        vote_results = self._vote(answers)
        
        # 选择最一致的答案
        best_answer = vote_results.most_common(1)[0][0]
        confidence = vote_results[best_answer] / len(answers)
        
        return {
            "problem": problem,
            "best_answer": best_answer,
            "confidence": confidence,
            "vote_distribution": dict(vote_results),
            "all_paths": paths,
            "all_answers": answers,
            "n_paths": n_paths
        }
    
    def _generate_diverse_paths(self, problem: str, n: int) -> List[Dict]:
        """生成多样化的推理路径"""
        paths = []
        
        for i in range(n):
            # 使用较高温度增加多样性
            config = PromptConfig(
                temperature=self.temperature + (i * 0.1),  # 递增温度
                max_tokens=512
            )
            
            # 调用CoT生成推理
            result = self.cot.solve(problem, mode="zero-shot")
            paths.append(result)
        
        return paths
    
    def _extract_final_answer(self, response: str) -> str:
        """
        从响应中提取最终答案
        
        尝试多种常见的答案格式
        """
        # 模式1: "答案是 X" 或 "答案：X"
        patterns = [
            r'答案[是为:]+\s*([^\n。]+)',
            r'答案[:：]\s*([^\n]+)',
            r'结果[是为:]+\s*([^\n。]+)',
            r'最终答案[:：]\s*([^\n]+)',
            r'[Tt]he answer is[:\s]+([^\n.]+)',
            r'[=＝]\s*([^\n]+)',  # 等号后的内容
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        
        # 如果没有匹配到，返回最后一行
        lines = [l.strip() for l in response.strip().split('\n') if l.strip()]
        return lines[-1] if lines else response
    
    def _vote(self, answers: List[str]) -> Counter:
        """对答案进行投票"""
        # 规范化答案（去除空格、标点等）
        normalized = []
        for ans in answers:
            norm = ans.lower().strip().rstrip('。').rstrip('.')
            normalized.append(norm)
        
        return Counter(normalized)
    
    def solve_with_verification(self,
                                problem: str,
                                n_paths: int = None) -> Dict[str, Any]:
        """
        带验证的自一致性推理
        
        不仅投票，还会检查推理过程的合理性
        """
        base_result = self.solve_with_voting(problem, n_paths)
        
        # 验证每个推理路径
        verified_paths = []
        for path in base_result["all_paths"]:
            verification = self._verify_reasoning(path["reasoning"])
            path["verification"] = verification
            verified_paths.append(path)
        
        # 只考虑通过验证的路径
        valid_paths = [p for p in verified_paths if p["verification"]["valid"]]
        
        if valid_paths:
            # 从有效路径中重新投票
            valid_answers = [self._extract_final_answer(p["answer"]) for p in valid_paths]
            vote_results = self._vote(valid_answers)
            best_answer = vote_results.most_common(1)[0][0]
        else:
            # 如果没有路径通过验证，回退到原始投票结果
            best_answer = base_result["best_answer"]
            valid_paths = verified_paths
        
        return {
            **base_result,
            "best_answer": best_answer,
            "verified_paths": verified_paths,
            "valid_paths_count": len(valid_paths)
        }
    
    def _verify_reasoning(self, reasoning: str) -> Dict:
        """
        验证推理过程的合理性
        
        简单的启发式验证，实际应用中可以使用更复杂的逻辑
        """
        checks = {
            "has_steps": len(reasoning.split('\n')) > 1,
            "has_numbers": bool(re.search(r'\d+', reasoning)),
            "reasonable_length": 10 < len(reasoning) < 1000,
        }
        
        valid = all(checks.values())
        
        return {
            "valid": valid,
            "checks": checks
        }


# ==================== LeastToMost类 ====================

class LeastToMost:
    """
    从简到繁提示技术实现
    
    实现了Zhou等人(2022)论文中的Least-to-Most Prompting技术：
    将复杂问题分解为一系列简单子问题，逐步解决。
    
    使用示例：
        l2m = LeastToMost(llm)
        result = l2m.solve("Amy5岁时身高3英尺，每年长高是前一年的1/3，10岁时多高？")
    """
    
    def __init__(self, llm: MockLLM):
        self.llm = llm
        self.decomposition_template = PromptTemplate(
            system_prompt="你是一个擅长将复杂问题分解的专家。",
            user_template="""请将以下复杂问题分解为一系列简单子问题，从最简单到最复杂：

问题：{problem}

要求：
1. 每个子问题应该可以独立回答
2. 后一个子问题可以基于前一个的答案
3. 只列出子问题，不要回答

子问题列表：""",
            input_variables=["problem"]
        )
    
    def solve(self, problem: str) -> Dict[str, Any]:
        """
        使用Least-to-Most策略解决问题
        
        步骤：
        1. 将问题分解为子问题
        2. 依次解决每个子问题
        3. 组合答案
        """
        # 第一步：分解问题
        subproblems = self._decompose(problem)
        
        # 第二步：逐步解决
        solutions = []
        context = ""
        
        for i, sub in enumerate(subproblems):
            # 构建带上下文的提示
            if context:
                prompt = f"基于之前的信息：{context}\n\n现在回答：{sub}"
            else:
                prompt = sub
            
            config = PromptConfig(temperature=0.3, max_tokens=256)
            answer = self.llm.generate(prompt, config)
            
            solutions.append({
                "subproblem": sub,
                "answer": answer,
                "step": i + 1
            })
            
            # 更新上下文
            context += f"\n问题{i+1}：{sub}\n答案：{answer}"
        
        # 第三步：综合答案
        final_answer = solutions[-1]["answer"] if solutions else ""
        
        return {
            "problem": problem,
            "subproblems": subproblems,
            "solutions": solutions,
            "final_answer": final_answer
        }
    
    def _decompose(self, problem: str) -> List[str]:
        """将问题分解为子问题"""
        prompt = self.decomposition_template.format(problem=problem)
        config = PromptConfig(temperature=0.5, max_tokens=256)
        
        response = self.llm.generate(prompt, config)
        
        # 解析子问题（假设每行一个，或以数字开头）
        subproblems = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line:
                # 去除序号前缀
                cleaned = re.sub(r'^[\d\s.、]+', '', line).strip()
                if cleaned:
                    subproblems.append(cleaned)
        
        return subproblems if subproblems else [problem]


# ==================== 完整演示 ====================

def demo_prompt_template():
    """演示PromptTemplate的使用"""
    print("=" * 60)
    print("演示1: PromptTemplate - 提示模板管理")
    print("=" * 60)
    
    # 创建基础模板
    template = PromptTemplate(
        system_prompt="你是一位专业的{role}。",
        user_template="请{action}以下内容：\n\n{content}",
        input_variables=["role", "action", "content"]
    )
    
    print("\n1. 基础模板：")
    print(template.preview())
    
    # 填充模板
    prompt = template.format(
        role="翻译官",
        action="将以下英文翻译成中文",
        content="Hello, how are you?"
    )
    print("\n2. 填充后的提示：")
    print(prompt)
    
    # 部分填充
    partial_template = template.partial(role="编辑")
    print("\n3. 部分填充后的模板：")
    print(partial_template.preview())
    
    # 聊天格式
    messages = template.format_chat(
        role="程序员",
        action="解释",
        content="什么是递归函数？"
    )
    print("\n4. 聊天格式：")
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:50]}...")
    
    # 从示例创建模板
    examples = [
        Example("猫", "动物"),
        Example("玫瑰", "植物"),
        Example("汽车", "交通工具")
    ]
    
    fs_template = PromptTemplate.from_examples(
        task_description="请将物品分类到正确的类别。",
        examples=examples,
        input_variables=["input"],
        suffix="输入：{input}\n输出："
    )
    
    print("\n5. Few-shot模板：")
    print(fs_template.format(input="飞机"))


def demo_few_shot_builder():
    """演示FewShotPromptBuilder的使用"""
    print("\n" + "=" * 60)
    print("演示2: FewShotPromptBuilder - 少样本提示构建")
    print("=" * 60)
    
    # 创建构建器
    builder = FewShotPromptBuilder(
        prefix="将电影评论分类为正面、负面或中性。",
        suffix="评论：{input}\n情感："
    )
    
    # 添加示例
    builder.add_examples([
        Example("这部电影真是太精彩了！", "正面"),
        Example("完全浪费时间的烂片。", "负面"),
        Example("一般般，没什么特别的。", "中性"),
        Example("演员演技出色，剧情紧凑。", "正面"),
        Example("剧情漏洞百出，无法直视。", "负面")
    ])
    
    # 构建提示
    prompt = builder.build("视觉效果很棒，但故事情节有点老套。")
    print("\n1. 构建的Few-shot提示：")
    print(prompt)
    
    # 不同选择策略
    print("\n2. 随机选择示例：")
    prompt_random = builder.build(
        "测试",
        n_examples=2,
        strategy=FewShotPromptBuilder.RANDOM
    )
    print(prompt_random)
    
    # 带推理的示例
    cot_builder = FewShotPromptBuilder()
    cot_builder.add_examples([
        Example(
            input="5 + 3 × 2 = ?",
            output="11",
            reasoning="先算乘法：3 × 2 = 6，再算加法：5 + 6 = 11"
        ),
        Example(
            input="10 - 2 × 4 = ?",
            output="2",
            reasoning="先算乘法：2 × 4 = 8，再算减法：10 - 8 = 2"
        )
    ])
    
    print("\n3. Chain of Thought示例：")
    cot_prompt = cot_builder.build("8 + 4 × 3 = ?", include_reasoning=True)
    print(cot_prompt)


def demo_chain_of_thought():
    """演示ChainOfThought的使用"""
    print("\n" + "=" * 60)
    print("演示3: ChainOfThought - 思维链推理")
    print("=" * 60)
    
    llm = MockLLM()
    cot = ChainOfThought(llm)
    
    # Zero-shot CoT
    print("\n1. Zero-shot Chain of Thought:")
    problem1 = "一个农场有若干鸡和兔子，共有35个头和94只脚。问鸡和兔子各有多少只？"
    result1 = cot.solve(problem1, mode="zero-shot")
    print(f"问题：{result1['problem']}")
    print(f"推理过程：\n{result1['reasoning']}")
    print(f"答案：{result1['answer']}")
    
    # Few-shot CoT
    print("\n2. Few-shot Chain of Thought:")
    problem2 = "罗杰有5个网球，他又买了2罐，每罐有3个网球。他现在有多少个网球？"
    result2 = cot.solve(problem2, mode="few-shot", n_examples=2)
    print(f"问题：{result2['problem']}")
    print(f"答案：{result2['answer']}")
    
    # 评估
    print("\n3. 批量评估：")
    test_cases = [
        {"problem": "5 + 3 = ?", "expected_answer": "8"},
        {"problem": "10 - 4 = ?", "expected_answer": "6"},
    ]
    eval_result = cot.evaluate(test_cases, mode="zero-shot")
    print(f"准确率：{eval_result['accuracy']:.2%}")
    print(f"正确数：{eval_result['correct']}/{eval_result['total']}")


def demo_self_consistency():
    """演示SelfConsistency的使用"""
    print("\n" + "=" * 60)
    print("演示4: SelfConsistency - 自一致性推理")
    print("=" * 60)
    
    llm = MockLLM()
    cot = ChainOfThought(llm)
    sc = SelfConsistency(llm, cot, n_paths=5)
    
    # 自一致性推理
    print("\n1. 自一致性投票：")
    problem = "15 - 4 × 2 = ?"
    result = sc.solve_with_voting(problem, n_paths=3)
    
    print(f"问题：{result['problem']}")
    print(f"最佳答案：{result['best_answer']}")
    print(f"置信度：{result['confidence']:.2%}")
    print(f"投票分布：{result['vote_distribution']}")
    
    print("\n2. 所有推理路径：")
    for i, path in enumerate(result['all_paths']):
        print(f"\n路径 {i+1}:")
        print(f"  推理：{path['reasoning'][:80]}...")
        print(f"  答案：{path['answer']}")


def demo_least_to_most():
    """演示LeastToMost的使用"""
    print("\n" + "=" * 60)
    print("演示5: LeastToMost - 从简到繁")
    print("=" * 60)
    
    llm = MockLLM()
    l2m = LeastToMost(llm)
    
    problem = "Amy在5岁时身高是3英尺。之后每年长高是前一年长高的1/3。现在她10岁，身高多少？"
    result = l2m.solve(problem)
    
    print(f"原始问题：{result['problem']}")
    print(f"\n分解的子问题：")
    for i, sub in enumerate(result['subproblems'], 1):
        print(f"  {i}. {sub}")
    
    print(f"\n逐步解答：")
    for sol in result['solutions']:
        print(f"  步骤{sol['step']}: {sol['subproblem']}")
        print(f"    → {sol['answer']}")
    
    print(f"\n最终答案：{result['final_answer']}")


def demo_complete_pipeline():
    """演示完整的提示工程流程"""
    print("\n" + "=" * 60)
    print("演示6: 完整流程 - 数学问题求解")
    print("=" * 60)
    
    llm = MockLLM()
    
    # 问题
    problem = "一个水箱可以装100升水。现在以每分钟5升的速度进水，
              "同时以每分钟3升的速度出水。问多长时间能装满水箱？"
    
    print(f"问题：{problem}")
    print("\n" + "-" * 40)
    
    # 方法1：直接提问（Zero-shot）
    print("\n方法1：Zero-shot")
    cot = ChainOfThought(llm)
    result1 = cot.solve(problem, mode="zero-shot")
    print(f"答案：{result1['answer']}")
    
    # 方法2：Few-shot CoT
    print("\n方法2：Few-shot Chain of Thought")
    result2 = cot.solve(problem, mode="few-shot")
    print(f"答案：{result2['answer']}")
    
    # 方法3：Self-Consistency
    print("\n方法3：Self-Consistency")
    sc = SelfConsistency(llm, cot)
    result3 = sc.solve_with_voting(problem, n_paths=3)
    print(f"答案：{result3['best_answer']} (置信度: {result3['confidence']:.2%})")
    
    # 方法4：Least-to-Most
    print("\n方法4：Least-to-Most")
    l2m = LeastToMost(llm)
    result4 = l2m.solve(problem)
    print(f"答案：{result4['final_answer']}")


def main():
    """主函数：运行所有演示"""
    print("\n" + "=" * 70)
    print(" " * 15 + "大语言模型与提示工程 - 动手实践")
    print("=" * 70)
    
    # 运行各个演示
    demo_prompt_template()
    demo_few_shot_builder()
    demo_chain_of_thought()
    demo_self_consistency()
    demo_least_to_most()
    demo_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("所有演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()


# ==================== 额外的实用工具函数 ====================

def extract_json_from_response(response: str) -> Optional[Dict]:
    """从模型响应中提取JSON"""
    # 尝试直接解析
    try:
        return json.loads(response)
    except:
        pass
    
    # 尝试从代码块中提取
    json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # 尝试从花括号中提取
    brace_match = re.search(r'\{.*\}', response, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except:
            pass
    
    return None


def create_classification_prompt(classes: List[str], text: str) -> str:
    """创建分类任务的提示"""
    class_list = "、".join(classes)
    return f"""请将以下文本分类到其中一个类别：{class_list}

文本："{text}"

类别："""


def create_extraction_prompt(entity_types: List[str], text: str) -> str:
    """创建实体提取任务的提示"""
    types_str = "、".join(entity_types)
    return f"""请从以下文本中提取所有{types_str}类型的实体。
以JSON格式输出：{{"实体类型": ["实体1", "实体2", ...]}}

文本："{text}"

提取结果："""


def create_summary_prompt(text: str, max_words: int = 100) -> str:
    """创建摘要任务的提示"""
    return f"""请用不超过{max_words}个字总结以下文本的主要内容。

文本：
{text}

摘要："""


def create_qa_prompt(context: str, question: str) -> str:
    """创建问答任务的提示"""
    return f"""基于以下上下文回答问题。如果上下文中没有答案，请说"无法找到答案"。

上下文：
{context}

问题：{question}

答案："""


# ==================== 练习题数据 ====================

PRACTICE_MATH_PROBLEMS = [
    {
        "id": 1,
        "difficulty": "基础",
        "problem": "小明有15元钱，买了3支铅笔，每支2元。他还剩多少钱？",
        "answer": "9元",
        "hint": "先计算花掉的钱，再用总数减去。"
    },
    {
        "id": 2,
        "difficulty": "进阶",
        "problem": "一个水池有两个进水管，A管单独注满需6小时，B管单独注满需4小时。两管同时开，多久能注满？",
        "answer": "2.4小时",
        "hint": "计算每小时的注水效率，然后求和。"
    },
    {
        "id": 3,
        "difficulty": "挑战",
        "problem": "一个三位数，各位数字之和为15，百位数字比十位数字大5，个位数字是十位数字的3倍。这个数是多少？",
        "answer": "726",
        "hint": "设十位数字为x，用x表示其他位。"
    }
]

PRACTICE_CLASSIFICATION_DATA = [
    {
        "text": "这个手机的电池续航太棒了，一整天都不用充电！",
        "label": "正面"
    },
    {
        "text": "物流太慢了，等了一个星期才到。",
        "label": "负面"
    },
    {
        "text": "产品符合描述，没有惊喜也没有失望。",
        "label": "中性"
    }
]


# ==================== 单元测试 ====================

def run_tests():
    """运行简单的单元测试"""
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # 测试1: PromptTemplate
    try:
        template = PromptTemplate(
            template="Hello, {name}!",
            input_variables=["name"]
        )
        result = template.format(name="World")
        assert result == "Hello, World!"
        print("✓ PromptTemplate测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ PromptTemplate测试失败: {e}")
        tests_failed += 1
    
    # 测试2: Example类
    try:
        ex = Example("input", "output", "reasoning")
        assert ex.input == "input"
        assert ex.output == "output"
        assert ex.reasoning == "reasoning"
        print("✓ Example类测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Example类测试失败: {e}")
        tests_failed += 1
    
    # 测试3: FewShotPromptBuilder
    try:
        builder = FewShotPromptBuilder()
        builder.add_example(Example("A", "B"))
        prompt = builder.build("C")
        assert "A" in prompt and "B" in prompt and "C" in prompt
        print("✓ FewShotPromptBuilder测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ FewShotPromptBuilder测试失败: {e}")
        tests_failed += 1
    
    # 测试4: SelfConsistency投票
    try:
        sc = SelfConsistency(MockLLM())
        answers = ["7", "7", "8", "7", "9"]
        vote_result = sc._vote(answers)
        assert vote_result.most_common(1)[0][0] == "7"
        print("✓ SelfConsistency投票测试通过")
        tests_passed += 1
    except Exception as e:
        print(f"✗ SelfConsistency投票测试失败: {e}")
        tests_failed += 1
    
    print(f"\n测试结果：通过 {tests_passed}，失败 {tests_failed}")
    return tests_failed == 0


if __name__ == "__main__":
    # 如果直接运行此文件，执行主程序
    main()
    # 运行单元测试
    run_tests()