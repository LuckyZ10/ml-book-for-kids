#!/usr/bin/env python3
"""
LLM提示工程工具集
Chapter 26: 大语言模型与提示工程
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """提示模板类"""
    template: str
    variables: List[str]
    
    def format(self, **kwargs) -> str:
        """填充模板变量"""
        return self.template.format(**kwargs)


class FewShotPromptBuilder:
    """Few-shot提示构建器"""
    
    def __init__(self):
        self.examples = []
        self.prefix = ""
        self.suffix = ""
    
    def add_example(self, input_text: str, output_text: str):
        """添加示例"""
        self.examples.append({"input": input_text, "output": output_text})
        return self
    
    def set_prefix(self, text: str):
        """设置前缀说明"""
        self.prefix = text
        return self
    
    def set_suffix(self, text: str):
        """设置后缀提示"""
        self.suffix = text
        return self
    
    def build(self, test_input: str) -> str:
        """构建完整提示"""
        lines = [self.prefix] if self.prefix else []
        
        for ex in self.examples:
            lines.extend([
                f"输入: {ex['input']}",
                f"输出: {ex['output']}",
                ""
            ])
        
        lines.extend([
            f"输入: {test_input}",
            "输出: " if not self.suffix else self.suffix
        ])
        
        return "\n".join(lines)


class ChainOfThoughtPrompt:
    """思维链提示生成器"""
    
    ZERO_SHOT_COT = "让我们一步步思考。"
    
    @staticmethod
    def create_zero_shot(question: str) -> str:
        """创建Zero-shot CoT提示""
        return f"""{question}

{ChainOfThoughtPrompt.ZERO_SHOT_COT}"""
    
    @staticmethod
    def create_few_shot(question: str, examples: List[Dict[str, str]]) -> str:
        """创建Few-shot CoT提示""
        prompt_parts = []
        
        for ex in examples:
            prompt_parts.extend([
                f"Q: {ex['question']}",
                f"A: {ex['reasoning']}",
                f"答案: {ex['answer']}",
                ""
            ])
        
        prompt_parts.extend([
            f"Q: {question}",
            "A: "
        ])
        
        return "\n".join(prompt_parts)


class SelfConsistencyDecoder:
    """自一致性解码器""
    
    def __init__(self, num_samples: int = 5, temperature: float = 0.7):
        self.num_samples = num_samples
        self.temperature = temperature
    
    def select_best_answer(self, answers: List[str]) -> str:
        """通过投票选择最一致的答案""
        from collections import Counter
        
        # 标准化答案（去除空格、转为小写等）
        normalized = [ans.strip().lower() for ans in answers]
        
        # 统计频率
        counter = Counter(normalized)
        most_common = counter.most_common(1)[0][0]
        
        # 返回原始格式的最常见答案
        for ans in answers:
            if ans.strip().lower() == most_common:
                return ans
        
        return answers[0]


class RolePrompt:
    """角色提示生成器""
    
    TEMPLATES = {
        'teacher': """请你扮演一位{level}的老师，用{style}的风格解释{topic}。
要求：
1. 使用类比和比喻帮助理解
2. 分步骤解释
3. 检查学生的理解程度",
        
        'expert': """你是{field}领域的专家，拥有20年从业经验。
请用专业但易懂的方式解答以下问题：{question}
要求提供具体的案例和数据支持。",
        
        'beginner': """假设你是一个刚接触{topic}的初学者，
请用简单直白的语言解释这个概念，避免使用专业术语。"
    }
    
    @classmethod
    def create(cls, role_type: str, **kwargs) -> str:
        """创建角色提示""
        template = cls.TEMPLATES.get(role_type, cls.TEMPLATES['teacher'])
        return template.format(**kwargs)


class PromptInjectionDetector:
    """提示注入检测器""
    
    DANGEROUS_KEYWORDS = [
        '忽略', 'ignore', '跳过', 'skip',
        '系统提示', 'system prompt', '指令', 'instruction',
        '忘记', 'forget', 'Reveal', 'previous'
    ]
    
    def __init__(self):
        self.blocked_patterns = self.DANGEROUS_KEYWORDS
    
    def check(self, user_input: str) -> Dict[str, Any]:
        """检查输入是否可能包含注入攻击""
        user_lower = user_input.lower()
        
        detected = []
        for keyword in self.blocked_patterns:
            if keyword.lower() in user_lower:
                detected.append(keyword)
        
        return {
            'is_safe': len(detected) == 0,
            'detected_keywords': detected,
            'risk_level': 'high' if len(detected) > 2 else ('medium' if len(detected) > 0 else 'low')
        }
    
    def sanitize(self, user_input: str) -> str:
        """清理潜在的危险输入""
        # 在实际应用中，这里会有更复杂的清理逻辑
        return user_input


class ReActPrompt:
    """ReAct (Reasoning + Acting) 提示生成器""
    
    TEMPLATE = """你可以使用以下工具：
{tools_description}

请使用以下格式回答问题：
思考1: [你的思考过程]
行动1: [工具名称[参数]]
观察1: [工具返回结果]
...
思考N: [最终思考]
答案: [最终答案]

问题: {question}
"""
    
    def __init__(self, tools: Dict[str, str]):
        """
        参数:
            tools: 工具名称到描述的映射
        """
        self.tools = tools
    
    def create_prompt(self, question: str) -> str:
        """创建ReAct提示""
        tools_desc = "\n".join([f"- {name}: {desc}" for name, desc in self.tools.items()])
        return self.TEMPLATE.format(
            tools_description=tools_desc,
            question=question
        )


# 使用示例
if __name__ == '__main__':
    # Few-shot示例
    builder = FewShotPromptBuilder()
    builder.set_prefix("请将以下英文翻译成中文：")           .add_example("Hello", "你好")           .add_example("Good morning", "早上好")
    
    prompt = builder.build("How are you?")
    print("Few-shot Prompt:")
    print(prompt)
    print()
    
    # Chain-of-Thought示例
    cot_prompt = ChainOfThoughtPrompt.create_zero_shot(
        "一个水箱有100升水，先用掉1/4，再用掉剩余的一半，还剩多少？"
    )
    print("Chain-of-Thought Prompt:")
    print(cot_prompt)
    print()
    
    # 角色提示示例
    role_prompt = RolePrompt.create(
        'teacher',
        level='小学',
        style='生动有趣',
        topic='神经网络'
    )
    print("Role Prompt:")
    print(role_prompt)
