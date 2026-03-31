# 第二十六章 大语言模型与提示工程

> **本章导读**：想象一下，你有一个聪明绝顶的朋友，他读过互联网上几乎所有的书籍、文章和对话。你问他任何问题，他都能给你一个答案——但答案的质量，往往取决于你如何提问。这就是大语言模型的魔法，也是提示工程的艺术。

---

## 26.1 费曼的考试秘诀

### 26.1.1 天才的困惑

想象一下，你正在准备一场突如其来的数学考试。考场门口，你发现三种不同的同学：

**第一种同学——小红**：她走进考场，什么都没准备，完全靠平时积累的知识答题。这叫做**裸考**。

**第二种同学——小明**：他在考场外快速看了几道例题，记住了解题模式，然后带着这些"模板"走进考场。这叫做**临时抱佛脚**。

**第三种同学——小华**：他不仅看了例题，还在草稿纸上写下每一步的思考过程："首先，我需要找出已知条件...然后，我应该使用什么公式...让我验证一下结果是否合理..."这叫做**写出演算过程**。

理查德·费曼曾经分享过他的学习秘诀："如果你不能简单地解释它，你就还没有真正理解它。"他发现，把思考过程写下来，不仅能帮助别人理解，更能帮助自己理清思路。

### 26.1.2 考试比喻与AI的对应

让我们把这个考试场景映射到大语言模型上：

| 考试场景 | AI术语 | 含义 |
|---------|--------|------|
| 裸考 | Zero-shot | 不给任何示例，直接提问 |
| 看一道例题 | One-shot | 提供一个示例 |
| 看几道例题 | Few-shot | 提供多个示例 |
| 写出演算过程 | Chain of Thought | 让模型展示推理步骤 |
| 多检查几遍 | Self-Consistency | 多次采样选择最一致答案 |
| 先易后难 | Least-to-Most | 将复杂问题分解 |

这个比喻的美妙之处在于：大语言模型就像一个拥有海量知识的"超级考生"，而提示工程就是教导我们如何成为优秀的"出题老师"。

---

## 26.2 什么是大语言模型

### 26.2.1 从GPT-3到GPT-4的进化

**GPT-3的诞生（2020年）**

2020年，OpenAI发布了GPT-3（Generative Pre-trained Transformer 3），这是一个拥有**1750亿参数**的语言模型。参数是什么？你可以把它们想象成模型大脑中的"神经元连接"。人脑大约有860亿个神经元，而GPT-3的1750亿参数，相当于一个庞大的人工神经网络。

Brown等人在他们的论文《Language Models are Few-Shot Learners》中展示了一个惊人的发现：**GPT-3不需要针对特定任务进行微调，只需要在提示中提供几个示例，就能完成各种任务**。

这就像你发现了一个学生，他虽然没学过翻译，但只要给他看几个"英文→中文"的例子，他就能帮你翻译了！

**GPT-3.5和ChatGPT（2022年）**

2022年，OpenAI在GPT-3的基础上进行了改进，通过人类反馈强化学习（RLHF），创造出了ChatGPT。这个模型不仅更擅长理解指令，还能进行连贯的多轮对话。

**GPT-4的飞跃（2023年）**

GPT-4代表了目前大语言模型的巅峰。虽然OpenAI没有公布具体参数数量，但据估计可能达到**数万亿级别**。更重要的是，GPT-4展现出了：

- **多模态能力**：能同时理解文本和图像
- **更强的推理能力**：在复杂的数学和逻辑问题上表现更好
- **更好的安全性**：更少产生有害或偏见的内容

### 26.2.2 大语言模型的工作原理

**Transformer架构**

大语言模型都基于一种叫做Transformer的神经网络架构。想象你在读一本书：

- **传统RNN**：像是一个字一个字地读，必须按顺序来，很慢
- **Transformer**：像是有无数双眼睛，可以同时看到整页的所有文字，还能理解它们之间的关系

Transformer的核心是**自注意力机制（Self-Attention）**，它让模型能够：

```
句子："猫坐在垫子上，因为它很温暖"

模型会思考：
- "它" 最可能指的是什么？
- "垫子" 和 "温暖" 之间有什么关系？
- 整个句子的含义是什么？
```

**下一个词预测**

大语言模型的核心任务其实很简单：**预测下一个词**。

给定"今天天气很"，模型要预测下一个词可能是"好"、"热"、"冷"等。通过在海量文本上训练，模型学会了：

1. 语法规则（主谓宾结构）
2. 世界知识（巴黎是法国的首都）
3. 推理能力（如果A=B且B=C，那么A=C）

这就像是让一个孩子读完了整个图书馆的书籍——他不仅会说话，还学会了很多知识。

### 26.2.3 大语言模型的能力边界

**能做什么：**

1. **文本生成**：写文章、故事、诗歌
2. **翻译**：中英文互译
3. **摘要**：把长文章浓缩成几句话
4. **问答**：回答各种问题
5. **代码生成**：写Python、JavaScript等程序
6. **推理**：解决数学问题、逻辑谜题

**不能做什么：**

1. **实时信息**：不知道今天的新闻
2. **真正理解**：没有意识，只是模式匹配
3. **精确计算**：大数乘法可能出错
4. **个性化知识**：不知道你个人的隐私信息

---

## 26.3 上下文学习：Zero-shot、One-shot、Few-shot

### 26.3.1 Zero-shot学习：裸考的艺术

**定义**：不给模型任何示例，直接提出问题。

就像走进考场，拿到试卷直接答题。

**示例**：

```
输入：
将以下英文翻译成中文：
"The quick brown fox jumps over the lazy dog."

输出：
"敏捷的棕色狐狸跳过了懒惰的狗。"
```

Kojima等人在2022年的论文《Large Language Models are Zero-Shot Reasoners》中发现了一个惊人的技巧：**只要在提示末尾加上"Let's think step by step"（让我们一步步思考）**，模型就能展现出推理能力！

**Zero-shot Chain of Thought示例**：

```
输入：
题目：罗杰有5个网球，他又买了2罐，每罐有3个网球。
      他现在有多少个网球？
      让我们一步步思考。

输出：
罗杰一开始有5个网球。
他买了2罐，每罐3个，所以是2 × 3 = 6个网球。
总共是5 + 6 = 11个网球。
答案是11。
```

**优点**：
- 简单直接，不需要准备示例
- 快速，省token（模型计费单位）

**缺点**：
- 对于复杂任务，准确率可能不高
- 模型可能误解任务意图

### 26.3.2 One-shot学习：一道例题的力量

**定义**：给模型提供一个示例，然后让模型按照示例的格式回答。

就像考试前快速看了一道例题，然后模仿它的解法。

**示例**：

```
输入：
将电影评论分类为正面或负面。

示例：
评论："这部电影真是太精彩了，演员的表演令人印象深刻！"
分类：正面

现在分类这条评论：
评论："浪费了我两个小时的生命，剧情漏洞百出。"
分类：

输出：
负面
```

**为什么One-shot有效？**

1. **格式指导**：模型学会了输出的格式
2. **任务理解**：模型明确了任务类型
3. **风格模仿**：模型可以模仿示例的语言风格

### 26.3.3 Few-shot学习：多道例题的威力

**定义**：给模型提供多个示例（通常是3-5个），然后提出问题。

Brown等人在GPT-3论文中发现：**随着示例数量的增加，模型性能持续提升**。

**Few-shot示例**：

```
输入：
以下是一些客户评论及其情感分类：

评论："产品质量很好，物流也很快！"
情感：正面

评论："完全不符合描述，退货流程太麻烦了。"
情感：负面

评论："一般般吧，没什么特别的。"
情感：中性

评论："客服态度太差了，再也不会买了。"
情感：负面

评论："超出预期，强烈推荐给大家！"
情感：正面

现在分类这条评论：
评论："包装破损，但产品本身没问题。"
情感：

输出：
中性（或混合情感）
```

**Few-shot学习的关键要素**：

1. **示例选择**：选择与目标任务相似的示例
2. **示例多样性**：覆盖不同的情况和边缘案例
3. **示例顺序**：通常将最相关的示例放在最后
4. **格式一致性**：所有示例保持相同的格式

**Few-shot vs Fine-tuning**：

| 方面 | Few-shot | Fine-tuning |
|------|----------|-------------|
| 训练成本 | 无 | 高（需要GPU） |
| 数据需求 | 几个示例 | 大量标注数据 |
| 灵活性 | 高（随时更换示例） | 低（需要重新训练） |
| 效果上限 | 受限于基础模型 | 可以超过基础模型 |

---

## 26.4 提示工程基础

### 26.4.1 什么是提示工程

**提示工程（Prompt Engineering）**是指设计和优化输入提示，以引导大语言模型产生期望输出的技术和实践。

这就像学习如何向一个非常聪明但有点"死板"的助手下达指令——你需要：

1. **清晰明确**：告诉他具体要什么
2. **提供上下文**：给他必要的背景信息
3. **设定格式**：告诉他如何组织答案
4. **给出示例**：展示你想要的输出样式

### 26.4.2 提示的基本结构

一个好的提示通常包含以下部分：

```
[角色设定] + [任务描述] + [上下文/背景] + [示例] + [待处理内容] + [输出格式]
```

**示例**：

```
【角色设定】
你是一位经验丰富的产品经理。

【任务描述】
请为以下功能写一段用户友好的描述。

【上下文】
我们的产品是一个在线学习平台。

【输出格式】
- 功能名称（10字以内）
- 功能描述（50字以内）
- 用户价值（30字以内）

【待处理内容】
功能：AI自动批改作业
```

### 26.4.3 提示设计原则

**原则1：具体明确**

❌ 差："写一封邮件"
✅ 好："写一封正式的商务邮件，向客户道歉产品延迟发货，并提供10%折扣补偿"

**原则2：提供上下文**

❌ 差："总结这篇文章"
✅ 好："请用3句话总结这篇文章的主要观点，面向没有技术背景的读者"

**原则3：使用分隔符**

使用```、"""、<>等分隔符来明确区分不同部分：

```
请翻译以下文本：

"""
The future belongs to those who believe in the beauty of their dreams.
"""

要求：
1. 保持诗意
2. 适合用作座右铭
```

**原则4：指定输出格式**

```
请分析以下产品的优缺点，并以JSON格式输出：

{
  "优点": ["...", "..."],
  "缺点": ["...", "..."],
  "总体评分": "1-10"
}
```

**原则5：给出示例（Few-shot）**

```
请将以下单词转换为过去式：

walk → walked
play → played
run → ?
```

### 26.4.4 常见的提示模式

**模式1：指令模式**

```
指令：写一篇关于人工智能的科普文章，字数500字左右，面向中学生。
```

**模式2：问答模式**

```
问题：什么是光合作用？
答案：
```

**模式3：续写模式**

```
故事开头：从前，有一个勇敢的小机器人...
请续写这个故事：
```

**模式4：转换模式**

```
请将以下口语转换为正式书面语：
口语："这个东西挺好的，我觉得可以。"
书面语：
```

**模式5：分析模式**

```
请分析以下代码中的bug，并给出修复建议：

```python
def divide(a, b):
    return a / b
```
```

---

## 26.5 思维链推理（Chain of Thought）

### 26.5.1 什么是思维链

**思维链（Chain of Thought, CoT）**是一种提示技术，通过让模型展示中间推理步骤来解决复杂问题。

Wei等人在2022年的论文中发现：**在Few-shot示例中加入推理过程，可以显著提升模型在数学和逻辑问题上的表现**。

就像费曼所说的："写出演算过程，答案自然显现。"

### 26.5.2 思维链示例

**普通Few-shot（错误）**：

```
问题：罗杰有5个网球，又买了2罐，每罐3个。他现在有多少个？
答案：11

问题：食堂有23个苹果，用了20个做午餐，又买了6个。现在有多少个？
答案：9

问题：服务器房有9台电脑，从周一到周四每天安装5台。现在有多少台？
答案：29（错误！）
```

**Chain of Thought Few-shot（正确）**：

```
问题：罗杰有5个网球，又买了2罐，每罐3个。他现在有多少个？
答案：罗杰开始有5个球。2罐每罐3个是6个球。5 + 6 = 11。答案是11。

问题：食堂有23个苹果，用了20个做午餐，又买了6个。现在有多少个？
答案：食堂原来有23个苹果。用了20个，剩下23 - 20 = 3个。又买了6个，3 + 6 = 9个。答案是9。

问题：服务器房有9台电脑，从周一到周四每天安装5台。现在有多少台？
答案：服务器房原来有9台电脑。从周一到周四是4天，每天安装5台，4 × 5 = 20台。9 + 20 = 29台。答案是29。
```

### 26.5.3 Zero-shot Chain of Thought

Kojima等人发现，即使不提供示例，只要在问题后加上"Let's think step by step"，模型就能自动产生推理过程。

```
问题：一个农场有若干鸡和兔子，共有35个头和94只脚。
      问鸡和兔子各有多少只？
      
让我们一步步思考。

答案：
设鸡有x只，兔子有y只。
根据头的数量：x + y = 35
根据脚的数量：2x + 4y = 94

从第一个方程：x = 35 - y
代入第二个方程：
2(35 - y) + 4y = 94
70 - 2y + 4y = 94
2y = 24
y = 12

所以兔子有12只，鸡有35 - 12 = 23只。
验证：23 × 2 + 12 × 4 = 46 + 48 = 94 ✓
答案是鸡23只，兔子12只。
```

### 26.5.4 思维链为什么有效

**1. 分解复杂问题**

复杂问题 → 多个简单步骤 → 逐步解决

**2. 提供更多计算机会**

每个推理步骤都是模型重新思考的机会，减少了"一步错，步步错"的风险。

**3. 可解释性**

我们可以看到模型的"思考过程"，更容易发现和纠正错误。

**4. 模拟人类认知**

人类解决复杂问题时也会写下中间步骤，思维链让模型模仿了这一过程。

### 26.5.5 思维链的适用场景

**适合使用CoT：**

- 数学问题（算术、代数、几何）
- 逻辑推理（谜题、条件推理）
- 符号操作（代码生成、公式推导）
- 多步决策（规划、策略游戏）

**不适合使用CoT：**

- 简单的事实问答（"法国首都是哪？"）
- 情感分析（正面/负面分类）
- 翻译任务
- 任何一步就能完成的任务

---

## 26.6 高级提示技术

### 26.6.1 Self-Consistency：自一致性

Wang等人在2022年提出：**让模型对同一个问题生成多个思维链，然后选择出现最频繁的答案**。

这就像考试时的"多检查几遍"：

```
问题：15 - 4 × 2 = ?

思维链1：
先算15 - 4 = 11
然后11 × 2 = 22
答案：22（错误！）

思维链2：
先算4 × 2 = 8
然后15 - 8 = 7
答案：7（正确）

思维链3：
乘法优先：4 × 2 = 8
减法：15 - 8 = 7
答案：7（正确）

最终答案：7（多数投票结果）
```

**实现步骤**：

1. 使用temperature > 0生成多个答案（temperature控制随机性）
2. 提取每个答案的最终结果
3. 对结果进行投票，选择最常见的答案

### 26.6.2 Least-to-Most：从简到繁

Zhou等人在2022年提出：**将复杂问题分解为一系列简单子问题，逐步解决**。

这就像解数学题时"先易后难"的策略：

```
复杂问题：Amy在5岁时身高是3英尺。
         之后每年长高是前一年长高的1/3。
         现在她10岁，身高多少？

分解：
1. 5岁到10岁是几年？
   → 5年

2. 每年长高多少？
   - 第1年（5→6岁）：长高1英尺
   - 第2年（6→7岁）：长高1/3英尺
   - 第3年（7→8岁）：长高1/9英尺
   - 第4年（8→9岁）：长高1/27英尺
   - 第5年（9→10岁）：长高1/81英尺

3. 总共长高多少？
   1 + 1/3 + 1/9 + 1/27 + 1/81 = ?

4. 最终身高？
   3 + (1 + 1/3 + 1/9 + 1/27 + 1/81)
```

### 26.6.3 Tree of Thoughts：思维树

**思维树（Tree of Thoughts, ToT）**将思维链扩展为树形结构，允许模型：

1. 探索多条推理路径
2. 评估中间步骤的质量
3. 回溯并尝试其他方案

这就像下棋时的思考过程：

```
当前局面：我走哪一步？

方案A：走兵
  → 评估：中等
  → 对方可能走...
  
方案B：走马
  → 评估：较好
  → 继续探索...
    → 走法B1
    → 走法B2（评分最高！）
    
方案C：走车
  → 评估：较差，放弃

最终选择：方案B2
```

### 26.6.4 ReAct：推理与行动结合

**ReAct（Reasoning + Acting）**将思维链与外部工具结合：

```
问题：2023年奥斯卡最佳男主角是谁？他主演了哪些电影？

思考1：我需要搜索2023年奥斯卡最佳男主角的信息。
行动1：搜索[2023年奥斯卡最佳男主角]
观察1：布兰登·费舍凭借《鲸》获得最佳男主角。

思考2：现在我需要查找布兰登·费舍主演的其他电影。
行动2：搜索[布兰登·费舍 电影作品]
观察2：他主演过《木乃伊》系列、《森林泰山》等。

思考3：我已经找到了所需信息，可以给出答案了。
答案：2023年奥斯卡最佳男主角是布兰登·费舍...
```

### 26.6.5 其他高级技巧

**1. 角色扮演（Role-playing）**

```
你是一位资深Python程序员。请审查以下代码，找出潜在的bug和性能问题。
```

**2. 思维反刍（Step-back Prompting）**

先问一个更普遍的问题，再回答具体问题：

```
一般问题：解决物理问题通常需要哪些步骤？

具体问题：一个球从10米高落下，求落地时的速度。
```

**3. 验证链（Chain of Verification）**

让模型先给出答案，然后检查答案中的事实：

```
初答：巴黎是法国的首都，人口约210万。

验证：
- 巴黎是法国首都？ ✓
- 人口210万？ 让我核实...
```

---

## 26.7 动手实现：提示模板和链式推理

### 26.7.1 环境准备

首先，我们需要导入必要的库并设置API：

```python
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
```

### 26.7.2 运行演示

保存上述代码到 `llm_prompting_demo.py`，然后运行：

```bash
python llm_prompting_demo.py
```

你将看到各个组件的演示输出，包括：

1. **PromptTemplate**：如何创建和管理提示模板
2. **FewShotPromptBuilder**：如何构建少样本提示
3. **ChainOfThought**：思维链推理的实现
4. **SelfConsistency**：自一致性投票机制
5. **LeastToMost**：从简到繁的问题分解

### 26.7.3 实际API集成

上述代码使用了模拟的LLM接口。在实际应用中，你需要替换为真实的API调用。以下是OpenAI API的集成示例：

```python
import openai

class OpenAILLM:
    """OpenAI API封装"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
    
    def generate(self, prompt: str, config: PromptConfig = None) -> str:
        config = config or PromptConfig()
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )
        
        return response.choices[0].message.content
    
    def generate_multiple(self, prompt: str, config: PromptConfig = None, 
                         n: int = 5) -> List[str]:
        config = config or PromptConfig()
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            n=n
        )
        
        return [choice.message.content for choice in response.choices]
```

---

## 26.8 练习题

### 26.8.1 基础练习题

**练习1：Zero-shot分类**

使用Zero-shot提示，让模型将以下评论分类为"电子产品"、"服装"或"食品"：

```
1. "这款手机的摄像头太棒了，夜景拍摄很清晰！"
2. "这件T恤面料很舒服，夏天穿很透气。"
3. "这个蛋糕甜度刚好，奶油很新鲜。"
```

**要求**：
- 不使用任何示例
- 确保分类结果准确

**参考答案思路**：
```
请将以下商品评论分类到"电子产品"、"服装"或"食品"类别：

评论："这款手机的摄像头太棒了，夜景拍摄很清晰！"
类别：
```

---

**练习2：Few-shot翻译**

构建一个Few-shot提示，将中文网络流行语翻译成地道英文：

```
示例：
"yyds" → "GOAT (Greatest Of All Time)"
"绝绝子" → "absolutely amazing / the best"
"躺平" → "lying flat"

待翻译："内卷"
```

**要求**：
- 至少提供3个示例
- 翻译要准确传达原意

**参考答案**：
```
请将以下中文网络流行语翻译成地道英文：

"yyds" → "GOAT (Greatest Of All Time)"
"绝绝子" → "absolutely amazing"
"躺平" → "lying flat"
"破防了" → "hit me right in the feels"

"内卷" → "involution / rat race"
```

---

**练习3：Chain of Thought数学**

使用Chain of Thought提示解决以下问题：

```
一个班级有40名学生。其中1/5的学生参加了数学竞赛，
参加数学竞赛的学生中有1/4获得了奖项。
问：获得奖项的学生有多少人？
```

**要求**：
- 展示完整的推理过程
- 每个步骤都要清晰说明

**参考答案**：
```
问题：一个班级有40名学生。其中1/5的学生参加了数学竞赛，
参加数学竞赛的学生中有1/4获得了奖项。获得奖项的学生有多少人？

让我一步步思考：

第一步：计算参加数学竞赛的学生人数
班级总人数：40人
参加竞赛的比例：1/5
参加竞赛的人数 = 40 × 1/5 = 8人

第二步：计算获得奖项的学生人数
参加竞赛的人数：8人
获奖比例：1/4
获奖人数 = 8 × 1/4 = 2人

答案是：2人。
```

---

### 26.8.2 进阶练习题

**练习4：构建Few-shot情感分析器**

使用本章实现的`FewShotPromptBuilder`类，构建一个情感分析器。

**要求**：
1. 准备至少5个标注好的示例（正面/负面/中性）
2. 使用相似度策略选择最相关的示例
3. 在测试集上评估准确率

**参考代码框架**：

```python
from llm_prompting_demo import FewShotPromptBuilder, Example, MockLLM

# 1. 准备训练示例
train_examples = [
    Example("产品质量很好，物流也很快！", "正面"),
    Example("完全不符合描述，退货流程太麻烦了。", "负面"),
    # ... 更多示例
]

# 2. 构建分类器
builder = FewShotPromptBuilder(prefix="将评论分类为正面、负面或中性。")
builder.add_examples(train_examples)

# 3. 测试
test_text = "包装破损，但产品本身没问题。"
prompt = builder.build(test_text, n_examples=3, strategy="similarity")

# 4. 调用LLM获取结果
llm = MockLLM()
result = llm.generate(prompt)
print(f"分类结果：{result}")
```

---

**练习5：实现Self-Consistency算术**

实现一个使用Self-Consistency解决算术问题的程序。

**要求**：
1. 生成5个不同的推理路径
2. 提取每个路径的最终答案
3. 通过投票选择最一致的答案
4. 计算置信度分数

**测试题目**：
```
1. 25 - 3 × 5 = ?
2. 100 ÷ (5 + 5) × 2 = ?
3. 一个长方形长8cm，宽比长少3cm，周长是多少？
```

**参考答案思路**：

```python
from llm_prompting_demo import SelfConsistency, ChainOfThought, MockLLM

llm = MockLLM()
cot = ChainOfThought(llm)
sc = SelfConsistency(llm, cot, n_paths=5)

problems = [
    "25 - 3 × 5 = ?",
    "100 ÷ (5 + 5) × 2 = ?",
    "一个长方形长8cm，宽比长少3cm，周长是多少？"
]

for p in problems:
    result = sc.solve_with_voting(p)
    print(f"问题：{p}")
    print(f"答案：{result['best_answer']} (置信度: {result['confidence']:.2%})")
    print(f"投票分布：{result['vote_distribution']}")
```

---

**练习6：Least-to-Most问题分解**

使用Least-to-Most技术解决以下复杂问题：

```
问题：一个水箱可以装120升水。A管单独注满需8小时，
B管单独注满需6小时，C管单独排空需12小时。
如果三管同时打开，多久能注满水箱？
```

**要求**：
1. 将问题分解为多个子问题
2. 按从简到繁的顺序解决
3. 展示每个子问题的解答

**分解示例**：
```
子问题1：A管每小时注水量是多少？
子问题2：B管每小时注水量是多少？
子问题3：C管每小时排水量是多少？
子问题4：三管同时开，每小时净注水量是多少？
子问题5：注满120升需要多少小时？
```

---

### 26.8.3 挑战练习题

**练习7：混合策略优化**

比较不同提示策略在GSM8K风格数学题上的表现。

**数据集**（5道题）：

```python
math_problems = [
    {
        "question": "小明买了3本书，每本25元。他付了100元，应该找回多少钱？",
        "answer": "25"
    },
    {
        "question": "一辆汽车每小时行驶60公里，行驶4小时后休息30分钟，然后再行驶2小时。总共行驶了多少公里？",
        "answer": "360"
    },
    {
        "question": "一个农场有鸡和兔子共20只，腿共56条。鸡有多少只？",
        "answer": "12"
    },
    {
        "question": "水箱原有水1/3满，加入40升后变成3/4满。水箱总容量是多少？",
        "answer": "96"
    },
    {
        "question": "甲、乙两人同时从A、B两地相向而行，甲速每小时5公里，乙速每小时4公里，2小时后相遇。A、B两地相距多远？",
        "answer": "18"
    }
]
```

**要求**：

1. 实现4种策略：
   - Zero-shot（直接提问）
   - Zero-shot CoT（加"让我们一步步思考"）
   - Few-shot CoT（提供2个示例）
   - Few-shot CoT + Self-Consistency（3个样本投票）

2. 记录每种策略的：
   - 准确率
   - 平均token消耗
   - API调用次数

3. 分析结果并给出结论

**评估报告模板**：

```
策略对比报告
================

| 策略 | 准确率 | 平均Tokens | API调用数 |
|------|--------|-----------|----------|
| Zero-shot | ?% | ? | ? |
| Zero-shot CoT | ?% | ? | ? |
| Few-shot CoT | ?% | ? | ? |
| CoT+Self-Consistency | ?% | ? | ? |

结论：
1. 最佳策略是...
2. 原因分析...
3. 实际应用建议...
```

---

**练习8：构建提示模板库**

设计并实现一个可复用的提示模板库，支持以下功能：

**功能需求**：

1. **模板注册系统**
   ```python
   registry = PromptRegistry()
   registry.register("sentiment", sentiment_template)
   registry.register("translation", translation_template)
   ```

2. **模板版本管理**
   ```python
   registry.register("sentiment", sentiment_v2, version="2.0")
   template = registry.get("sentiment", version="2.0")
   ```

3. **A/B测试支持**
   ```python
   # 随机选择不同版本的模板
   template = registry.select_for_ab_test("sentiment", 
                                         variants=["1.0", "2.0"],
                                         weights=[0.5, 0.5])
   ```

4. **效果追踪**
   ```python
   registry.log_result("sentiment", version="2.0", 
                      accuracy=0.92, latency=0.5)
   ```

**实现要求**：
- 使用本章的`PromptTemplate`作为基础
- 支持从JSON/YAML文件加载模板
- 提供模板效果分析报表

---

**练习9：实现Tree of Thoughts**

实现Tree of Thoughts算法解决24点游戏。

**24点游戏规则**：
- 给定4个数字
- 使用加、减、乘、除和括号
- 每个数字必须使用且只能使用一次
- 最终结果等于24

**示例**：
```
输入：[4, 7, 8, 8]
输出：4 * 7 - 8 + 8 = 24
      (4 - 8/8) * 7 = 24
```

**Tree of Thoughts实现要求**：

1. **状态表示**：当前已使用的数字和计算结果
2. **动作空间**：选择一个运算符和两个操作数
3. **评估函数**：评估当前状态离目标的距离
4. **搜索策略**：BFS或DFS探索不同路径
5. **回溯机制**：当路径走不通时回溯

**框架代码**：

```python
class TreeOfThoughts24:
    def __init__(self, llm):
        self.llm = llm
    
    def solve(self, numbers: List[int], target: int = 24) -> Optional[str]:
        """
        使用ToT解决24点问题
        
        Returns:
            找到的计算表达式，或None
        """
        # 实现你的算法
        pass
    
    def evaluate_state(self, state: Dict) -> float:
        """评估状态的潜力"""
        # 返回0-1之间的分数
        pass
    
    def generate_actions(self, state: Dict) -> List[Dict]:
        """生成可能的下一步动作"""
        pass

# 测试
tot = TreeOfThoughts24(MockLLM())
result = tot.solve([4, 7, 8, 8])
print(f"解：{result}")
```

**进阶挑战**：
- 实现束搜索（Beam Search）限制搜索宽度
- 使用LLM评估状态质量
- 添加可视化展示搜索树

---

## 26.9 总结与展望

### 26.9.1 本章要点回顾

**核心概念**：

1. **大语言模型**：基于Transformer架构，通过预测下一个词学习语言和世界知识
2. **上下文学习**：Zero-shot、One-shot、Few-shot三种模式
3. **提示工程**：设计和优化输入以获得更好输出的艺术
4. **思维链推理**：通过展示中间步骤提升复杂任务表现
5. **高级技术**：Self-Consistency、Least-to-Most、Tree of Thoughts等

**费曼比喻回顾**：

| AI概念 | 考试比喻 |
|--------|---------|
| Zero-shot | 裸考 |
| Few-shot | 看例题再考 |
| Chain of Thought | 写演算过程 |
| Self-Consistency | 多检查几遍 |
| Least-to-Most | 先易后难 |

### 26.9.2 最佳实践清单

**✅ 应该做的**：

- [ ] 明确指定任务和输出格式
- [ ] 使用分隔符区分不同部分
- [ ] 提供高质量的Few-shot示例
- [ ] 对于复杂任务，使用Chain of Thought
- [ ] 对于关键任务，使用Self-Consistency投票
- [ ] 测试不同策略并比较效果
- [ ] 记录和版本管理提示模板

**❌ 不应该做的**：

- [ ] 提示过于模糊或不完整
- [ ] 示例质量不一致
- [ ] 在不必要的情况下使用复杂技术
- [ ] 忽视token消耗和成本
- [ ] 不对输出进行验证

### 26.9.3 未来发展方向

1. **自动提示优化**：使用机器学习自动寻找最优提示
2. **多模态提示**：结合文本、图像、音频的统一提示框架
3. **提示安全**：防止提示注入攻击和恶意使用
4. **可解释性**：更好地理解模型为什么产生特定输出
5. **高效推理**：减少推理时间和成本的优化技术

---

## 参考文献

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33, 1877-1901. https://arxiv.org/abs/2005.14165

Kojima, T., Gu, S. S., Reid, M., Matsuo, Y., & Iwasawa, Y. (2022). Large language models are zero-shot reasoners. *Advances in Neural Information Processing Systems*, 35, 22199-22213. https://arxiv.org/abs/2205.11916

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., ... & Zhou, D. (2022). Self-consistency improves chain of thought reasoning in language models. *arXiv preprint arXiv:2203.11171*. https://arxiv.org/abs/2203.11171

Wei, J., Wang, X., Schuurmans, D., Bosma, M., ichter, b., Xia, F., ... & Zhou, D. (2022). Chain of thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824-24837. https://arxiv.org/abs/2201.11903

Zhou, D., Schärli, N., Hou, L., Scales, N., Min, Y., Fu, X., ... & Le, Q. (2022). Least-to-most prompting enables complex reasoning in large language models. *arXiv preprint arXiv:2205.10625*. https://arxiv.org/abs/2205.10625

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., & Narasimhan, K. (2023). Tree of thoughts: Deliberate problem solving with large language models. *arXiv preprint arXiv:2305.10601*. https://arxiv.org/abs/2305.10601

OpenAI. (2023). *GPT-4 technical report*. arXiv preprint arXiv:2303.08774. https://arxiv.org/abs/2303.08774

---

> **课后思考**：在费曼的物理课堂上，他常说："如果你认为你理解了某样东西，试着教给一个孩子。"提示工程的本质也是如此——我们需要用尽可能清晰、简洁的方式，向这个"超级智能但有点固执的学生"（大语言模型）传达我们的意图。当你掌握了这项技能，你就拥有了一把打开AI无限可能性的钥匙。

---

*本章完*
