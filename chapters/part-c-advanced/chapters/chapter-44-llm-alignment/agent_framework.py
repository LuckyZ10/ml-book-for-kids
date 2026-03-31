# 第四十六章 AI Agents与多智能体系统

## 46.1 什么是AI Agent？

### 从聊天机器人到智能代理

想象你有一个非常聪明的朋友——他能回答各种问题、写诗、解数学题，但他被关在房间里，只能通过门缝传递纸条与人交流。这就是**大语言模型（LLM）**的现状：拥有海量知识，却困在对话框里。

现在，给这位朋友配上：
- 📱 一部手机（可以搜索信息）
- 💻 一台电脑（可以写代码）
- 🗂️ 一个文件柜（可以记住事情）
- 🤖 几个助手（可以分工合作）

他不再是只能对话的" brain in a jar "（罐中之脑），而是能够**观察环境、做出决策、执行行动**的智能实体。这就是**AI Agent（人工智能代理）**。

> **费曼时间**：想象一个厨师👨‍🍳
> - LLM就像是厨师的大脑——知道所有食谱和烹饪技巧
> - Agent则是完整的厨房团队——厨师不仅知道怎么做菜，还能指挥助手采购食材、控制火候、摆盘上桌
> - 没有手脚的大脑只能"纸上谈兵"，有了工具的Agent才能真正"做出一道菜"

### Agent的核心组件

一个完整的AI Agent通常包含四个核心组件：

```
┌─────────────────────────────────────────────────────┐
│                  AI Agent 架构                      │
├─────────────────────────────────────────────────────┤
│  🧠 大脑 (LLM Core)                                  │
│     └── 推理、规划、决策的核心引擎                   │
├─────────────────────────────────────────────────────┤
│  👁️ 感知 (Perception)                                │
│     └── 接收环境信息：文本、图像、传感器数据         │
├─────────────────────────────────────────────────────┤
│  🎯 规划 (Planning)                                  │
│     └── 任务分解、策略制定、步骤编排                 │
├─────────────────────────────────────────────────────┤
│  🛠️ 行动 (Action)                                    │
│     └── 调用工具、执行代码、与环境交互               │
├─────────────────────────────────────────────────────┤
│  💾 记忆 (Memory)                                    │
│     └── 短期记忆（对话历史）+ 长期记忆（知识库）     │
└─────────────────────────────────────────────────────┘
```

**1. 大脑（LLM Core）**

大语言模型是Agent的"认知引擎"。它负责：
- 理解用户意图
- 进行逻辑推理
- 生成行动计划
- 评估执行结果

**2. 感知（Perception）**

Agent需要感知环境，包括：
- **文本输入**：用户指令、文档内容
- **视觉信息**：图像、视频、屏幕截图
- **结构化数据**：数据库查询结果、API响应
- **传感器数据**：温度、位置、设备状态（对具身智能体）

**3. 规划（Planning）**

面对复杂任务，Agent需要制定计划：
- **任务分解**：将"帮我订一张去纽约的机票"分解为：
  1. 查询航班信息
  2. 比较价格和时段
  3. 选择最优选项
  4. 填写预订信息
  5. 确认支付

- **策略选择**：决定使用哪些工具、按什么顺序执行
- **错误恢复**：当某步失败时，如何调整计划

**4. 行动（Action）**

这是Agent与外部世界交互的接口，包括：
- **工具调用**：搜索引擎、计算器、代码解释器
- **API调用**：预订服务、数据库查询、发送邮件
- **代码执行**：编写并运行Python脚本
- **物理动作**（对机器人）：移动、抓取、操作物体

**5. 记忆（Memory）**

Agent需要记住事情才能持续工作：
- **短期记忆**：当前对话的上下文
- **长期记忆**：用户偏好、历史交互、学到的知识
- **外部记忆**：向量数据库存储的海量信息

### Agent vs LLM：关键区别

| 特性 | 传统LLM | AI Agent |
|------|---------|----------|
| **交互方式** | 一问一答 | 自主循环 |
| **信息获取** | 依赖训练数据 | 实时搜索、工具调用 |
| **任务执行** | 仅生成文本 | 可以调用API、执行代码 |
| **任务复杂度** | 适合单步任务 | 可以处理多步复杂任务 |
| **自主性** | 被动响应 | 主动规划、执行、反思 |
| **记忆** | 有限的上下文窗口 | 持久化的记忆系统 |

> **类比**：
> - LLM像是一位**百科全书式的学者**——学识渊博但只能回答问题
> - Agent像是一位**能干的助理**——不仅懂知识，还能帮你订餐厅、写报告、安排日程

### Agent的工作循环

一个典型的Agent遵循"**感知-思考-行动**"循环：

```
     ┌──────────────┐
     │   感知环境   │◄────────────────┐
     └──────┬───────┘                 │
            │ 接收输入                 │
            ▼                          │
     ┌──────────────┐                 │
     │   思考规划   │                 │
     │  (LLM推理)   │                 │
     └──────┬───────┘                 │
            │ 生成行动                 │
            ▼                          │
     ┌──────────────┐    观察结果      │
     │   执行行动   │─────────────────┘
     │ (工具调用)   │
     └──────────────┘
```

这个循环会持续进行，直到任务完成或达到终止条件。

### Agent的类型

根据能力范围，Agent可以分为：

**1. 单任务Agent（Single-Task Agent）**
- 专注于特定领域
- 例如：专门写代码的Copilot、专门画图的DALL-E Agent

**2. 通用Agent（General-Purpose Agent）**
- 可以处理多种任务
- 例如：AutoGPT、Claude with Tools

**3. 多智能体系统（Multi-Agent System）**
- 多个Agent协作完成任务
- 例如：MetaGPT（产品经理+架构师+程序员+测试员）

**4. 具身智能体（Embodied Agent）**
- 有物理形态，可以与世界物理交互
- 例如：机器人、自动驾驶汽车

---

## 46.2 单智能体架构：ReAct与工具使用

### ReAct：推理与行动的协同

2022年，Google Research的Yao等人提出了**ReAct（Reasoning + Acting）**框架，这是Agent领域最具影响力的工作之一。

> **核心思想**：让LLM交替进行**推理（Reasoning）**和**行动（Acting）**，形成"思考→行动→观察→再思考"的循环。

**费曼比喻：ReAct像侦探破案🔍**

想象福尔摩斯在破案：
1. **思考**："现场有脚印，凶手应该是男性，身高约180cm..."
2. **行动**：去查询失踪人口数据库
3. **观察**：发现3名符合条件的嫌疑人
4. **再思考**："其中一人的鞋底纹路和现场匹配..."
5. **再行动**：调取该人的监控录像
6. **...**

侦探不会一开始就得出最终结论，而是通过**迭代式**的思考-行动-观察循环，逐步逼近真相。ReAct正是模拟了这种人类解决问题的模式。

### ReAct的数学形式

在ReAct框架中，Agent维护一个轨迹（trajectory）：

$$
\tau = [(t_1, a_1, o_1), (t_2, a_2, o_2), ..., (t_n, a_n, o_n)]
$$

其中：
- $t_i$：第$i$步的**思考（Thought）**
- $a_i$：第$i$步的**行动（Action）**
- $o_i$：第$i$步的**观察（Observation）**

在每个时间步，LLM根据当前轨迹生成下一步的思考：

$$
t_{n+1}, a_{n+1} = \text{LLM}(\text{prompt}, \tau)
$$

然后执行行动获得观察：

$$
o_{n+1} = \text{Environment}(a_{n+1})
$$

### ReAct示例

让我们看一个具体例子：

**用户问题**："《泰坦尼克号》和《阿凡达》的导演是谁？这两部电影的票房总和是多少？"

**ReAct执行过程**：

```
思考1: 我需要找出《泰坦尼克号》的导演。
行动1: Search["泰坦尼克号 导演"]
观察1: 《泰坦尼克号》由詹姆斯·卡梅隆执导。

思考2: 现在我需要找出《阿凡达》的导演。
行动2: Search["阿凡达 导演"]
观察2: 《阿凡达》同样由詹姆斯·卡梅隆执导。

思考3: 两部电影都是卡梅隆导演。现在需要查找票房。
行动3: Search["泰坦尼克号 全球票房"]
观察3: 《泰坦尼克号》全球票房约22.6亿美元。

思考4: 现在查找《阿凡达》的票房。
行动4: Search["阿凡达 全球票房"]
观察4: 《阿凡达》全球票房约29.2亿美元。

思考5: 计算总和：22.6 + 29.2 = 51.8亿美元
行动5: Calculate["22.6 + 29.2"]
观察5: 51.8

思考6: 我已经获得所有需要的信息，可以回答了。
行动6: Finish["两部电影都由詹姆斯·卡梅隆导演，票房总和约51.8亿美元。"]
```

### ReAct的优势

相比单纯让LLM直接回答，ReAct有显著优势：

**1. 可解释性（Interpretability）**
- Agent的"思考过程"对人类是透明的
- 可以追溯每一步的推理逻辑
- 便于调试和优化

**2. 事实准确性（Factual Accuracy）**
- 通过搜索等工具获取实时信息
- 避免依赖训练数据中的过时知识
- 减少幻觉（Hallucination）

**3. 处理复杂任务（Complex Tasks）**
- 多步推理可以分解复杂问题
- 每步专注于一个子问题
- 错误可以在后续步骤中纠正

**4. 鲁棒性（Robustness）**
- 当某步行动失败时，可以调整策略
- 不依赖单次推理的正确性

### 工具学习：让Agent"学会"使用工具

工具是Agent扩展能力的关键。2023年，Meta的Schick等人提出了**Toolformer**，让语言模型学会自己决定何时、如何使用工具。

**Toolformer的核心思想**：
- 在训练数据中插入工具调用标记
- 模型学会预测何时需要调用工具
- 模型生成工具调用，执行后填入结果

**工具调用的数学表示**：

给定输入序列$x$，模型需要预测：
1. 是否需要调用工具
2. 调用哪个工具
3. 工具的参数

$$
p(a_t | x_{<t}) = \text{softmax}(W_a \cdot h_t)
$$

其中$a_t \in \{\text{API}_1, \text{API}_2, ..., \text{NoAPI}\}$

### 常见工具类型

**1. 信息检索工具**
- 搜索引擎（Google、Bing）
- 知识库查询（Wikipedia、企业内部文档）
- 数据库查询（SQL）

**2. 计算工具**
- 计算器（简单算术）
- Python解释器（复杂计算、数据分析）
- 符号计算系统（Mathematica、Wolfram Alpha）

**3. 代码执行工具**
- 代码解释器（执行Python、Bash）
- 代码编辑器（读写文件）
- 版本控制（Git操作）

**4. 外部服务API**
- 邮件发送
- 日历管理
- 预订服务（机票、酒店）
- 支付接口

**5. 专业领域工具**
- 图像生成（DALL-E、Stable Diffusion）
- 语音识别/合成
- 科学计算库

### 工具选择的决策过程

Agent如何选择使用哪个工具？这涉及**工具选择（Tool Selection）**问题：

给定用户请求$q$和可用工具集合$T = \{t_1, t_2, ..., t_n\}$，Agent需要：

1. **工具选择**：选择最合适的工具$t^*$

$$
t^* = \arg\max_{t \in T} P(t | q, \text{context})
$$

2. **参数填充**：为选定工具生成参数$\theta$

$$
\theta = \text{LLM}(q, t^*, \text{context})
$$

3. **执行与验证**：执行工具调用，验证结果

### Reflexion：自我反思的Agent

2023年，Shinn等人提出了**Reflexion**，在ReAct基础上增加了**自我反思**机制。

**核心思想**：Agent不仅执行任务，还要**评估自己的表现**，并从错误中学习。

**Reflexion的三组件**：

1. **执行者（Actor）**
   - 基于ReAct生成行动
   - 与环境交互完成任务

2. **评估者（Evaluator）**
   - 评估执行结果的质量
   - 给出一个分数或反馈

3. **自我反思（Self-Reflection）**
   - 分析失败原因
   - 生成改进建议
   - 存储到记忆中供未来使用

**Reflexion的工作流程**：

```
┌────────────┐     失败      ┌──────────────┐
│   Actor    │──────────────►│  Evaluator   │
│  (执行者)  │               │   (评估者)   │
└────────────┘               └──────┬───────┘
      ▲                             │ 给出反馈
      │                             ▼
      │      改进后重试      ┌──────────────┐
      └──────────────────────┤ Self-Reflect │
                             │  (自我反思)  │
                             └──────────────┘
```

**费曼比喻：Reflexion像运动员回看比赛录像🏃**

想象一个篮球运动员：
1. **比赛（执行）**：上场打球
2. **回看录像（评估）**：教练指出"你刚才的传球选择不好"
3. **反思总结**："我当时应该看到队友已经空位了，下次要先观察再传球"
4. **下次改进**：下场比赛应用学到的经验

---

## 46.3 规划与推理：从CoT到Tree of Thoughts

### 链式思考（Chain-of-Thought, CoT）

2022年，Google的Wei等人发现：让LLM"一步步思考"可以显著提升推理能力。

**标准提示**：
```
问题：一个农场有35只鸡和28只兔子，共有多少只脚？
答案：____
```

**CoT提示**：
```
问题：一个农场有35只鸡和28只兔子，共有多少只脚？
让我们一步步思考：
- 每只鸡有2只脚，35只鸡有 35 × 2 = 70 只脚
- 每只兔子有4只脚，28只兔子有 28 × 4 = 112 只脚
- 总脚数 = 70 + 112 = 182
答案：182
```

**CoT的数学原理**：

CoT实际上是在增加计算的**有效深度**。标准提示下，模型需要直接映射：

$$
P(y|x) = \prod_{t=1}^{T} P(y_t | y_{<t}, x)
$$

而CoT通过中间步骤$z = [z_1, z_2, ..., z_k]$分解问题：

$$
P(y|x) = \sum_{z} P(y|z, x) \cdot P(z|x)
$$

这使得复杂的推理可以分解为多个简单的步骤。

### 自一致性（Self-Consistency）

CoT的一个问题是：如果某步推理出错，整个答案就错了。**自一致性**通过采样多条推理路径来解决这个问题。

**算法**：
1. 使用CoT采样$N$个不同的推理路径
2. 每条路径产生一个答案
3. 选择出现频率最高的答案

$$
y^* = \arg\max_{y} \sum_{i=1}^{N} \mathbb{1}(y_i = y)
$$

这就像"三人成虎"——多个独立推理路径达成一致时，结果更可靠。

### 思维树（Tree of Thoughts, ToT）

CoT是一条直线式的推理，而**Tree of Thoughts**将其扩展为树形搜索。

> **核心思想**：允许Agent探索多条推理路径，像下棋一样评估不同选择，回溯并尝试其他方案。

**费曼比喻：ToT像迷宫探索🌀**

想象你在一个迷宫中寻找出口：
- **CoT**：选定一条路走到底，不管对不对
- **ToT**：在每个岔路口都探索一下，评估哪条路更有希望，可以回溯重试

**ToT的四个核心操作**：

1. **分解（Decomposition）**
   - 将问题分解为思考步骤
   - 每个步骤是一个中间状态

2. **生成（Generation）**
   - 从当前状态生成候选下一步
   - 可以使用采样或提议生成

3. **评估（Evaluation）**
   - 评估每个候选状态的价值
   - 可以使用价值函数或投票

4. **搜索（Search）**
   - 使用BFS、DFS或束搜索探索树
   - 选择最有希望的路径

**ToT的数学框架**：

定义思考状态$s$，行动$a$（生成下一步思考），价值函数$V(s)$。

搜索目标是找到最优的思考序列：

$$
s^* = \arg\max_{s \in \mathcal{T}} V(s)
$$

其中$\mathcal{T}$是思考树。

**ToT算法（广度优先搜索版本）**：

```
输入：问题x，思考分解步骤数k，每步候选数b，评估函数V
输出：最优解决方案

1. 初始化：S₀ = {x}  # 只有根节点
2. for i = 1 to k:
3.     # 生成候选
4.     Sᵢ = GenerateCandidates(Sᵢ₋₁, b)
5.     # 评估候选
6.     Vᵢ = Evaluate(Sᵢ)
7.     # 选择最有希望的b个
8.     Sᵢ = SelectTopK(Sᵢ, Vᵢ, b)
9. return 最佳状态的解决方案
```

### 规划算法的对比

| 方法 | 结构 | 适用场景 | 计算成本 |
|------|------|----------|----------|
| **Standard** | 直接回答 | 简单问题 | 低 |
| **CoT** | 线性链 | 多步推理 | 中 |
| **Self-Consistency** | 多链投票 | 需要高可靠性 | 高 |
| **ToT** | 树形搜索 | 需要探索的复杂问题 | 很高 |

---

## 46.4 多智能体系统：协作的力量

### 为什么需要多智能体？

单Agent虽然强大，但面临限制：
- **能力瓶颈**：一个Agent难以同时精通所有领域
- **单点故障**：一个错误可能导致整体失败
- **效率限制**：串行执行耗时

**多智能体系统（Multi-Agent System, MAS）**通过协作解决这些问题。

> **费曼比喻：多智能体像一家公司🏢**
> 
> 想象一下：
> - **产品经理**：规划产品方向
> - **架构师**：设计系统结构
> - **程序员**：编写代码
> - **测试员**：质量保证
> 
> 一个人很难同时做好所有这些角色。但一个团队可以分工协作，发挥各自的专长。

### 多智能体架构

**1. 水平协作（Horizontal Collaboration）**

多个Agent平等协作，没有明确的领导者。

```
Agent A ◄──────► Agent B
   ▲    \      /    ▲
   └─────► Agent C ◄─┘
```

特点：
- 民主决策
- 信息充分共享
- 适合讨论、头脑风暴

**代表系统**：ChatEval（多个Agent辩论评估）

**2. 垂直协作（Vertical Collaboration）**

有明确的层级结构，一个领导者协调其他Agent。

```
     Coordinator
         │
    ┌────┼────┐
    ▼    ▼    ▼
 Agent A  B    C
```

特点：
- 效率高（减少无效讨论）
- 责任明确
- 适合复杂项目管理

研究发现：有领导者的团队比纯平等团队**快约10%**完成任务。

**代表系统**：MetaGPT

**3. 动态协作（Dynamic Collaboration）**

根据任务需要动态组建团队、分配角色。

```
阶段1：Recruitment（招募）
阶段2：Collaboration（协作）
阶段3：Execution（执行）
阶段4：Evaluation（评估）
   ↓
（循环直到完成）
```

**代表系统**：AgentVerse、DyLAN

### 多智能体通信机制

Agent之间需要"交谈"才能协作。常见通信模式：

**1. 消息传递（Message Passing）**

Agent直接发送消息给对方：

```python
message = {
    "from": "Agent_A",
    "to": "Agent_B",
    "content": "我已经完成了数据库设计，请开始后端开发",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

**2. 黑板系统（Blackboard）**

所有Agent共享一个公共信息板：

```
┌─────────────────────────────────┐
│           Blackboard            │
├─────────────────────────────────┤
│ 任务：开发一个网站               │
│ 状态：设计阶段已完成             │
│ 当前负责人：后端开发Agent        │
│ 已完成：需求分析、UI设计        │
│ 待办：后端API、前端实现、测试   │
└─────────────────────────────────┘
```

**3. 发布-订阅（Publish-Subscribe）**

Agent订阅感兴趣的主题，当有新消息时自动接收：

```
Agent A 发布："design_completed"
   ↓
订阅了该主题的 Agent B 和 C 收到通知
```

### MetaGPT：多智能体编程框架

2023年，Deep Wisdom团队提出了**MetaGPT**，将软件开发流程映射到多智能体系统。

**核心思想**：用**标准操作程序（SOP）**规范多智能体协作。

**MetaGPT的角色**：
- **产品经理**：撰写产品需求文档（PRD）
- **架构师**：设计系统架构和技术选型
- **项目经理**：制定任务计划
- **工程师**：编写代码
- **QA工程师**：编写测试用例并测试

**MetaGPT的工作流程**：

```
用户需求
   ↓
产品经理 → 输出PRD
   ↓
架构师 → 设计文档 + 接口定义
   ↓
项目经理 → 任务分解
   ↓
工程师 → 代码实现
   ↓
QA工程师 → 测试 + 反馈
   ↓
（如有问题，返回修改）
   ↓
交付
```

**为什么有效？**

1. **角色专业化**：每个Agent专注于一个领域
2. **结构化输出**：每个角色有明确的交付物格式
3. **流程标准化**：遵循软件工程最佳实践

### 多智能体强化学习

当Agent需要在环境中学习协作时，涉及**多智能体强化学习（Multi-Agent Reinforcement Learning, MARL）**。

**核心挑战**：

1. **非平稳环境（Non-stationarity）**
   - 其他Agent在同时学习，环境不断变化
   - 今天有效的策略明天可能失效

2. **信用分配（Credit Assignment）**
   - 团队成功时，如何确定每个Agent的贡献？

3. **协调问题（Coordination）**
   - Agent需要学会配合，而不仅仅是各自为战

**MADDPG算法**（Multi-Agent Deep Deterministic Policy Gradient）是经典方法：

为每个Agent$i$维护：
- **策略网络** $\mu_i$：决定行动
- **Q网络** $Q_i$：评估状态-行动对

关键创新：**集中训练，分散执行（CTDE）**
- 训练时：每个Agent可以看到其他Agent的信息
- 执行时：每个Agent只根据自己的观察行动

$$
\nabla_{\theta_i} J \approx \mathbb{E}[\nabla_{\theta_i} \mu_i(a_i|o_i) \nabla_{a_i} Q_i^{\mu}(x, a_1, ..., a_N)|_{a_i=\mu_i(o_i)}]
$$

其中$x = (o_1, ..., o_N)$是所有Agent的联合观察。

---

## 46.5 前沿应用与展望

### AutoGPT：自主AI的先驱

2023年，AutoGPT横空出世，展示了Agent的自主能力。

**AutoGPT的特点**：
- **目标导向**：给定一个高层目标，自动分解并执行
- **互联网接入**：可以搜索信息
- **代码执行**：可以编写和运行代码
- **长期记忆**：使用向量数据库存储信息

**典型用例**：
```
用户：帮我研究电动汽车市场，写一份投资报告

AutoGPT:
1. 搜索"2024电动汽车市场规模"
2. 搜索"主要电动汽车厂商市场份额"
3. 搜索"电动汽车技术趋势"
4. 分析收集到的信息
5. 生成报告大纲
6. 撰写完整报告
7. 保存为文件
```

### Devin：AI软件工程师

2024年，Cognition Labs推出了**Devin**，被称为"第一个完全自主的AI软件工程师"。

**Devin的能力**：
- 理解自然语言需求
- 搭建开发环境
- 编写代码
- 调试错误
- 部署应用

**技术突破**：
- 端到端的软件开发流程
- 与现有工具链集成（GitHub、AWS等）
- 可以处理复杂的工程任务

### 科学发现Agent

AI Agent正在进入科学研究领域：

**1. 材料发现**
- Agent搜索文献、提出假设
- 设计实验、分析结果
- 发现新材料

**2. 药物研发**
- Agent分析分子结构
- 预测药物-靶点相互作用
- 加速新药发现

**3. 数学证明**
- 探索猜想
- 生成证明思路
- 验证证明正确性

### 具身智能体

将Agent与物理世界连接：

**1. 机器人**
- 理解自然语言指令
- 规划物理动作序列
- 与环境交互完成任务

**2. 自动驾驶**
- 感知周围环境
- 预测其他车辆行为
- 做出安全驾驶决策

**3. 智能家居**
- 协调多个智能设备
- 学习用户习惯
- 主动提供服务

### 未来挑战与方向

**1. 安全与对齐**
- Agent有更大的行动能力，风险也更大
- 需要确保Agent的目标与人类对齐
- 防止Agent被恶意利用

**2. 可解释性**
- Agent的决策过程需要透明
- 用户需要理解Agent为什么这样做
- 便于调试和信任建立

**3. 效率与成本**
- 多步推理和工具调用成本高
- 需要更高效的架构
- 边缘部署的挑战

**4. 长期记忆与学习**
- 当前Agent的"学习"能力有限
- 如何实现持续学习而不遗忘
- 个性化与隐私的平衡

---

## 46.6 完整代码实现

### 基础Agent框架

```python
"""
AI Agent基础框架
包含：Agent基类、ReAct循环、工具系统、记忆系统
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import deque


class ActionType(Enum):
    """行动类型枚举"""
    THINK = "think"
    SEARCH = "search"
    CALCULATE = "calculate"
    EXECUTE_CODE = "execute_code"
    API_CALL = "api_call"
    FINISH = "finish"


@dataclass
class Thought:
    """思考步骤"""
    content: str
    step: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class Action:
    """行动步骤"""
    action_type: ActionType
    params: Dict[str, Any]
    step: int
    timestamp: float = field(default_factory=time.time)
    
    def to_string(self) -> str:
        """转换为字符串表示"""
        if self.action_type == ActionType.SEARCH:
            return f"Search[{self.params.get('query', '')}]"
        elif self.action_type == ActionType.CALCULATE:
            return f"Calculate[{self.params.get('expression', '')}]"
        elif self.action_type == ActionType.EXECUTE_CODE:
            return f"Execute[{self.params.get('code', '')[:50]}...]"
        elif self.action_type == ActionType.FINISH:
            return f"Finish[{self.params.get('answer', '')[:100]}...]"
        else:
            return f"{self.action_type.value}[{json.dumps(self.params)}]"


@dataclass
class Observation:
    """观察结果"""
    content: str
    step: int
    timestamp: float = field(default_factory=time.time)


class Tool(ABC):
    """工具基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, **params) -> str:
        """执行工具，返回结果字符串"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具参数模式"""
        return {
            "name": self.name,
            "description": self.description
        }


class SearchTool(Tool):
    """搜索工具（模拟）"""
    
    def __init__(self):
        super().__init__(
            name="search",
            description="搜索网络信息，参数：query（搜索关键词）"
        )
        # 模拟知识库
        self.knowledge_base = {
            "泰坦尼克号 导演": "《泰坦尼克号》由詹姆斯·卡梅隆执导，1997年上映。",
            "阿凡达 导演": "《阿凡达》由詹姆斯·卡梅隆执导，2009年上映。",
            "泰坦尼克号 票房": "《泰坦尼克号》全球票房约22.6亿美元。",
            "阿凡达 票房": "《阿凡达》全球票房约29.2亿美元。",
            "詹姆斯·卡梅隆": "詹姆斯·卡梅隆是加拿大电影导演，代表作有《泰坦尼克号》《阿凡达》《终结者2》等。"
        }
    
    def execute(self, query: str) -> str:
        """模拟搜索"""
        # 简单的关键词匹配
        for key, value in self.knowledge_base.items():
            if query.lower() in key.lower() or key.lower() in query.lower():
                return value
        return f"未找到关于'{query}'的信息。"


class CalculatorTool(Tool):
    """计算器工具"""
    
    def __init__(self):
        super().__init__(
            name="calculate",
            description="执行数学计算，参数：expression（数学表达式）"
        )
    
    def execute(self, expression: str) -> str:
        """安全计算表达式"""
        try:
            # 只允许安全操作
            allowed_names = {
                "abs": abs, "max": max, "min": min,
                "sum": sum, "pow": pow
            }
            code = compile(expression, "<string>", "eval")
            result = eval(code, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"计算错误: {str(e)}"


class Memory:
    """记忆系统"""
    
    def __init__(self, max_short_term: int = 10):
        self.short_term = deque(maxlen=max_short_term)  # 短期记忆
        self.long_term: Dict[str, Any] = {}  # 长期记忆
    
    def add_to_short_term(self, item: Any):
        """添加到短期记忆"""
        self.short_term.append(item)
    
    def get_short_term(self) -> List[Any]:
        """获取短期记忆"""
        return list(self.short_term)
    
    def store_long_term(self, key: str, value: Any):
        """存储长期记忆"""
        self.long_term[key] = value
    
    def retrieve_long_term(self, key: str) -> Optional[Any]:
        """检索长期记忆"""
        return self.long_term.get(key)
    
    def clear_short_term(self):
        """清空短期记忆"""
        self.short_term.clear()


class BaseAgent:
    """基础Agent类"""
    
    def __init__(self, name: str, llm_interface: Optional[Callable] = None):
        self.name = name
        self.llm = llm_interface or self._default_llm
        self.tools: Dict[str, Tool] = {}
        self.memory = Memory()
        self.trajectory: List[Any] = []  # 执行轨迹
        self.max_steps = 10
    
    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
    
    def _default_llm(self, prompt: str) -> str:
        """默认LLM接口（模拟）"""
        # 实际使用时，这里调用OpenAI API等
        return f"[模拟LLM响应] 收到提示: {prompt[:50]}..."
    
    def think(self, context: str) -> Thought:
        """思考步骤"""
        prompt = self._build_think_prompt(context)
        response = self.llm(prompt)
        thought = Thought(content=response, step=len(self.trajectory))
        self.trajectory.append(thought)
        self.memory.add_to_short_term(thought)
        return thought
    
    def act(self, thought: Thought) -> Action:
        """根据思考生成行动"""
        prompt = self._build_act_prompt(thought.content)
        response = self.llm(prompt)
        
        # 解析行动（简化版，实际需要更健壮的解析）
        action = self._parse_action(response)
        self.trajectory.append(action)
        return action
    
    def observe(self, action: Action) -> Observation:
        """执行行动并观察结果"""
        if action.action_type == ActionType.FINISH:
            obs = Observation(
                content="任务完成",
                step=len(self.trajectory)
            )
        elif action.action_type in [ActionType.SEARCH, ActionType.CALCULATE]:
            # 执行工具调用
            tool_name = action.action_type.value
            if tool_name in self.tools:
                result = self.tools[tool_name].execute(**action.params)
                obs = Observation(
                    content=result,
                    step=len(self.trajectory)
                )
            else:
                obs = Observation(
                    content=f"错误: 工具 '{tool_name}' 未找到",
                    step=len(self.trajectory)
                )
        else:
            obs = Observation(
                content="无观察结果",
                step=len(self.trajectory)
            )
        
        self.trajectory.append(obs)
        self.memory.add_to_short_term(obs)
        return obs
    
    def _build_think_prompt(self, context: str) -> str:
        """构建思考提示"""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
        prompt = f"""你是一个AI助手，需要解决以下问题：

{context}

可用工具：
{tools_desc}

请思考下一步应该做什么。简要说明你的推理过程。"""
        
        return prompt
    
    def _build_act_prompt(self, thought: str) -> str:
        """构建行动提示"""
        prompt = f"""基于以下思考，选择下一步行动：

思考：{thought}

可用行动格式：
- Search[查询内容] - 搜索信息
- Calculate[数学表达式] - 执行计算  
- Finish[最终答案] - 完成任务

请输出下一步行动（使用上述格式）："""
        
        return prompt
    
    def _parse_action(self, response: str) -> Action:
        """解析行动（简化实现）"""
        response = response.strip()
        
        if "Finish[" in response or "finish[" in response:
            # 提取答案
            start = response.find("[") + 1
            end = response.rfind("]")
            answer = response[start:end] if end > start else response
            return Action(
                action_type=ActionType.FINISH,
                params={"answer": answer},
                step=len(self.trajectory)
            )
        elif "Calculate[" in response or "calculate[" in response:
            start = response.find("[") + 1
            end = response.rfind("]")
            expr = response[start:end] if end > start else "0"
            return Action(
                action_type=ActionType.CALCULATE,
                params={"expression": expr},
                step=len(self.trajectory)
            )
        elif "Search[" in response or "search[" in response:
            start = response.find("[") + 1
            end = response.rfind("]")
            query = response[start:end] if end > start else ""
            return Action(
                action_type=ActionType.SEARCH,
                params={"query": query},
                step=len(self.trajectory)
            )
        else:
            return Action(
                action_type=ActionType.THINK,
                params={"content": response},
                step=len(self.trajectory)
            )
    
    def run(self, task: str) -> str:
        """运行Agent完成任务"""
        print(f"\\n{'='*50}")
        print(f"🤖 Agent '{self.name}' 开始执行任务")
        print(f"任务: {task}")
        print(f"{'='*50}\\n")
        
        context = task
        step = 0
        
        while step < self.max_steps:
            # 思考
            print(f"\\n📍 Step {step + 1}")
            thought = self.think(context)
            print(f"💭 思考: {thought.content[:100]}...")
            
            # 行动
            action = self.act(thought)
            print(f"🔧 行动: {action.to_string()}")
            
            # 如果是结束行动，返回结果
            if action.action_type == ActionType.FINISH:
                answer = action.params.get("answer", "")
                print(f"\\n✅ 任务完成！")
                print(f"📤 最终答案: {answer}")
                return answer
            
            # 观察
            observation = self.observe(action)
            print(f"👁️  观察: {observation.content[:100]}...")
            
            # 更新上下文
            context = f"{context}\\n思考: {thought.content}\\n行动: {action.to_string()}\\n观察: {observation.content}"
            
            step += 1
        
        print(f"\\n⚠️ 达到最大步数限制 ({self.max_steps})")
        return "任务未完成（达到步数限制）"


# ==================== ReAct Agent ====================

class ReActAgent(BaseAgent):
    """ReAct风格的Agent"""
    
    def __init__(self, name: str = "ReActAgent", llm_interface: Optional[Callable] = None):
        super().__init__(name, llm_interface)
        self.max_steps = 8
    
    def _build_think_prompt(self, context: str) -> str:
        """ReAct风格的思考提示"""
        trajectory_str = self._format_trajectory()
        
        prompt = f"""你正在使用ReAct（推理+行动）方法解决问题。

问题：{context}

执行历史：
{trajectory_str}

请按照以下格式输出：
思考：[你的推理过程，分析当前情况]
行动：[Search/Calculate/Finish][参数]

注意：
- 如果有足够信息回答问题，使用 Finish[答案]
- 如果需要外部信息，使用 Search[查询]
- 如果需要计算，使用 Calculate[表达式]

请输出下一步的思考和行动："""
        
        return prompt
    
    def _format_trajectory(self) -> str:
        """格式化执行轨迹"""
        lines = []
        for item in self.trajectory:
            if isinstance(item, Thought):
                lines.append(f"思考: {item.content}")
            elif isinstance(item, Action):
                lines.append(f"行动: {item.to_string()}")
            elif isinstance(item, Observation):
                lines.append(f"观察: {item.content}")
        return "\\n".join(lines[-6:])  # 只显示最近6步
    
    def think_and_act(self, task: str) -> tuple[Thought, Action]:
        """联合思考和行动生成"""
        prompt = self._build_think_prompt(task)
        response = self.llm(prompt)
        
        # 解析思考和行动
        thought_content = ""
        action_str = ""
        
        if "思考：" in response or "思考:" in response:
            parts = response.split("行动：") if "行动：" in response else response.split("行动:")
            thought_content = parts[0].replace("思考：", "").replace("思考:", "").strip()
            if len(parts) > 1:
                action_str = parts[1].strip()
        else:
            thought_content = response
            action_str = "Finish[未解析到行动]"
        
        thought = Thought(content=thought_content, step=len(self.trajectory))
        self.trajectory.append(thought)
        
        # 解析行动
        action = self._parse_action(action_str)
        
        return thought, action
    
    def run(self, task: str) -> str:
        """ReAct循环执行"""
        print(f"\\n{'='*60}")
        print(f"🤖 ReAct Agent '{self.name}' 开始任务")
        print(f"任务: {task}")
        print(f"{'='*60}\\n")
        
        step = 0
        while step < self.max_steps:
            print(f"\\n{'─'*40}")
            print(f"📍 第 {step + 1} 轮 ReAct 循环")
            print(f"{'─'*40}")
            
            # 联合思考和行动
            thought, action = self.think_and_act(task)
            print(f"\\n💭 思考: {thought.content}")
            print(f"🔧 行动: {action.to_string()}")
            
            # 检查是否结束
            if action.action_type == ActionType.FINISH:
                answer = action.params.get("answer", "")
                print(f"\\n{'='*60}")
                print(f"✅ 任务完成！")
                print(f"📤 答案: {answer}")
                print(f"{'='*60}")
                return answer
            
            # 执行行动并观察
            observation = self.observe(action)
            print(f"👁️  观察: {observation.content}")
            
            step += 1
        
        return "任务未完成"


# ==================== 演示 ====================

def demo_simple_agent():
    """演示基础Agent"""
    print("\\n" + "="*60)
    print("演示1: 基础Agent")
    print("="*60)
    
    # 创建Agent
    agent = BaseAgent(name="SimpleAgent")
    
    # 注册工具
    agent.register_tool(SearchTool())
    agent.register_tool(CalculatorTool())
    
    # 模拟LLM（实际应调用真实API）
    def mock_llm(prompt: str) -> str:
        """模拟LLM响应"""
        if "思考" in prompt and "票房" in prompt:
            return "我需要计算两部电影的票房总和"
        elif "行动" in prompt and "票房" in prompt:
            return "Calculate[22.6 + 29.2]"
        return "让我思考一下"
    
    agent.llm = mock_llm
    
    # 运行任务
    result = agent.run("计算泰坦尼克号和阿凡达的票房总和")
    print(f"\\n结果: {result}")


def demo_react_agent():
    """演示ReAct Agent"""
    print("\\n" + "="*60)
    print("演示2: ReAct Agent")
    print("="*60)
    
    agent = ReActAgent(name="ReActAgent")
    agent.register_tool(SearchTool())
    agent.register_tool(CalculatorTool())
    
    # 模拟更智能的LLM响应
    step_count = [0]
    
    def mock_llm(prompt: str) -> str:
        step_count[0] += 1
        step = step_count[0]
        
        if step == 1:
            return """思考：我需要先查找泰坦尼克号的导演信息
行动：Search[泰坦尼克号 导演]"""
        elif step == 2:
            return """思考：已知泰坦尼克号导演是卡梅隆。现在查找阿凡达的导演
行动：Search[阿凡达 导演]"""
        elif step == 3:
            return """思考：两部电影都是卡梅隆导演。现在查找泰坦尼克号的票房
行动：Search[泰坦尼克号 票房]"""
        elif step == 4:
            return """思考：泰坦尼克号票房是22.6亿。现在查找阿凡达的票房
行动：Search[阿凡达 票房]"""
        elif step == 5:
            return """思考：阿凡达票房是29.2亿。现在计算总和
行动：Calculate[22.6 + 29.2]"""
        elif step == 6:
            return """思考：计算结果是51.8亿美元。我已经有足够信息回答问题了
行动：Finish[两部电影都由詹姆斯·卡梅隆导演，票房总和约51.8亿美元。]"""
        
        return "思考：让我继续分析\\n行动：Finish[完成]"
    
    agent.llm = mock_llm
    
    result = agent.run("《泰坦尼克号》和《阿凡达》的导演是谁？这两部电影的票房总和是多少？")
    return result


if __name__ == "__main__":
    demo_simple_agent()
    demo_react_agent()


# ==================== 46.7 应用场景 ====================

"""
## 46.7 应用场景与前沿方向

### 科研Agent：加速科学发现

AI Agent正在改变科学研究的方式：

**文献综述Agent**
- 自动检索和分析海量文献
- 识别研究趋势和空白
- 生成综述报告

**实验设计Agent**
- 根据研究问题设计实验方案
- 优化实验参数
- 预测实验结果

**数据分析Agent**
- 自动清洗和预处理数据
- 选择合适的统计方法
- 生成可视化图表

> **费曼比喻**：科研Agent就像一位不知疲倦的研究助理
> - 可以24小时阅读文献
> - 同时跟踪多个研究方向
> - 快速整合跨学科知识
> - 但最终的科学洞察仍需要人类科学家的创造力

### 编程Agent：从Copilot到Devin

编程Agent的演进：

**代码补全（GitHub Copilot）**
- 根据上下文生成代码片段
- 单行/函数级别的辅助
- 被动等待开发者触发

**代码生成（CodeT5, CodeBERT）**
- 从自然语言描述生成代码
- 代码到代码的转换（翻译、优化）
- 需要清晰的规格说明

**自主编程（Devin, MetaGPT）**
- 从需求到代码的端到端实现
- 可以自主调试和修复Bug
- 与人类开发者协作

**编程教育的应用**
- 个性化编程辅导
- 自动批改和反馈
- 生成练习题和解答

### 具身智能：Agent与物理世界

**机器人Agent**
- 自然语言指令理解："把桌上的红色积木放到篮子里"
- 任务规划：分解为"定位→抓取→移动→放置"
- 动作执行：控制机械臂完成操作
- 错误恢复：掉落后重新尝试

**自动驾驶Agent**
- 多传感器融合感知
- 轨迹预测和决策
- 实时路径规划
- 安全冗余机制

**智能家居Agent**
- 场景理解：识别用户活动和意图
- 设备协调：统一控制多个智能设备
- 习惯学习：自动调整环境设置
- 主动服务：预测用户需求

### 多模态Agent：统一感知与行动

结合视觉、语言、听觉的Agent：

**视觉问答Agent**
- 看图回答用户问题
- 定位图中特定物体
- 理解场景关系

**视频理解Agent**
- 总结视频内容
- 回答关于视频的问题
- 提取关键事件

**具身问答（Embodied QA）**
- 在3D环境中导航和探索
- 回答关于环境的问题
- 例如："厨房里有什么水果？"

### Agent安全与对齐

随着Agent能力的增强，安全问题日益重要：

**风险类型**
1. **工具滥用**：Agent调用危险工具（删除数据、发送邮件）
2. **目标错误**：误解用户意图，执行错误操作
3. **连锁故障**：Agent A的错误导致Agent B的错误
4. **恶意操控**：被攻击者利用执行有害操作

**安全机制**
1. **权限控制**：限制Agent可使用的工具和API
2. **人工确认**：关键操作需要人类批准
3. **行为监控**：实时监控Agent行为，检测异常
4. **沙箱执行**：在隔离环境中运行Agent代码

**对齐技术**
- RLHF：通过人类反馈对齐Agent行为
- 宪法AI：让Agent遵循预定义的安全准则
- 透明性：记录和可解释Agent决策过程

### 未来展望

**2024-2025：Agent能力提升**
- 更可靠的长期规划和执行
- 更好的错误恢复和自适应
- 多Agent协作的标准化

**2025-2026：Agent生态成熟**
- Agent应用商店和平台
- 跨平台Agent互操作性
- Agent服务化（Agent-as-a-Service）

**2026+：通用智能体**
- 接近人类水平的通用Agent
- 自主学习新技能
- 与人类社会深度协作

---

## 46.8 练习题

### 基础题（必做）

**46.1** 请用自己的话解释AI Agent和LLM的主要区别。为什么LLM需要"Agent化"才能处理复杂任务？

**46.2** ReAct框架中的"思考→行动→观察"循环有什么优势？请举例说明为什么这种迭代式方法比一次性回答更有效。

**46.3** Tree of Thoughts与Chain-of-Thought的主要区别是什么？在什么情况下ToT会比CoT表现更好？

### 进阶题（选做）

**46.4** 阅读Reflexion论文（Shinn et al., 2023），分析自我反思机制如何帮助Agent从失败中学习。这种能力与人类学习有什么相似之处？

**46.5** MetaGPT使用SOP（标准操作程序）来规范多智能体协作。请设计一个简单的SOP流程，描述如何用3个Agent（研究员、写手、编辑）协作完成一篇研究博客文章。

**46.6** 在多智能体系统中，通信机制的选择（消息队列vs黑板系统）对系统性能有什么影响？什么场景适合消息队列，什么场景适合黑板系统？

### 挑战题（深入研究）

**46.7** 实现一个简单的代码生成Agent，能够：
- 接收自然语言描述（如"写一个计算阶乘的函数"）
- 生成Python代码
- 执行代码并验证结果
- 如果出错，分析错误并修复

提示：可以结合本章的ReAct框架和Reflexion的自反思机制。

**46.8** 设计一个多Agent系统来解决24点游戏。系统应包含：
- 一个生成候选计算的Agent
- 一个评估计算结果的Agent
- 一个决定是否继续搜索的Agent
- 使用黑板系统共享当前最佳方案

**46.9** 调研当前的Agent安全研究，选择一种攻击方式（如Prompt Injection、Tool Misuse），分析其原理和可能的防御措施。

---

## 46.9 参考文献

### 核心论文

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. *arXiv preprint arXiv:2210.03629*.

2. Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 35, 24824-24837.

3. Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y., & Narasimhan, K. (2023). Tree of Thoughts: Deliberate problem solving with large language models. *arXiv preprint arXiv:2305.10601*.

4. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Self-reflective agents. *arXiv preprint arXiv:2303.11366*.

5. Hong, S., Zheng, X., Chen, J., Cheng, Y., Wang, J., Zhang, C., Wang, Z., Yau, S. K. S., Lin, Z., Zhou, L., et al. (2023). MetaGPT: Meta programming for a multi-agent collaborative framework. *arXiv preprint arXiv:2308.00352*.

6. Wu, Q., Bansal, G., Zhang, J., Wu, Y., Zhang, S., Zhu, E., Li, B., Jiang, L., Zhang, X., & Wang, C. (2023). AutoGen: Enabling next-gen LLM applications via multi-agent conversation. *arXiv preprint arXiv:2308.08155*.

7. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. *arXiv preprint arXiv:2302.04761*.

8. Wooldridge, M., & Jennings, N. R. (1995). Intelligent agents: Theory and practice. *The Knowledge Engineering Review*, 10(2), 115-152.

9. Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology*, 1-22.

10. Qian, C., Cong, X., Yang, C., Chen, W., Su, Y., Xu, J., Liu, Z., & Sun, M. (2023). Communicative agents for software development. *arXiv preprint arXiv:2307.07924*.

### 推荐资源

- **AutoGPT**: https://github.com/Significant-Gravitas/AutoGPT
- **MetaGPT**: https://github.com/geekan/MetaGPT
- **AutoGen**: https://github.com/microsoft/autogen
- **LangChain Agents**: https://python.langchain.com/docs/modules/agents/
- **OpenAI Function Calling**: https://platform.openai.com/docs/guides/function-calling

---

*本章代码实现参考了上述论文的核心思想，教育目的使用。*
*完整代码可在配套代码仓库中找到：multi_agent_system.py, tree_of_thoughts.py, reflexion_agent.py*
"""
