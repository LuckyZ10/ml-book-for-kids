### 进阶练习

**练习4**：实现NSP任务

在预训练代码的基础上，添加下一句预测（NSP）任务。需要：
1. 修改数据集，生成50%连续的句子对和50%随机句子对
2. 添加NSP输出头
3. 修改损失函数为MLM + NSP

**练习5**：不同微调策略对比

实现并比较以下微调策略在情感分类任务上的表现：
- 完整微调
- 只微调顶层
- 逐层解冻
- LoRA (r=4, 8, 16)

记录每种方法的：
- 收敛速度（达到90%准确率需要的epoch）
- 最终准确率
- 训练时间和内存占用

**练习6**：灾难性遗忘实验

设计实验验证灾难性遗忘现象：
1. 先在任务A（如情感分析）上预训练/微调
2. 然后在任务B（如主题分类）上微调
3. 测试模型在任务A上的性能是否下降
4. 尝试使用EWC或其他正则化方法缓解遗忘

### 挑战题目

**挑战1**：实现一个简单的GPT模型

基于本章的Transformer代码，实现一个GPT风格的自回归语言模型（只使用解码器，因果掩码）。训练它在给定前文的情况下预测下一个字符。

**挑战2**：多任务学习

扩展分类器，使其能够同时处理多个任务（如情感分析、主题分类、语言识别），通过任务特定的提示或标签来区分不同任务。

**挑战3**：探索Prompt Tuning

实现Prompt Tuning：
1. 冻结整个预训练模型
2. 在输入前添加可学习的虚拟token
3. 只训练这些虚拟token的嵌入
4. 与LoRA的效果进行对比

---

## 九、本章小结

### 核心概念回顾

1. **预训练**（Pre-training）：在大规模无标注数据上学习通用表示
2. **微调**（Fine-tuning）：在特定任务数据上调整模型
3. **MLM**（Masked Language Model）：BERT的预训练目标，完形填空
4. **CLM**（Causal Language Model）：GPT的预训练目标，自回归预测
5. **参数高效微调**：LoRA、Adapter、Prompt Tuning等

### 重要公式

**BERT总损失**：
$$\mathcal{L}_{BERT} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$

**LoRA低秩适应**：
$$W = W_0 + BA$$

**对比学习损失**：
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(h_i, h_i^+) / \tau)}{\sum_j \exp(\text{sim}(h_i, h_j) / \tau)}$$

### 关键时间线

```
2013 - Word2Vec: 静态嵌入
2017 - Transformer: "Attention Is All You Need"
2018 - ELMo: 上下文相关词表示
2018 - ULMFiT: 预训练+微调范式确立
2018 - GPT-1: 生成式预训练
2018 - BERT: 双向编码器，NLP新时代
2019 - GPT-2: 零样本能力
2019 - RoBERTa: BERT优化版
2019 - DistilBERT: 知识蒸馏
2020 - GPT-3: 1750亿参数，少样本学习
2021 - LoRA: 参数高效微调
2022 - InstructGPT: RLHF
2023 - GPT-4: 多模态，推理能力飞跃
```

### 进一步阅读

**经典论文**：
1. Devlin et al. (2018) - BERT
2. Radford et al. (2018) - GPT-1
3. Radford et al. (2019) - GPT-2
4. Brown et al. (2020) - GPT-3
5. Liu et al. (2019) - RoBERTa
6. Hu et al. (2021) - LoRA

**推荐资源**：
- Hugging Face Transformers文档
- "Natural Language Processing with Transformers"（书籍）
- Stanford CS224N课程

---

## 参考文献
