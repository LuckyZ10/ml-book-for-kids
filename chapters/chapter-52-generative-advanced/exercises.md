## 练习题

### 基础概念题

**52.1** 一致性模型为什么能够实现单步生成？请用"时光机"比喻解释一致性约束的含义。

**52.2** 潜在扩散模型相比像素空间扩散模型的优势是什么？解释为什么VAE能够将图像压缩48倍而不会丢失关键信息。

**52.3** Classifier-Free Guidance是如何在"遵循文本提示"和"保持图像多样性"之间取得平衡的？如果guidance scale设置过大会有什么副作用？

### 数学推导题

**52.4** 证明耦合层的雅可比矩阵行列式可以高效计算。给定变换：
$$\mathbf{x}_{1:d} = \mathbf{z}_{1:d}, \quad \mathbf{x}_{d+1:D} = \mathbf{z}_{d+1:D} \odot \exp(s(\mathbf{z}_{1:d})) + t(\mathbf{z}_{1:d})$$
推导 $\log|\det \frac{\partial \mathbf{x}}{\partial \mathbf{z}}|$ 的表达式。

**52.5** 一致性模型的损失函数可以表示为：
$$\mathcal{L}(\theta, \theta^-) = \mathbb{E}_{t, t', \mathbf{x}_0, \mathbf{x}_t}\left[ d\left(f_\theta(\mathbf{x}_t, t), f_{\theta^-}(\mathbf{x}_{t'}, t')\right) \right]$$
请解释为什么需要使用EMA参数 $\theta^-$ 来计算目标，而不是直接使用 $\theta$？

**52.6** 证明Classifier-Free Guidance等价于对条件概率的对数进行缩放：
$$\log p_{\text{CFG}}(\mathbf{z} | \mathbf{c}) = \log p(\mathbf{z}) + w \cdot (\log p(\mathbf{z} | \mathbf{c}) - \log p(\mathbf{z}))$$
并解释这与贝叶斯定理的关系。

### 编程实践题

**52.7** 实现一个改进的一致性模型训练循环，支持：
- 渐进式噪声调度（从易到难）
- 自适应EMA更新频率
- 混合精度训练

**52.8** 实现潜在扩散模型的CFG采样，并比较不同guidance scale（1.0, 3.0, 7.5, 15.0）下的生成效果。编写代码可视化这些差异。

**52.9** 使用预训练的VAE实现潜在空间算术：给定文本提示"国王"、"女人"、"男人"，尝试生成"国王 - 男人 + 女人"对应的图像（即"女王"）。分析潜在空间是否真的有这种语义算术性质。

---

## 参考文献
