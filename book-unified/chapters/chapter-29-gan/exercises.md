# 第二十九章 练习题

## 题目说明

本章练习题共分为三个难度等级：
- **基础题 (3道)**：检验对核心概念的理解
- **进阶题 (3道)**：深入分析和应用
- **挑战题 (2道)**：拓展研究和创新思考

建议完成时间：基础题30分钟/题，进阶题60分钟/题，挑战题120分钟/题。

---

## 一、基础题 (Basic)

### 练习题 1：扩散过程计算 ⭐

**题目**：

给定一个DDPM模型，参数如下：
- 总时间步数 $T = 1000$
- 线性噪声调度：$\beta_t$ 从 $10^{-4}$ 线性增加到 $0.02$

对于一张归一化后的图像 $x_0$ (假设像素值在 $[-1, 1]$ 范围内)，请计算：

**(a)** 在时间步 $t = 500$ 时，$\bar{\alpha}_t$ 的值是多少？

**(b)** 如果在 $t = 500$ 时添加标准高斯噪声 $\epsilon \sim \mathcal{N}(0, I)$，加噪后的图像 $x_{500}$ 与原始图像 $x_0$ 的比例关系是什么？

**(c)** 在 $t = 999$ 时，$x_{999}$ 与纯噪声有多接近？(计算信噪比)

**提示**：
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$
- $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$

---

### 练习题 2：训练目标理解 ⭐

**题目**：

DDPM的训练目标被简化为噪声预测的均方误差：

$$\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

请回答：

**(a)** 为什么让网络预测噪声 $\epsilon$ 比直接预测原始图像 $x_0$ 更好？

**(b)** 在代码实现中，训练时是如何采样时间步 $t$ 的？为什么要这样设计？

**(c)** 假设网络预测的噪声是 $\epsilon_\theta$，如何从 $x_t$ 和 $\epsilon_\theta$ 恢复对 $x_0$ 的估计？写出公式。

---

### 练习题 3：采样过程分析 ⭐

**题目**：

DDPM的采样算法如下：

```
1. 从 N(0,I) 采样 x_T
2. 对于 t = T, T-1, ..., 1：
3.   如果 t > 1，采样 z ~ N(0,I)，否则 z = 0
4.   x_{t-1} = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) ε_θ(x_t,t)) + σ_t z
5. 返回 x_0
```

请回答：

**(a)** 解释第4行公式的含义。其中各项分别代表什么？

**(b)** 为什么当 $t = 1$ 时不添加噪声(即 $z = 0$)？

**(c)** 如果采样过程中完全不添加随机噪声(即令所有 $z = 0$)，会发生什么？这对应于哪种采样方法？

---

## 二、进阶题 (Advanced)

### 练习题 4：方差调度设计 ⭐⭐

**题目**：

DDPM论文中使用了线性方差调度，但后续研究提出了其他调度策略。

**(a)** 解释余弦调度(cosine schedule)的公式，并分析它相比线性调度的优势：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s = 0.008$。

**(b)** 绘制(或使用Python代码生成)线性调度和余弦调度下 $\sqrt{\bar{\alpha}_t}$ 和 $\sqrt{1 - \bar{\alpha}_t}$ 随时间变化的曲线，并分析两者在信号-噪声比例上的差异。

**(c)** 设计一个新的方差调度策略，并论证其可能的优势。

---

### 练习题 5：Classifier-Free Guidance 推导 ⭐⭐

**题目**：

CFG的核心公式为：

$$\hat{\epsilon}(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))$$

**(a)** 证明上述公式等价于：

$$\hat{\epsilon}(x_t, t, y) = (1 + w) \cdot \epsilon_\theta(x_t, t, y) - w \cdot \epsilon_\theta(x_t, t, \emptyset)$$

**(b)** 解释为什么CFG可以提高生成样本与条件的对齐度。提示：考虑贝叶斯定理和对数概率。

**(c)** 假设引导强度 $w = 7.5$，分析当 $w \to \infty$ 时会发生什么。这在实际应用中有什么限制？

---

### 练习题 6：DDIM采样算法实现 ⭐⭐

**题目**：

DDIM的核心更新公式为：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0^{\text{pred}} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t z$$

其中 $x_0^{\text{pred}} = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$

**(a)** 证明当 $\sigma_t = \sqrt{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$ 时，DDIM退化为DDPM。

**(b)** 解释为什么当 $\sigma_t = 0$ 时，DDIM变成确定性采样。

**(c)** 请用Python伪代码实现DDIM采样函数，支持可配置的子序列采样步数。

```python
def ddim_sample(model, shape, ddim_steps=50, eta=0.0):
    """
    DDIM采样实现
    
    参数:
        model: 训练好的噪声预测模型
        shape: 输出形状 (batch, channels, height, width)
        ddim_steps: 实际采样的时间步数
        eta: 随机性参数 (0为确定性)
    返回:
        生成的样本
    """
    # 请完成实现
    pass
```

---

## 三、挑战题 (Challenge)

### 练习题 7：与分数匹配的联系 ⭐⭐⭐

**题目**：

扩散模型与分数匹配(score matching)有密切的理论联系。

**(a)** 定义分数函数 $s(x) = \nabla_x \log p(x)$。证明对于前向过程 $q(x_t | x_0)$，有：

$$\nabla_{x_t} \log q(x_t | x_0) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}}$$

其中 $\epsilon$ 是添加到 $x_0$ 中的噪声。

**(b)** 解释为什么训练DDPM预测噪声 $\epsilon$ 等价于学习分数函数的缩放版本。

**(c)** Langevin动力学采样使用以下更新规则：

$$x_{t-1} = x_t + \frac{\eta}{2} \nabla_x \log p(x_t) + \sqrt{\eta} z$$

证明当选择合适的步长时，DDPM的采样过程等价于离散化的Langevin动力学。

**(d)** 讨论：分数匹配视角对理解扩散模型有什么优势？在设计新的生成模型时，这种联系如何指导我们？

---

### 练习题 8：潜在扩散模型的设计 ⭐⭐⭐

**题目**：

Stable Diffusion使用潜在扩散模型(LDM)在预训练VAE的潜在空间中执行扩散过程。

**(a)** 分析LDM相比像素空间扩散模型的优势和潜在缺点：
- 计算效率方面
- 生成质量方面
- 训练稳定性方面
- 应用领域限制方面

**(b)** 假设VAE的编码器将 $256 \times 256 \times 3$ 的图像压缩为 $32 \times 32 \times 4$ 的潜在表示，计算：
- 像素空间与潜在空间的维度比
- 单次U-Net前向传播在潜在空间的计算节省(假设计算量与空间像素数成正比)
- 如果原始DDPM需要1000步采样，LDM实际需要多少时间步才能达到相似效果？

**(c)** 设计实验：如何评估VAE重建质量对最终LDM生成质量的影响？提出至少3个评估指标和相应的实验方案。

**(d)** 创新思考：除了使用VAE，还有哪些方法可以获得适合扩散模型的紧凑表示？提出你的想法并分析可行性。

---

## 参考答案要点

### 练习题1

**(a)** 对于线性调度：
```python
beta_t = linspace(1e-4, 0.02, 1000)
alpha_t = 1 - beta_t
alpha_bar_500 = prod(alpha_t[:500]) ≈ 0.606
```

**(b)** 比例系数：
- $x_0$ 的系数：$\sqrt{\bar{\alpha}_{500}} \approx 0.778$
- $\epsilon$ 的系数：$\sqrt{1 - \bar{\alpha}_{500}} \approx 0.628$

**(c)** 信噪比：$\frac{\bar{\alpha}_{999}}{1 - \bar{\alpha}_{999}} \approx 0.0001$ (几乎全是噪声)

---

### 练习题3

**(c)** 当所有 $z = 0$ 时，采样变为确定性过程。这对应于**DDIM**的确定性采样模式，或者Flow模型的ODE采样。

---

### 练习题5

**(b)** 直观解释：
- $\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset)$ 表示条件$y$提供的"方向"
- 通过放大这个方向，采样过程被"推动"向更符合条件的区域
- 这与贝叶斯后验采样 $p(x|y) \propto p(x)p(y|x)$ 有理论联系

---

### 练习题6

**(c)** 伪代码要点：
```python
def ddim_sample(model, shape, ddim_steps=50, eta=0.0):
    # 选择子序列时间步
    c = total_steps // ddim_steps
    timesteps = list(range(0, total_steps, c)) + [total_steps - 1]
    
    x = randn(shape)
    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_prev = timesteps[i-1] if i > 0 else 0
        
        # 预测噪声
        eps = model(x, t)
        
        # 预测x0
        x0_pred = (x - sqrt(1-alpha_bar[t]) * eps) / sqrt(alpha_bar[t])
        
        # 计算方向
        sigma = eta * sqrt((1-alpha_bar[t_prev])/(1-alpha_bar[t]) * 
                          (1-alpha_bar[t]/alpha_bar[t_prev]))
        
        # 更新
        noise = randn_like(x) if i > 0 else 0
        x = sqrt(alpha_bar[t_prev]) * x0_pred + \
            sqrt(1-alpha_bar[t_prev]-sigma**2) * eps + \
            sigma * noise
    
    return x
```

---

## 编程实践建议

1. **实现最小DDPM**：在MNIST或CIFAR-10上实现简化版DDPM
2. **可视化扩散过程**：编写代码展示图像如何逐渐被噪声覆盖
3. **比较不同调度**：实验线性、余弦、二次方调度对生成质量的影响
4. **DDIM加速**：实现DDIM采样并测量生成时间
5. **CFG探索**：训练条件DDPM并实验不同引导强度的效果

---

*练习题编写时间：2026年3月*
