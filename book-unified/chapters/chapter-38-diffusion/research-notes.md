# 扩散模型核心论文研究笔记

> 为第38章重写准备的深度研究资料  
> 生成时间：2026-03-25

---

## 目录
1. [技术演进时间线](#1-技术演进时间线)
2. [核心论文深度解析](#2-核心论文深度解析)
3. [费曼比喻素材](#3-费曼比喻素材)
4. [数学推导要点](#4-数学推导要点)
5. [Stable Diffusion架构详解](#5-stable-diffusion架构详解)

---

## 1. 技术演进时间线

```
2015 ──────────────────────────────────────────────────────────────────────►
│
│  ICML 2015
│  Sohl-Dickstein et al.
│  "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
│  └── 扩散模型奠基之作，首次提出扩散概率模型框架
│
2019 ──────────────────────────────────────────────────────────────────────►
│
│  NeurIPS 2019
│  Song & Ermon
│  "Generative Modeling by Estimating Gradients of the Data Distribution"
│  └── Score-based生成模型（NCSN），引入分数匹配
│
2020 ──────────────────────────────────────────────────────────────────────►
│
│  NeurIPS 2020
│  Ho, Jain, Abbeel
│  "Denoising Diffusion Probabilistic Models (DDPM)"
│  └── 首个高质量图像生成扩散模型，与Score-based模型建立联系
│
2021 ──────────────────────────────────────────────────────────────────────►
│
│  ICLR 2021 (Jan)
│  Song, Meng, Ermon
│  "Denoising Diffusion Implicit Models (DDIM)"
│  └── 非马尔可夫采样，实现20-50倍加速
│
│  ICLR 2021 (May)
│  Song et al.
│  "Score-Based Generative Modeling through SDEs"
│  └── 统一框架，将DDPM和NCSN视为SDE的离散化
│
│  NeurIPS 2021
│  Dhariwal & Nichol
│  "Diffusion Models Beat GANs on Image Synthesis"
│  └── 扩散模型首次在图像生成质量上超越GAN
│
│  NeurIPS 2021 Workshop
│  Ho & Salimans
│  "Classifier-Free Diffusion Guidance"
│  └── 无需外部分类器的引导生成技术
│
2022 ──────────────────────────────────────────────────────────────────────►
│
│  CVPR 2022
│  Rombach et al.
│  "High-Resolution Image Synthesis with Latent Diffusion Models"
│  └── LDM/Stable Diffusion，潜空间扩散，大幅降低计算成本
│
│  Aug 2022
│  Stability AI发布Stable Diffusion 1.x
│  └── 首个开源大规模文本到图像生成模型
│
2023 ──────────────────────────────────────────────────────────────────────►
│
│  ICML 2023
│  Song, Dhariwal, Chen, Sutskever
│  "Consistency Models"
│  └── 单步生成，一致性模型，实现实时生成
│
2024+ ─────────────────────────────────────────────────────────────────────►
│
│  后续发展
│  ├── DiT (Diffusion Transformers)
│  ├── Flow Matching
│  └── 视频生成模型 (Sora等)
```

---

## 2. 核心论文深度解析

### 2.1 DDPM: Denoising Diffusion Probabilistic Models

**论文信息**：Ho et al., NeurIPS 2020

#### 核心贡献
1. **高质量图像生成**：首次证明扩散模型可以生成与GAN相媲美的高质量图像
2. **简化训练目标**：提出简化的均方误差损失，替代复杂的变分下界
3. **与Score-based模型建立联系**：证明DDPM与Score Matching的等价性

#### 关键公式

**前向过程（扩散）**：
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

**重参数化技巧**：
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$

**训练目标（简化版）**：
$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

**反向采样**：
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

#### 算法流程

```
训练算法：
1. 重复：
2.   x₀ ~ q(x₀)                # 从数据分布采样
3.   t ~ Uniform({1,...,T})     # 随机时间步
4.   ε ~ N(0,I)                 # 采样噪声
5.   计算 x_t = √ᾱ_t x₀ + √(1-ᾱ_t) ε
6.   梯度下降优化 ||ε - ε_θ(x_t, t)||²
7. 直到收敛

采样算法：
1. x_T ~ N(0,I)                # 从纯噪声开始
2. for t = T,...,1:
3.   z ~ N(0,I) (if t>1 else 0)
4.   x_{t-1} = (1/√α_t)[x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ(x_t,t)] + σ_t z
5. return x₀
```

---

### 2.2 Score-based Generative Models (NCSN)

**论文信息**：Song & Ermon, NeurIPS 2019; ICLR 2021 (SDE扩展)

#### 核心贡献
1. **分数匹配**：直接学习数据分布的分数函数（对数密度的梯度）
2. **多尺度噪声**：使用不同噪声水平的扰动数据训练多个分数模型
3. **Langevin采样**：通过Langevin动力学从学习的分数函数中采样

#### 关键公式

**分数函数**：
$$s_\theta(x) \approx \nabla_x \log p(x)$$

**去噪分数匹配目标**：
$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t} \left[ \lambda(t) \mathbb{E}_{x_0} \mathbb{E}_{x_t|x_0} \left[ \| s_\theta(x_t, t) - \nabla_{x_t} \log q(x_t|x_0) \|^2 \right] \right]$$

**Score与噪声的关系**：
$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

**Langevin采样**：
$$x_{i+1} = x_i + \epsilon \nabla_x \log p(x) + \sqrt{2\epsilon} z_i$$

---

### 2.3 Score SDE: Score-Based Generative Modeling through SDEs

**论文信息**：Song et al., ICLR 2021 (Oral)

#### 核心贡献
1. **统一框架**：将SMLD和DDPM统一到连续时间SDE框架
2. **Reverse-Time SDE**：利用Anderson的结果推导反向时间SDE
3. **Predictor-Corrector采样**：结合数值SDE求解器和MCMC校正
4. **Probability Flow ODE**：推导出等价的确定性ODE，支持精确似然计算

#### 关键公式

**前向SDE（一般形式）**：
$$dx = f(x, t)dt + g(t)dw$$

**VP-SDE（DDPM的连续版本）**：
$$dx = -\frac{1}{2}\beta(t)x dt + \sqrt{\beta(t)} dw$$

**VE-SDE（SMLD的连续版本）**：
$$dx = \sqrt{\frac{d[\sigma^2(t)]}{dt}} dw$$

**反向时间SDE**：
$$dx = \left[ f(x,t) - g(t)^2 \nabla_x \log p_t(x) \right] dt + g(t) d\bar{w}$$

**Probability Flow ODE**：
$$dx = \left[ f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x) \right] dt$$

#### SDE类型对比

| SDE类型 | 漂移项 f(x,t) | 扩散项 g(t) | 对应离散方法 |
|---------|--------------|-------------|-------------|
| VE-SDE | 0 | $\sqrt{d\sigma^2(t)/dt}$ | SMLD |
| VP-SDE | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)}$ | DDPM |
| Sub-VP SDE | $-\frac{1}{2}\beta(t)x$ | $\sqrt{\beta(t)(1-e^{-2\int_0^t \beta(s)ds})}$ | 新提出 |

---

### 2.4 LDM: Latent Diffusion Models

**论文信息**：Rombach et al., CVPR 2022

#### 核心贡献
1. **潜空间扩散**：在VAE压缩的潜空间而非像素空间进行扩散，计算量减少64倍
2. **跨注意力条件机制**：通过Transformer跨注意力注入文本/图像条件
3. **多模态条件**：支持文本、语义图、类别等多种条件
4. **开源生态**：Stable Diffusion成为最具影响力的开源生成模型

#### 关键公式

**VAE编码**：
$$z = \mathcal{E}(x), \quad x \in \mathbb{R}^{3 \times H \times W}, z \in \mathbb{R}^{4 \times (H/8) \times (W/8)}$$

**LDM训练目标**：
$$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y)) \|_2^2 \right]$$

**条件注入**：
- 文本编码器 $\tau_\theta$（如CLIP）将条件 $y$ 转换为嵌入序列
- UNet中的跨注意力层：$\text{Attention}(Q=\text{UNet特征}, K=V=\text{文本嵌入})$

#### 架构组件

| 组件 | 功能 | 关键参数 |
|------|------|----------|
| VAE Encoder | 图像→潜空间 | 4通道，8倍下采样 |
| VAE Decoder | 潜空间→图像 | 重建图像 |
| UNet | 去噪网络 | 约860M参数 |
| Text Encoder | 文本→嵌入 | CLIP ViT-L/14，77tokens |
| Cross-Attention | 条件注入 | 多头注意力机制 |

---

### 2.5 DDIM: Denoising Diffusion Implicit Models

**论文信息**：Song, Meng, Ermon, ICLR 2021

#### 核心贡献
1. **非马尔可夫扩散过程**：推广DDPM到非马尔可夫链，保持相同训练目标
2. **确定性采样**：通过设置方差为0，实现确定性ODE-like采样
3. **加速采样**：可跳过步骤，实现10-50倍加速，质量损失很小

#### 关键公式

**非马尔可夫前向过程**：
$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 I\right)$$

**DDIM采样（通用形式）**：
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon$$

其中 $\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}$

**确定性DDIM（σ=0）**：
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$

**ODE解释**：
DDIM可以看作是一个ODE的Euler离散化：
$$\frac{dx}{dt} = -\frac{1}{2}\beta(t)x - \frac{\beta(t)}{2\sqrt{\bar{\alpha}_t(1-\bar{\alpha}_t)}} \epsilon_\theta(x_t, t)$$

---

### 2.6 Classifier-Free Guidance (CFG)

**论文信息**：Ho & Salimans, NeurIPS 2021 Workshop

#### 核心贡献
1. **无需外部分类器**：通过训练时随机丢弃条件，实现无分类器引导
2. **灵活的引导强度**：通过guidance scale控制生成质量与多样性的权衡
3. **文本到图像的关键**：成为Stable Diffusion等模型的标准配置

#### 关键公式

**训练时条件丢弃**：
- 以概率 $p_{\text{uncond}}$ 将条件 $c$ 替换为空标记 $\emptyset$
- 单个模型同时学习条件生成和无条件生成

**CFG采样**：
$$\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot [\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset)]$$

或等价地：
$$\tilde{\epsilon}_\theta(x_t, t, c) = (1-w) \cdot \epsilon_\theta(x_t, t, \emptyset) + w \cdot \epsilon_\theta(x_t, t, c)$$

**其中**：
- $w$ 是guidance scale（通常7.5-10）
- $w=1$：标准条件生成
- $w>1$：增强条件引导，提高质量但降低多样性

#### 与Classifier Guidance对比

| 方法 | 需要额外分类器 | 训练方式 | 计算成本 | 效果 |
|------|--------------|----------|----------|------|
| Classifier Guidance | 是 | 单独训练分类器 | 2x（分类器+扩散） | 好 |
| Classifier-Free | 否 | 联合训练 | 2x（两次前向） | 更好 |

---

### 2.7 Consistency Models

**论文信息**：Song, Dhariwal, Chen, Sutskever, ICML 2023

#### 核心贡献
1. **单步生成**：直接学习噪声到数据的映射，支持单步生成
2. **一致性蒸馏**：将预训练扩散模型蒸馏为一致性模型
3. **一致性训练**：从零开始训练一致性模型
4. **零样本编辑**：支持图像修复、上色、超分辨率等零样本任务

#### 关键公式

**一致性函数定义**：
$$f: (x_t, t) \mapsto x_\epsilon$$

**自一致性性质**：
对于同一轨迹上的任意两点 $(x_t, t)$ 和 $(x_{t'}, t')$：
$$f(x_t, t) = f(x_{t'}, t') = x_\epsilon$$

**一致性蒸馏损失**：
$$\mathcal{L}_{\text{CD}}^N = \mathbb{E}\left[ \lambda(t_n) d(f_\theta(x_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat{x}_{t_n}, t_n)) \right]$$

其中：
- $\hat{x}_{t_n} = x_{t_{n+1}} - (t_n - t_{n+1}) \cdot t_{n+1} \cdot s_\phi(x_{t_{n+1}}, t_{n+1}})$ （使用预训练分数模型）
- $\theta^-$ 是EMA目标网络

**一致性训练损失**：
$$\mathcal{L}_{\text{CT}}^N = \mathbb{E}\left[ \lambda(t_n) d(f_\theta(x + t_{n+1}z, t_{n+1}), f_{\theta^-}(x + t_n z, t_n)) \right]$$

#### 性能表现

| 数据集 | 方法 | 1步FID | 2步FID |
|--------|------|--------|--------|
| CIFAR-10 | CD | 3.55 | 2.93 |
| ImageNet 64x64 | CD | 6.20 | 4.70 |
| LSUN Bedroom 256 | CD | 7.80 | 5.22 |

---

## 3. 费曼比喻素材

### 3.1 扩散过程比喻

#### 墨水扩散（经典物理）
> "想象一滴蓝色墨水滴入清水中。起初，墨水集中在一个点上，形成清晰的蓝色斑点。随着时间推移，墨水分子通过随机运动逐渐扩散，最终整杯水变成均匀的淡蓝色。"

**对应概念**：
- 清晰斑点 = 原始图像
- 随机运动 = 高斯噪声添加
- 均匀蓝色 = 纯噪声分布

#### 逆向雕塑
> "想象一个雕塑家从一块大理石开始工作。起初，石头只是一块粗糙的材料（像纯噪声）。雕塑家一步步地去除多余的部分，逐渐显现出雕像的形状。每一步，他都根据当前石头的状态决定下一刀该怎么切。"

**对应概念**：
- 大理石块 = 纯噪声
- 雕塑过程 = 反向扩散/去噪
- 雕塑家 = 神经网络（噪声预测器）

### 3.2 潜空间比喻

#### 地图压缩
> "想象你有一张巨大的详细地图（原始图像），但你发现每次查看或携带都非常不便。于是你创建了一张缩略图（潜空间表示），它保留了主要的地形特征和道路网络，但尺寸大大减小。当你需要时，可以根据缩略图还原出详细地图。"

**对应概念**：
- 详细地图 = 高分辨率图像（512×512×3）
- 缩略图 = 潜空间表示（64×64×4）
- 8倍压缩 = 计算量减少64倍

### 3.3 去噪比喻

#### 信号从噪声中浮现
> "想象你在调收音机，起初只能听到沙沙的静电噪音（纯噪声）。随着你慢慢调节，音乐旋律开始从噪音中浮现出来，越来越清晰，直到你听到完整的歌曲（生成的图像）。"

**对应概念**：
- 静电噪音 = 纯高斯噪声
- 调节过程 = 逐步去噪
- 完整歌曲 = 最终生成图像

### 3.4 分数函数比喻

#### 山坡上的球
> "想象一个球放在崎岖不平的山坡上。分数函数就像是告诉你球应该往哪个方向滚动的指南针——它指向最陡的下降方向。在生成模型中，这个'山坡'是数据分布的概率密度，分数函数指引我们如何从高概率区域（真实数据）向低概率区域（噪声）移动，或者反过来。"

**对应概念**：
- 山坡 = 概率密度 landscape
- 分数 = 对数密度的梯度 $\nabla_x \log p(x)$
- 滚动方向 = 采样方向

### 3.5 引导生成比喻

#### 导航系统
> "想象你在雾中开车（扩散过程），你想去一个特定的地方（条件/文本描述）。Classifier-Free Guidance就像是在车里同时使用GPS导航（条件路径）和随机驾驶（无条件路径）。你可以调节两者的比例——更相信GPS会得到更准确的路线，但可能错过一些有趣的风景；更随机驾驶则探索性更强，但可能偏离目标。"

**对应概念**：
- 雾中驾驶 = 生成过程
- GPS导航 = 条件生成
- 随机驾驶 = 无条件生成
- 调节比例 = guidance scale

---

## 4. 数学推导要点

### 4.1 DDPM前向过程的闭式解

**目标**：证明 $x_t$ 可以直接从 $x_0$ 采样，无需迭代

**推导**：
1. 定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$
2. 重参数化：$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1}$
3. 递归展开：
   $$x_t = \sqrt{\alpha_t}(\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}} \epsilon_{t-2}) + \sqrt{1-\alpha_t} \epsilon_{t-1}$$
4. 利用高斯分布的可加性：
   $$\sqrt{\alpha_t}\sqrt{1-\alpha_{t-1}} \epsilon_{t-2} + \sqrt{1-\alpha_t} \epsilon_{t-1} \sim \sqrt{1-\alpha_t\alpha_{t-1}} \bar{\epsilon}_{t-2}$$
5. 最终得到：
   $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$$

### 4.2 反向过程的变分下界

**目标**：推导DDPM的变分下界（VLB）

**关键步骤**：
1. 联合分布：$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$
2. 负对数似然：$-\log p_\theta(x_0) \leq -\mathbb{E}_q[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}]$
3. 展开后得到KL散度之和：
   $$\mathcal{L}_{\text{VLB}} = \mathbb{E}_q \left[ D_{\text{KL}}(q(x_T|x_0) \| p(x_T)) + \sum_{t>1} D_{\text{KL}}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t)) \right]$$
4. 由于两者都是高斯分布，KL散度有闭式解
5. 简化后得到均方误差损失

### 4.3 Score Matching与DDPM等价性

**目标**：证明预测噪声 $\epsilon_\theta$ 与学习分数函数等价

**推导**：
1. Score定义：$s(x_t, t) = \nabla_{x_t} \log p(x_t)$
2. 对于前向过程，条件分数：
   $$\nabla_{x_t} \log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$
3. 因此：$\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} s_\theta(x_t, t)$
4. 这意味着预测噪声等价于估计（缩放的）分数函数

### 4.4 Reverse-Time SDE推导

**Anderson 1982的结果**：

对于SDE：$dx = f(x,t)dt + g(t)dw$

反向时间SDE为：
$$dx = [f(x,t) - g(t)^2 \nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

**直观理解**：
- 反向过程需要"修正"正向过程的漂移
- 修正项包含分数函数，用于确保逆向过程回到正确的数据分布
- 扩散系数保持不变

### 4.5 CFG的数学解释

**贝叶斯分解**：
$$\nabla_x \log p(x|c) = \nabla_x \log p(x) + \nabla_x \log p(c|x)$$

**CFG近似**：
1. 分类器梯度近似：$\nabla_x \log p(c|x) \approx \frac{1}{w} [\epsilon_\theta(x,c) - \epsilon_\theta(x,\emptyset)]$
2. 因此：
   $$\nabla_x \log p(x|c) \approx \epsilon_\theta(x,\emptyset) + w[\epsilon_\theta(x,c) - \epsilon_\theta(x,\emptyset)]$$

**解释**：
- 无条件分数指向"任何真实图像"的方向
- 条件分数与无条件分数的差指向"满足条件的方向"
- Guidance scale放大这个方向，使生成更加符合条件

---

## 5. Stable Diffusion架构详解

### 5.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Stable Diffusion Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Text Prompt ──► Text Encoder ──► Text Embeddings (77×768)      │
│                                      │                          │
│                                      ▼                          │
│  Random Noise ──► UNet (w/ Cross-Attention) ──► Denoised Latent │
│       ▲                      ▲                                  │
│       │                      │                                  │
│   Timesteps              Text Embeddings                        │
│                                                                 │
│  Denoised Latent ──► VAE Decoder ──► Generated Image            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 VAE（变分自编码器）

**结构**：
- Encoder：4层下采样（ResNet + 下采样），将512×512×3 → 64×64×4
- Decoder：4层上采样（ResNet + 上采样），将64×64×4 → 512×512×3
- 后量化：潜空间离散化（VQ-VAE风格）

**关键参数**：
- 压缩比：8×8 = 64倍
- 潜空间维度：4通道
- 缩放因子：0.18215（标准化用）

### 5.3 UNet架构

**整体结构**：
```
Input (64×64×4)
    │
    ├──► DownBlock1 (64×64×320) ─────────────┐
    │      + Cross-Attention                    │
    │      + ResNet                             │
    ▼                                           │
Downsample (32×32×320)                          │
    │                                           │
    ├──► DownBlock2 (32×32×640) ──────────────┤
    │      + Cross-Attention                    │   Skip
    │      + ResNet                             │   Connections
    ▼                                           │
Downsample (16×16×640)                          │
    │                                           │
    ├──► DownBlock3 (16×16×1280) ─────────────┤
    │      + Cross-Attention                    │
    │      + ResNet                             │
    ▼                                           │
Downsample (8×8×1280)                          │
    │                                           │
    └──► MidBlock (8×8×1280) ◄─────────────────┘
              + Self-Attention
              + ResNet
              ▲
    ┌─────────┴─────────────────────────────────┐
    │                                           │
Upsample (16×16×1280)                          │
    │                                           │
    ├──► UpBlock1 (16×16×1280) ◄───────────────┘
    │      + Cross-Attention
    │      + ResNet
    ▼
Upsample (32×32×640)
    │
    ├──► UpBlock2 (32×32×640)
    │      + Cross-Attention
    │      + ResNet
    ▼
Upsample (64×64×320)
    │
    └──► UpBlock3 (64×64×320)
              + Cross-Attention
              + ResNet
              ▼
Output (64×64×4)
```

**关键组件**：

1. **ResNet Block**：
   - GroupNorm + SiLU + Conv3×3
   - 时间步嵌入注入（通过MLP）
   - 残差连接

2. **Cross-Attention Block**：
   ```
   Q = Linear(UNet特征)
   K = Linear(文本嵌入)
   V = Linear(文本嵌入)
   Attention = Softmax(Q·K^T / √d) · V
   ```

3. **Self-Attention Block**：
   - 位于UNet中间层（bottleneck）
   - 捕获全局空间关系

### 5.4 Text Encoder

**Stable Diffusion 1.x**：CLIP ViT-L/14 Text Encoder
- 输入：文本token序列（最大77个token）
- 输出：文本嵌入（77×768）
- 架构：12层Transformer

**Stable Diffusion 2.x**：OpenCLIP ViT-H/14
- 输出维度：1024
- 更大容量，更好的文本理解

### 5.5 训练和推理流程

**训练流程**：
1. 图像 $x$ 经过VAE编码器得到 $z = \mathcal{E}(x)$
2. 文本 $y$ 经过Text Encoder得到 $\tau_\theta(y)$
3. 随机采样时间步 $t$ 和噪声 $\epsilon$
4. 计算加噪潜变量：$z_t = \sqrt{\bar{\alpha}_t}z + \sqrt{1-\bar{\alpha}_t}\epsilon$
5. UNet预测噪声：$\epsilon_\theta(z_t, t, \tau_\theta(y))$
6. 优化MSE损失：$\mathcal{L} = \|\epsilon - \epsilon_\theta\|^2$

**推理流程**：
1. 从标准高斯分布采样 $z_T$
2. 文本编码得到条件嵌入
3. 对于 $t = T, ..., 1$：
   - UNet预测噪声（使用CFG）
   - 根据采样器（DDPM/DDIM/DPM++等）更新 $z_{t-1}$
4. VAE解码器生成图像：$x = \mathcal{D}(z_0)$

### 5.6 条件注入机制

**Cross-Attention详细过程**：
```python
# UNet特征: [batch, seq_len_unet, dim]
# 文本嵌入: [batch, seq_len_text, text_dim]

# 1. 投影到相同维度
Q = linear_q(unet_features)  # [batch, seq_unet, head_dim * n_heads]
K = linear_k(text_embeddings) # [batch, seq_text, head_dim * n_heads]
V = linear_v(text_embeddings) # [batch, seq_text, head_dim * n_heads]

# 2. 多头分割
Q = Q.view(batch, seq_unet, n_heads, head_dim).transpose(1, 2)
K = K.view(batch, seq_text, n_heads, head_dim).transpose(1, 2)
V = V.view(batch, seq_text, n_heads, head_dim).transpose(1, 2)

# 3. 注意力计算
attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(head_dim)
attn_weights = F.softmax(attn_scores, dim=-1)
attn_output = torch.matmul(attn_weights, V)

# 4. 合并多头
attn_output = attn_output.transpose(1, 2).contiguous()
attn_output = attn_output.view(batch, seq_unet, dim)

# 5. 残差连接
output = unet_features + linear_out(attn_output)
```

---

## 参考资料

### 核心论文
1. Sohl-Dickstein et al. (2015) - Deep Unsupervised Learning using Nonequilibrium Thermodynamics
2. Song & Ermon (2019) - Generative Modeling by Estimating Gradients of the Data Distribution
3. Ho et al. (2020) - Denoising Diffusion Probabilistic Models
4. Song et al. (2020) - Denoising Diffusion Implicit Models
5. Song et al. (2021) - Score-Based Generative Modeling through SDEs
6. Dhariwal & Nichol (2021) - Diffusion Models Beat GANs on Image Synthesis
7. Ho & Salimans (2022) - Classifier-Free Diffusion Guidance
8. Rombach et al. (2022) - High-Resolution Image Synthesis with Latent Diffusion Models
9. Song et al. (2023) - Consistency Models

### 技术博客与资源
- Yang Song's Blog: https://yang-song.net/blog/2021/score/
- Lilian Weng's Blog: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- Hugging Face Diffusers Documentation
- Stable Diffusion Paper and Code

---

*本研究笔记为《机器学习：写给中小学生的AI启蒙书》第38章重写准备*
