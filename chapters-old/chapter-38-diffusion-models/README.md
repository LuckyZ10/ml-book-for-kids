

---

## 38.9 应用场景与前沿方向

### 38.9.1 图像生成

扩散模型在图像生成领域取得了革命性突破。

```python
class ImageGenerationApplications:
    """图像生成应用概览"""
    
    @staticmethod
    def text_to_image_models():
        """
        文本到图像生成模型
        
        1. DALL-E 2 (OpenAI, 2022)
           - 架构: CLIP + GLIDE (扩散模型)
           - 特点: 生成高质量、多样化图像
           - 分辨率: 1024×1024
           - 创新: unCLIP，使用CLIP特征作为中间表示
        
        2. Stable Diffusion (Stability AI, 2022)
           - 架构: LDM + CFG
           - 特点: 开源，可在消费级GPU运行
           - 影响: 推动了AI艺术民主化
        
        3. Imagen (Google, 2022)
           - 架构: 大型语言模型 + 级联扩散
           - 特点: 使用T5-XXL文本编码器
           - 突破: DrawBench评测领先
        
        4. Midjourney
           - 特点: 专注于艺术风格
           - 应用: Discord社区驱动
           - 影响: 定义了AI艺术美学
        """
        pass
    
    @staticmethod
    def image_editing_applications():
        """
        图像编辑应用
        
        • Inpainting: 填充图像缺失区域
        • Outpainting: 扩展图像边界
        • Style Transfer: 风格迁移
        • Super-resolution: 超分辨率
        • Colorization: 黑白照片上色
        """
        pass
    
    @staticmethod
    def personalization():
        """
        个性化生成
        
        • DreamBooth: 用3-5张图片学习新概念
        • LoRA: 低秩适配，高效微调
        • Textual Inversion: 学习新token表示
        • ControlNet: 精细控制生成过程
        """
        pass

print("\n" + "=" * 70)
print("应用场景与前沿方向")
print("=" * 70)

image_gen_section = """
╔═══════════════════════════════════════════════════════════════════════╗
║                       图像生成应用                                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  文本到图像生成:                                                       ║
║  ───────────────                                                      ║
║                                                                       ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       ║
║  │    DALL-E 2     │  │ Stable Diffusion│  │     Imagen      │       ║
║  │   (OpenAI)      │  │  (开源)         │  │   (Google)      │       ║
║  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤       ║
║  │ • CLIP+GLIDE    │  │ • LDM+CFG       │  │ • T5-XXL+级联   │       ║
║  │ • 1024×1024     │  │ • 512×512       │  │ • 1024×1024     │       ║
║  │ • API访问       │  │ • 开源免费      │  │ • 闭源          │       ║
║  └─────────────────┘  └─────────────────┘  └─────────────────┘       ║
║                                                                       ║
║  图像编辑:                                                            ║
║  ────────                                                             ║
║  • Inpainting:  智能填充（去除水印、修复老照片）                        ║
║  • Outpainting: 边界扩展（调整构图、生成全景图）                        ║
║  • ControlNet:  精确控制（姿态、边缘、深度图）                          ║
║                                                                       ║
║  个性化:                                                              ║
║  ───────                                                              ║
║  • DreamBooth:  "我的狗" → 特定概念学习                                ║
║  • LoRA:        轻量级微调（仅训练几MB参数）                            ║
║  • IP-Adapter:  图像提示适配器                                         ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
print(image_gen_section)
```

### 38.9.2 视频生成

```python
class VideoGeneration:
    """视频生成应用"""
    
    @staticmethod
    def sora_description():
        """
        Sora (OpenAI, 2024)
        
        突破性进展：
        • 生成长达60秒的高质量视频
        • 支持复杂场景、多角度镜头
        • 理解物理世界规律
        • 文本到视频、图像到视频
        
        技术架构（推测）：
        • 时空patch化
        • 视频作为3D数据（空间+时间）
        • 类DiT (Diffusion Transformer)架构
        • 大规模训练（推测）
        
        影响：
        • 影视制作、游戏开发、虚拟现实
        • 世界模型(World Model)研究
        """
        pass
    
    @staticmethod
    def other_video_models():
        """
        其他视频生成模型：
        
        • Video Diffusion Models (2022)
        • Make-A-Video (Meta, 2022)
        • Imagen Video (Google, 2022)
        • AnimateDiff (2023)
        • VideoCrafter (2023)
        • Lumiere (Google, 2024)
        """
        pass

video_section = """
视频生成:

Sora (OpenAI, 2024) - 里程碑式突破
─────────────────────────────────────
能力:
• 60秒连续视频生成
• 复杂场景理解（物理规律、因果关系）
• 多镜头切换
• 角色一致性

技术特点:
• 时空联合建模
• 大规模扩展
• 视频压缩表示

应用:
• 影视前期制作
• 游戏场景生成
• 虚拟现实内容
• 世界模型研究

其他模型:
• AnimateDiff: 为SD添加动画能力
• Lumiere: 时空U-Net，一次生成全视频
• VideoCrafter: 开源视频生成
"""
print(video_section)
```

### 38.9.3 3D生成

```python
class ThreeDGeneration:
    """3D生成应用"""
    
    @staticmethod
    def dreamfusion():
        """
        DreamFusion (Google, 2022)
        
        核心创新：
        • 使用2D扩散模型生成3D内容
        • Score Distillation Sampling (SDS)
        • 无需3D训练数据
        
        工作原理：
        1. 随机初始化3D表示（NeRF）
        2. 渲染2D图像
        3. 用2D扩散模型打分
        4. 优化3D表示以匹配文本描述
        """
        pass
    
    @staticmethod
    def point_e():
        """
        Point-E (OpenAI, 2022)
        
        • 文本/图像到3D点云
        • 两阶段生成：先GLIDE生成视图，再点云扩散
        • 快速生成（1-2分钟）
        """
        pass

three_d_section = """
3D生成:

DreamFusion (2022)
──────────────────
突破: 用2D扩散模型指导3D生成（无需3D数据）
方法: Score Distillation Sampling
输出: 3D NeRF表示
应用: 3D资产创作、AR/VR内容

Point-E (OpenAI)
────────────────
输入: 文本或图像
输出: 3D点云
速度: 1-2分钟
优势: 快速、可直接3D打印

Magic3D (NVIDIA)
────────────────
改进: 两阶段优化（低分辨率→高分辨率）
质量: 超越DreamFusion
速度: 更快收敛

应用前景:
• 游戏3D资产
• AR/VR内容
• 3D打印模型
• 建筑可视化
"""
print(three_d_section)
```

### 38.9.4 其他应用

```python
class OtherApplications:
    """其他扩散模型应用"""
    
    @staticmethod
    def audio_generation():
        """
        音频生成:
        • AudioLDM: 文本到音频
        • MusicLM (Google): 文本到音乐
        • Noise2Music: 噪声到音乐
        • VoiceBox (Meta): 语音生成
        """
        pass
    
    @staticmethod
    def molecule_design():
        """
        分子设计:
        • Diffusion models for molecule generation
        • 3D分子构象生成
        • 药物发现
        """
        pass
    
    @staticmethod
    def robotics():
        """
        机器人策略 (Diffusion Policy):
        • 行为克隆作为去噪
        • 多模态动作分布
        • 比GMM和VAE更好
        """
        pass

other_apps = """
其他前沿应用:

音频生成:
─────────
• AudioLDM: 文本→音效/音乐
• MusicLM: 高保真音乐生成
• VoiceBox: 多语言语音合成

分子设计:
─────────
• 3D分子生成
• 药物发现加速
• 材料科学

机器人策略 (Diffusion Policy):
──────────────────────────────
• 将行为克隆建模为去噪过程
• 处理多模态动作分布
• 优于传统模仿学习

数据增强:
─────────
• 合成训练数据
• 隐私保护数据生成
• 少样本学习
"""
print(other_apps)
```

### 38.9.5 前沿研究方向

```python
class ResearchFrontiers:
    """扩散模型前沿研究方向"""
    
    @staticmethod
    def frontiers():
        """
        前沿方向:
        
        1. 高效采样
           • 一步/少步生成
           • 一致性模型
           • 蒸馏技术
        
        2. 可控生成
           • 更精细的控制
           • 多条件组合
           • 实时交互
        
        3. 多模态统一
           • 文本、图像、视频、3D统一
           • 任意模态到任意模态
        
        4. 世界模型
           • 理解物理规律
           • 因果推理
           • 长期预测
        
        5. 安全与伦理
           • 深度伪造检测
           • 内容溯源
           • 负责任AI
        """
        pass

frontiers_section = """
╔═══════════════════════════════════════════════════════════════════════╗
║                    扩散模型前沿研究方向                                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  1. 高效采样                                                          ║
║  ─────────                                                           ║
║  • 目标: 单步或几步生成高质量内容                                       ║
║  • 方法: Consistency Models, 蒸馏, ODE求解器优化                        ║
║  • 挑战: 保持质量同时大幅降低计算成本                                    ║
║                                                                       ║
║  2. 可控生成                                                          ║
║  ─────────                                                           ║
║  • 目标: 更精确、更细粒度的控制                                         ║
║  • 方法: ControlNet, T2I-Adapter, 布局控制                              ║
║  • 应用: 专业设计、影视制作                                             ║
║                                                                       ║
║  3. 多模态统一                                                        ║
║  ───────────                                                         ║
║  • 目标: 统一处理文本、图像、视频、3D、音频                              ║
║  • 方法: 统一扩散框架，共享表示空间                                      ║
║  • 愿景: 任意模态→任意模态                                              ║
║                                                                       ║
║  4. 世界模型 (World Models)                                            ║
║  ─────────────────────────                                           ║
║  • 目标: 理解物理世界，预测未来                                          ║
║  • 方向: 视频预测、物理模拟、因果推理                                    ║
║  • 意义: 迈向通用人工智能的重要一步                                      ║
║                                                                       ║
║  5. 安全与伦理                                                         ║
║  ───────────                                                         ║
║  • 深度伪造检测与防御                                                   ║
║  • 内容溯源 (水印技术)                                                  ║
║  • 偏见消除与公平性                                                     ║
║  • 版权保护                                                             ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
print(frontiers_section)
```

---

## 38.10 练习题

### 基础概念题

**练习题 38.1**

解释扩散模型与GAN、VAE的本质区别。为什么扩散模型在训练稳定性方面优于GAN？

<details>
<summary>点击查看答案</summary>

**答案要点:**

1. **训练目标不同:**
   - GAN使用对抗损失（minimax游戏），训练不稳定
   - VAE使用ELBO，包含重建项和KL散度
   - 扩散模型使用简单的MSE损失（预测噪声），训练非常稳定

2. **生成过程不同:**
   - GAN/VAE是单步生成
   - 扩散模型是多步迭代去噪

3. **训练稳定性:**
   - GAN需要平衡生成器和判别器，容易崩溃
   - 扩散模型损失函数简单，梯度稳定
   - 扩散模型每一步都是监督学习，易于优化

</details>

**练习题 38.2**

什么是重参数化技巧？为什么在DDPM的前向过程中它如此重要？

<details>
<summary>点击查看答案</summary>

**答案要点:**

**重参数化技巧:**
```
x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε,  ε ~ N(0,I)
```

**重要性:**
1. **计算效率:** 避免逐步采样，直接从x_0计算任意t的x_t
2. **训练效率:** 可以在任意时间步训练，增加数据多样性
3. **数学等价:** 与马尔可夫链逐步加噪数学等价
4. **实现简洁:** 单次前向计算，无需循环

**对比:**
- 无重参数化: 需要T步循环，O(T)复杂度
- 有重参数化: 一步计算，O(1)复杂度

</details>

**练习题 38.3**

解释Classifier-Free Guidance (CFG)的工作原理。为什么在推理时使用CFG可以提高生成质量？

<details>
<summary>点击查看答案</summary>

**答案要点:**

**CFG工作原理:**
1. **训练阶段:** 以概率p_uncond随机丢弃条件，让模型同时学会无条件和条件生成
2. **推理阶段:** 同时计算无条件和条件预测，外推增强条件效果

**公式:**
```
ε_guided = ε_uncond + w·(ε_cond - ε_uncond)
```

**为什么有效:**
1. **放大条件信号:** w>1时，增强条件影响
2. **抑制无条件偏差:** 去除无关的"平均"模式
3. **平衡质量与多样性:** 通过w调节（通常7.5-10）

**优势:**
- 无需额外分类器
- 训练开销小
- 效果通常优于Classifier Guidance

</details>

### 数学推导题

**练习题 38.4**

推导DDPM中从噪声预测到分数函数的转换关系。

<details>
<summary>点击查看答案</summary>

**推导过程:**

给定:
- 噪声预测: ε_θ(x_t, t)
- 分数函数: s_θ(x_t, t) = ∇_{x_t} log p(x_t)

**步骤1:** 前向过程
```
x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
```

**步骤2:** 扰动分布
```
q(x_t | x_0) = N(x_t; √(ᾱ_t)·x_0, (1-ᾱ_t)I)
```

**步骤3:** 计算分数（对数密度梯度）
```
log q(x_t | x_0) = -||x_t - √(ᾱ_t)·x_0||²/(2(1-ᾱ_t)) + const

∇_{x_t} log q(x_t | x_0) = -(x_t - √(ᾱ_t)·x_0)/(1-ᾱ_t)
                         = -√(1-ᾱ_t)·ε/(1-ᾱ_t)
                         = -ε/√(1-ᾱ_t)
```

**步骤4:** 代入x_0表达式
由 x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε，得:
```
ε = (x_t - √(ᾱ_t)·x_0)/√(1-ᾱ_t)
```

**结论:**
```
s(x_t, t) = -ε_θ(x_t, t) / √(1-ᾱ_t)

ε_θ(x_t, t) = -s(x_t, t) · √(1-ᾱ_t)
```

证毕。

</details>

**练习题 38.5**

证明DDPM的简化损失函数 L_simple 是变分下界的一个加权形式。

<details>
<summary>点击查看答案</summary>

**证明:**

**完整的变分下界:**
```
ELBO = L_T + Σ_{t=2}^T L_{t-1} + L_0
```

其中 L_{t-1} 是KL散度:
```
L_{t-1} = KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))
```

**两个高斯分布:**
- q(x_{t-1}|x_t,x_0) = N(x_{t-1}; μ̃_t, β̃_t I)
- p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)

**KL散度:**
```
KL = (1/2σ_t²)||μ̃_t - μ_θ||² + const
```

**参数化:**
```
μ_θ = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε_θ)
```

**代入:**
```
||μ̃_t - μ_θ||² = C·||ε - ε_θ||²
```

其中C是与θ无关的常数。

**简化损失:**
```
L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t,t)||²]
```

这正是ELBO中各项（除常数项）的加权和！

</details>

**练习题 38.6**

推导DDIM的确定性采样公式，解释为什么当η=0时采样是确定性的。

<details>
<summary>点击查看答案</summary>

**推导:**

**非马尔可夫前向过程:**
```
q_σ(x_{t-1}|x_t,x_0) = N(x_{t-1}; √ᾱ_{t-1}·x_0 + √(1-ᾱ_{t-1}-σ_t²)·(x_t-√ᾱ_t·x_0)/√(1-ᾱ_t), σ_t²I)
```

**反向过程（去噪）:**
```
p_θ(x_{t-1}|x_t) = q_σ(x_{t-1}|x_t, (x_t-√(1-ᾱ_t)·ε_θ)/√ᾱ_t)
```

**均值:**
```
μ_θ = √ᾱ_{t-1}·(x_t-√(1-ᾱ_t)·ε_θ)/√ᾱ_t + √(1-ᾱ_{t-1}-σ_t²)·ε_θ
```

**DDIM更新:**
```
x_{t-1} = √ᾱ_{t-1}·x_0^{pred} + √(1-ᾱ_{t-1})·ε_θ + σ_t·ε

其中:
x_0^{pred} = (x_t - √(1-ᾱ_t)·ε_θ)/√ᾱ_t
σ_t = η·√[(1-ᾱ_{t-1})/(1-ᾱ_t)]·√[1-ᾱ_t/ᾱ_{t-1}]
```

**当η=0时:**
```
σ_t = 0
x_{t-1} = √ᾱ_{t-1}·x_0^{pred} + √(1-ᾱ_{t-1})·ε_θ
```

此时无随机项，采样完全由模型决定，是确定性的！

</details>

### 编程挑战题

**练习题 38.7**

实现一个简化的DDPM训练循环，使用MNIST数据集训练一个生成手写数字的扩散模型。

<details>
<summary>点击查看提示</summary>

**提示:**

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 准备数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 2. 定义简单的U-Net（适合28x28图像）
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 实现编码器-解码器结构
        # 时间步嵌入
        pass
    
    def forward(self, x, t):
        # 返回预测噪声
        pass

# 3. 定义DDPM调度器
# 参考本章代码实现

# 4. 训练循环
# 参考本章train_ddpm.py

# 5. 采样并可视化结果
```

</details>

**练习题 38.8**

实现Classifier-Free Guidance（CFG）训练。修改你的DDPM模型，支持条件生成并在推理时使用CFG。

<details>
<summary>点击查看提示</summary>

**提示:**

```python
class CFGDDPM:
    def train_step(self, x0, y, optimizer):
        # 以概率p_uncond随机丢弃条件
        mask = torch.rand(len(y)) > self.p_uncond
        y_masked = y.clone()
        y_masked[~mask] = -1  # -1表示无条件
        
        # 正常训练...
        
    def sample_cfg(self, shape, y, guidance_scale=7.5):
        # 1. 无条件预测
        eps_uncond = self.model(x, t, torch.full_like(y, -1))
        
        # 2. 条件预测
        eps_cond = self.model(x, t, y)
        
        # 3. CFG
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        
        # 4. 更新x
        # ...
```

**挑战:**
- 在MNIST上训练类别条件模型
- 测试不同guidance_scale的效果
- 比较CFG vs 无CFG的生成质量

</details>

**练习题 38.9**

实现DDIM采样器，并与DDPM采样器比较生成速度和质量。

<details>
<summary>点击查看提示</summary>

**提示:**

```pythonnclass DDIMSampler:
    def __init__(self, model, num_train_timesteps=1000):
        self.model = model
        self.num_train_timesteps = num_train_timesteps
    
    def sample(self, shape, num_inference_steps=50, eta=0.0):
        # 设置推理时间步（均匀间隔子集）
        timesteps = torch.linspace(
            self.num_train_timesteps-1, 0, num_inference_steps
        ).long()
        
        x = torch.randn(shape)
        
        for t in timesteps:
            # 预测噪声
            eps = self.model(x, t)
            
            # 计算x_0预测
            alpha_t = self.alphas_cumprod[t]
            x0_pred = (x - torch.sqrt(1-alpha_t)*eps) / torch.sqrt(alpha_t)
            
            # 计算下一个alpha
            # ...
            
            # DDIM更新
            x = torch.sqrt(alpha_next) * x0_pred + ...
        
        return x
```

**实验要求:**
1. 在相同模型上，比较DDPM(1000步)和DDIM(50步)的FID分数
2. 测试不同eta值(0, 0.5, 1.0)对生成质量的影响
3. 测量并比较采样时间

</details>

---

## 参考文献

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *Advances in Neural Information Processing Systems*, 33, 6840-6851.

2. Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., & Ganguli, S. (2015). Deep unsupervised learning using nonequilibrium thermodynamics. *International Conference on Machine Learning*, 2256-2265.

3. Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *Advances in Neural Information Processing Systems*, 32.

4. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *International Conference on Learning Representations*.

5. Nichol, A. Q., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *International Conference on Machine Learning*, 8162-8171.

6. Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. *Advances in Neural Information Processing Systems*, 34, 8780-8794.

7. Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *arXiv preprint arXiv:2207.12598*.

8. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 10684-10695.

9. Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *International Conference on Learning Representations*.

10. Salimans, T., & Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. *International Conference on Learning Representations*.

11. Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency models. *International Conference on Machine Learning*, 32211-32252.

12. Luo, S., Tan, Y., Patil, S., Gu, D., von Platen, P., Passos, A., ... & Zhao, B. (2023). LCM-LoRA: A universal stable-diffusion acceleration module. *arXiv preprint arXiv:2311.05556*.

---

## 本章总结

**第三十八章 扩散模型 - 核心要点回顾:**

```
╔═══════════════════════════════════════════════════════════════════════╗
║                         本章知识图谱                                   ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  基础概念                                                              ║
║  ────────                                                             ║
║  • 正向扩散：马尔可夫链加噪 q(x_t|x_{t-1})                              ║
║  • 反向去噪：学习 p_θ(x_{t-1}|x_t)                                      ║
║  • 重参数化：从x_0直接采样x_t                                           ║
║  • 简化损失：预测噪声的MSE目标                                           ║
║                                                                       ║
║                              ↓                                        ║
║                                                                       ║
║  三个视角的统一                                                         ║
║  ─────────────                                                         ║
║  • DDPM：离散时间，变分推断                                             ║
║  • Score-based：分数函数，Langevin动力学                                 ║
║  • Score SDE：连续时间，统一框架                                         ║
║                                                                       ║
║                              ↓                                        ║
║                                                                       ║
║  条件生成                                                              ║
║  ────────                                                             ║
║  • Classifier Guidance：分类器梯度引导                                   ║
║  • CFG：无需分类器的引导（革命性）                                       ║
║  • CLIP+Diffusion：文本到图像                                            ║
║                                                                       ║
║                              ↓                                        ║
║                                                                       ║
║  高效实现                                                              ║
║  ────────                                                             ║
║  • LDM：潜在空间扩散（4-64倍加速）                                       ║
║  • DDIM：确定性采样（50步≈1000步质量）                                   ║
║  • Consistency Models：单步生成                                          ║
║                                                                       ║
║                              ↓                                        ║
║                                                                       ║
║  前沿应用                                                              ║
║  ────────                                                             ║
║  • 图像：DALL-E, Stable Diffusion, Midjourney                           ║
║  • 视频：Sora, Video Diffusion                                          ║
║  • 3D：DreamFusion, Point-E                                             ║
║  • 其他：音频、分子、机器人                                              ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
```

**关键公式速查:**

| 公式 | 说明 |
|------|------|
| $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$ | 重参数化采样 |
| $L_{\text{simple}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$ | 简化损失 |
| $s(x,t) = -\epsilon_\theta(x,t) / \sqrt{1-\bar{\alpha}_t}$ | 噪声→分数转换 |
| $\epsilon_{\text{guided}} = \epsilon_{\text{uncond}} + w \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})$ | CFG公式 |

**实践建议:**
1. 从简单的无条件DDPM开始，使用MNIST/CIFAR-10
2. 逐步添加条件生成、CFG等功能
3. 使用预训练模型（Stable Diffusion）进行微调
4. 关注高效采样技术（DDIM、LCM）
5. 尝试多模态应用（图像编辑、视频生成）

---

*本章完*
