

# 第五十二章：生成模型进阶——从扩散模型到一致性模型

> *"如果生成是一步步去除噪声的过程，那为什么不能一步到位？"*

---

## 52.1 引言：生成模型的进化之路

在上一章中，我们探索了扩散模型如何通过学习逆转噪声过程来生成逼真图像。这种"渐进式去噪"的方法虽然强大，但存在采样速度慢的根本性问题。本章我们将深入研究生成模型的最新进展，特别关注**一致性模型**（Consistency Models）如何通过单步或少步生成来解决这一困境。

### 52.1.1 从多步到单步：效率与质量的平衡

**扩散模型的困境**：
- 需要50-1000步迭代才能生成高质量图像
- 每一步都是一次完整的神经网络前向传播
- 实时应用受到严重限制

**一致性模型的突破**：
- 理论上只需**1步**即可生成
- 实践中通常使用**2-4步**达到高质量
- 在图像质量与推理速度之间找到新的平衡点

### 52.1.2 本章路线图

```
52.1 引言：生成模型的进化之路
52.2 一致性模型：穿越时间的魔法
52.3 一致性蒸馏：从教师到学生
52.4 一致性训练：从零开始学习
52.5 应用与展望
```

---

### 52.2.1 扩散模型的慢速采样困境

在上一章中，我们学习了扩散模型如何通过"去噪"过程生成图像。这个过程如同一位画家，从一张全是噪声的画布开始，经过50步、100步甚至1000步的精心雕琢，最终呈现出一幅精美的作品。

但这种逐步细化的方式带来了一个根本性的矛盾：**质量需要步骤的积累，而效率要求步骤的压缩**。

想象你正在使用一个AI绘画工具生成一张猫咪的图片。如果使用DDPM，你可能需要等待1000步才能完成——每一步都是一次神经网络前向传播。即使使用DDIM这样的加速采样器，通常也需要20-50步才能获得满意的质量。

这种延迟在以下场景中是无法接受的：
- **实时交互**：用户拖动滑块时，图像需要即时响应
- **视频生成**：每秒需要生成24帧图像
- **移动设备**：算力有限，无法承担多次迭代

### 52.2.2 一致性模型的核心思想

2023年，OpenAI的宋飏博士提出了**一致性模型**（Consistency Models），这个工作的核心洞察令人拍案叫绝：

> **如果扩散过程中的任意中间状态都对应同一个最终图像，那么是否存在一个"捷径"函数，可以直接从任意噪声状态映射到清晰图像？**

让我们用数学语言来描述这个想法。

在扩散模型中，我们有前向过程（加噪）：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

这意味着，给定一张清晰图像 $\mathbf{x}_0$，我们可以通过加噪得到任意时刻 $t$ 的噪声图像 $\mathbf{x}_t$。

关键观察是：**对于同一张 $\mathbf{x}_0$，无论我们从哪个时刻 $t$ 开始，去噪的终点都是同一个 $\mathbf{x}_0$**。

一致性模型定义了一个映射函数 $f_\theta$，它将任意时刻的噪声状态直接映射到起点：

$$f_\theta: (\mathbf{x}_t, t) \mapsto \mathbf{x}_0$$

并且要求这个映射满足**一致性约束**：

$$f_\theta(\mathbf{x}_t, t) = f_\theta(\mathbf{x}_{t'}, t') \quad \text{对于所有 } t, t' \in [\epsilon, T]$$

这里的 $\epsilon$ 是一个很小的正数（避免 $t=0$ 的数值问题），$T$ 是最大噪声水平。

> **费曼法比喻：时光机概念**
> 
> 想象时间是河流，扩散过程就像把一块石头扔进河里，激起层层涟漪。常规扩散模型像是一位耐心的渔夫，他站在河边，观察涟漪一圈圈扩散，然后逆着时间一点点还原石头入水的瞬间。
> 
> 一致性模型则像是一台**时光机**。无论你身处时间河流的哪个位置——是刚刚有涟漪的源头，还是已经扩散很远的下游——这台时光机都能瞬间把你带回石头入水的那一刻。
> 
> 更神奇的是，这台时光机是"自洽"的：无论你从哪个时间点出发，最终到达的都是同一个目的地。这就是"一致性"的精髓。

### 52.2.3 一致性损失的数学推导

如何训练这样一个"时光机"呢？我们需要设计一个损失函数，确保模型满足一致性约束。

**关键洞察**：如果我们知道从 $\mathbf{x}_t$ 到 $\mathbf{x}_{t'}$ 的转移（其中 $t' < t$），那么一致性要求：

$$f_\theta(\mathbf{x}_t, t) \approx f_\theta(\mathbf{x}_{t'}, t')$$

我们可以通过**扩散模型的ODE（常微分方程）轨迹**来估计 $\mathbf{x}_{t'}$。

在扩散模型中，概率流ODE描述了如何确定性从噪声还原图像：

$$\frac{d\mathbf{x}_t}{dt} = -\frac{1}{2}g(t)^2 \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$

其中 $g(t)$ 是扩散系数，$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$ 是得分函数（score function）。

如果我们有一个预训练的扩散模型 $\phi$，它可以估计得分函数，那么我们可以通过数值积分（如Euler方法）从 $\mathbf{x}_t$ 估计出 $\mathbf{x}_{t'}$：

$$\mathbf{x}_{t'}^{\phi} = \mathbf{x}_t + \int_t^{t'} \frac{d\mathbf{x}_s}{ds} ds \approx \mathbf{x}_t + (t' - t) \cdot \text{ODE}(\mathbf{x}_t, t)$$

**一致性损失**定义为：

$$\mathcal{L}(\theta, \theta^-; \phi) = \mathbb{E}_{t \sim U[\epsilon, T], t' \sim U[\epsilon, t), \mathbf{x}_0 \sim p_{\text{data}}, \mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)} \left[ d\left(f_\theta(\mathbf{x}_t, t), f_{\theta^-}(\mathbf{x}_{t'}^{\phi}, t')\right) \right]$$

其中：
- $\theta^-$ 是参数 $\theta$ 的**指数移动平均（EMA）**，用于稳定训练
- $d(\cdot, \cdot)$ 是距离度量，通常使用LPIPS（感知损失）或简单的L2距离
- $\mathbf{x}_{t'}^{\phi}$ 是用教师模型 $\phi$ 估计的ODE轨迹上的点

> **直观理解**：这个损失函数在说："如果你从时刻 $t$ 坐时光机回到起点，和从中间站 $t'$ 坐时光机回到起点，你应该到达同一个地方。"教师模型 $\phi$ 提供了一个"参考答案"，告诉我们如何从 $t$ 走到 $t'$。

### 52.2.4 蒸馏训练 vs 独立训练

一致性模型有两种主要的训练方式：

**1. 蒸馏训练（Distillation）**

需要一个预训练的扩散模型（教师），通过模仿教师模型定义的ODE轨迹来训练一致性模型。

优点：
- 可以利用已有扩散模型的知识
- 训练相对稳定
- 生成质量接近教师模型

缺点：
- 依赖于预训练教师模型
- 训练教师模型本身就很耗时

**2. 独立训练（Isolation/Standalone）**

不依赖预训练模型，直接从数据学习一致性映射。

损失函数修改为：

$$\mathcal{L}(\theta, \theta^-) = \mathbb{E}_{t \sim U[\epsilon, T], t' \sim U[\epsilon, t), \mathbf{x}_0 \sim p_{\text{data}}, \mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_0)} \left[ \lambda(t) \cdot d\left(f_\theta(\mathbf{x}_t, t), f_{\theta^-}(\mathbf{x}_{t'}, t')\right) \right]$$

其中 $\mathbf{x}_{t'}$ 是从 $q(\mathbf{x}_{t'}|\mathbf{x}_0)$ 直接采样的（不通过ODE），$\lambda(t)$ 是权重函数。

优点：
- 不需要预训练扩散模型
- 训练流程更简单
- 可以从头训练

缺点：
- 训练可能需要更多数据
- 理论保证稍弱于蒸馏版本

### 52.2.5 代码实现：一致性模型

让我们实现一个简化版的一致性模型，展示其核心思想：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# ===== 时间嵌入模块 =====

class TimeEmbedding(nn.Module):
    """
    正弦位置编码，将时间步t映射为高维向量
    类似于Transformer中的位置编码
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        """
        t: [batch_size] 时间步
        返回: [batch_size, dim] 时间嵌入
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings

# ===== U-Net风格的骨干网络 =====

class ResidualBlock(nn.Module):
    """带时间条件的残差块"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 时间嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
        
        self.gn1 = nn.GroupNorm(8, in_channels)
        self.gn2 = nn.GroupNorm(8, out_channels)
    
    def forward(self, x, t_emb):
        """
        x: [batch, channels, h, w]
        t_emb: [batch, time_emb_dim]
        """
        residual = self.residual_conv(x)
        
        h = self.gn1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # 加入时间信息
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t
        
        h = self.gn2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        return h + residual

class SimpleUNet(nn.Module):
    """简化版U-Net用于一致性模型"""
    def __init__(self, in_channels=3, model_channels=64, time_emb_dim=128):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 编码器
        self.encoder1 = ResidualBlock(in_channels, model_channels, time_emb_dim)
        self.down1 = nn.Conv2d(model_channels, model_channels, 3, stride=2, padding=1)
        
        self.encoder2 = ResidualBlock(model_channels, model_channels * 2, time_emb_dim)
        self.down2 = nn.Conv2d(model_channels * 2, model_channels * 2, 3, stride=2, padding=1)
        
        # 瓶颈
        self.bottleneck = ResidualBlock(model_channels * 2, model_channels * 2, time_emb_dim)
        
        # 解码器
        self.up2 = nn.ConvTranspose2d(model_channels * 2, model_channels * 2, 4, stride=2, padding=1)
        self.decoder2 = ResidualBlock(model_channels * 4, model_channels, time_emb_dim)
        
        self.up1 = nn.ConvTranspose2d(model_channels, model_channels, 4, stride=2, padding=1)
        self.decoder1 = ResidualBlock(model_channels * 2, model_channels, time_emb_dim)
        
        self.out_conv = nn.Conv2d(model_channels, in_channels, 1)
    
    def forward(self, x, t):
        """
        x: [batch, 3, h, w] 噪声图像
        t: [batch] 时间步
        """
        t_emb = self.time_embed(t)
        
        # 编码
        e1 = self.encoder1(x, t_emb)
        e2 = self.down1(e1)
        e2 = self.encoder2(e2, t_emb)
        e3 = self.down2(e2)
        
        # 瓶颈
        b = self.bottleneck(e3, t_emb)
        
        # 解码（带跳跃连接）
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2, t_emb)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1, t_emb)
        
        return self.out_conv(d1)

# ===== 一致性模型 =====

class ConsistencyModel(nn.Module):
    """
    一致性模型：从任意噪声状态直接映射到清晰图像
    
    核心思想: f_theta(x_t, t) -> x_0
    """
    def __init__(self, unet_model, sigma_min=0.002, sigma_max=80.0, sigma_data=0.5):
        super().__init__()
        self.model = unet_model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        # EMA模型（用于稳定训练）
        self.ema_model = None
        
    def get_ema_model(self):
        """获取或创建EMA模型"""
        if self.ema_model is None:
            self.ema_model = type(self.model)(
                **{k: v for k, v in self.model.__dict__.items() 
                   if k not in ['_modules', '_parameters', '_buffers']}
            )
            self.ema_model.load_state_dict(self.model.state_dict())
            for param in self.ema_model.parameters():
                param.requires_grad = False
        return self.ema_model
    
    def update_ema(self, decay=0.9999):
        """更新EMA模型参数"""
        ema = self.get_ema_model()
        with torch.no_grad():
            for param_ema, param in zip(ema.parameters(), self.model.parameters()):
                param_ema.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    def c_skip(self, sigma):
        """跳跃连接权重"""
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
    
    def c_out(self, sigma):
        """输出缩放因子"""
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
    
    def c_in(self, sigma):
        """输入缩放因子"""
        return 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
    
    def c_noise(self, sigma):
        """时间步编码（sigma的对数）"""
        return 0.25 * torch.log(sigma + 1e-44)
    
    def forward(self, x, sigma):
        """
        前向传播：预测x_0
        
        x: [batch, 3, h, w] 噪声图像
        sigma: [batch] 噪声水平（对应时间步）
        
        返回: [batch, 3, h, w] 预测的清晰图像
        """
        # 对输入进行缩放
        c_in = self.c_in(sigma).view(-1, 1, 1, 1)
        x_scaled = x * c_in
        
        # 获取时间嵌入
        t = self.c_noise(sigma)
        
        # 通过模型预测
        F_theta = self.model(x_scaled, t)
        
        # 组合跳跃连接和模型输出
        c_skip = self.c_skip(sigma).view(-1, 1, 1, 1)
        c_out = self.c_out(sigma).view(-1, 1, 1, 1)
        
        return c_skip * x + c_out * F_theta
    
    def consistency_loss(self, x_0, teacher_model=None):
        """
        计算一致性损失
        
        x_0: [batch, 3, h, w] 真实图像
        teacher_model: 教师扩散模型（用于蒸馏训练），如果为None则使用独立训练
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # 随机采样两个时间步
        # z_t = z_0 + sigma * noise
        sigma_t = torch.exp(
            torch.rand(batch_size, device=device) * 
            (np.log(self.sigma_max) - np.log(self.sigma_min)) + 
            np.log(self.sigma_min)
        )
        
        # 采样sigma_s < sigma_t
        u = torch.rand(batch_size, device=device)
        sigma_s = torch.exp(
            u * torch.log(sigma_t + 1e-44) + (1 - u) * np.log(self.sigma_min)
        )
        
        # 加噪得到x_t
        noise_t = torch.randn_like(x_0)
        x_t = x_0 + sigma_t.view(-1, 1, 1, 1) * noise_t
        
        # 学生模型的预测
        pred_x_0 = self.forward(x_t, sigma_t)
        
        # 目标：使用EMA模型预测
        if teacher_model is not None:
            # 蒸馏训练：使用教师模型估计x_s
            with torch.no_grad():
                # 使用Euler方法从t走到s
                # dx = (x_0_pred - x_t) / sigma_t * (sigma_s - sigma_t)
                x_0_pred_teacher = teacher_model(x_t, sigma_t)
                x_s = x_t + (sigma_s - sigma_t).view(-1, 1, 1, 1) * (x_0_pred_teacher - x_t) / sigma_t.view(-1, 1, 1, 1)
                target = self.get_ema_model()(x_s, sigma_s)
        else:
            # 独立训练：直接从数据采样x_s
            noise_s = torch.randn_like(x_0)
            x_s = x_0 + sigma_s.view(-1, 1, 1, 1) * noise_s
            with torch.no_grad():
                target = self.get_ema_model()(x_s, sigma_s)
        
        # 使用L2距离作为损失
        loss = F.mse_loss(pred_x_0, target)
        return loss
    
    @torch.no_grad()
    def sample(self, batch_size, image_size, device='cpu', num_steps=1):
        """
        单步采样！这就是一致性模型的核心优势
        
        batch_size: 生成图像数量
        image_size: 图像尺寸 (h, w)
        num_steps: 采样步数（默认为1，可以增加到2-4步以提高质量）
        """
        h, w = image_size
        
        # 从最大噪声水平采样
        x = torch.randn(batch_size, 3, h, w, device=device) * self.sigma_max
        
        if num_steps == 1:
            # 单步生成！
            sigma = torch.ones(batch_size, device=device) * self.sigma_min
            return self.forward(x, sigma)
        else:
            # 多步生成（可选，提高质量）
            sigmas = torch.exp(
                torch.linspace(np.log(self.sigma_max), np.log(self.sigma_min), num_steps + 1)
            ).to(device)
            
            for i in range(num_steps):
                sigma = torch.ones(batch_size, device=device) * sigmas[i]
                x = self.forward(x, sigma)
                
                # 添加少量噪声（除了最后一步）
                if i < num_steps - 1:
                    sigma_next = torch.ones(batch_size, device=device) * sigmas[i + 1]
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(sigma_next ** 2 - self.sigma_min ** 2).view(-1, 1, 1, 1) * noise
            
            return x

# ===== 训练示例 =====

def train_consistency_model():
    """训练一致性模型的示例"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    unet = SimpleUNet(in_channels=3, model_channels=64)
    cm = ConsistencyModel(unet).to(device)
    
    optimizer = torch.optim.Adam(cm.model.parameters(), lr=1e-4)
    
    # 假设我们有数据加载器
    # for epoch in range(num_epochs):
    #     for x_0 in dataloader:
    #         x_0 = x_0.to(device)
    #         
    #         # 计算一致性损失
    #         loss = cm.consistency_loss(x_0, teacher_model=None)
    #         
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         
    #         # 更新EMA
    #         cm.update_ema()
    
    # 单步采样演示
    generated = cm.sample(batch_size=4, image_size=(32, 32), device=device, num_steps=1)
    print(f"Generated images shape: {generated.shape}")
    
    return cm

# 注意：完整训练需要真实数据集和教师模型
```

---

## 52.3 潜在扩散模型：在思维中作画

### 52.3.1 像素空间的困境

传统扩散模型直接在像素空间进行操作。对于一张256×256的RGB图像，这意味着在16384维（256×256×3）的空间中进行扩散和去噪。

这就像试图在足球场大小的画布上画一幅肖像——每一个像素都需要单独处理，效率极低。

**计算成本的直观对比**：
- 训练扩散模型在256×256图像上需要数百个GPU天
- 大部分计算花费在感知上无关紧要的细节上
- 高频噪声和低频结构被同等对待

### 52.3.2 潜在空间：压缩的魔法

人类画家作画时，脑海中首先浮现的是**概念和构图**——"一只猫坐在椅子上"——而不是像素级别的RGB值。只有在落笔时，这些抽象概念才被转化为具体的图像。

**潜在扩散模型**（Latent Diffusion Models, LDM）正是借鉴了这种思想：

1. **编码阶段**：使用变分自编码器（VAE）将图像压缩到低维潜在空间
2. **扩散阶段**：在低维潜在空间中进行扩散过程
3. **解码阶段**：使用VAE解码器将潜在表示还原为像素图像

Stable Diffusion的潜在空间尺寸：
- 输入图像：512×512×3 = 786,432维
- 潜在表示：64×64×4 = 16,384维
- **压缩比：48倍！**

这意味着在潜在空间中，扩散模型只需要处理1/48的数据量，显著提升了训练和推理效率。

> **费曼法比喻：思维压缩**
> 
> 想象你正在给朋友描述一部电影。你不会逐帧描述每一个像素的RGB值——那样需要讲述数百万个数字！相反，你会使用概念和语言："有一个太空战士，穿着黑色盔甲，手持光剑，站在沙漠星球上。"
> 
> 这种**概念化描述**就是"潜在表示"——它捕捉了图像的本质信息，舍弃了无关紧要的细节。当朋友听完你的描述，他们可以在脑海中"重建"出电影场景，虽然不是逐像素精确，但抓住了关键特征。
> 
> 潜在扩散模型就像这样一位**概念画师**：它在"思维空间"（潜在空间）中构思图像，只在最后才把想法转化为具体的像素。这既高效又富有表现力。

### 52.3.3 Stable Diffusion架构解析

Stable Diffusion（SD）是目前最流行的开源文本到图像模型，其核心架构包含三个组件：

**1. VAE（变分自编码器）**

VAE负责像素空间和潜在空间之间的转换。

编码器：$\mathcal{E}: \mathbf{x} \in \mathbb{R}^{H \times W \times 3} \mapsto \mathbf{z} \in \mathbb{R}^{h \times w \times c}$

解码器：$\mathcal{D}: \mathbf{z} \in \mathbb{R}^{h \times w \times c} \mapsto \hat{\mathbf{x}} \in \mathbb{R}^{H \times W \times 3}$

其中 $h = H/8$, $w = W/8$，即空间分辨率压缩8倍。

VAE的训练目标：

$$\mathcal{L}_{\text{VAE}} = \|\mathbf{x} - \mathcal{D}(\mathcal{E}(\mathbf{x}))\|_2^2 + \beta \cdot \text{KL}[\mathcal{E}(\mathbf{x}) \| \mathcal{N}(0, \mathbf{I})]$$

**2. U-Net去噪网络**

U-Net是扩散过程的核心，负责预测噪声或预测 $v$-prediction（见第38章）。

关键特性：
- **跨注意力层**（Cross-Attention）：注入文本条件
- **残差连接**：保留空间信息
- **时间嵌入**：让模型知道当前噪声水平

**3. CLIP文本编码器**

CLIP（Contrastive Language-Image Pre-training）将文本提示编码为语义向量。

文本编码器：$\tau_\theta: \mathbf{y} \mapsto \mathbf{c} \in \mathbb{R}^{d}$

其中 $\mathbf{y}$ 是文本提示，$\mathbf{c}$ 是条件向量，通过跨注意力机制注入U-Net。

### 52.3.4 Classifier-Free Guidance：控制力与创造力的平衡

在条件生成中，我们面临一个永恒的矛盾：
- **遵循条件**：生成的图像应该符合文本描述
- **保持多样性**：生成的图像应该有创意和变化

**Classifier-Free Guidance（CFG）**提供了优雅的解决方案。

核心思想：同时训练**条件生成**和**无条件生成**，在采样时进行外推。

训练时，以概率 $p_{\text{uncond}}$ 丢弃条件：

$$\mathbf{c} = \begin{cases} \tau_\theta(\mathbf{y}) & \text{以概率 } 1-p_{\text{uncond}} \\ \emptyset & \text{以概率 } p_{\text{uncond}} \end{cases}$$

采样时，预测是有条件和无条件预测的加权外推：

$$\hat{\epsilon}_\theta(\mathbf{z}_t, \mathbf{c}) = \epsilon_\theta(\mathbf{z}_t, \emptyset) + w \cdot (\epsilon_\theta(\mathbf{z}_t, \mathbf{c}) - \epsilon_\theta(\mathbf{z}_t, \emptyset))$$

其中 $w \geq 1$ 是**guidance scale**（引导尺度）：
- $w = 1$：标准条件生成
- $w > 1$：增强条件遵循，但可能降低多样性
- $w = 7.5$：SD的默认设置

**数学解释**：

CFG等价于在贝叶斯框架下增强条件信号。根据贝叶斯定理：

$$\log p(\mathbf{z} | \mathbf{c}) = \log p(\mathbf{z}) + \log p(\mathbf{c} | \mathbf{z}) - \log p(\mathbf{c})$$

CFG通过对数概率的缩放来增强条件影响：

$$\log p_{\text{CFG}}(\mathbf{z} | \mathbf{c}) = \log p(\mathbf{z}) + w \cdot \log p(\mathbf{c} | \mathbf{z})$$

### 52.3.5 代码实现：潜在扩散模型

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ===== VAE编码器和解码器 =====

class VAEEncoder(nn.Module):
    """
    VAE编码器：将图像压缩到潜在空间
    输入: [batch, 3, H, W]
    输出: [batch, 4, H/8, W/8] (均值和对数方差)
    """
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        
        # 下采样块
        self.conv_in = nn.Conv2d(in_channels, 128, 3, padding=1)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=2, padding=1),  # /2
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # /4
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),  # /8
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),
        )
        
        # 输出均值和对数方差
        self.conv_out = nn.Conv2d(512, latent_dim * 2, 3, padding=1)
    
    def forward(self, x):
        """
        返回均值和对数方差
        """
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        h = self.conv_out(h)
        
        # 分割为均值和对数方差
        mean, logvar = h.chunk(2, dim=1)
        return mean, logvar

class VAEDecoder(nn.Module):
    """
    VAE解码器：将潜在表示还原为图像
    输入: [batch, 4, H/8, W/8]
    输出: [batch, 3, H, W]
    """
    def __init__(self, latent_dim=4, out_channels=3):
        super().__init__()
        
        self.conv_in = nn.Conv2d(latent_dim, 512, 3, padding=1)
        
        # 上采样块
        self.up1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),  # *2
        )
        
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # *4
        )
        
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),  # *8
        )
        
        self.conv_out = nn.Conv2d(128, out_channels, 3, padding=1)
    
    def forward(self, z):
        h = self.conv_in(z)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.conv_out(h)
        return h

class VAE(nn.Module):
    """完整的VAE模型"""
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)
        self.latent_dim = latent_dim
    
    def encode(self, x):
        """编码图像到潜在空间"""
        mean, logvar = self.encoder(x)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def decode(self, z):
        """解码潜在表示到图像"""
        return self.decoder(z)
    
    def forward(self, x):
        """完整的前向传播"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar
    
    def get_latent(self, x):
        """获取确定性潜在表示（使用均值）"""
        mean, _ = self.encode(x)
        return mean

# ===== 文本条件U-Net =====

class CrossAttention(nn.Module):
    """跨注意力层：将文本条件注入图像特征"""
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
    
    def forward(self, x, context):
        """
        x: [batch, hw, query_dim] 图像特征
        context: [batch, seq_len, context_dim] 文本特征
        """
        batch_size = x.shape[0]
        
        # 计算Q, K, V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        #  reshape为多注意力头
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.query_dim if hasattr(self, 'query_dim') else q.shape[2] * self.num_heads)
        
        # 修正形状计算
        out = out.view(batch_size, -1, self.num_heads * self.head_dim)
        return self.to_out(out)

class ResnetBlock(nn.Module):
    """带文本条件的ResNet块"""
    def __init__(self, in_channels, out_channels, time_emb_dim, context_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # 时间嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        ) if time_emb_dim is not None else None
        
        # 残差连接
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # 跨注意力（如果提供上下文）
        self.attn = None
        if context_dim is not None:
            self.norm = nn.GroupNorm(32, out_channels)
            self.attn = CrossAttention(out_channels, context_dim)
    
    def forward(self, x, time_emb=None, context=None):
        h = self.conv1(x)
        
        if time_emb is not None and self.time_mlp is not None:
            h = h + self.time_mlp(time_emb)[:, :, None, None]
        
        h = self.conv2(F.silu(h))
        
        # 应用跨注意力
        if self.attn is not None and context is not None:
            batch, c, h_dim, w = h.shape
            h_flat = h.view(batch, c, -1).transpose(1, 2)  # [b, hw, c]
            h_flat = self.attn(self.norm(h_flat), context)
            h = h_flat.transpose(1, 2).view(batch, c, h_dim, w)
        
        return h + self.residual_conv(x)

class TextConditionedUNet(nn.Module):
    """
    文本条件的U-Net用于潜在扩散
    """
    def __init__(self, in_channels=4, model_channels=320, context_dim=768):
        super().__init__()
        
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 编码器
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, 3, padding=1),
            ResnetBlock(model_channels, model_channels, time_emb_dim, context_dim),
            ResnetBlock(model_channels, model_channels, time_emb_dim, context_dim),
        ])
        
        # 中间层
        self.middle_block = ResnetBlock(model_channels, model_channels, time_emb_dim, context_dim)
        
        # 解码器
        self.output_blocks = nn.ModuleList([
            ResnetBlock(model_channels * 2, model_channels, time_emb_dim, context_dim),
            ResnetBlock(model_channels * 2, model_channels, time_emb_dim, context_dim),
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
        ])
    
    def forward(self, x, t, context=None):
        """
        x: [batch, 4, h, w] 潜在表示
        t: [batch] 时间步
        context: [batch, seq_len, context_dim] 文本嵌入
        """
        # 时间嵌入
        t_emb = timestep_embedding(t, self.model_channels if hasattr(self, 'model_channels') else 320)
        t_emb = self.time_embed(t_emb)
        
        # 编码
        hs = []
        h = x
        for module in self.input_blocks:
            if isinstance(module, ResnetBlock):
                h = module(h, t_emb, context)
            else:
                h = module(h)
            hs.append(h)
        
        # 中间
        h = self.middle_block(h, t_emb, context)
        
        # 解码
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if isinstance(module, ResnetBlock):
                h = module(h, t_emb, context)
            else:
                h = module(h)
        
        return h

def timestep_embedding(timesteps, dim, max_period=10000):
    """创建正弦时间步嵌入"""
    half = dim // 2
    freqs = torch.exp(
        -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# ===== 潜在扩散模型主类 =====

class LatentDiffusionModel(nn.Module):
    """
    潜在扩散模型 (LDM / Stable Diffusion)
    """
    def __init__(self, vae, unet, text_encoder=None, 
                 num_timesteps=1000, beta_start=0.00085, beta_end=0.012):
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.text_encoder = text_encoder
        self.num_timesteps = num_timesteps
        
        # 冻结VAE参数（通常预训练且不更新）
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # 定义噪声调度（余弦调度）
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 预计算扩散参数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def encode_text(self, text):
        """编码文本提示（简化版，实际使用CLIP）"""
        if self.text_encoder is None:
            return None
        return self.text_encoder(text)
    
    def forward_diffusion(self, z_0, t, noise=None):
        """
        前向扩散：加噪到潜在表示
        z_0: [batch, 4, h, w] 干净潜在表示
        t: [batch] 时间步
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(z_0.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1).to(z_0.device)
        
        z_t = sqrt_alpha * z_0 + sqrt_one_minus_alpha * noise
        return z_t, noise
    
    def predict_noise(self, z_t, t, context=None, guidance_scale=7.5):
        """
        预测噪声，支持Classifier-Free Guidance
        """
        if guidance_scale > 1.0 and context is not None:
            # CFG: 同时预测条件和无条件
            z_t_double = torch.cat([z_t, z_t])
            t_double = torch.cat([t, t])
            context_double = torch.cat([context, torch.zeros_like(context)])
            
            noise_pred = self.unet(z_t_double, t_double, context_double)
            noise_cond, noise_uncond = noise_pred.chunk(2)
            
            # CFG外推
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = self.unet(z_t, t, context)
        
        return noise_pred
    
    @torch.no_grad()
    def sample(self, batch_size, image_size, text=None, num_inference_steps=50, 
               guidance_scale=7.5, device='cpu'):
        """
        DDIM采样生成图像
        
        batch_size: 生成数量
        image_size: 图像尺寸 (H, W)
        text: 文本提示
        num_inference_steps: 采样步数
        guidance_scale: CFG引导尺度
        """
        H, W = image_size
        latent_h, latent_w = H // 8, W // 8
        
        # 编码文本
        context = None
        if text is not None and self.text_encoder is not None:
            context = self.encode_text(text)
        
        # 从噪声开始
        z = torch.randn(batch_size, 4, latent_h, latent_w, device=device)
        
        # DDIM采样调度
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
        
        for i in range(num_inference_steps):
            t = torch.full((batch_size,), timesteps[i], device=device, dtype=torch.long)
            
            # 预测噪声
            noise_pred = self.predict_noise(z, t, context, guidance_scale)
            
            # DDIM更新
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            alpha_prev = self.alphas_cumprod[timesteps[i-1] if i > 0 else 0].view(-1, 1, 1, 1) if i > 0 else torch.ones_like(alpha_t)
            
            pred_x0 = (z - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            if i < num_inference_steps - 1:
                noise = torch.randn_like(z) if i < num_inference_steps - 1 else 0
                z = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * noise_pred
            else:
                z = pred_x0
        
        # 解码到像素空间
        images = self.vae.decode(z)
        return images

# ===== 使用示例 =====

def create_ldm_model():
    """创建简化的LDM模型"""
    vae = VAE(in_channels=3, latent_dim=4)
    unet = TextConditionedUNet(in_channels=4, model_channels=320)
    ldm = LatentDiffusionModel(vae, unet)
    return ldm

# 注意：完整训练需要大规模数据集和CLIP文本编码器
```

---

## 52.4 进阶主题与模型对比

### 52.4.1 生成模型家族大对比

下表总结了我们在第38章和本章中讨论的主要生成模型：

| 模型类型 | 训练目标 | 采样速度 | 似然计算 | 模式覆盖 | 主要优势 | 主要局限 |
|---------|---------|---------|---------|---------|---------|---------|
| **GANs** | 对抗训练（min-max） | ⚡ 单步 | ❌ 无法直接计算 | 差（模式坍塌） | 生成质量高、速度快 | 训练不稳定、无显式似然 |
| **VAEs** | ELBO（重构+KL） | ⚡ 单步 | ✅ 可计算下界 | 好 | 原理简单、可解释性强 | 生成质量一般、后验坍塌 |
| **Flows** | 最大似然 | ⚡ 单步 | ✅ 精确计算 | 好 | 可逆变换、精确似然 | 架构受限、表达能力有限 |
| **Diffusion** | 去噪分数匹配 | 🐢 多步（50-1000） | ❌ 无法直接计算 | 优秀 | 质量最佳、训练稳定 | 采样慢、需要多步迭代 |
| **Consistency** | 一致性损失 | ⚡ 单步（或多步） | ❌ 无法直接计算 | 好 | 单步生成、质量接近扩散 | 训练复杂、需要EMA稳定 |

**速度 vs 质量的权衡**：

```
生成质量 ↑
    │
    │         ★ Diffusion (1000步)
    │        ╱
    │       ╱  ★ Diffusion (50步)
    │      ╱
    │     ★ Consistency (4步)
    │    ╱
    │   ★ Consistency (1步)
    │  ╱
    │ ★ GAN / Flow / VAE
    │
    └────────────────→ 采样速度
```

### 52.4.2 2024年最新进展

**1. Adversarial Diffusion Distillation (ADD)**

由Stability AI提出，结合了对抗训练和蒸馏：
- 使用预训练的扩散模型作为教师
- 引入判别器提供对抗信号
- 在1-4步内达到接近50步扩散模型的质量

**2. 渐进蒸馏（Progressive Distillation）**

将多步扩散模型逐步蒸馏为少步模型：
- 第一轮：50步 → 25步
- 第二轮：25步 → 12步
- 第三轮：12步 → 6步
- 最终：6步 → 单步

**3. LCM-LoRA：加速扩散的利器**

Latent Consistency Models + LoRA微调：
- 在预训练的Stable Diffusion上应用LoRA
- 仅需少量微调步骤（约10k步）
- 实现2-4步高质量生成
- 社区广泛使用，可在消费级GPU上运行

```python
# LCM-LoRA使用示例（伪代码）
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

# 只需2-4步！
image = pipe(prompt="a beautiful sunset", num_inference_steps=4).images[0]
```

---

## 52.5 实战案例

### 52.5.1 案例一：一致性模型单步图像生成

以下代码演示如何使用一致性模型进行单步图像生成：

```python
import torch
import matplotlib.pyplot as plt

# 加载预训练的一致性模型（这里使用简化版演示）
# 实际使用时加载官方权重：
# from consistency_models import load_consistency_model
# cm = load_consistency_model("consistency_model_lsun_bedroom")

def demo_single_step_generation():
    """演示单步生成"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型（实际使用时加载预训练权重）
    unet = SimpleUNet(in_channels=3, model_channels=128)
    cm = ConsistencyModel(unet, sigma_min=0.002, sigma_max=80.0).to(device)
    
    # 单步生成！
    print("正在进行单步图像生成...")
    with torch.no_grad():
        images = cm.sample(
            batch_size=4, 
            image_size=(64, 64),  # 可以从低分辨率开始
            device=device,
            num_steps=1  # 只需要1步！
        )
    
    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        img = (images[i].cpu().permute(1, 2, 0) + 1) / 2  # 从[-1,1]转换到[0,1]
        ax.imshow(img.clip(0, 1))
        ax.axis('off')
    plt.suptitle("单步生成结果")
    plt.show()
    
    return images

# 对比多步生成质量
def compare_steps():
    """比较不同步数的生成质量"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    unet = SimpleUNet(in_channels=3, model_channels=128)
    cm = ConsistencyModel(unet).to(device)
    
    steps_to_try = [1, 2, 4]
    results = {}
    
    for steps in steps_to_try:
        torch.manual_seed(42)  # 固定种子以便比较
        images = cm.sample(batch_size=1, image_size=(64, 64), 
                          device=device, num_steps=steps)
        results[steps] = images[0]
    
    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, (steps, img) in enumerate(results.items()):
        img = (img.cpu().permute(1, 2, 0) + 1) / 2
        axes[i].imshow(img.clip(0, 1))
        axes[i].set_title(f"{steps}步生成")
        axes[i].axis('off')
    plt.show()
```

### 52.5.2 案例二：Stable Diffusion文本到图像生成

```python
from diffusers import StableDiffusionPipeline
import torch

def text_to_image_demo():
    """使用Stable Diffusion进行文本到图像生成"""
    
    # 加载模型（需要约4-7GB显存）
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 文本提示
    prompt = "a photograph of an astronaut riding a horse, highly detailed"
    negative_prompt = "blurry, low quality, distorted"
    
    # 生成参数
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,  # 采样步数
        guidance_scale=7.5,       # CFG引导尺度
        height=512,
        width=512,
        seed=42
    ).images[0]
    
    image.save("astronaut_riding_horse.png")
    return image
```

### 52.5.3 案例三：潜在空间插值与编辑

```python
def latent_interpolation_demo(vae, text_encoder, unet):
    """
    在潜在空间进行插值，实现图像平滑过渡
    """
    device = 'cuda'
    ldm = LatentDiffusionModel(vae, unet, text_encoder).to(device)
    
    # 编码两个不同的文本提示
    prompt1 = "a peaceful lake at sunrise"
    prompt2 = "a stormy ocean at midnight"
    
    with torch.no_grad():
        # 获取文本嵌入
        z1 = ldm.encode_text([prompt1])
        z2 = ldm.encode_text([prompt2])
        
        # 在潜在空间插值
        num_interpolation = 8
        alphas = torch.linspace(0, 1, num_interpolation).to(device)
        
        interpolated_images = []
        for alpha in alphas:
            # 球形插值（SLERP）比线性插值更自然
            z_interp = slerp(z1, z2, alpha)
            
            # 从插值的潜在表示生成图像
            image = ldm.sample_from_embedding(z_interp)
            interpolated_images.append(image)
    
    return interpolated_images

def slerp(v0, v1, t):
    """球形线性插值"""
    dot = (v0 * v1).sum() / (v0.norm() * v1.norm())
    dot = dot.clamp(-1, 1)
    theta = torch.acos(dot)
    
    if theta < 1e-5:
        return v0 * (1 - t) + v1 * t
    
    sin_theta = torch.sin(theta)
    return v0 * torch.sin((1 - t) * theta) / sin_theta + \
           v1 * torch.sin(t * theta) / sin_theta
```

---

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

Dinh, L., Krueger, D., & Bengio, Y. (2015). NICE: Non-linear independent components estimation. *Workshop at ICLR 2015*. https://arxiv.org/abs/1410.8516

Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density estimation using Real NVP. *ICLR 2017*. https://arxiv.org/abs/1605.08803

Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS 2020*. https://arxiv.org/abs/2006.11239

Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *NeurIPS 2022 Workshop on Deep Generative Models*. https://arxiv.org/abs/2207.12598

Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1×1 convolutions. *NeurIPS 2018*. https://arxiv.org/abs/1807.03039

Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR 2014*. https://arxiv.org/abs/1312.6114

Luo, S. (2023). Understanding diffusion models: A unified perspective. *arXiv preprint*. https://arxiv.org/abs/2208.11970

Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing flows for probabilistic modeling and inference. *Journal of Machine Learning Research, 22*(57), 1-64. https://jmlr.org/papers/v22/19-427.html

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models. *CVPR 2022*. https://arxiv.org/abs/2112.10752

Saharia, C., Chan, W., Saxena, S., Li, L., Whang, J., Denton, E. L., ... & Norouzi, M. (2022). Photorealistic text-to-image diffusion models with deep language understanding. *NeurIPS 2022*. https://arxiv.org/abs/2205.11487

Salimans, T., & Ho, J. (2022). Progressive distillation for fast sampling of diffusion models. *ICLR 2022*. https://arxiv.org/abs/2202.00512

Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR 2021*. https://arxiv.org/abs/2010.02502

Song, Y., & Dhariwal, P. (2024). Improved techniques for training consistency models. *ICLR 2024*. https://arxiv.org/abs/2310.14189

Song, Y., Dhariwal, P., Chen, M., & Sutskever, I. (2023). Consistency models. *ICML 2023*. https://arxiv.org/abs/2303.01469

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). Score-based generative modeling through stochastic differential equations. *ICLR 2021*. https://arxiv.org/abs/2011.13456

Zhang, L., Rao, A., & Agrawala, M. (2023). Adding conditional control to text-to-image diffusion models. *ICCV 2023*. https://arxiv.org/abs/2302.05543

---

*"生成模型的美妙之处在于：我们不是在拟合数据，而是在学习创造的艺术。"*

**本章完**
