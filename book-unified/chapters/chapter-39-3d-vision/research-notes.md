# 第39章：NeRF与3D视觉生成研究笔记

## 研究概述

本笔记为《机器学习趣味入门》第39章"3D视觉生成"提供深度技术支撑，涵盖NeRF（神经辐射场）及其后续发展的核心论文、技术原理和数学基础。

---

## 一、核心论文详解

### 1.1 NeRF: Representing Scenes as Neural Radiance Fields (Mildenhall et al., ECCV 2020)

**核心贡献：**
- 开创性地将3D场景表示为连续的神经辐射场
- 使用MLP网络隐式表示场景的几何和外观
- 实现了从稀疏2D图像合成高质量新视角照片的效果

**关键公式：**

1. **5D辐射场函数：**
```
F(x, d) → (c, σ)
```
其中：
- x = (x, y, z)：3D空间坐标
- d = (θ, φ)：观察方向（方位角和极角）
- c = (r, g, b)：RGB颜色
- σ：体积密度（不透明度）

2. **体积渲染方程（离散形式）：**
```
Ĉ = Σᵢ Tᵢ · (1 - exp(-σᵢδᵢ)) · cᵢ
其中 Tᵢ = exp(-Σⱼ₌₀ⁱ⁻¹ σⱼδⱼ)
```
- Tᵢ：透射率（光线到达该点未被遮挡的概率）
- δᵢ：采样点间距
- αᵢ = 1 - exp(-σᵢδᵢ)：不透明度

3. **位置编码（Positional Encoding）：**
```
γ(p) = [sin(2⁰πp), cos(2⁰πp), ..., sin(2ᴸ⁻¹πp), cos(2ᴸ⁻¹πp)]
```
- L=10用于位置坐标，L=4用于方向
- 克服MLP的频谱偏置，学习高频细节

**算法流程：**
1. 沿相机射线采样3D点
2. 用MLP预测每个点的密度和颜色
3. 通过体积渲染积分计算像素颜色
4. 最小化与真实图像的重建损失

---

### 1.2 Mip-NeRF: A Multiscale Representation for Anti-Aliasing (Barron et al., ICCV 2021)

**核心贡献：**
- 解决NeRF的抗锯齿（anti-aliasing）问题
- 用圆锥形视锥体替代NeRF的射线采样
- 引入集成位置编码（Integrated Positional Encoding, IPE）

**关键技术：**
1. **圆锥采样：** 对每个像素发射一个圆锥视锥体而非单条射线
2. **IPE公式：**
```
γ_IPE(x, Σ) = [sin(2ᵏπx)·exp(-½(2ᵏπ)²diag(Σ)), 
               cos(2ᵏπx)·exp(-½(2ᵏπ)²diag(Σ))]
```
- 对视锥体内的区域进行高斯建模
- 自动适应不同尺度的细节

**与NeRF对比：**
| 特性 | NeRF | Mip-NeRF |
|-----|------|----------|
| 采样方式 | 射线 | 圆锥视锥体 |
| 位置编码 | 正弦编码 | 集成位置编码 |
| 抗锯齿 | 无 | 有 |
| 多尺度 | 需多模型 | 单模型支持 |

---

### 1.3 NeRF++: Analyzing and Improving Neural Radiance Fields (Zhang et al., 2020)

**核心贡献：**
- 分析辐射场的形状-光照歧义性（shape-radiance ambiguity）
- 提出**倒球参数化（Inverted Sphere Parameterization）**处理360°无界场景
- 将场景分为近景单位球内和远景球外两个区域

**倒球参数化公式：**
```
对于单位球外的点 x (||x|| > 1):
x' = x / ||x||² + (2 - 1/||x||) · (x / ||x||)
```
- 将球外无穷远区域压缩到有限空间
- 远景区域使用不同的MLP处理

---

### 1.4 Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields (Barron et al., CVPR 2022)

**核心贡献：**
- 将Mip-NeRF扩展到无界360°场景
- 引入场景参数化处理远景
- 改进的采样策略和正则化

**关键技术：**
1. **同心圆参数化（Concentric Sphere Parameterization）**
2. **在线蒸馏（Online Distillation）**：从精细模型向粗糙模型传递知识
3. **更紧的正则化**：减少浮点（floaters）伪影

---

## 二、加速方法

### 2.1 PlenOctrees (Yu et al., ICCV 2021)

**核心贡献：**
- 将训练好的NeRF转换为八叉树表示
- 实现**实时光线追踪渲染**（>100 FPS）

**技术路线：**
1. 用NeRF生成密集采样点的颜色和密度
2. 将球谐函数（SH）拟合到八叉树的叶节点
3. 基于八叉树结构进行快速光线追踪

**球谐函数颜色表示：**
```
c(v) = Σₗ₌₀ᴸ Σₘ₌₋ₗˡ cₗₘ · Yₗₘ(v)
```
- Yₗₘ：球谐基函数
- cₗₘ：可学习的系数

---

### 2.2 Instant-NGP: Instant Neural Graphics Primitives (Müller et al., SIGGRAPH 2022)

**核心贡献：**
- **多分辨率哈希编码（Multiresolution Hash Encoding）**
- 5秒训练NeRF级别质量的场景
- 支持NeRF、SDF、神经图像等多种图元

**哈希编码公式：**
```
h(x) = ⊕ᵢ₌₁ᴸ lookup(hash(⌊x · Nᵢ⌋ mod T))
```
- L：层级数（通常16层）
- Nᵢ：第i层分辨率
- T：哈希表大小（通常2¹⁴到2²⁰）

**技术亮点：**
1. **可训练的多分辨率哈希表**：存储特征向量
2. **小MLP**：仅用2层隐藏层（64或32单元）
3. **高效GPU实现**：利用CUDA并行

**性能对比：**
- 训练时间：原始NeRF 1-2天 → Instant-NGP 5秒
- 渲染速度：实时（>60 FPS at 1920×1080）

---

### 2.3 MobileNeRF (Chen et al., CVPR 2023)

**核心贡献：**
- 利用**多边形光栅化流水线**在移动设备上高效渲染
- 将神经场转换为纹理化多边形网格

**技术路线：**
1. **多边形化**：将场景表示为带纹理的三角形网格
2. **二值不透明度（Binary Opacity）**：每个三角形要么完全透明要么完全不透明
3. **特征纹理（Feature Texture）**：存储神经特征而非直接RGB
4. **小MLP后处理**：将特征转换为最终颜色

**优势：**
- 利用移动GPU的硬件光栅化
- 可与现有3D渲染引擎集成
- 在手机上达到30+ FPS

---

## 三、3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023)

### 核心贡献

**SIGGRAPH 2023最佳论文之一**，革命性地将场景表示为**显式的3D高斯点云**，实现：
- 照片级真实感渲染质量
- **实时渲染**（≥30 FPS at 1080p）
- 有竞争力的训练时间

### 3D高斯表示

每个高斯由以下参数定义：
```
G(x) = exp(-½(x - μ)ᵀ Σ⁻¹(x - μ))
```
- **位置 μ ∈ ℝ³**：高斯中心
- **协方差 Σ ∈ ℝ³ˣ³**：高斯形状和方向
  - 分解为：Σ = RSSᵀRᵀ
  - R：旋转矩阵（用四元数表示）
  - S：缩放矩阵
- **不透明度 α ∈ ℝ**：透明度
- **颜色 c**：球谐函数系数（通常3阶，16个系数）

### 渲染管线

1. **投影**：将3D高斯投影到2D图像平面
   ```
   Σ' = JWΣWᵀJᵀ
   ```
   - W：视图变换矩阵
   - J：投影变换的仿射近似雅可比矩阵

2. **Alpha混合**：按深度排序后进行alpha合成
   ```
   C = Σᵢ cᵢ αᵢ Tᵢ, 其中 Tᵢ = Πⱼ₌₁ⁱ⁻¹(1 - αⱼ)
   ```

### 优化策略

1. **自适应密度控制**：
   - **分裂（Split）**：对高梯度的高斯进行分裂
   - **克隆（Clone）**：对欠重建区域克隆高斯
   - **剪枝（Prune）**：移除透明度过低的高斯

2. **交错优化**：每迭代一定次数进行密度控制

### 与NeRF对比

| 特性 | NeRF | 3D Gaussian Splatting |
|-----|------|----------------------|
| 表示 | 隐式MLP | 显式高斯点云 |
| 训练时间 | 小时-天 | 分钟（10-30） |
| 渲染速度 | 慢（<1 FPS） | 实时（>30 FPS） |
| 内存占用 | 较低 | 较高 |
| 编辑性 | 困难 | 容易（直接操作点云） |
| 几何提取 | 困难 | 较容易 |

---

## 四、文本到3D生成

### 4.1 DreamFusion (Poole et al., 2022)

**核心贡献：**
- 开创**Score Distillation Sampling (SDS)**损失
- 无需3D训练数据，利用预训练2D扩散模型生成3D

**SDS损失公式：**
```
∇θL_SDS = Eₜ,w[ω(t) (ε̂_φ(zₜ; y, t) - ε) ∂x/∂θ]
```
- zₜ：扩散时间步t的加噪渲染图像
- ε̂_φ：预训练扩散模型的噪声预测
- y：文本条件
- x = g(θ)：可微渲染器渲染的图像
- θ：3D表示参数（NeRF权重）

**算法流程：**
1. 随机初始化NeRF
2. 从随机相机视角渲染图像
3. 将渲染图像加噪到随机时间步
4. 用扩散模型预测噪声
5. 根据SDS梯度更新NeRF参数

**问题：**
- 过饱和（oversaturation）
- 多面问题（Janus problem，物体有多个正面）
- 生成速度慢（需长时间优化）

---

### 4.2 Magic3D (Lin et al., 2023)

**核心贡献：**
- **两阶段优化**框架，提升分辨率和质量
- 粗阶段：低分辨率NeRF（Instant-NGP）
- 细阶段：高分辨率可微网格渲染

**技术路线：**
```
阶段1（粗）：文本 → 低分辨率NeRF（64×64）
阶段2（细）：NeRF初始化 → 可微网格优化 → 高分辨率纹理（512×512）
```

**优势：**
- 比DreamFusion更好的几何细节
- 更高分辨率输出
- 可提取高质量网格

---

### 4.3 MVDream (Shi et al., 2023)

**核心贡献：**
- 解决多视角一致性问题
- 训练**多视角扩散模型**作为3D先验

**技术要点：**
1. **多视角注意力机制**：替换标准自注意力为跨视角注意力
2. **联合训练**：在真实图像和合成多视角数据上训练
3. **多视角SDS**：从多个视角同时蒸馏知识

**公式：**
```
∇θL_MV-SDS = Σᵢ₌₁⁴ Eₜ,w[ω(t) (ε̂_φ(zₜⁱ; y, t, cameraᵢ) - εⁱ) ∂xⁱ/∂θ]
```
- 同时生成4个视角（前、右、后、左）
- 相机条件确保几何一致性

**解决Janus问题：**
- 多视角同时优化防止不一致性
- 相比单视角DreamFusion显著提升几何一致性

---

## 五、技术演进时间线（2020-2024）

```
2020年 ─────────────────────────────────────────────────────
│
├─ March: NeRF (Mildenhall et al., ECCV 2020) - 奠基之作
│
├─ Oct: NeRF++ (Zhang et al.) - 360°无界场景
│
2021年 ─────────────────────────────────────────────────────
│
├─ ICCV: Mip-NeRF (Barron et al.) - 抗锯齿
│
├─ ICCV: PlenOctrees (Yu et al.) - 实时渲染加速
│
├─ CVPR: pixelNeRF (Yu et al.) - 单/少图像生成
│
└─ NeRF获得SIGGRAPH最佳论文荣誉奖
│
2022年 ─────────────────────────────────────────────────────
│
├─ CVPR: Mip-NeRF 360 (Barron et al.) - 无界场景优化
│
├─ CVPR: Plenoxels (Fridovich-Keil et al.) - 无神经网络
│
├─ SIGGRAPH: Instant-NGP (Müller et al.) - 哈希编码革命
│  └─ 训练从小时级降到秒级
│
├─ DreamFusion (Poole et al.) - 文本到3D开山之作
│
└─ ECCV: TensoRF (Chen et al.) - 张量分解
│
2023年 ─────────────────────────────────────────────────────
│
├─ CVPR: MobileNeRF (Chen et al.) - 移动端渲染
│
├─ SIGGRAPH: 3D Gaussian Splatting (Kerbl et al.) - 革命！
│  └─ SIGGRAPH最佳论文
│
├─ CVPR: Magic3D (Lin et al.) - 两阶段文本到3D
│
├─ ICCV: MVDream (Shi et al.) - 多视角一致性
│
├─ Instant3D (Li et al.) - 前馈文本到3D
│
└─ DreamGaussian (Tang et al.) - 高斯文本到3D
│
2024年 ─────────────────────────────────────────────────────
│
├─ 4D Gaussian Splatting - 动态场景
│
├─ GaussianEditor - 高斯场景编辑
│
├─ 大量3DGS加速和压缩工作
│
└─ 神经场+扩散模型融合趋势
│
2025+ ─────────────────────────────────────────────────────
│
└─ 多模态、实时、可编辑成为主要方向
```

---

## 六、费曼比喻素材

### 6.1 体渲染（Volume Rendering）

**比喻1：雾中的风景**
> "想象你站在浓雾中看一座山。雾越浓的地方（密度高），你能看到山的机会就越小。体渲染就像是计算每缕雾气后面的风景有多少能穿透雾到达你的眼睛。"

**比喻2：千层蛋糕**
> "体渲染就像切千层蛋糕。每一层都有自己的颜色和透明度。我们从蛋糕底部开始，逐层叠加，前面的层会挡住后面的层。最终的颜色是每一层贡献的加权平均。"

**比喻3：X光透视**
> "医生看X光片时，可以看到骨骼和器官的叠加。体渲染类似，我们沿着一条射线看进去，累积所有点的颜色和密度信息。"

### 6.2 神经辐射场（Neural Radiance Field）

**比喻1：智能调色盘**
> "想象一个超级智能的调色盘，你告诉它'我在房间的角落，面向窗户'，它就会告诉你那个位置和方向应该是什么颜色。NeRF就是这样的调色盘——一个神经网络，学会根据位置和视角预测颜色和密度。"

**比喻2：记忆宫殿**
> "记忆宫殿是一种记忆技巧，你把要记住的东西放在想象中的房间里。NeRF就像一个AI的记忆宫殿，它'记住'了整个3D场景，可以从任何角度回忆出来。"

**比喻3：光学魔法**
> "哈利波特的隐身衣让光线弯曲。NeRF像一个反向的魔法——它学习如何让光线'应该'如何弯曲，从而从2D照片重建3D世界。"

### 6.3 高斯点云（3D Gaussian Splatting）

**比喻1：彩色泡泡**
> "想象数以万计的彩色肥皂泡漂浮在空中，每个泡泡中心最浓，边缘渐渐消失。当我们从某个角度看，这些泡泡叠加在一起就形成了一幅画面。这就是3D高斯点云——用软软的'电子泡泡'表示世界。"

**比喻2：印象派绘画**
> "印象派画家用许多小色点组成画面。3DGS类似，但每个'色点'是一个3D高斯，有位置、大小、颜色和透明度，可以在不同视角呈现不同面貌。"

**比喻3：数字雕塑**
> "传统雕塑家用黏土一点点塑形。3DGS就像是用无数个发光的黏土小球，每个都可以变形、移动、变色，组合起来就成了逼真的数字世界。"

### 6.4 位置编码（Positional Encoding）

**比喻：音阶与和弦**
> "想象你只听一个低音C，很难分辨旋律。但如果你加上C的高八度、再高八度，形成和弦，就能听出更多细节。位置编码就像给坐标加上'高八度'，让神经网络能感知更精细的空间细节。"

### 6.5 哈希编码（Hash Encoding）

**比喻：图书馆索引卡**
> "想象一个巨型图书馆，每本书的位置由一个索引卡决定。Instant-NGP的多分辨率哈希编码就像是这样一套索引系统——从粗到细快速找到每本书（特征），而不需要遍历整个图书馆（大MLP）。"

### 6.6 Score Distillation Sampling (SDS)

**比喻：雕塑家的批评家**
> "想象一个雕塑家（3D模型）在工作，旁边站着一个艺术评论家（扩散模型）。雕塑家从不同角度展示自己的作品，评论家根据文本描述给出建议：'这里应该更像一只猫'。SDS就是评论家指导雕塑家改进的过程。"

---

## 七、数学推导要点

### 7.1 体渲染方程推导

**从连续到离散：**

1. **连续体渲染方程：**
```
C = ∫₀ᴰ T(t) · σ(t) · c(t) dt
其中 T(t) = exp(-∫₀ᵗ σ(s) ds)
```

2. **微分关系：**
```
dT/dt = -T(t) · σ(t)
```

3. **离散化（分段常数假设）：**
在区间 [tᵢ, tᵢ₊₁] 假设 σ 和 c 为常数：
```
∫_{tᵢ}^{tᵢ₊₁} T(t)σc dt = cᵢ · Tᵢ · (1 - exp(-σᵢδᵢ))
其中 δᵢ = tᵢ₊₁ - tᵢ
Tᵢ = exp(-Σⱼ₌₀ⁱ⁻¹ σⱼδⱼ)
```

4. **最终离散形式：**
```
Ĉ = Σᵢ₌₁ᴺ Tᵢ · αᵢ · cᵢ
其中 αᵢ = 1 - exp(-σᵢδᵢ)
Tᵢ = Πⱼ₌₁ⁱ⁻¹(1 - αⱼ)
```

**物理解释：**
- αᵢ：光线在该点被吸收的概率
- Tᵢ：光线到达该点未被吸收的概率
- 乘积 Tᵢ·αᵢ：光线恰好在此点首次被阻挡的概率

### 7.2 位置编码原理

**动机：**
- MLP有频谱偏置，倾向于学习低频函数
- 需要高频细节来重建精细纹理

**正弦编码：**
```
γ(p) = [p, sin(2⁰πp), cos(2⁰πp), ..., sin(2ᴸ⁻¹πp), cos(2ᴸ⁻¹πp)]
```

**映射到高维：**
- 输入：3D坐标 (x, y, z) ∈ ℝ³
- 输出：ℝ^(3+6L)，通常L=10，输出维度63

**频率选择：**
- 频率呈指数增长：2⁰, 2¹, 2², ..., 2ᴸ⁻¹
- 覆盖从低频到高频的全部频谱

**集成位置编码（IPE）推导：**

对于圆锥内的区域，假设服从高斯分布 N(μ, Σ)：
```
E[sin(ωᵀx)] = sin(ωᵀμ) · exp(-½ωᵀΣω)
E[cos(ωᵀx)] = cos(ωᵀμ) · exp(-½ωᵀΣω)
```
- 这是正弦函数在高斯分布下的期望
- 协方差Σ导致高频成分的衰减

### 7.3 球谐函数（Spherical Harmonics）

**定义：**
球谐函数 Yₗᵐ(θ, φ) 是球坐标系下拉普拉斯方程的角度解：
```
Yₗᵐ(θ, φ) = Kₗᵐ · Pₗ^{|m|}(cosθ) · e^{imφ}
```
其中：
- Pₗᵐ：连带勒让德多项式
- Kₗᵐ：归一化常数
- l：阶数（band），l ≥ 0
- m：次数，-l ≤ m ≤ l

**实值球谐函数：**
```
yₗᵐ = {
  √2 · Re(Yₗᵐ)  if m > 0
  Yₗ⁰           if m = 0
  √2 · Im(Yₗᵐ)  if m < 0
}
```

**前3阶（9个基函数）：**
```
l=0: 1个常数基函数
l=1: 3个线性基函数（对应x, y, z）
l=2: 5个二次基函数
```

**用于3DGS颜色表示：**
```
c(v) = Σₗ₌₀ᴸ⁻¹ Σₘ₌₋ₗˡ cₗₘ · yₗᵐ(v)
```
- v：观察方向（单位向量）
- cₗₘ：可学习的系数（每个高斯3通道×系数数量）
- L=3时，共9个系数，27个参数

**优势：**
- 正交基函数，独立控制不同频率
- 紧凑表示视角依赖的颜色变化
- 旋转时系数有简单变换规则

### 7.4 3D高斯协方差矩阵

**参数化（保证正定性）：**
```
Σ = RSSᵀRᵀ
```
- S = diag(s)：缩放矩阵，s ∈ ℝ³
- R ∈ SO(3)：旋转矩阵，用四元数 q ∈ ℝ⁴ 表示

**投影到2D：**
```
Σ' = JWΣWᵀJᵀ
```
- W：世界到相机的变换
- J：投影变换的仿射近似雅可比矩阵
```
J = [∂u/∂x, ∂u/∂y, ∂u/∂z]
    [∂v/∂x, ∂v/∂y, ∂v/∂z]
```

---

## 八、关键概念速查

### 8.1 术语表

| 术语 | 英文 | 定义 |
|-----|------|------|
| 神经辐射场 | Neural Radiance Field | 用神经网络表示的连续5D辐射场 |
| 体积渲染 | Volume Rendering | 沿光线积分密度和颜色 |
| 位置编码 | Positional Encoding | 将低维坐标映射到高维频率空间 |
| 透射率 | Transmittance | 光线未被遮挡的概率 |
| 不透明度 | Opacity/Alpha | 光线被阻挡的概率 |
| 球谐函数 | Spherical Harmonics | 球面上的正交基函数 |
| 高斯溅射 | Gaussian Splatting | 将3D高斯投影到2D并混合 |
| SDS | Score Distillation Sampling | 从扩散模型蒸馏知识优化3D |
| 哈希编码 | Hash Encoding | 多分辨率哈希表存储特征 |
| 多视角一致性 | Multi-view Consistency | 不同视角渲染结果的一致性 |

### 8.2 损失函数汇总

**NeRF重建损失：**
```
L = Σ ||C_rendered - C_gt||²
```

**SDS损失：**
```
∇θL_SDS = E[(ε_pred - ε) · ∇θx]
```

**3DGS损失：**
```
L = (1-λ)||C - C_gt||₁ + λ(1-SSIM(C, C_gt))
```

### 8.3 性能基准

| 方法 | 训练时间 | 渲染FPS (1080p) | PSNR |
|-----|---------|----------------|------|
| NeRF | 1-2天 | <1 | ~31 |
| Mip-NeRF 360 | 48h | 0.07 | ~33 |
| PlenOctrees | 预处理 | 100+ | ~30 |
| Instant-NGP | 5秒-5分钟 | 60+ | ~32 |
| 3D Gaussian Splatting | 10-30分钟 | 100+ | ~33 |
| MobileNeRF | 预处理 | 30+ (移动端) | ~29 |

---

## 九、参考文献

### 核心论文

1. **NeRF**: Mildenhall et al. "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", ECCV 2020
2. **Mip-NeRF**: Barron et al. "Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields", ICCV 2021
3. **NeRF++**: Zhang et al. "NeRF++: Analyzing and Improving Neural Radiance Fields", arXiv 2020
4. **Mip-NeRF 360**: Barron et al. "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields", CVPR 2022
5. **PlenOctrees**: Yu et al. "PlenOctrees for Real-Time Rendering of Neural Radiance Fields", ICCV 2021
6. **Instant-NGP**: Müller et al. "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding", SIGGRAPH 2022
7. **MobileNeRF**: Chen et al. "MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures", CVPR 2023
8. **3D Gaussian Splatting**: Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
9. **DreamFusion**: Poole et al. "DreamFusion: Text-to-3D using 2D Diffusion", ICLR 2023
10. **Magic3D**: Lin et al. "Magic3D: High-Resolution Text-to-3D Content Creation", CVPR 2023
11. **MVDream**: Shi et al. "MVDream: Multi-view Diffusion for 3D Generation", ICCV 2023

### 综述论文

12. **NeRF Survey**: Gao et al. "NeRF: Neural Radiance Field in 3D Vision, Introduction and Review", arXiv 2022
13. **3D Generation Survey**: Cao et al. "A Survey on 3D-aware Image Synthesis", 2023

---

## 十、延伸阅读建议

### 入门路径
1. 先理解传统计算机图形学基础（光线追踪、光栅化）
2. 阅读NeRF原始论文并运行开源代码
3. 体验Instant-NGP快速训练
4. 尝试3D Gaussian Splatting

### 进阶方向
- **效率优化**: TensoRF, VMRF, Gaussian compression
- **动态场景**: D-NeRF, 4D Gaussian Splatting
- **生成模型**: DreamFusion后续工作, Diffusion+3D
- **几何提取**: NeuS, Neuralangelo, SuGaR

---

*研究笔记完成时间：2025年3月*
*适用章节：第39章 - 3D视觉生成*
