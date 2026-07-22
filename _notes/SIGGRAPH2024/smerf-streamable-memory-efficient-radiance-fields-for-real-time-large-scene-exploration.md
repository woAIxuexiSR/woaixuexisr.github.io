---
title: "SMERF: Streamable Memory Efficient Radiance Fields for Real-Time Large-Scene Exploration"
authors:
  - "Daniel Duckworth"
  - "Peter Hedman"
  - "Christian Reiser"
  - "Peter Zhizhin"
  - "Jean-François Thibert"
  - "Mario Lucic"
  - "Richard Szeliski"
  - "Jonathan T. Barron"
category: "Rendering"
track: "Journal"
source: "arxiv"
institution:
  - "Google"
  - "University of Tübingen"
tags:
  - "Neural Radiance Fields"
  - "Real-Time Rendering"
  - "Novel View Synthesis"
  - "Volumetric Representation"
  - "Distillation"
  - "Large-Scale Scenes"
links:
  paper: "https://doi.org/10.1145/3658193"
  project: "https://smerf-3d.github.io"
---

## 一句话总结

SMERF 把一个高质量但慢得离谱的 Zip-NeRF 教师模型，蒸馏进一族以 MERF 为骨干、按相机位置分块的学生子模型：每个子模型都能表示整个场景但只把容量集中在自己负责的区域，渲染任意视角只需加载一个子模型。配合"色彩+几何"双重蒸馏、光线抖动数据增强与子模型一致性正则，SMERF 在浏览器里、在手机和笔记本上都能实时漫游数百平方米的大场景，质量逼近离线 SOTA，而渲染速度比教师快三个数量级。

## 研究背景

辐射场（Radiance Field）已成为重建与重渲染照片级真实 3D 场景的强力表示。它常以神经网络存储、用体渲染光线步进（ray marching）来渲染。这既是它最大的优点也是最大的软肋：神经网络能在足够算力预算下紧凑地表达复杂几何与视相关效果，但作为体积表示，渲染一张图的运算量随像素数而非图元数增长——最好的模型（如 Zip-NeRF）渲染一帧要做上千万次网络求值，一帧要数秒，完全无法实时。

于是实时辐射场只能在质量、速度、模型大小之间做妥协，一个悬而未决的问题是：它们能否与 3D Gaussian Splatting（3DGS）这类光栅化方法竞争？同时，现有实时方法（包括作者此前的 MERF）在中小场景表现不错，但难以扩展到多房间的大场景：单套三平面（triplane）容量有限，大量表面点投影到同一 2D 平面位置，无法同时表达多个表面的高频细节；靠提高分辨率来缓解则会让显存开销爆炸。

SMERF 的目标是肯定地回答上述问题：在消费级硬件（含智能手机）上实时渲染大空间，质量接近离线 SOTA，且显存预算与场景大小无关。

## 核心方法

SMERF 建立在 MERF 之上。先回顾 MERF：它把 3D 位置 $\mathbf{x}$ 映射到 8 维特征，用高分辨率三平面 $(\mathbf{P}_x,\mathbf{P}_y,\mathbf{P}_z)$ 与低分辨率稀疏体素网格 $\mathbf{V}$ 组合表示：

$$\mathbf{t}_{\text{MERF}}(\mathbf{x}) = \mathbf{P}_x(\mathbf{x}) + \mathbf{P}_y(\mathbf{x}) + \mathbf{P}_z(\mathbf{x}) + \mathbf{V}(\mathbf{x})$$

特征拆成密度、漫反射色和视相关特征，沿光线做体渲染 alpha 合成，最后用 SNeRG 式的延迟着色（deferred shading）小 MLP 解出像素颜色。SMERF 的两大贡献是**分层模型架构**与**蒸馏训练**。

### 贡献一：分层分块架构

SMERF 是一个三级层次结构，均以相机原点 $\mathbf{o}$ 为索引：(i) 选一个子模型；(ii) 子模型内按插值算出延迟着色网络参数；(iii) 局部体素邻域内用特征门控确定特征表示。

**坐标空间分块（Coordinate Space Partitioning）。** 把所有训练相机平移缩放到 $[-K/2, K/2]^3$ 立方体，再切成 $K^3$ 个 $[-1,1]^3$ 子体。每个子体关联一个独立的 MERF-like 子模型，并有自己的收缩空间（contraction）。关键点在于：**每个子模型仍表示整个场景**，只是各自把高分辨率容量分配给自己负责的格子，格子外的区域粗略建模。因此渲染某个相机只需一个子模型在显存中。相机到子模型的分配按原点所在子体决定：

$$s^* = \arg\min_{k\in\{1..K^3\}} D(\mathbf{o}, k), \quad D(\mathbf{o},k) = \min_{\mathbf{x}\in R_k} \|\mathbf{o}-\mathbf{x}\|_\infty$$

该分配对训练集外相机同样有效，从而支持光线抖动、子模型重分配、任意测试相机放置与 ping-pong 缓冲。只对含至少一个训练相机的子体实例化子模型，把子模型数从 $K^3$ 降到 $O(K^2)$。

**延迟着色网络分块（Deferred Appearance Network Partitioning）。** MERF 用一个小 MLP 解码视相关颜色，但其容量不足以复现大场景复杂的视相关效果，直接增大 MLP 又会拖慢渲染。SMERF 把每个子模型的域切成每轴 $P$ 个顶点的格子，每个格点关联一组网络参数 $\theta_{uvw}$，按相机原点三线性插值：

$$\theta = \text{Trilerp}(\mathbf{o}, \{\theta_{uvw}: (u,v,w)\in\{1,..,P\}^3\})$$

用三线性（而非最近邻）插值对防止相机移动时高光"跳变（popping）"至关重要。由于视相关 MLP 参数相比整体表示微不足道，这一分块几乎"免费"提升模型容量。

**特征门控（Feature Gating）。** MERF 把四路特征简单相加，会把高低分辨率信息纠缠在一起。SMERF 改用低分辨率体素特征去"门控"高分辨率三平面特征：取体素贡献的最后一维 $w(\mathbf{x}) = [\mathbf{V}(\mathbf{x})]_8$ 作为缩放因子：

$$\hat{\mathbf{t}}(\mathbf{x}) = w(\mathbf{x})\cdot\big(\mathbf{P}_x(\mathbf{x}) + \mathbf{P}_y(\mathbf{x}) + \mathbf{P}_z(\mathbf{x}) + \mathbf{V}(\mathbf{x})\big)$$

$$\mathbf{t}(\mathbf{x}) = \hat{\mathbf{t}}(\mathbf{x}) \oplus \mathbf{V}(\mathbf{x})$$

这样模型可在白墙等低频区域"关掉"高分辨率三平面，把精力集中到细节丰富处。

### 贡献二：蒸馏训练

模型容量增大是把双刃剑：它缺少 SOTA 离线模型那种鼓励合理重建的归纳偏置。SMERF 因此从一个已训练好的高质量 Zip-NeRF 教师蒸馏而来（教师在优化中冻结），同时蒸馏外观与几何。

**外观蒸馏。** 用教师从任意相机渲染的 $3\times3$ 图块作为"真值"，学生与教师图块间用 RMSE 加 DSSIM 组合损失：

$$\mathcal{L}_c = 1.5\cdot\text{DSSIM}(\mathbf{C}, \mathbf{C}^*) + \sum_{\mathbf{c}\in\mathbf{C}} \|\mathbf{c}-\mathbf{c}^*\|^2$$

**几何蒸馏。** 查询教师得到沿光线的加权区间 $\{((t_i,t_{i+1}), w_i^T)\}$，蒸馏其体渲染合成权重——因为权重是密度的函数，对权重的监督间接让师生的密度场在可见区域一致：

$$\mathcal{L}_\tau = \sum_i \big| w_i^T - w_i^S \big|$$

**数据增强（光线抖动 Ray Jittering）。** 蒸馏可在欧氏空间任意光线上监督学生。均匀采光线会命中物体内部或无关区域，只用数据集相机光线又太局限。折中做法是对数据集相机光线做随机扰动：原点加各向同性高斯噪声，方向在 $\epsilon$ 邻域均匀采样：

$$\hat{\mathbf{o}}\sim\mathcal{N}(\mathbf{o},\sigma^2\mathbf{I}), \quad \hat{\mathbf{d}}\sim\mathcal{U}(\{\mathbf{v}\in\mathbb{R}^3: \|\mathbf{v}-\mathbf{d}\|_2<\epsilon, \|\mathbf{v}\|_2=1\})$$

实验取 $\sigma=0.03K$、$\epsilon=0.03$。消融显示光线抖动是单项贡献里最关键的：去掉它在 treehill 场景训练会发散。

**子模型一致性。** 坐标分块相当于并行训练多个独立学生子模型，测试时跨子模型切换需时序一致。SMERF 不做多子模型混合（太慢），而是引入光度一致性损失：每条训练光线用其"归属"子模型 $s$ 和一个随机相邻子模型 $\hat{s}$ 各渲染一次，约束两者颜色一致：

$$\mathcal{L}_s = \|\mathbf{c}_s(\mathbf{r}) - \mathbf{c}_{\hat{s}}(\mathbf{r})\|^2$$

此外构造训练批次时，先把每条光线分给其"归属"子模型，再随机把 20% 光线重分给相邻子模型以提供额外视角。

### 渲染与实时查看器

训练后"烘焙（baking）"出预计算资产：对最高分辨率 $2048^3$ 占据网格做 $3\times3\times3$ 中值滤波去浮游物，产出每个子模型的高分辨率 2D 特征图与稀疏低分辨率 3D 特征网格（量化字节数组），延迟网络参数保留浮点，资产用 gzip 压缩存储。查看器基于 MERF 的 OpenGL 片元着色器，新增子模型支持、参数插值、距离网格加速结构等优化，帧率比 MERF 提升超 70%。为隐藏加载延迟，用"ping-pong"策略：相机进入新子体时后台加载新子模型，旧子模型继续渲染，就绪后切换——峰值显存至多两个子模型。延迟网络参数插值因整帧共享相机原点，每帧只需在 CPU 上插值一次。

## 实验与结果

**大场景（Zip-NeRF 数据集：Berlin、Alameda、London、Nyc）。** 随分块分辨率 $K$ 增大，SMERF 精度逼近 Zip-NeRF 教师，$K=5$ 时差距不到 0.1 PSNR、0.01 SSIM。定量结果（越大越好 / 越小越好）：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Disk (MB)↓ |
|------|-------|-------|--------|------|------------|
| MERF (ours) | 23.49 | 0.746 | 0.444 | 283 | 128 |
| 3DGS | 25.50 | 0.810 | 0.369 | 441 | 212 |
| Ours (K=1) | 25.44 | 0.777 | 0.412 | 329 | 118 |
| Ours (K=3) | 27.09 | 0.823 | 0.350 | 220 | 1628 |
| Ours (K=5) | 27.28 | 0.829 | 0.339 | 204 | 4108 |
| Zip-NeRF (教师) | 27.37 | 0.836 | 0.305 | 0.25 | 607 |

在大场景上比现有实时 SOTA 高 **1.78 dB** PSNR，渲染比 Zip-NeRF 快约三个数量级。

**mip-NeRF 360（无界室内外场景）。** 这些场景较小，无需分块（$K=1$）即可高质量。$K=1$ 版本在图像质量上超越所有已有实时模型，且渲染速度与 3DGS 相当：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ | Disk (MB)↓ |
|------|-------|-------|--------|------|------------|
| 3DGS | 27.20 | 0.815 | 0.214 | 260 | 740 |
| MERF (ours) | 24.95 | 0.728 | 0.302 | 278 | 153 |
| Ours (K=1) | 27.98 | 0.818 | 0.212 | 217 | 139 |
| Zip-NeRF (教师) | 28.78 | 0.836 | 0.177 | 0.25 | 607 |

比现有实时 SOTA 高 **0.78 dB** PSNR。

**跨设备实时性。** SMERF 在 iPhone、MacBook、桌面工作站上均能实时（手机上降分辨率）。作者的 MERF 实现在 iPhone 上达 58.3 FPS、桌面 187 FPS，$K=1$ 版本 iPhone 55.4 FPS、桌面 142 FPS，而 3DGS 缺少非 CUDA 设备的官方实现，无法在 iPhone/MacBook 运行。

**消融（mip-NeRF 360，$K=1$）。** 逐项关闭各贡献验证价值：光线抖动是单项最重要贡献（去掉会发散）；超参调优与更长训练（200k vs 25k 步）对细节重建关键；无几何监督（改用学习式 proposal 网络）明显掉点；无特征门控、无 MLP 网格均有非可忽略下降。**空间分辨率消融**表明：相比直接提高烘焙资产分辨率，通过增加子模型数 $K$ 提升分辨率能达到更高质量，且实时查看器显存受限于两个最大子模型的大小、增长缓慢，而单体模型基线显存随分辨率超线性增长。

## 贡献与局限

**贡献：**
- 提出面向大场景实时辐射场的**分块模型架构**，能在从手机到桌面工作站的硬件上高保真表示大场景，显存与场景大小解耦。
- 提出**辐射场蒸馏训练流程**，把精确但笨重的 Zip-NeRF 教师的泛化能力与归纳偏置迁移到高度一致的学生子模型。
- 首个能在消费级硬件、浏览器中令人信服地实时渲染无界多房间大空间的方法；大场景质量近乎与 Zip-NeRF 无法区分。

**局限：**
- **存储成本高。** 大场景（$K=5$）烘焙资产可达数 GB，实时查看器会有加载事件和高网络占用。
- **训练开销大。** 除训练教师外还要优化 100k–200k 步；端到端一个代表性 Zip-NeRF 场景约需 33 小时（教师 3h + 蒸馏 24h + 烘焙 6h，16×A100），mip-NeRF 360 场景约 17 小时。
- 虽平均质量高于 3DGS，但并非在所有场景所有区域都更细致，作者归因于表示施加的体素结构。

## 延伸思考

SMERF 的核心思想——"用一个慢而准的离线模型当教师，蒸馏出一族快而一致的分块学生"——把"分而治之的大场景 NeRF"（如 Block-NeRF）从需要多子模型混合的昂贵渲染，推进到"单子模型即可渲染任意视角"的实时范式，关键在于让每个子模型都能表示整个场景、仅在容量分配上有别。这与同期用蒸馏做实时视图合成的 HybridNeRF（蒸馏 SDF）殊途同归，共同印证蒸馏是弥合"离线质量"与"实时速度"鸿沟的有效路径。其对高存储与长训练的局限，也指向后续在资产压缩、流式传输，以及减少对重量级教师依赖等方向的改进空间。
