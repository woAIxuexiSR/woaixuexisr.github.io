---
title: "Editable Physically-based Reflections in Raytraced Gaussian Radiance Fields"
authors:
  - "Yohan Poirier-Ginter"
  - "Jeffrey Hu"
  - "Jean-François Lalonde"
  - "George Drettakis"
category: "Rendering"
track: "Conference"
source: "arxiv"
institution:
  - "Université Laval"
  - "Inria"
tags:
  - "Gaussian Splatting"
  - "Ray Tracing"
  - "Path Tracing"
  - "Specular Reflections"
  - "Scene Editing"
  - "Differentiable Rendering"
  - "Radiance Fields"
  - "Intrinsic Decomposition"
links:
  paper: "https://doi.org/10.1145/3757377.3763971"
  project: "https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/"
  code: "https://repo-sam.inria.fr/nerphys/editable-gaussian-reflections/"
---

## 问题背景

3D Gaussian Splatting (3DGS) 与 NeRF 能从多视角照片中实现高质量、实时的新视角合成，也能重建大部分镜面反射。但它们对反射的建模方式存在根本缺陷：反射被"烘焙"进了方向相关的辐射分量（3DGS 用球谐、NeRF 用 MLP），并且往往表现为放置在反射面背后的"假几何"（fake mirror geometry）。这种表示把反射与几何纠缠在一起，导致无法一致地编辑场景——当移动物体或改变材质时，反射不会随之正确更新。

本文提出首个允许对辐射场进行**反射一致的交互式编辑**的方法。核心思路是：用两个独立但并行的优化过程重建同一个场景——一个负责漫反射分量，一个负责镜面分量，分别由输入的漫反射缓冲与镜面缓冲监督。这里的"specular"泛指一切非漫反射（从粗糙光泽到纯镜面）。

## 核心方法

### 漫反射/镜面分离与物理反射

方法基于渲染方程，将 BRDF 近似为漫反射与镜面两项之和 $$f = f_d + f_s$$，从而将出射辐射拆为两个积分项：

$$L_o(x, \omega_o) = \int_\Omega L_i(x,\omega_i) f_d(\omega_i, n)\cos\theta_i\, d\omega_i + \int_\Omega L_i(x,\omega_i) f_s(\omega_i, n, \omega_o)\cos\theta_i\, d\omega_i.$$

用 Heckbert 的光路记法，第一项计算所有 $$L((D|S)^*)DE$$ 路径，第二项计算所有 $$L((D|S)^*)SE$$ 路径。关键做法是：漫反射分量被监督重建成一个"缓存版本"的漫反射光照 $$\hat{L}_d$$，其作用类似于已经乘过反照率的辐照度缓存（irradiance cache），从而避免了完整路径追踪中最昂贵的第一项：

$$L_o(x, \omega_o) = \hat{L}_d + \int_\Omega L_i(x,\omega_i) f_s(\omega_i, n, \omega_o)\cos\theta_i\, d\omega_i.$$

该缓存中"烘焙"了阴影和漫反射颜色渗透。镜面项则通过对高斯基元做路径追踪计算，天然支持多次反弹（实践中最大路径长度设为 3，兼顾性能与精度）。

材质采用 Cook-Torrance BRDF（Disney BRDF 公式的子集），参数为粗糙度 $$\rho$$ 和基础反射率 $$F_0$$。每个高斯基元不使用球谐，而是存单一漫反射 RGB 值，并附带法线、粗糙度、$$F_0$$ 等属性。光线沿途用 alpha-blending 累积这些量，在期望终止深度处求交，用累积得到的法线、粗糙度、$$F_0$$ 依据 Cook-Torrance BRDF 重要性采样反射光线。训练时用 1 spp，离线渲染用 128 spp 后接 OptiX 降噪器。

### 单一场景的稳定优化

以往方法（如 NeRF-casting、EnvGS）发现梯度直接穿过反射会破坏几何重建，因而用两套独立辐射场分开建模主场景和反射——但这样就无法做多反弹编辑，也难以重建输入视角中未直接看到的反射物体。本文用漫反/镜面分离来消解纹理与被反射物体之间的歧义，从而在保持单一场景表示的同时维持训练稳定。

关键设计：法线与 BRDF 直接通过输入缓冲监督恢复，**不**对 RGB 图像做逆向渲染求导，因此即使梯度穿过反射，场景仍保持良好形态。训练损失包含三部分：

- 漫反射损失（L1）：$$L_d = \lambda_d\, \ell(\hat{d}, d^*)$$
- 镜面损失，匹配路径后续段贡献之和：$$L_r = \lambda_r\, \ell\!\left(\sum_{i=2}^{K} \hat{c}_i,\; s^*\right)$$
- 附加属性损失（仅作用于第一段路径）：$$L_a = \lambda_t\,\ell(\hat{t}, t^*) + \lambda_n\,\ell(\hat{n}, n^*) + \lambda_\rho\,\ell(\hat{\rho}, \rho^*) + \lambda_{F_0}\,\ell(\hat{F}_0, F_0^*)$$

最终目标为两个子目标同时优化：

$$\min_{\theta, d} \mathbb{E}[L_d + L_r] + \min_{\theta, t, n, \rho, F_0} \mathbb{E}[L_a].$$

权重取 $$\lambda_d = 5.0,\ \lambda_r = 3.0,\ \lambda_t = 2.5,\ \lambda_n = 2.5,\ \lambda_\rho = 1.0,\ \lambda_{F_0} = 1.0$$，所有损失在线性空间计算。训练调度先用 750 次迭代仅优化第一段光线做漫反射预热，随后开启全部反射，并加入 $$M = 75000$$ 个远场高斯（围绕场景原点按 $$\sigma = 4S$$ 的正态分布采样、在 $$3\sigma$$ 处截断，$$S$$ 为场景范围），用于逐步重建环境中被反射的物体。此外采用稠密初始化替代稠密化以加速收敛。

### 高效多反弹光线追踪器

为满足优化与渲染的速度需求，作者在四方面改进高斯光线追踪：

- **OBB 包围盒**：用 OptiX 硬件加速的 AABB 实例化变换来模拟有向包围盒（OBB），避免显式计算协方差矩阵、也无需自行求逆高斯变换。
- **避免多次 BVH 遍历**：3DGRT 需要多次遍历 BVH 来分组排序最近基元；本文改为单次遍历，将所有相交高斯存入逐像素链表（PPLL）"回放缓冲"，再多次循环收集、排序、积分最近 16 个高斯，带来 10–50% 提速。
- **前向/后向融合**：每个像素在前向后立即启动后向，用 PPLL 存储后向所需数据；前向加后向仅比单独前向慢 2.23 倍（3DGS 为 3.31 倍）。
- **激进的基元截断**：透射率沿光线快速衰减，远处高斯贡献极小。直接丢弃会使梯度有偏，因此对保留基元用其颜色近似来缩放梯度，从而在优化期间也能安全地做透射率阈值截断。

### 从图像推断分离层的网络

由于现有方法都不预测本文所需的非漫反射缓冲，作者将预训练的 Stable Diffusion 2 微调成一个单步逆向渲染网络，用文本提示切换预测不同缓冲（漫反射、镜面、深度、法线、粗糙度），沿用 RGB↔X 的思路。训练混合使用 Hypersim 与 InteriorVerse 数据集，采用潜空间 MSE 损失以加快训练。

## 实验与结果

由于没有其他方法能做一致的镜面反射编辑，作者以"漫反/镜面分离质量"作为尽力而为的评测代理指标。

- **合成场景**：三个由已有工作改造、调亮材质、去除透明度的场景（1152×768）。在仅用网络预测输入时，本文方法在漫反/镜面分离上平均优于 Gaussian Shader、3DGS-DR、Reflective-GS 以及并行工作 EnvGS。用真值缓冲时（proof-of-concept）分离质量大幅领先。由于本方法不制造"假几何"来复现输入 RGB，最终图像的 PSNR 反而更低，但质量已足够支撑编辑。
- **优化速度**：本文方法训练明显快于所有对比方法，约为次优 EnvGS 的 5 倍。
- **真实场景**：用微调网络推断分离层，尽管预测质量远非完美，仍能完成类似的编辑演示；对比 EnvGS 可见其漫反射层残留镜面内容。
- **光线追踪器对比**：作为 3DGS 的直接替换（1 spp、仅 RGB、7k 次迭代、无球谐），本文追踪器在 MipNeRF 各场景上相比 3DGRT 训练提速约 4 倍、FPS 提升约 2 倍，PSNR 相当。
- **编辑演示**：可一致地移动反射物体、复制反射物体、改变基础反射率 $$F_0$$、旋转物体、修改粗糙度（让物体变漫反射或金属化），反射均随编辑正确更新，并支持多反弹。方法还能重建仅通过镜面间接可见的物体（如仅在镜中出现的书封面），甚至重建相机背后的环境。

## 局限与结论

主要局限在于用于提取分离层的网络性能（作者相信会持续改进）；其次不支持透明度（需处理反射与折射两条光线）；阴影被"烘焙"进漫反射层，编辑时不会更新，完整场景编辑需要至少部分重光照能力；即使有完美的分离层，进一步提升图像质量仍需更精确地重建被反射的场景。

总体上，本文通过漫反/镜面分离与多反弹反射支持，实现了辐射场中实时、一致的镜面反射编辑，关键在于对漫反射层与镜面层分别监督、并配合一系列高斯光线追踪的性能改进。
