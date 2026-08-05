---
title: "NeRO: Neural Geometry and BRDF Reconstruction of Reflective Objects from Multiview Images"
authors:
  - "Yuan Liu"
  - "Peng Wang"
  - "Cheng Lin"
  - "Xiaoxiao Long"
  - "Jiepeng Wang"
  - "Lingjie Liu"
  - "Taku Komura"
  - "Wenping Wang"
category: "Reconstruction"
track: "Journal"
source: "arxiv"
institution:
  - "The University of Hong Kong"
  - "Max Planck Institute for Informatics"
  - "Texas A&M University"
tags:
  - "Inverse Rendering"
  - "BRDF"
  - "Neural SDF"
  - "Multiview Reconstruction"
  - "Reflective Objects"
links:
  paper: "https://doi.org/10.1145/3592134"
  project: "https://liuyuan-pal.github.io/NeRO/"
  code: "https://github.com/liuyuan-pal/NeRO"
---

## 一句话总结

NeRO 只用一组已知相机位姿、无掩膜、无已知光照的多视角 RGB 图像，就能把强反光物体的表面几何和 BRDF 材质一起重建出来，输出可直接用于重打光的三角网格。

## 研究背景

- 领域现状：多视角三维重建近年在传统 MVS 与神经隐式表面（IDR、VolSDF、NeuS 等）方向都取得了很好的效果，能处理相当复杂的物体。
- 核心痛点：反光物体的镜面高光是视角相关的，破坏了绝大多数重建方法赖以工作的"多视角一致性"假设，导致 COLMAP、NeuS 这类方法在反光表面上重建出错误几何。现有能建模环境光的材质估计方法又普遍依赖物体掩膜、且大多只考虑远处的直接光，难以处理由物体自身或邻近区域反射造成的间接光。
- 本文 idea：把渲染方程显式地嵌入神经表面重建框架，用 BRDF 与环境光的交互来解释视角相关的高光，从而让镜面变化"反过来"帮助几何重建；并设计一套区分直接光/间接光的光照表示，配合两阶段流程使渲染方程可被 tractable 地求解。

## 方法

整体框架：NeRO 用一个 MLP 表示的神经 SDF 表达表面（沿用 NeuS 的体渲染），但把颜色函数换成基于微表面 BRDF 的着色函数。整个流程分两阶段——第一阶段在无掩膜条件下重建几何，第二阶段固定几何、用更精确的蒙特卡洛采样精修 BRDF 与环境光。

```mermaid
flowchart LR
  A["多视角 RGB + 位姿"] --> B["Stage I: 神经 SDF + 体渲染"]
  B --> C["split-sum 近似 + 积分方向编码 IDE"]
  C --> D["直接/间接光表示 + 遮挡概率"]
  D --> E["重建几何 (SDF 零水平集)"]
  E --> F["Stage II: 固定几何, MC 重要性采样"]
  F --> G["精确 BRDF + 环境光"]
  G --> H["带材质三角网格 → 重打光"]
```

关键设计：

1. 微表面 BRDF + 渲染方程作为颜色函数。每个采样点由材质 MLP 预测金属度 $$m$$、粗糙度 $$\rho$$、反照率 $$\boldsymbol{a}$$，颜色由渲染方程给出：
$$c(\omega_o) = \int_{\Omega} L(\omega_i) f(\omega_i, \omega_o)(\omega_i \cdot \boldsymbol{n})\, d\omega_i$$
其中 BRDF 分为漫反射项 $$(1-m)\boldsymbol{a}/\pi$$ 与镜面项 $$DFG / [4(\omega_i\cdot\boldsymbol{n})(\omega_o\cdot\boldsymbol{n})]$$。显式的着色机制让网络能拟合高频镜面色变，而不是像 NeuS 那样只用视角方向硬拟合，从而避免几何被"拧歪"。

2. split-sum 近似让积分可算。直接求解每个体渲染采样点上的光照积分是 intractable 的，Stage I 借用实时渲染常用的 split-sum，把"光照 × BRDF"的积分拆成光照积分与 BRDF 积分两部分。镜面部分近似为
$$c_{\text{specular}} \approx \Big(\int_{\Omega} L(\omega_i) D(\rho, \boldsymbol{t})\, d\omega_i\Big)\cdot\Big(\int_{\Omega}\frac{DFG}{4(\omega_o\cdot\boldsymbol{n})}\, d\omega_i\Big)$$
BRDF 积分部分可用预计算标量直接算出，只剩下光照积分是未知量。

3. 区分直接光与间接光的光照表示。以物体外接球为界：来自球外的是直接光，被球内表面反射的是间接光。光照写作
$$L(\omega_i) = [1 - s(\omega_i)]\, g_{\text{direct}}(SH(\omega_i)) + s(\omega_i)\, g_{\text{indirect}}(SH(\omega_i), \boldsymbol{p})$$
其中 $$g_{\text{direct}}$$ 只依赖方向（提供全局一致的强先验），$$g_{\text{indirect}}$$ 额外吃位置 $$\boldsymbol{p}$$（间接光随空间变化），$$s(\omega_i)\in[0,1]$$ 是由 MLP 预测的遮挡概率。光照积分用积分方向编码（IDE）近似，无需预滤波环境贴图。

4. 遮挡损失与两阶段。若放任 $$g_{\text{occ}}$$ 只从渲染损失学遮挡概率，它会与重建几何严重不一致、导致训练发散；因此用神经 SDF 沿反射方向去约束遮挡概率的监督（遮挡损失）。总损失为渲染损失、Eikonal 正则、遮挡损失，以及前 1k 步的稳定化正则之和。Stage II 固定几何后，改用蒙特卡洛重要性采样（在漫反射瓣与镜面瓣上分别采样）取代近似，得到更准确的 BRDF。

## 实验结果

作者构建了合成数据集 Glossy-Synthetic（8 个低粗糙度强反光物体）与真实数据集 Glossy-Real（5 个物体），用 Chamfer 距离评几何、用重打光图像的 PSNR 评 BRDF。下表为 Glossy-Synthetic 上的几何重建 Chamfer 距离对比（越低越好），其中 NDR 与 NDRMC 使用了真值掩膜，其余方法无掩膜：

| 方法 | Bell | Teapot | TBell | Angel | Avg.↓ |
|------|------|--------|-------|-------|-------|
| NeuS | 0.0146 | 0.0546 | 0.0348 | 0.0035 | 0.0233 |
| Ref-NeRF | 0.0137 | 0.0143 | 0.0216 | 0.0291 | 0.0241 |
| NDR* | 0.0122 | 0.0530 | 0.0821 | 0.0056 | 0.0329 |
| NDRMC* | 0.0045 | 0.0052 | 0.0046 | 0.0034 | 0.0253 |
| NeRO (本文) | 0.0032 | 0.0037 | 0.0035 | 0.0034 | 0.0042 |

NeRO 在平均 Chamfer 距离上把误差压到约基线的五分之一到八分之一，且在几乎每个物体上都取得最优或接近最优；即使对手用了真值掩膜也依然领先。重打光质量（PSNR）上同样优于 NDR、NDRMC、MII、NeILF 等材质估计方法。真实数据集 Glossy-Real 上结论一致：基线方法在多个反光物体上出现噪声、扭曲或空洞，NeRO 能稳定重建所有反光表面。

## 亮点与局限

- 亮点：
  - 把渲染方程显式引入神经表面重建，让"高光是麻烦"转变为"高光可利用"，无需物体掩膜即可重建强反光物体。
  - 直接光/间接光分离 + 遮挡概率的光照表示，让间接光也能被稳定建模，这是以往方法的普遍短板。
  - 两阶段设计（先几何、后 BRDF）把 tractable 的近似与精确的 MC 采样各用其所，产出可直接用于重打光的带材质网格。
- 局限：
  - 训练开销大：单物体几何重建约 20 小时、BRDF 估计约 5 小时（2080Ti），主要瓶颈在体渲染的海量采样点，作者也指出体素化表示有望加速。
  - 依赖已知相机位姿，且以外接球划分直接/间接光的假设更贴合"单个物体"场景，对开放大场景的适用性有限。
  - split-sum 与 IDE 是近似着色，第一阶段的 BRDF/光照并不精确，需要第二阶段专门精修。

## 延伸思考

NeRO 的核心洞见——用物理着色模型把视角相关外观从"噪声"变成几何线索——与后续在高斯泼溅上处理反射（如各类反射感知的 3D Gaussian Splatting 工作）思路相通，只是表示从 SDF 换成了显式基元。其两阶段"先几何后材质"的解耦在实践中很稳健，但也提出一个值得追问的点：能否让几何与 BRDF 在单阶段内联合、可微地相互约束而不发散？此外，把外接球式的直接/间接光划分推广到多物体乃至完整场景的逆渲染，是把这类方法从"物体级"推向"场景级"的关键一步。
