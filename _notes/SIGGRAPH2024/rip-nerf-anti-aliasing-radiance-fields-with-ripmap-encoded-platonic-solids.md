---
title: "Rip-NeRF: Anti-aliasing Radiance Fields with Ripmap-Encoded Platonic Solids"
authors:
  - "Junchen Liu"
  - "Wenbo Hu"
  - "Zhuo Yang"
  - "Jianteng Chen"
  - "Guoliang Wang"
  - "Xiaoxue Chen"
  - "Yantong Cai"
  - "Huan-ang Gao"
  - "Hao Zhao"
category: "Rendering"
track: "Conference"
source: "arxiv"
institution:
  - "Beihang University"
  - "Tencent AI Lab"
  - "Beijing Institute of Technology"
  - "Tsinghua University"
tags:
  - "NeRF"
  - "Anti-Aliasing"
  - "Novel View Synthesis"
  - "Radiance Field"
  - "Area Sampling"
links:
  paper: "https://doi.org/10.1145/3641519.3657402"
  project: "https://github.com/JunchenLiu77/Rip-NeRF"
---

## 一句话总结

用「柏拉图立体投影 + Ripmap 编码」把锥形采样产生的各向异性 3D 高斯，精确且高效地投影到多个方向的 2D 平面并做各向异性区域采样，让不同锥体的各向异性区域在特征上可区分，从而在保持高效训练的同时渲染出高保真、抗锯齿的新视角图像。

## 研究背景

神经辐射场（NeRF）用神经网络把场景表示为连续的 5D 函数，在新视角合成、几何重建、内容生成等任务上推动了显著进展。但由于对连续物理世界做离散采样，渲染结果常出现锯齿与模糊等走样伪影。

围绕抗锯齿有两条主要思路：

- 提高采样率（多重采样）：Zip-NeRF 在高效的哈希网格混合表示上做多重采样，效果好但为刻画单个区域需要大量样本，在渲染质量与计算/存储开销间陷入两难。
- 降低场景频率（预滤波，即区域采样）：Mip-NeRF 通过积分位置编码开创了 NeRF 抗锯齿，但纯隐式表示训练与渲染都很慢；Tri-MipRF 用三张正交 mipmap 做区域采样，兼顾效率与紧凑，但其各向同性机制无法刻画锥体投射普遍产生的各向异性区域。

问题的核心在于：锥形采样得到的锥台可用各向异性 3D 高斯来刻画，而各向同性的区域采样会把来自不同锥体、形状不同的两个各向异性区域模糊地映射到同一个采样区域，导致表示歧义（如麦克风表面周期性网格被渲染得模糊）。本文要解决的正是如何精确、高效地对各向异性 3D 区域做特征化。

## 方法

### 整体框架

给定一组标定好的多视角图像，为每个像素投射一个锥体并划分为多个锥台，每个锥台用均值与协方差 $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 的各向异性 3D 高斯刻画；再把该 3D 高斯投影到某个柏拉图立体的各个不平行面上、在每个面上用 Ripmap 编码做各向异性区域采样查询特征，聚合后由一个小 MLP 估计颜色与密度，最后体渲染成像，整个系统用光度损失端到端优化。

```mermaid
flowchart LR
    A[像素锥体投射<br/>划分锥台] --> B[各向异性 3D 高斯<br/>均值 μ / 协方差 Σ]
    B --> C[柏拉图立体投影<br/>投到 n 个不平行面]
    C --> D[每面 2D 高斯<br/>μ_proj / Σ_proj]
    D --> E[Ripmap 编码<br/>四线性插值查特征 f_i]
    E --> F[聚合 f_i + 方向编码]
    F --> G[小 MLP F_θ<br/>估计颜色 c / 密度 σ]
    G --> H[体渲染成像<br/>光度损失端到端优化]
```

### 关键设计

Ripmap 编码（各向异性区域采样）。Tri-MipRF 用 2D mipmap 支持各向同性圆盘的区域采样，采样区域是正方形，无法精确刻画投影后各向异性 2D 高斯（其轴对齐包围盒本质是矩形）。作者借鉴传统图形学中处理视线与 UV 轴接近对齐时走样的 Ripmap（各向异性 mipmap），提出可学习的 Ripmap 编码。编码 $R$ 含 $L\times L$ 个层级，基层 $R_{0,0}$ 是形状为 $H\times W\times C$ 的可学习特征网格 $F$，其余层级由对低层做各向异性平均池化得到：

$$R_{i,j}=\begin{cases}\mathrm{Avg}_{2\times 1}(R_{i,j-1}) & j\neq 0\\ \mathrm{Avg}_{1\times 2}(R_{i-1,j}) & i\neq 0\ \&\ j=0\\ F & \text{otherwise}\end{cases}$$

只有基层可学习，其余层级由它派生，因此编码紧凑且层级间一致。查询时对各向异性 2D 高斯做四线性插值 $\boldsymbol{f}=R(p_x,p_y,l_x,l_y)$，其中查询位置与层级由投影高斯的均值与协方差决定：

$$p_d=\mu_d,\qquad l_d=\log_2\!\left(\frac{w\,\sigma_d}{r}\right),\quad d\in\lbrace x,y\rbrace$$

$\sigma_x,\sigma_y=\sqrt{\mathrm{diag}(\boldsymbol{\Sigma}_{\text{proj}})}$ 为两轴标准差，$w$ 调节覆盖多少概率质量，$r$ 为重建场景包围球半径。位置对应采样区域位置、层级对应采样区域大小。

柏拉图立体投影（3D 空间分解）。把 3D 空间分解成若干 2D 平面已被证明有效且紧凑（内存从 $O(n^3)$ 降到 $O(n^2)$）。但只用三张正交平面时，主轴沿立方体两条不同体对角线的两个 3D 椭球会得到相同的 2D 轴对齐包围盒，在 Ripmap 编码下无法区分。作者提出把 3D 高斯投影到更多、方向分布均匀的平面上再拼接特征，使不同高斯的 2D 包围盒更可区分、特征更具判别力。作者比较了柏拉图立体面、黄金螺旋、球面蓝噪声等均匀分布方案与球面白噪声对照组，发现均匀分布组明显更好且彼此接近，故为简单起见选用柏拉图立体（四面体、立方体、八面体、十二面体、二十面体），其各面为全等正多边形、二面角相等。记平面及外法向为 $\lbrace P_i\rbrace$、$\lbrace \boldsymbol{\phi}_i\rbrace$，平面局部 2D 轴需与法向垂直：

$$\begin{aligned}&x_i=X,\ y_i=Y &&\text{if }\boldsymbol{\phi}_i=Z\\ &x_i=Z\times\boldsymbol{\phi}_i,\ y_i=x_i\times\boldsymbol{\phi}_i &&\text{otherwise}\end{aligned}$$

各向异性 3D 高斯的特征化。用投影矩阵 $M_i=[x_i,y_i]$ 把 3D 高斯投到每个平面得到 2D 高斯：

$$M_i=[x_i,y_i],\quad \boldsymbol{\mu}^{i}_{\text{proj}}=M_i^{\top}\boldsymbol{\mu},\quad \boldsymbol{\Sigma}^{i}_{\text{proj}}=M_i^{\top}\boldsymbol{\Sigma}\,M_i$$

在每个平面的 Ripmap 编码上查询得 $\boldsymbol{f}_i$，把所有平面的特征拼接成 3D 高斯的最终特征 $\boldsymbol{f}$。柏拉图立体面数不同还提供了质量与效率（训练时间、显存）之间的灵活权衡；默认采用二十面体（含十个不平行面），基层设为 $H=W=512$、$C=16$，$w=2.0$。

## 实验结果

在多尺度 Blender 数据集上（1×/2×/4×/8× 平均），Rip-NeRF 在 PSNR、LPIPS 上超过所有对比方法，SSIM 与 Zip-NeRF 相当但更快更省：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | 训练时间 | 模型大小 |
|------|-------|-------|--------|---------|---------|
| Tri-MipRF | 35.30 | 0.976 | 0.028 | 5.5 min | 48 MB |
| Zip-NeRF | 36.69 | 0.985 | 0.021 | 4.5 h | 592 MB |
| 3DGS | 30.05 | 0.963 | 0.039 | 7.5 min | 27 MB |
| Rip-NeRF25k | 36.16 | 0.979 | 0.024 | 32 min | 160 MB |
| Rip-NeRF（本文） | 37.23 | 0.984 | 0.019 | 2.6 h | 160 MB |

相比 Zip-NeRF，本文训练时间约为一半（2.6h 对 4.5h）、模型约为四分之一（160MB 对 592MB）、训练显存约 20GB（对约 80GB）、渲染约 3 FPS（对 0.25 FPS）；Rip-NeRF25k 仅用 11.76% 的训练时间即可媲美 Zip-NeRF。在单尺度 Blender 上 Rip-NeRF 的 PSNR 也最高（35.44）。在自采真实数据集（四个含精细周期结构的物体）上，Rip-NeRF 平均 PSNR 38.89，优于 Tri-MipRF（38.09）与 Zip-NeRF（37.84）。

消融显示：单独使用柏拉图立体投影（PSP）仅提升约 0.37% PSNR，单独使用 Ripmap 编码（RE）反而略降；两者结合才产生协同效应（PSNR 平均提升约 3.44%），说明整体大于部分之和。平面数量并非越多越好（PS4 相比 PS3 各指标反而下降），关键在于各向异性高斯在不同平面上的投影足够多样，从而突破模型容量限制。

## 亮点与局限

亮点：

- 抓住"锥形采样产生各向异性区域、而各向同性区域采样无法区分"这一痛点，用 Ripmap 各向异性预滤波精确刻画椭圆足迹，每个区域仅需一次采样。
- 柏拉图立体投影把 3D 空间分解为多方向 2D 平面，既解决三正交平面的投影歧义，又保持 $O(n^2)$ 的紧凑内存；正交三平面可视为立方体的特例。
- 在质量、效率、存储三方面同时占优：以远小于 Zip-NeRF 的训练时间、显存与模型大小取得更好或相当的渲染质量，并提供面数可调的质量-效率权衡。
- PSP 与 RE 的协同效应清晰，消融验证了两者缺一不可。

局限：

- 面向有界场景，难以处理无界场景。作者推测两点原因：非近似凸形状会把自遮挡位置的信息投影到同一 2D 区域；Mip-NeRF 360 的空间扭曲机制与 2D 式表示不易兼容。
- 相比 Tri-MipRF、3DGS，模型偏大（160MB）、训练时间偏长（完整版 2.6h）。

## 延伸思考

Rip-NeRF 把图形学中经典的 Ripmap 各向异性纹理滤波思想迁移到神经渲染的区域采样，指出"预滤波是否精确"取决于采样核能否匹配各向异性足迹，为区域采样类抗锯齿方法提供了更本质的视角。柏拉图立体投影这种"多方向平面分解 + 特征拼接"的思路，也可看作对 tri-plane 表示的自然推广，对如何在紧凑 2D 网格上无歧义地编码 3D 各向异性信息具有借鉴意义。而其在无界场景上的局限，指向了把空间扭曲、遮挡感知投影纳入这类平面分解表示的后续改进方向。
