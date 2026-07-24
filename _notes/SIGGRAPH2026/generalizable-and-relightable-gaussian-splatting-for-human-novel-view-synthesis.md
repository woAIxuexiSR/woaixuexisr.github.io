---
title: "Generalizable and Relightable Gaussian Splatting for Human Novel View Synthesis"
authors:
  - "Yipengjing Sun"
  - "Shengping Zhang"
  - "Chenyang Wang"
  - "Shunyuan Zheng"
  - "Zonglin Li"
  - "Xiangyang Ji"
category: "Rendering"
track: "Conference"
source: "arxiv"
institution:
  - "Harbin Institute of Technology"
  - "Tsinghua University"
tags:
  - "3D Gaussian Splatting"
  - "Relighting"
  - "Novel View Synthesis"
  - "Physically Based Rendering"
  - "Human Rendering"
links:
  paper: "https://doi.org/10.1145/3799902.3811132"
  project: "https://sypj-98.github.io/grgs/"
---

## 一句话总结

GRGS 用一套前馈、全监督的框架，把多视角 2D 观测中的几何、材质、光照线索投影到 3D 高斯上，实现无需逐人优化、可泛化到新身份和新光照的高保真人体新视角合成与重光照。

## 研究背景

- 领域现状：人体新视角合成（NVS）追求在任意视角生成真实图像，进一步加入重光照能力后可在合成新视角的同时编辑光照。目前主流有三条路线——基于 mesh 的传统重建、NeRF/3DGS 的神经表示逆渲染、以及基于 2D 图像的编解码/扩散重光照。
- 核心痛点：3D 方法（NeRF、3DGS 逆渲染）依赖逐人优化，计算成本高，且逆渲染在任意光照下本身是病态问题，几何与材质难以准确解耦，泛化能力差；2D 图像方法虽然泛化好、效率高，但缺乏 3D 一致性约束，视角受控渲染时会出现闪烁，且往往忽略物理可解释性。
- 本文 idea：不做逐场景逆渲染优化，而是用监督式、数据驱动的策略，学习从多视角 2D 观测前馈地预测 3D 高斯的几何、材质与光照属性，并把物理渲染（PBR）嵌入进来保证光照传输的物理一致性。

## 方法

GRGS 的整体 pipeline 由两个核心模块串联：先用 Lighting-robust Geometry Refinement（LGR）模块从稀疏视角图像估计对光照鲁棒的深度与法线，再用 Physically Grounded Neural Rendering（PGNR）模块在这套几何基础上前馈回归高斯参数与内蕴属性，并融合物理渲染完成可编辑重光照。

```mermaid
flowchart LR
  A["稀疏多视角图像 + HDR 环境图"] --> B["LGR: 立体深度 + 法线细化"]
  B --> C["精确深度 / 法线几何"]
  C --> D["PGNR: 几何感知解码器"]
  D --> E["高斯参数 + 材质 + 可见性 + 间接光"]
  E --> F["物理渲染 PBR + 光栅化"]
  F --> G["任意视角重光照结果"]
```

关键设计分为三部分：

1. 光照鲁棒的几何细化（LGR）。借鉴 GPS-Gaussian 的泛化思路，用 RAFT-Stereo 式的立体匹配以视差作为跨视角几何约束来估计深度。关键在于：预训练深度估计器对光照变化敏感，因此作者用一个在大规模多视角"重打光"数据集上训练的共享特征提取器 $$\xi_{img}$$，学习对光照不变的反射率特征，从而缓解光照引起的特征错配。深度模块内部构建低分辨率 3D 相关体，用 GRU 迭代细化后再凸上采样得到全分辨率深度。由于凸上采样会丢失高频细节、产生棋盘伪影，再用一个轻量 U-Net 的法线细化模块，结合多尺度反射率特征与从深度梯度算出的粗法线 $$\mathbf{N}_c$$ 预测偏移 $$\Delta \mathbf{N}$$，得到细化法线 $$\mathbf{N}_f = \frac{\mathbf{N}_c + \Delta \mathbf{N}}{\lVert \mathbf{N}_c + \Delta \mathbf{N} \rVert}$$。

2. 物理接地的神经渲染（PGNR）。每个高斯点被参数化为四类属性：普通属性（位置、旋转、尺度、不透明度）、几何属性（法线、SH 编码的光可见性 $$\mathbf{v}\in\mathbb{R}^{16}$$）、材质属性（反照率、粗糙度）、光照属性（SH 编码的间接光 $$\mathbf{l}_{ind}\in\mathbb{R}^{48}$$）。这些 3D 属性都以像素对齐深度图为桥梁在 2D 图上表达，从而无需优化即可直接推断高斯点云。几何感知编码器 $$\xi_{geo}$$ 从深度和法线抽取几何特征，与反射率特征融合后经解码器与各头预测高斯参数与材质；反照率用残差项 $$\Delta \mathbf{A}$$ 加速收敛并起到去光照作用：$$\mathbf{A} = \text{Sigmoid}(\mathbf{I} + \Delta \mathbf{A})$$。

3. 光照参数化与物理渲染。对高分辨率 HDR 环境图做预滤波得到卷积版 $$\mathbf{L}_d'$$ 近似入射光积分，但会偏暗，于是引入直接光缩放因子 $$s_d$$ 做全局亮度补偿；同时用光照编解码器预测间接光图。最终按渲染方程对每个高斯点计算 PBR 颜色，入射辐射按可见性拆成直接光与间接光两部分：$$\mathbf{L}(\boldsymbol{\omega}_i) = V(\boldsymbol{\omega}_i)\left(s_d \mathbf{L}_d'(\boldsymbol{\omega}_i)\right) + \left(1 - V(\boldsymbol{\omega}_i)\right)\mathbf{L}_{ind}(\boldsymbol{\omega}_i)$$，BRDF 采用简化的 Disney 模型，最后光栅化成图。

此外，2D-to-3D 投影训练策略是效率关键：用环境光遮蔽（AO）、直接光、间接光图作为光度监督信号，直接从 2D 图像监督在 3D 空间优化光照参数，避免显式光线追踪的昂贵开销；其中 AO 与直接光联合监督可见性估计，并用梯度截断的硬阴影融合方案加速间接光收敛。

## 实验结果

在合成数据集上与两个 3DGS 逆渲染方法（R3DGS、ARGS）在法线、反照率、AO、重光照四个维度对比，GRGS 在全部指标上领先：

| 方法 | Normal MAE↓ | Albedo PSNR↑ | AO PSNR↑ | Relighting PSNR↑ | Relighting LPIPS↓ |
|------|------|------|------|------|------|
| R3DGS | 10.208 | 23.584 | 20.983 | 21.983 | 0.162 |
| ARGS | 6.941 | 25.937 | 23.721 | 23.879 | 0.147 |
| Ours | 5.369 | 27.536 | 24.470 | 27.977 | 0.099 |

消融方面，几何评估中 LGR 模块相比 GPS-Gaussian 把 NVS 的 PSNR 从 30.444 提升到 31.694、深度端点误差 EPE 从 1.463 降到 0.692，验证了光照鲁棒特征与法线细化的作用；光照传输消融表明直接光缩放因子 $$s_d$$ 能修复卷积环境光带来的偏暗，加入可见性与间接光后能准确重现手臂、球体等遮挡阴影。方法只需每高斯采样约 40 条光线即可平衡质量与效率，配合 TensorRT 加速推理可达 20 FPS。此外还在 GPS-Gaussian、DNA-Rendering 的真实数据上验证泛化性，并与 2D 方法 IC-Light、SwitchLight 做了定性对比。

## 亮点与局限

- 亮点：
  - 首次把可泛化前馈重建与物理接地的重光照结合在 3D 高斯框架里，摆脱了逐人优化和病态逆渲染的束缚。
  - LGR 通过在合成重打光数据上学习光照不变特征，显著提升了不同光照下的几何一致性，成为连接 2D 图像空间与 3D 高斯域的关键桥梁。
  - 2D-to-3D 投影训练用 AO/直接光/间接光图监督，绕开显式光线追踪，兼顾物理一致性与推理效率（20 FPS）。

- 局限：
  - 强依赖大规模高质量人体扫描与 HDR 环境图合成的重打光数据（Twindom、THuman2.0、384 张环境图），数据构建成本高，且合成到真实的域差异仍需验证。
  - 训练开销大（单卡 4090 约四天，LGR 10 万步 + 整体 30 万步）。
  - 采用简化 Disney BRDF 和 SH 编码的可见性/间接光，对高频镜面、复杂材质（如半透明、毛发）的表达能力有限；输入依赖多视角且需要相对准确的相机标定。

## 延伸思考

GRGS 走的是"用监督数据把逆渲染问题变成前馈预测"的路线，和 GPS-Gaussian 的可泛化重建、以及 relightable 3DGS 系列的逐人逆渲染形成有趣对照——本质是用大规模合成数据的先验替换掉在线优化。值得追问的是：这套 2D-to-3D 光照监督能否扩展到动态序列（时间一致性）与全身动画驱动；SH 编码的可见性/间接光在高频阴影和强镜面下的上限在哪；以及在缺乏配对合成 GT 的野外场景里，如何减少对 LightStage 式数据的依赖。与扩散式重光照方法结合，或许能在泛化性与 3D 一致性之间取得更好折中。
