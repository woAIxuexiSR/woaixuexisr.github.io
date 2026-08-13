---
title: "DirectL: Efficient Radiance Fields Rendering for 3D Light Field Displays"
authors:
  - "Zongyuan Yang"
  - "Baolin Liu"
  - "Yingde Song"
  - "Lan Yi"
  - "Yongping Xiong"
  - "Zhaohe Zhang"
  - "Xunbo Yu"
category: "Rendering"
track: "Journal"
source: "arxiv"
institution: "Beijing University of Posts and Telecommunications"
tags:
  - "Radiance Fields"
  - "Light Field Displays"
  - "Autostereoscopic Display"
  - "NeRF"
  - "3D Gaussian Splatting"
  - "Real-Time Rendering"
links:
  paper: "https://doi.org/10.1145/3687897"
  project: "https://direct-l.github.io/"
---

## 一句话总结

DirectL 通过深入分析柱透镜光场显示屏中"空间光线—屏幕子像素"的交织映射关系，提出子像素复用与逐光线（ray-order）渲染范式，使 NeRF 与 3DGS 能够直接渲染编码图像而非多视点图像，在裸眼 3D 光场屏上实现最高约 40 倍的渲染加速且几乎不损失视觉质量。

## 研究背景

光场显示（Light Field Displays, LFDs），尤其是无需佩戴设备、支持多用户观看的柱透镜自动立体显示（Autostereoscopic Displays），近年来随着 LCD 面板分辨率提升与成本下降逐渐走向消费级市场（如 Looking Glass、X-real）。然而其大规模应用长期受两大难题制约：

- **内容创作难**：非专业用户难以制作 3D 显示内容。将 2D 资源转为网格（mesh）门槛高，且辐射场导出网格存在几何误差与拓扑混乱。
- **渲染开销大**：由于多个空间视点共享同一 LCD 面板像素，为保证每个视点的感知分辨率，需渲染大量像素。标准范式要求为一帧渲染至少 45~60 个略微偏移的高分辨率视点，再交织（interleaving）成一张编码图像，实时渲染极为困难。

辐射场（Radiance Fields）作为新兴 3D 表示，能从视频甚至单张图像重建高质量场景，天然重建了光场，非常适配光场显示。但已有工作仅通过高层 SDK 式集成将两者简单拼接，未在底层原理上探究其结合，仍受制于低渲染效率。本文的动机正是在更底层将辐射场与光场显示有机结合。

## 方法

### 整体框架

DirectL 的核心洞察是：在标准范式渲染的多视点图像中，超过 90% 的像素在交织后被丢弃；即便只渲染所需像素，仍需渲染约 3 倍原始分辨率（因为编码图像每个像素的 RGB 三个子像素来自不同视点）。DirectL 分三步（离线预计算 + 在线渲染）解决该问题：

```mermaid
flowchart LR
    A[光场屏硬件参数] --> B[离线: 计算光线配置<br/>+ 子像素复用<br/>+ 重排序索引 IG/IB]
    B --> C[在线: 逐光线视点无关渲染<br/>NeRF ray-casting / Ray-order 3DGS]
    C --> D[在线: 按索引重排 G/B 通道]
    D --> E[Reshape 成编码图像<br/>直接显示]
```

1. **离线预计算**：根据柱透镜屏的硬件参数（倾斜角 $$\alpha$$、线数 $$L_x$$、偏移 $$K_{offset}$$、视点数 $$N_v$$），计算每条所需光线相对于正面光场中心的变换矩阵，并确定 G、B 通道重排为编码图像所需的索引矩阵 $$I_G, I_B$$。
2. **在线逐光线渲染**：对 NeRF 与 3DGS 均执行视点无关的 ray-order 渲染，得到每条光线颜色。
3. **在线重排**：按索引重排 G、B 通道，取前 $$w h$$ 行 reshape 成 $$(h, w, 3)$$ 的编码图像帧直接上屏。

### 交织原理与子像素复用（Subpixel Repurposing）

柱透镜屏中，任一子像素 $$(x, y, k)$$ 到其栅格单元左缘的距离决定了它属于哪个视点：

$$d_{offset} = 3 \cdot y + 3 \cdot x \cdot \tan(\alpha) + k - K_{offset}$$

$$x_{offset} = d_{offset} \bmod L_x$$

$$v = \left\lfloor \frac{N_v \cdot x_{offset}}{L_x} \right\rfloor$$

由此得到唯一确定于硬件参数的视点编号矩阵 $$M_v$$。关键观察：同一像素位置的 R、G、B 子像素因 $$x_{offset}$$ 不同而分属不同视点，若只采样 R 通道，则同位置的 G、B 子像素被闲置。作者据此提出**子像素复用**：依据"同一栅格单元内，R 子像素会与最邻近的同视点 G、B 子像素一同进入人眼"这一成像规律，将闲置子像素启发式地重新分配到编码图像中的合适位置。该策略进一步将需渲染的像素数量再减少约 55%，视觉质量影响可忽略。为避免同视点 RGB 子像素相距过远导致画质下降，算法将面板划分为若干含 $$P_w$$ 个栅格单元的互不相交区域，在每个区域内独立执行复用（搜索范围见原文式 14）。

### NeRF 的视点无关 ray-casting

给定光场中心外参 $$L_0$$，各视点相机外参为：

$$T_{v \times 4 \times 4} = R_{v \times 4 \times 4} \cdot L_0^{4 \times 4}$$

再依据预计算的光线集合 $$\tilde{R}_{rays}$$ 逐条计算光线原点 $$\boldsymbol{o}_i$$ 与方向 $$\boldsymbol{d}_i$$。NeRF 天然支持逐像素渲染，作者用预计算光线替换单相机发射的全部光线；虽然 NeRF 颜色计算是视点相关的，但渲染整张光场图像是视点无关的（因为它对应仅由硬件参数决定的离散正面光场）。该范式同样适用于任何支持 image-order 渲染的方法（如 NeLFs）。

### Ray-order 3DGS

原生 3DGS 采用基于 tile 的物体序（object-order）光栅化，多视点渲染的计算成本与视点数 $$N_v$$ 成正比。由于所需像素在各视点中不规则但均匀分布，无法用方形 tile 高效求交。作者改用**基于光线投射的逐光线 3DGS**：

- 借助光线追踪算法，从 3D 高斯构建紧凑的 SAH 8-wide BVH（每个叶节点为高斯的 AABB），SAH 代价为：

$$SAH = \sum_{n \in I} A_n \cdot c_{node} + \sum_{n \in L} A_n \cdot P_n \cdot c_{prim}$$

- 对每条光线求与高斯的最大概率交点，得到线性解：

$$t_j = \frac{(\boldsymbol{\mu}_j - \boldsymbol{o}_i)^\top \Sigma_j^{-1} \boldsymbol{d}_i}{\boldsymbol{d}_i^\top \Sigma_j^{-1} \boldsymbol{d}_i}$$

- 每条光线维护固定容量 $$c_h$$ 的最大堆记录交点距离，遍历结束后排序并 alpha 混合。堆容量 $$c_h$$ 在画质与效率间构成权衡。

作者也坦承其局限：优化仍用光栅化，导致训练—推理不一致带来轻微退化；无 tile 划分使单视点同分辨率渲染慢于光栅化。并提出可微光线投射微调、2D 高斯切平面加速、OIT 免排序等未来方向。

## 实验结果

在 7.9 英寸 2K（48 视点）、15.6 英寸 4K（60 视点）、65 英寸 8K（96 视点）三块柱透镜屏上评测，硬件为 Intel Xeon Silver 4110 + 单张 NVIDIA RTX 3090。以标准范式 HR（原生屏分辨率）编码图像为基准用 RMSE 度量，并对 30 名受试者做用户视觉感知打分。

NeRF（Instant-NGP / SNeRG）在合成 Blender 数据集上的渲染帧率（FPS）对比（节选自原文主实验）：

| 方法 | Instant-NGP 7.9″2K | Instant-NGP 15.6″4K | SNeRG 7.9″2K | SNeRG 15.6″4K |
|---|---|---|---|---|
| Multi-View HR | 0.576 | 0.150 | 4.238 | 0.859 |
| Ray-Order Standard（无复用） | 9.709 | 2.967 | 12.305 | 4.163 |
| Ray-Order Ours | **18.519** | **6.803** | **25.406** | **8.402** |

要点：

- 即便不使用子像素复用，仅靠逐光线渲染 + 预计算光线（Standard），FPS 已全面超过所有 Multi-View 方法；对 NeRF 而言 Standard 生成的光场图像与 HR 完全一致。
- 加入子像素复用后（Ours），帧率在 HR 基础上平均再提升一个数量级；SNeRG 在 7.9 英寸 2K 上达到 25.4 FPS，成为首个能在 48 视点 2K 光场屏上实时（≥25 FPS）高质量渲染辐射场的方法。
- 论文摘要报告相较标准范式最高约 **40 倍**加速。
- 用户感知打分（3D 效果 / 清晰度 / 舒适度加权）显示 Ours 质量介于 MR 与 HR 之间，在较大屏幕上接近或优于 HR。
- 从渲染像素比例看，子像素复用使 Ours 相对原生分辨率仅需约 1.37~1.45 倍像素，而标准 HR 需约 2.6~4.17 倍。

## 亮点与局限

亮点：

- 首次在**底层原理层面**打通辐射场与光场显示，而非高层 SDK 拼接，揭示两者的天然协同。
- 子像素复用是一个巧妙的启发式：利用被丢弃的闲置子像素，进一步减少约 55% 渲染像素，且几乎不损画质，尤其利于 image-order 辐射场渲染。
- **仅修改渲染流程**、不改动训练，可无缝集成到下游辐射场重建/生成任务，通用性强（同样适用 NeLFs 等 image-order 方法）。

局限：

- Ray-order 3DGS 因训练用光栅化、推理用光线投射而存在不一致，带来轻微质量退化。
- 缺少 tile 划分，单视点同分辨率下 3DGS 渲染速度慢于原生光栅化。
- 65 英寸 8K + SNeRG 因 WebGL 显存溢出、索引无法一次性加载等工程限制未能报告完整数据，8K 光场渲染仍是挑战。
- 缺乏编码图像感知质量的客观定量指标，评价主要依赖 30 人的主观用户研究。

## 延伸思考

- DirectL 把"多视点渲染 + 交织"重构为"逐光线渲染 + 通道重排"，本质是把光场显示的物理成像约束前移到渲染阶段，这一思路可能启发其他显示形态（如全息、集成成像）的高效渲染。
- 子像素复用揭示了显示硬件的冗余在算法层可被榨取的空间，与"渲染必要像素"的思想呼应；若能建立编码图像感知质量的可微客观指标，或可将复用从启发式升级为端到端优化。
- Ray-order 3DGS 与近期高斯光线追踪工作方向一致，若结合 2D 高斯切平面求交与免排序 OIT，有望缩小与光栅化的单视点速度差距，并支持动态场景（需动态 BVH）。
- 该工作把裸眼 3D 内容创作门槛从"做网格"降到"拍视频重建辐射场"，对光场显示的消费级普及具有现实推动意义。
