---
title: "Lumiere: A Space-Time Diffusion Model for Video Generation"
authors:
  - "Omer Bar-Tal"
  - "Hila Chefer"
  - "Omer Tov"
  - "Charles Herrmann"
  - "Roni Paiss"
  - "Shiran Zada"
  - "Ariel Ephrat"
  - "Junhwa Hur"
  - "Guanghui Liu"
  - "Amit Raj"
  - "Yuanzhen Li"
  - "Michael Rubinstein"
  - "Tomer Michaeli"
  - "Oliver Wang"
  - "Deqing Sun"
  - "Tali Dekel"
  - "Inbar Mosseri"
category: "Neural & Generative"
track: "Conference"
source: "arxiv"
institution:
  - "Google Research"
  - "Weizmann Institute"
  - "Tel-Aviv University"
  - "Technion"
tags:
  - "Text-to-Video"
  - "Diffusion Model"
  - "Video Generation"
  - "Space Time Unet"
  - "Image-to-Video"
links:
  paper: "https://doi.org/10.1145/3680528.3687614"
---

## 一句话总结

Lumiere 用一个 Space-Time U-Net（STUNet）在时间与空间上同时下/上采样，一次前向直接生成整段全帧率视频，摆脱了"稀疏关键帧 + 时间超分级联"范式，从而获得全局连贯的运动。

## 研究背景

文本到图像（T2I）扩散模型已能生成高分辨率、贴合复杂文本的照片级图像，但训练大规模文本到视频（T2V）模型仍是难题：运动建模对误差敏感，时间维度带来显著的显存与算力压力，且需要更大规模数据来拟合更复杂的分布。

现有 T2V 模型普遍采用级联设计：基础模型先生成相距较远的关键帧，再由时间超分（TSR）模块在不重叠片段中补齐中间帧。作者指出这种范式在全局运动连贯性上存在固有缺陷：

- 基础模型输出的关键帧被激进降采样，快速运动出现时间混叠、语义模糊；
- TSR 模块只能在固定的小时间窗内工作，无法在整段视频上一致地消解混叠歧义（如周期性的行走运动）；
- 级联训练存在域差异——TSR 在真实下采样帧上训练，推理时却要插值生成帧，误差不断累积。

一个被以往工作忽视的事实是：几乎所有 inflation 方案都只做空间下/上采样，全网络维持固定的时间分辨率。

## 方法

Lumiere 框架由两部分组成：一个在粗空间分辨率上生成整段视频的基础模型，以及一个时间感知的空间超分（SSR）模型。基础模型输出 $$128 \times 128$$、80 帧、16fps（5 秒）的视频，SSR 上采样到 $$1024 \times 1024$$。

```mermaid
flowchart LR
    T[文本提示] --> STUNet[STUNet 基础模型<br/>一次生成全部 80 帧<br/>128x128, 16fps]
    STUNet --> MD[SSR + MultiDiffusion<br/>重叠时间窗聚合]
    MD --> OUT[1024x1024, 80 帧<br/>全局连贯视频]
```

### 关键设计一：Space-Time U-Net（STUNet）

在预训练 T2I U-Net 中交错插入时间模块，并在每个空间 resize 模块后插入时间下/上采样模块，使网络能在压缩的时空表示上完成大部分计算：

- 除最粗层外，各层插入分解式时空卷积（factorized space-time convolution），相比全 3D 卷积降低算力并增加非线性，相比 1D 卷积增强表达力；
- 时间注意力的算力随帧数二次增长，因此只在最粗层（时空高度压缩）使用，此处可低开销地堆叠多层时间注意力。

只训练新增的时间参数，冻结预训练 T2I 权重。由于引入了时间下/上采样，无法像常规 inflation 那样在初始化时让 T2V 等价于 T2I；作者发现将这些模块初始化为最近邻下/上采样能提供良好起点。

### 关键设计二：时间维度的 MultiDiffusion

SSR 网络因显存限制只能处理短片段，直接拼接会在窗口边界产生外观不一致。作者将 MultiDiffusion 扩展到时间轴：把带噪视频 $$J \in \mathbb{R}^{H \times W \times T \times 3}$$ 切成重叠片段 $$\{J_i\}_{i=1}^{N}$$，每段做 SSR 得到 $$\Phi(J_i)$$，再通过求解

$$\arg\min_{J'} \sum_{i=1}^{n} \lVert J' - \Phi(J_i) \rVert^2$$

来协调各片段预测，其解等价于在重叠窗口上线性组合，从而获得整段视频的全局连贯结果。

### 关键设计三：条件生成的统一接口

对基础模型追加两路输入：掩码条件视频 $$C \in \mathbb{R}^{T \times H \times W \times 3}$$ 与二值掩码 $$M \in \mathbb{R}^{T \times H \times W \times 1}$$，与噪声视频拼接成 $$\langle J, C, M \rangle \in \mathbb{R}^{T \times H \times W \times 7}$$，把首层卷积通道从 3 扩展到 7 并微调。通过不同的 $$C, M$$ 配置，统一支持图像到视频（首帧作为条件）、视频修复/外扩、以及 Cinemagraph（仅在指定区域内动、其余静止）。风格化生成则在空间层的微调权重 $$W_{style}$$ 与原始权重 $$W_{orig}$$ 之间线性插值：$$W_{interpolate} = \alpha \cdot W_{style} + (1-\alpha) \cdot W_{orig}$$，其中 $$\alpha \in [0.5, 1]$$。

## 实验结果

在 UCF101 上进行零样本文本到视频生成评测，报告 FVD 与 IS：

| 方法 | FVD ↓ | IS ↑ |
| --- | --- | --- |
| MagicVideo | 655.00 | - |
| Emu Video | 606.20 | 42.70 |
| Video LDM | 550.61 | 33.45 |
| Show-1 | 394.46 | 35.42 |
| Make-A-Video | 367.23 | 33.00 |
| PYoCo | 355.19 | 47.76 |
| SVD | 242.02 | - |
| **Lumiere (Ours)** | **332.49** | **37.54** |

Lumiere 取得有竞争力的 FVD 与 IS。作者同时指出这些指标不能忠实反映人类感知，且 UCF101 协议只取生成视频的 16 帧、无法刻画长时运动。在 2AFC 用户研究中（每个基线约 400 份判断），Lumiere 在文本到视频、图像到视频上的视频质量与文本对齐度均优于 ImagenVideo、Pika、ZeroScope、AnimateDiff、Gen-2 与 SVD 等基线。

## 亮点与局限

亮点：
- 用单一基础模型一次性生成整段全帧率视频，避免时间超分级联，从架构上解决全局运动连贯性问题；
- 首次在 inflation 方案中引入时间下/上采样，在压缩时空表示上计算，兼顾效率与表达力；
- 无 TSR 级联使得下游应用（图像到视频、修复、风格化、SDEdit 视频编辑）都能以统一直观的接口实现，可直接套用现成编辑方法。

局限：
- 不能生成多镜头或含场景转场的视频；
- 基于像素空间 T2I 模型构建，需要额外的空间超分模块；作者认为设计原则同样适用于潜空间视频扩散，值得进一步探索；
- 存在被滥用于制造虚假/有害内容的风险，需配套检测工具。

## 延伸思考

- 把 STUNet 的时空下/上采样思想迁移到潜空间扩散（如 Stable Diffusion 系）是自然的下一步，可望进一步降低算力门槛同时保留全局连贯性。
- "一次生成整段"与"自回归/关键帧插值"是两条根本不同的路线：前者上限在于单次显存下能容纳的时长，如何在保持全局一致的同时扩展到分钟级、多镜头叙事仍是开放问题。
- 时间维度的 MultiDiffusion 提供了一个把"局部窗口预测聚合为全局解"的通用范式，可能对其他需要长序列一致性的生成任务（长音频、长时序仿真）有借鉴价值。
