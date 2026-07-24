---
title: "SynthVerse: A Large-Scale Diverse Synthetic Dataset for Point Tracking"
authors:
  - "Weiguang Zhao"
  - "Haoran Xu"
  - "Xingyu Miao"
  - "Qin Zhao"
  - "Rui Zhang"
  - "Kaizhu Huang"
  - "Ning Gao"
  - "Peizhou Cao"
  - "Mingze Sun"
  - "Mulin Yu"
  - "Tao Lu"
  - "Linning Xu"
  - "Junting Dong"
  - "Jiangmiao Pang"
category: "Image & Video"
track: "Conference"
source: "arxiv"
institution:
  - "Shanghai AI Laboratory"
  - "University of Liverpool"
  - "Zhejiang University"
tags:
  - "Point Tracking"
  - "Synthetic Dataset"
  - "Benchmark"
  - "3D Point Tracking"
  - "Video Tracking"
links:
  paper: "https://doi.org/10.1145/3799902.3811183"
  project: "https://weiguangzhao.github.io/SynthVerse/"
  code: "https://github.com/weiguangzhao/SynthVerse"
---

## 一句话总结

SynthVerse 用 Blender + Isaac Sim 双引擎搭了一套跨平台数据生成流水线，产出了一个规模达 581.6 万训练帧、覆盖 8 大领域的合成点跟踪数据集与评测基准，并证明用它微调能显著提升现有跟踪器在多域场景下的泛化能力。

## 研究背景

- 领域现状：点跟踪（point tracking）要在复杂运动、遮挡和视角变化下持续追踪画面上的视觉点，是场景理解、4D 重建和机器人感知的基础能力。随着基础模型的出现，2D 与 3D 点跟踪都进步很快。
- 核心痛点：进展被高质量训练数据卡住。真实场景很难拿到准确的轨迹真值（尤其 3D 还需要精确深度、相机位姿和时序稳定的 3D 轨迹），人工标注昂贵且带误差。现有合成数据集（Kubric、PointOdyssey、Dynamic Replica 等）规模有限、多样性不足：基本都是单一渲染管线、以第三人称（allocentric）视角为主，缺少铰接物体、具身操作、导航式动态和第一人称观测，导致训练分布偏置、跨域时性能骤降。
- 本文 idea：造一个足够大、足够多样的合成数据集和基准。用跨平台管线补齐此前缺失的领域与物体类型（动画电影内容、具身操作、场景导航、手物交互、铰接与可变形物体），并系统性地评测现有跟踪器在广泛域偏移下的表现。

## 方法

整体上，SynthVerse 是一条"原始素材采集 → 场景构建 → 多模态标注"的统一数据生产管线，横跨 Blender 与 Isaac Sim 两个引擎，从底层物理状态直接导出可靠的 3D 轨迹真值，再投影得到 2D 轨迹与可见性标注。

```mermaid
flowchart LR
  A["原始素材: 3D/4D 资产 / HDR 环境 / 网格场景 / 镜头级工程"] --> B["场景构建: 具身 / 导航 / 动画 / 3DHOI"]
  B --> C["Blender / Isaac Sim 渲染"]
  C --> D["原始数据: RGB / 点云 / 轨迹"]
  D --> E["标注: 轨迹投影 + 遮挡检测"]
  E --> F["数据集与基准"]
```

关键设计分四点：

1. **跨平台双引擎管线**：Blender 擅长影视级渲染与可变形/动画资产，Isaac Sim 擅长物理仿真与具身/机器人交互。两者结合让数据集同时支持第一人称与第三人称视角，并覆盖单一管线做不到的领域。所有部分以统一格式导出，便于扩展到新模拟器。

2. **广域素材采集与资产组织**：从多来源收集并整理成若干资产类别——具身操作来自 GenManip（VLA 驱动、文本指令控制机器人，每个场景用环绕、俯视和两个手部相机采集）；人物约 2 万角色 + 2000 动作序列，另加 Mixamo 类人资产；动物覆盖 75 个物种、每种 20+ 动作；物体分刚体（OmniObject3D）、铰接物体（PartNet-Mobility、Infinite Mobility 的 URDF 转 USD 进 Isaac Sim）、可变形物体；动画电影用 Blender Studio 公开的镜头级工程；导航用 InternScenes 室内场景 + NavDP 生成相机轨迹；手物交互复用 Hot3D 的物体与手部状态轨迹在 Blender 里重建。

3. **场景构建与运动调度**：场景级资产（具身、动画、导航）尽量保留原始布局和相机配置以维持真实构图；实例级资产（人、动物、手交互、铰接物体）则按 Kubric/PointOdyssey 的思路放进 HDR 环绕环境、用多视角相机（四个固定视角 + 一个环绕相机）渲染。针对不同资产标准化运动调度：铰接物体在关节有效范围内随机采样角度，带预设动画的资产则变速播放以拓宽运动分布。手交互子集用 MANO 手模型加高保真手部纹理合成交互动作。

4. **数据增强与可靠标注**：对相机初始位姿和轨迹加高斯噪声、随机化 FOV 覆盖近景与远景，扰动光源位置与强度，替换物体纹理/材质（如用 HTML 手部纹理替换默认 MANO 纹理）。标注上，先在 3D 空间跟踪表面点得到轨迹，再用相机内外参投影到各视角得到 2D 轨迹；可见性通过视锥检查、渲染深度一致性和实例分割一致性三者结合来判定遮挡。

## 实验结果

作者以开源且带训练/评测代码的 TAPIP3D 作为主模型做微调验证。下面这张主实验展示"在 SynthVerse 上微调后，跨三类基准（自建合成 / 公开合成 / 公开真实）的泛化提升"，指标为 3D 平均杰卡德 AJ3D（越高越好），带星号为微调后模型：

| 模型配置 | SynthVerse-mAverage | Dynamic Replica | LSFOdyssey | 公开真实(ADT/DriveTrack/PStudio) |
|------|------|------|------|------|
| TAPIP3D-world | 33.3 | 55.5 | 72.2 | 23.5 / 14.9 / 18.1 |
| TAPIP3D-world* | 41.6 | 57.2 | 73.9 | 24.3 / 15.4 / 18.6 |
| TAPIP3D-camera | 33.3 | 53.7 | 68.3 | 21.6 / 14.6 / 18.1 |
| TAPIP3D-camera* | 41.8 | 56.6 | 70.8 | 22.9 / 15.1 / 18.6 |

可以看到，在自建基准上 AJ3D 从 33.3 提升到约 41.6~41.8（增益最大，因为分布最匹配），在与 TAPIP3D 原训练分布相近的公开合成集上提升较温和，在真实基准上也一致改善，说明 SynthVerse 提供的是互补性的监督信号、有助于缩小 sim-to-real 域差距。

基准评测方面的关键结论：SynthVerse 基准分 Nav、Human、Animal、Objects、Embodied、Film、Interaction 八个子集，其中 Nav 与 Film 对所有方法都最难（视角快速变化、大幅相机运动、遮挡频繁），Interaction 与 Animal 相对容易；3D 跟踪整体比 2D 退化更明显、方差更大，且没有单一方法在所有域都最好，印证了基准的多样性和现有跟踪器对分布偏移的敏感。定性上，微调后 TAPIP3D 在手部等快速铰接/遮挡区域的轨迹丢失和漂移问题都明显缓解。

## 亮点与局限

- 亮点：
  - 规模与多样性上都大幅超越既有合成数据集（581.6 万训练帧、48K 序列，含约 1K 铰接、6K 可变形、4K 人物、75 动物序列），补齐了铰接物体、具身、导航、手交互、动画电影等此前缺失的域。
  - 跨平台双引擎 + 从物理状态直接导出轨迹，标注可靠且天然支持第一/第三人称视角。
  - 建立了统一的多域基准，系统揭示了现有 SOTA 跟踪器在域偏移下的短板，评测维度（AJ/APD 的 2D/3D + 遮挡准确率 OA）较全面。

- 局限：
  - 效验主要绑定单一模型 TAPIP3D 做微调，其他跟踪器只做了基准评测而非微调，数据集的普适训练价值论证还不够广（作者也把"benchmark 更多模型"列为未来工作）。
  - 提升幅度在真实基准上相对有限，sim-to-real 差距仍在；合成数据的外观/物理真实度对最终迁移效果的影响未做深入消融。
  - 数据主要靠聚合多个现成来源（GenManip、Hot3D、InternScenes 等），部分领域（如动物仅 75 物种）覆盖仍有天花板。

## 延伸思考

这篇本质是"数据即方法"的工作，思路和 Kubric、PointOdyssey 一脉相承，但把具身仿真（Isaac Sim）引入点跟踪数据合成是有意思的一步——它让机器人操作、导航这类第一人称、强交互的动态进入了跟踪训练分布，这恰是真实机器人感知最需要的。值得追问的是：合成多样性的边际收益在哪里？当域已经足够多时，继续堆量还是转向更精细的物理/外观真实度更划算？另外，基准揭示"没有方法在所有域都最好"，这暗示未来可能需要域自适应或混合专家式的跟踪器，而不是单一通用模型。对做 4D 重建、世界模型的人来说，这套带完整深度/位姿/轨迹/可见性的多模态标注也可以直接复用到跟踪之外的任务。
