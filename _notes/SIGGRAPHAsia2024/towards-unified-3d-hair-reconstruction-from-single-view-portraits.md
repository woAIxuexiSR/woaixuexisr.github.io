---
title: "Towards Unified 3D Hair Reconstruction from Single-View Portraits"
authors:
  - Yujian Zheng
  - Yuda Qiu
  - Leyang Jin
  - Chongyang Ma
  - Haibin Huang
  - Di Zhang
  - Pengfei Wan
  - Xiaoguang Han
track: "Conference"
source: arxiv
category: "Reconstruction"
institution:
  - The Chinese University of Hong Kong, Shenzhen
  - Kuaishou Technology
tags:
  - 3D Hair Reconstruction
  - Single View
  - 3D Gaussian Splatting
  - Diffusion Prior
  - Score Distillation Sampling
links:
  paper: "https://doi.org/10.1145/3680528.3687597"
---

## 一句话总结

提出一套统一流水线，仅凭单视图人像即可重建包含辫发与非辫发在内的多样 3D 发型：先用合成多视图数据集训练两个面向头发的扩散先验，再以 3D 高斯为表示、经视图级与像素级两级细化优化出高质量、多视一致的 3D 头发。

## 研究背景

单视图 3D 头发重建的核心难点在于发型形态差异极大，从简单波波头到带辫子、发髻、扭花的复杂造型都要覆盖。现有主流方法存在明显局限：

- 基于检索或全监督学习的方法依赖规模小、风格窄的 3D 头发数据集，只能处理简单的非辫发，且难以恢复被遮挡的不可见区域。
- 针对辫发的专用方法（如辫元识别）虽能处理特定造型，但无法生成非可见部分，也不是自动化流程。
- 直接套用通用的 image-to-3D 生成方法有两大障碍：一是现有扩散先验训练于通用物体而非头发领域，重建效果次优；二是头发纹理要求高频、丝状细节，通用方法质量有限。同时缺乏足够大规模、足够多样的 3D 头发数据来训练理想先验。

作者的目标是用同一条流水线同时处理辫发与非辫发，摆脱对昂贵 3D 头发几何数据的依赖，转而从 2D 扩散先验中优化出 3D 头发。

## 方法

整体框架是一个粗到细的优化式 2D 提升（2D-lifting）方法，以 3D 高斯作为底层头发表示。给定单张人像，先做图像预处理对齐，再蒸馏出粗糙高斯 $$\Theta_0$$，随后经视图级细化得到 $$\Theta_1$$、像素级细化得到最终 $$\Theta_2$$。

```mermaid
flowchart LR
    I[输入人像 I] --> A[人脸关键点对齐<br/>+ 头发分割]
    A --> Ia[对齐图像 Ia]
    Ia --> C[SDS 优化<br/>HairSynthesizer]
    C --> T0[粗糙高斯 Θ0]
    T0 --> V[视图级细化<br/>HairSynthesizer]
    V --> T1[细化高斯 Θ1]
    T1 --> P[像素级细化<br/>HairEnhancer]
    P --> T2[最终高斯 Θ2]
    T2 --> R[任意视角渲染]
```

关键设计一：SynMvHair 合成多视图数据集。从原始 3D 头发资源中筛除粗糙、模糊数据后，得到 2,396 个 3D 发型（含 1,544 非辫发、852 辫发）与 82,682 张纹理贴图，每个发型有 5-188 张 UV 纹理。所有模型经艺术家归一化并对齐到同一模板身体，再在上半球随机相机位渲染出标定准确的多视图图像，用于训练 3D 感知的扩散先验。相比以往仅数百个非辫发模型的数据集，在规模与风格（尤其辫发）上都有显著扩展。

关键设计二：两个面向头发的扩散先验。HairSynthesizer 采用与 Zero-1-to-3 相同架构，在对齐图像 $$I_a$$ 与相对相机位姿 $$(R,T)$$ 条件下从噪声生成新视图 $$I_n = S(N\vert (I_a, R, T))$$；HairEnhancer 则以模糊头发图为条件，生成带丝状纹理的细节增强图 $$I_{enhance} = E(N\vert I_{blur})$$，用于弥补纹理细节。

关键设计三：粗糙高斯优化。沿用 DreamGaussian，在包含模板身体上半身的包围盒内初始化 $$\Theta_0$$，通过 SDS 损失与参考损失联合优化：

$$\nabla_{\Theta} \mathcal{L}_{SDS} = \mathbb{E}_{t,p,\epsilon}\left[(\epsilon_S(I^{(R,T)}_{\Theta_0}; t, I_a, (R,T)) - \epsilon)\frac{\partial I^{(R,T)}_{\Theta_0}}{\partial \Theta_0}\right]$$

参考损失约束参考视角的 RGB 与掩膜一致：$$\mathcal{L}_{ref} = \vert \vert I_a - I'_a\vert \vert _1 + \vert \vert M_a - M'_a\vert \vert _1$$。

关键设计四：两级高斯细化。视图级细化受扩散图像编辑启发，每步把 $$\Theta_0$$ 渲染图与噪声按系数混合后经 HairSynthesizer 去噪得到 $$I_{refine} = S((\gamma \cdot I_{\Theta_0} + (1-\gamma)\cdot N)\vert (I_a, R, T))$$，并以 L1 加感知损失优化；$$\gamma$$ 在优化中逐步增大，兼顾纹理质量与视图一致性。像素级细化再借 HairEnhancer 进一步优化：

$$\mathcal{L}_{enhance} = \vert \vert I_{\Theta_1} - E(N\vert I_{\Theta_1})\vert \vert _1 + \beta \cdot \vert \vert \phi(I_{\Theta_1}) - \phi(E(N\vert I_{\Theta_1}))\vert \vert _1$$

其中 $$\phi(\cdot)$$ 为 VGG 特征提取。最终 $$\Theta_2$$ 获得丝状细节纹理，并与统一模板身体对齐。

## 实验结果

在 SynMvHair 上按约 9:1 划分训练/测试，训练集 2,155 个发型（73,862 个带纹理网格），241 个发型用于评测。消融实验在测试子集上验证两级细化的有效性，指标随 $$\Theta_0 \to \Theta_1 \to \Theta_2$$ 逐级改善：

| 方法 | $$L_1 \downarrow$$ | $$Perc. \downarrow$$ | $$PSNR \uparrow$$ |
| --- | --- | --- | --- |
| HairSynthesizer | 0.01565 | 3.262 | 26.07 |
| $$\Theta_0$$ | 0.02377 | 4.427 | 24.92 |
| $$\Theta_1$$ | 0.01249 | 2.403 | 28.99 |
| $$\Theta_2$$（完整） | 0.01236 | 2.352 | 29.60 |

此外，针对 10 个随机发型、32 名志愿者的用户研究中，本方法在合理性（9.08）与纹理质量（8.96）两项均显著领先 DreamGaussian、One-2-3-45++、One-2-3-45 等扩散式 image-to-3D 方法。

## 亮点与局限

亮点：

- 首次在单条统一流水线中同时重建辫发与非辫发多样发型，摆脱对昂贵 3D 头发几何数据的依赖。
- 贡献了规模与风格显著扩展的 SynMvHair 合成多视图数据集，惠及后续头发研究。
- 尽管先验仅在合成数据上训练，对真实野外人像仍展现良好泛化能力。
- 输出的 Gaussian 头发可快速、高质量、多视一致地渲染，还能把单视图 3D 发丝重建转化为更成熟的多视图发丝捕捉任务。

局限：

- 依赖头-发对齐与头发掩膜分割等中间步骤，这些环节的误差会影响最终结果。
- 对复杂真实光照、发饰处理不佳，可能造成纹理或视图不一致。
- 在波浪、卷曲、螺旋等发型上表现仍不理想。
- 侧视图输入需额外处理（关键点检测在侧脸困难），且对齐方法对背视图无效。

## 延伸思考

这项工作最具启发性的一点，是把「缺乏高质量 3D 几何数据」这一长期瓶颈，转化为「用合成多视图渲染训练领域专用扩散先验 + 优化式 2D 提升」的思路——只要能渲染出标定准确的多视图图像，就能绕开对精细 3D 发丝拓扑的直接需求。视图级与像素级两级细化的拆分也颇具借鉴意义：前者靠底层 3D 高斯保证多视一致性，后者靠专用增强先验补齐高频细节，二者分工明确地攻克了「一致但模糊」与「清晰但不一致」这对矛盾。未来若能把波浪/卷发建模、复杂光照解耦、以及侧/背视图鲁棒对齐纳入统一框架，配合已展示的「转多视图发丝重建」能力，有望进一步逼近真正端到端、可编辑的数字人头发生产管线。
