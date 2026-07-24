---
title: "UniVidX: A Unified Multimodal Framework for Versatile Video Generation via Diffusion Priors"
authors:
  - "Houyuan Chen"
  - "Hong Li"
  - "Xianghao Kong"
  - "Tianrui Zhu"
  - "Shaocong Xu"
  - "Weiqing Xiao"
  - "Yuwei Guo"
  - "Chongjie Ye"
  - "Lvmin Zhang"
  - "Hao Zhao"
  - "Anyi Rao"
category: "Neural & Generative"
track: "Journal"
source: "arxiv"
institution: "HKUST"
tags:
  - "Video Generation"
  - "Diffusion Model"
  - "Multimodal Learning"
  - "Intrinsic Decomposition"
  - "Video Matting"
  - "LoRA"
links:
  paper: "https://doi.org/10.1145/3811304"
  project: "https://houyuanchen111.github.io/UniVidX.github.io/"
  code: "https://github.com/houyuanchen111/UniVidX"
---

## 一句话总结

UniVidX 把多种像素对齐的图形任务统一成"在共享多模态空间里做条件生成"，用一个视频扩散模型（VDM）加三项轻量设计，就能在极少训练数据（<1k 视频）下同时胜任内在分解、正/反向渲染、抠图、文生 RGBA 等 15 类任务。

## 研究背景

- 领域现状：预训练视频扩散模型已成为强大的"世界先验引擎"，被越来越多工作复用到内在分解、法线估计、抠图、可控生成等下游图形任务上。
- 核心痛点：现有做法几乎都是"一个任务训一个模型"，把网络锁死在固定的输入→输出映射（如 RGB→alpha、intrinsic→X）。这样既缺乏灵活性，又忽略了不同视觉模态之间的联合相关性，导致要么只能做单模态生成，要么靠串行推理，最终模态栈之间互相不一致。
- 本文 idea：能不能设计一个统一的生成框架，让一组对齐模态里任意子集充当"条件"或"目标"，从而在多模态之间自由生成？作者据此提出 UniVidX，把所有任务重写成同一个多模态空间里的条件生成问题。

## 方法

UniVidX 以文生视频（T2V）模型 Wan2.1-T2V-14B 为骨干，选它是因为它天生能吃纯文本、且潜空间可扩展以容纳视觉模态。整体思路是：把所有视觉模态编码进同一潜空间，训练时动态划分谁是条件、谁是目标，只对目标模态加噪并预测速度场，从而让一个模型学会"全向"生成。

```mermaid
flowchart LR
  A["多模态输入 RGB / Albedo / Normal ..."] --> B["VAE 编码进共享潜空间"]
  B --> C["SCM 随机划分 条件保持干净 / 目标加噪"]
  C --> D["DiT Blocks + DGL 按模态门控 LoRA"]
  D --> E["CMSA 跨模态自注意力 共享 K/V"]
  E --> F["Flow Matching 预测目标模态速度场"]
```

关键设计分三块：

1. **Stochastic Condition Masking（SCM，随机条件掩码）**：训练时把所有模态潜变量随机切成互斥的两组——目标子集加噪（时间步 $$t\in[0,1]$$ 上在高斯噪声 $$\epsilon$$ 与干净数据之间线性插值），条件子集固定在 $$t=1$$ 保持干净。流匹配目标只对目标子集预测速度场 $$v=x_T-\epsilon$$，即 $$L_{\text{uni}}=\mathbb{E}\lVert v_\theta(z_t^T\mid z_1^C,c_{\text{txt}})-v\rVert_2^2$$。条件集可以为空（对应纯文生 X）。推理时按任务自定义划分，一个模型覆盖 Text→X、X→X、Text&X→X 三种范式。

2. **Decoupled Gated LoRA（DGL，解耦门控 LoRA）**：不同模态分布差异大，共享参数会互相破坏。DGL 给每个模态挂一套独立的低秩更新 $$\Delta W_k=B_kA_k$$，并用门控 $$W_k'=W+m_k\cdot\Delta W_k$$ 控制：只有当某模态是"生成目标"（加噪）时才开（$$m_k=1$$），当它作"条件"（干净）时关闭（$$m_k=0$$），直接走原始冻结权重、最大化利用 VDM 的原生编码能力。这样既避免参数干扰、又防止全量微调式的灾难性遗忘。LoRA 秩取 32，可训练参数仅 385M。

3. **Cross-Modal Self-Attention（CMSA，跨模态自注意力）**：标准 VDM 的自注意力对每个模态各算各的，捕捉不到跨模态依赖。CMSA 把各模态的键/值拼成共享上下文 $$k_{\text{shared}}=[k_1,\dots,k_n]$$、$$v_{\text{shared}}=[v_1,\dots,v_n]$$，而查询保持模态专属，让每个模态在生成时都"看见"其他模态，从而保证跨模态一致与对齐。

作者据此实例化两个模型：**UniVid-Intrinsic**（RGB 视频与其内在图 albedo/irradiance/normal）和 **UniVid-Alpha**（混合 RGB、alpha 遮罩、前景 FG、背景 BG）。二者都只在 4×H100 上、用不到 1k 视频、几千步训练完成（InteriorVid 900 段室内合成片段 / VideoMatte240K 的 484 段）。

## 实验结果

主实验是 UniVid-Intrinsic 在 InteriorVid-Test 上的反向渲染（inverse rendering）对比：既超过内在分解基线，也在法线估计上压过专用估计器，取得最低 MAE 11.09°。下表节选 albedo 与 normal 的核心指标：

| 方法 | Albedo PSNR↑ | Albedo LPIPS↓ | Normal MAE↓ | Normal 11.25°↑ |
|------|-----|-----|-----|-----|
| RGB↔X | 11.64 | 0.3324 | 18.48 | 50.88 |
| NormalCrafter | - | - | 12.49 | 64.13 |
| Diffusion Renderer | 13.59 | 0.2624 | 15.76 | 54.42 |
| Ouroboros | 14.21 | 0.2639 | 14.52 | 57.58 |
| UniVid-Intrinsic | 16.89 | 0.2248 | 11.09 | 70.52 |

其余实验用文字概述：在真实世界 albedo 基准 MAW 上取得最优 intensity 误差 0.44（chromaticity 3.60 有竞争力），说明纯合成训练也能迁移到真实场景；法线估计的 Sintel 基准上与专用模型持平，但训练帧数仅 19K，比视频专用的 NormalCrafter 少 45 倍以上，凸显数据效率；视频抠图在 VideoMatte 上以最低 MAD 4.24 击败包括需要额外掩码引导（MG）在内的所有方法，且能额外生成干净背景。消融显示：换成常见的"通道拼接"策略会因破坏扩散先验而在小数据下结构崩溃，而 UniVidX 走"批维拼接"无需改动输入输出层；去掉 DGL 的解耦/门控会导致共享提示下生成失败、注意力图混乱。

## 亮点与局限

- 亮点：
  - 用"条件/目标随机划分 + 门控 LoRA + 跨模态注意力"三件套，把碎片化的任务统一进单个 VDM，一个模型覆盖 15 类任务且模态间一致。
  - 极致数据效率：<1k 视频、几千步训练即达到或超越专用 SOTA，充分说明是在"引导先验"而非"从零学表征"。
  - 几乎不改骨干结构（批维拼接 + LoRA），最大化保留 VDM 原生先验，还能组合任务支持重光照、重纹理、材质编辑、视频修复等下游应用。

- 局限：
  - 因缺乏同时标注内在与 alpha 标签的数据，两种能力目前只能分别实例化为两个模型，尚未真正合一。
  - 14B 骨干显存开销大，最多 4 个模态、21 帧、480p，规模受限。
  - 强依赖先验使其对训练分布偏置敏感，在玻璃等透明物理边角案例（如法线估计）上表现不稳定；作者认为这是数据依赖而非结构问题，可用针对性数据补齐。

## 延伸思考

这篇的核心范式很有启发：把"判别/感知"和"生成"都塑造成同一潜空间里的条件生成，用门控 LoRA 隔离模态分布差异，本质上是在冻结的大模型上做"可组合的任务插件"。它与 RGB↔X、Diffusion Renderer、Ouroboros 等"复用扩散先验做图形任务"的路线一脉相承，但把"固定映射"松绑成"任意子集互为条件"，这一步抽象是关键增量。值得追问的方向：能否把模态数与分辨率随骨干蒸馏/量化一起放大？统一内在与 alpha 后是否会出现模态冲突？以及在缺乏干净合成标签的真实数据上，这种"引导先验"式训练的上限在哪里。
