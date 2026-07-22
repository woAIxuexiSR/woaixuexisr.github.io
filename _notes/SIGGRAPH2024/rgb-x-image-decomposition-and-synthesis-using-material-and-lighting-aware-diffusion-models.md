---
title: "RGB↔X: Image decomposition and synthesis using material- and lighting-aware diffusion models"
authors:
  - "Zheng Zeng"
  - "Valentin Deschaintre"
  - "Iliyan Georgiev"
  - "Yannick Hold-Geoffroy"
  - "Yiwei Hu"
  - "Fujun Luan"
  - "Ling-Qi Yan"
  - "Miloš Hašan"
category: "Neural & Generative"
track: "Conference"
source: "arxiv"
institution:
  - "Adobe Research"
  - "University of California, Santa Barbara"
tags:
  - "Diffusion Model"
  - "Intrinsic Decomposition"
  - "Inverse Rendering"
  - "Image Synthesis"
  - "Material Editing"
links:
  paper: "https://doi.org/10.1145/3641519.3657445"
  project: "https://github.com/zheng95z/rgbx"
---

## 一句话总结

在同一个条件扩散框架下，用 RGB→X 模型把室内场景图像分解为逐像素的材质与光照本征通道，再用 X→RGB 模型从（完整或部分）本征通道合成逼真图像，从而统一了图像分解与图像合成两个方向。

## 研究背景

真实感前向渲染、逐像素逆向渲染、生成式图像合成通常被视为图形学与视觉中互不相干的子领域。经典的本征图像分解问题由于材质与光照之间的固有歧义而高度欠约束；近期逐像素逆向渲染工作能估计漫反射反照率、粗糙度、金属度以及各种空间变化的光照表示，作者把这些信息缓冲统称为本征通道 X，把从图像估计它们的问题称为 RGB→X。

另一方面，物理渲染长期专注于反向任务：把包含几何、光照、材质的场景描述转换成逼真图像，作者称之为 X→RGB。而基于大型扩散模型的生成式方法则从纯噪声出发迭代去噪产生高真实感图像。作者认为这三者应被统一研究：既做材质/光照估计，又做以材质/光照为条件的图像合成，都放进同一个扩散框架。前人 Kocsis 等已用扩散架构改进本征通道估计，本文在此基础上进一步改进 RGB→X（并新增光照估计），并首次提出能从本征通道合成图像的扩散式 X→RGB 模型。

## 方法

整体框架由两个基于 Stable Diffusion 2.1 微调的潜空间扩散模型组成：RGB→X 负责分解，X→RGB 负责合成。本征通道 X 包含法线 n、反照率 a、粗糙度 r、金属度 m 与漫反射辐照度 E（作为光照表示）。

```mermaid
flowchart LR
    I[输入 RGB 图像] -->|编码 E(I)| RX[RGB→X 扩散 UNet]
    RX --> X[本征通道 X: 法线/反照率/粗糙度/金属度/辐照度]
    X -->|编码并拼接| XR[X→RGB 扩散 UNet]
    P[文本提示] --> XR
    XR --> OUT[合成 RGB 图像]
    X -.可编辑/部分给定.-> XR
```

关键设计：

- RGB→X 的多通道输出。直接扩大潜空间通道数会迫使输入输出卷积层从头训练，把模型"震"到新域、训练困难且质量下降。作者改为一次只输出一个本征通道，把在该任务中本无用途的文本提示复用为"开关"，用五个固定提示词（"normal"、"albedo"、"roughness"、"metallicity"、"irradiance"）指定当前输出通道。这样只需微调并存储一个网络权重，效果与为每种模态单独微调模型相当。训练采用 v-prediction，损失为

$$L_{\theta}=\left\lVert v_{t}^{\text{RGB}\to \text{X}}-\hat v_{\theta}^{\text{RGB}\to \text{X}}\!\left(t, z_{t}^{X}, E(I), \tau(p^{X})\right)\right\rVert_{2}^{2}$$

- X→RGB 的条件拼接与通道 dropout。目标潜变量取 $$z_{0}^{RGB}=E(I)$$，把编码后的本征通道拼接到噪声潜变量上作为条件，只需新增输入卷积层从头训练，不会破坏原模型的去噪能力。为处理不同数据集缺失不同通道的异质数据，采用条件通道 dropout，使某个通道要么被编码要么置零：

$$z_{t}^{X}=\left(P(n), P(a), P(r), P(m), P(E)\right),\quad P(x)\in\{E(x), 0\}$$

这既能训练异质数据，又允许推理时只给部分通道，模型用先验补全其余部分。

- 低分辨率光照。辐照度 E 不经编码器，而是直接下采样到潜空间分辨率，作为粗略"光照提示"而非逐像素精确控制，便于编辑法线时对光照做更粗粒度的控制。

- 数据。作者拼接四个异质数据集：InteriorVerse、Hypersim、自建的 Evermotion（提供可靠的粗糙度与金属度），以及用 RGB→X 自行估计通道的 ImageDecomp（5 万张高质量室内图像）。文本描述由 BLIP-2 预先生成。为支持局部编辑（材质替换、物体插入），额外微调出带掩码输入的 inpainting 版 X→RGB。

## 实验结果

RGB→X 在 Hypersim（反照率、法线、辐照度）与 Evermotion（粗糙度、金属度）测试集上，除无可比方法的辐照度外，各通道 PSNR 与 LPIPS 均优于前人方法。

| 方法 | 反照率 PSNR↑ / LPIPS↓ | 法线 PSNR↑ / LPIPS↓ |
| --- | --- | --- |
| Our RGB→X | 17.4 / 0.18 | 19.8 / 0.18 |
| Zhu 等 | 11.7 / 0.54 | 16.5 / 0.45 |
| Careaga 与 Aksoy | 13.5 / 0.34 | — |
| PVT-normal | — | 18.8 / 0.30 |
| Kocsis 等 | 12.1 / 0.41 | — |

粗糙度、金属度、辐照度上，Our RGB→X 分别取得 PSNR 11.2 / 12.1 / 14.1、LPIPS 0.52 / 0.44 / 0.22，明显优于 Zhu 等与 Kocsis 等。X→RGB 在未参与训练的合成厨房场景上与蒙特卡洛路径追踪参考接近，材质外观与全局光照吻合；给定部分通道（如仅法线与反照率）时可用文本提示控制光照或物体颜色。两模型结合可实现材质替换与物体插入等编辑应用。

## 亮点与局限

亮点：把分解与合成统一进单一扩散框架；用"提示词开关"巧妙解决 RGB→X 的多通道输出问题，避免扩大潜空间带来的训练退化；通道 dropout 使异质数据集可混合训练，并支持推理时任意子集输入与部分规格下的合理"幻想"补全；辐照度下采样给出可控光照提示。

局限：粗糙度与金属度可靠训练数据最少，且只有在合适的高频光照下才显著影响表面反射，导致采样方差大、"好样本"产出率低；材质系统较粗糙（如无法表示玻璃，仅按零粗糙度零金属度处理）；文本控制在物体较多时难以精确指定单个物体的颜色；作者不专门解决重光照问题。

## 延伸思考

RGB→X 与 X→RGB 构成一个可闭环的"分析—合成"回路，天然适合作为可编辑管线：先分解、在本征域编辑、再合成。把无用的文本条件复用为任务开关，是在不改动潜空间维度前提下让单一扩散模型多任务化的一个可迁移思路。用自身模型估计通道来扩充训练数据（ImageDecomp）体现了生成式先验对数据瓶颈的缓解，但也可能引入模型自身偏差，值得关注其对材质细节（如磨损）的影响。
