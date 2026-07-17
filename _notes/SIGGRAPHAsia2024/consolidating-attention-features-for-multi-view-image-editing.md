---
title: "Consolidating Attention Features for Multi-view Image Editing"
authors:
  - Or Patashnik
  - Rinon Gal
  - Daniel Cohen-Or
  - Jun-Yan Zhu
  - Fernando De la Torre
track: "Conference"
source: author-page
category: "Neural & Generative"
institution:
  - Tel Aviv University
  - NVIDIA
  - Carnegie Mellon University
tags:
  - Multi View Editing
  - Diffusion Model
  - Self Attention
  - Neural Radiance Fields
  - Geometric Editing
links:
  paper: "https://doi.org/10.1145/3680528.3687611"
---

## 一句话总结

针对同一物体多视角图像的几何编辑（如改变姿态、体型），本文提出把扩散模型自注意力层的 query 特征蒸馏进一个神经辐射场（QNeRF），在去噪过程中反复渲染出 3D 一致的 query 并软注入回网络，从而让多视角编辑结果保持几何一致。

## 研究背景

大规模文生图模型让基于文本或空间控制的图像编辑变得强大，但把这些方法直接套用到描绘同一场景的多视角图像集上时，各视角独立编辑会产生 3D 不一致的结果。

已有多视角编辑工作（如 Instruct-NeRF2NeRF、CSD）依赖"数据集更新"范式：先在原始数据上训练 NeRF，再迭代地编辑单张图像并回填训练。这类方法隐含假设编辑幅度足够小，底层 3D 表示能够对不一致的改动做平均。该假设对纹理、外观类改动成立，但面对关节运动、体型变化等大幅几何编辑时会失效——例如把人物手臂抬起来时，部分视角编辑到新位置、部分留在原位，最终 NeRF 会出现"幽灵四肢"这类重影伪影。

本文聚焦基于空间控制（骨架、松散深度图、包围盒）的几何操作，目标是在保持原场景外观的同时，让大幅几何编辑在所有视角间保持一致。

## 方法

整体基于预训练 Stable Diffusion 与 MasaCtrl 的编辑流程：图像先经 DDIM 反演，再在 ControlNet 提供的粗略空间控制下重新合成，并通过注入原图自注意力层的 key/value 来保留外观。作者的关键观察是：自注意力层中的 query 决定了对应像素的语义与图像结构。因此只要让各视角的 query 在生成过程中保持一致，就能对齐物体形状。

```mermaid
flowchart LR
    A[多视角图像并行去噪] --> B[从UNet提取自注意力Q]
    B --> C[用Q训练QNeRF]
    C --> D[渲染3D一致的Q]
    D --> E[软注入回各视角网络]
    E --> A
```

**QNeRF：query 特征空间的辐射场。** 在每个区间末尾，从 UNet 解码器中分辨率为 16、32、64 的层提取自注意力 query，共 9 组 query 作为训练集。QNeRF 基于 depth-nerfacto，改动包括：为每个坐标输出 9 个 query 值（在基础网络上挂 9 个 head，每个 head 对应一层 query，输出维度等于该层通道数 1280/640/320）；9 个 head 共享同一密度以稳定几何、跨层共享信息；去掉对视角方向的依赖，因为 query 表示的是与视角无关的几何。训练损失为 q-loss 加深度损失：

$$
L_q = \sum_{r\in R}\sum_{l}\left\lVert \hat{Q}_l(r) - q_r(l)\right\rVert
$$

其中 $$\hat{Q}_l(r)$$ 用标准体渲染得到，只是把 RGB 值 $$c$$ 换成自注意力 query。

**软注入的 query 引导。** 直接用渲染出的 query 替换网络生成的 query 会产生伪影。作者改用"软引导"：每个引导步先前向一次 UNet 提取自然生成的 query，再对输入 latent 做一步优化，使其接近 QNeRF 渲染的 query：

$$
z^v_t \leftarrow z^v_t - \alpha \nabla_{z^v_t}\sum_l \left\lVert q^v(l) - q^v_{\text{ren}}(l)\right\rVert_2
$$

随后再执行一步 DDIM 去噪。

**区间化去噪。** 不为每个去噪步都训练 QNeRF，而是分区间进行。每个区间跨 $$2\tau$$ 步：先用上一区间的 QNeRF 做 $$\tau$$ 个引导步（并缓存此时 latent），再做 $$\tau$$ 个不带引导的原生 MasaCtrl 步让 query 自由演化；区间末提取更新后的 query 训练新 QNeRF，然后回到缓存的 latent 从 $$T_i-\tau$$ 开始下一区间。这样兼顾了"相邻时间步特征相似可复用"与"给 query 留出自由演化空间"两点。

**渐进式巩固。** 每个区间训练 QNeRF 的 query 都受上一区间 QNeRF 渲染 query 的影响，作者用"拉链"作比喻：每一步都建立在前序部分已闭合、更靠拢的基础上，从而逐步把各视角的 query 轨迹拉近，避免它们在被重新巩固前漂移过远。

## 实验结果

在 "statue"、"person"、"alligator-toy" 三个场景上，与 IN2N-CN、CSD-CN、TokenFlow 对比。用 KID/FID 衡量与原图的图像质量，用用户研究评估从编辑结果训练出的 NeRF 深度图与目标姿态的对齐程度和质量（收集 20 名用户共 120 份反馈）。

| 指标 | IN2N-CN | CSD-CN | TokenFlow | 本文 |
|------|---------|--------|-----------|------|
| KID (↓) | 0.280 | 0.090 | 0.440 | 0.072 |
| FID (↓) | 201 | 87 | 295 | 73 |
| 用户研究排名 (↓) | 2.26 | 2.12 | 3.70 | 1.90 |
| 用户研究胜率 (↑) | 14.16% | 34.16% | 0.83% | 50.83% |

本文方法在图像保真度和用户偏好上均领先。消融显示：独立编辑（MasaCtrl）导致视角间不一致；直接注入渲染 query 会使结果过度偏离原图（截断腿部、抬高平台）；非渐进式巩固会带来更多伪影（如缺失的腿）；完整方法则同时保持一致性与对原场景的忠实。

## 亮点与局限

亮点：
- 在注意力特征空间（而非像素空间）做多视角巩固，更贴近真实图像流形，避免像素平均带来的重影伪影。
- 通过共享密度的 9-head QNeRF 与软注入机制，实现大幅几何编辑（关节运动、体型变化）下的多视角一致性，这是"数据集更新"类方法难以处理的。
- 方法通用，可适配多种基于扩散、修改图像布局的编辑技术，且控制方式（骨架、粗深度、包围盒）宽松易用。

局限：
- 继承了文生图模型的固有弱点，如难以生成手部、可能幻觉细节。方法只巩固形状，高细节物体的细节、被遮挡区域重新暴露的背景仍不一致，导致训练 NeRF 时出现模糊区域。
- QNeRF 用黑盒优化器训练，可能对离群值做"平均"而非鲁棒地过滤。
- 作者展望用高斯泼溅等其他 3D 表示，以及鲁棒统计手段来改进特征巩固。

## 延伸思考

- 将扩散模型内部特征蒸馏进 3D 表示、再回注引导生成，这一"特征即几何"的思路可推广到视频编辑、多视角风格迁移等需要跨帧/跨视角一致性的任务。
- QNeRF 只巩固与视角无关的 query（结构），而 key/value（外观）仍走注入通道，这种"几何—外观"解耦值得在更强的 3D 一致生成中进一步探索。
- 用高斯泼溅替换 NeRF 有望大幅缩短每区间的训练时间，缓解本文在大规模图像集（如近 500 张的场景）上高昂的计算开销。
