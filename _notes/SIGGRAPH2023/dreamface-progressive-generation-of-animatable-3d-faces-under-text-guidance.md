---
title: "DreamFace: Progressive Generation of Animatable 3D Faces under Text Guidance"
authors:
  - "Longwen Zhang"
  - "Qiwei Qiu"
  - "Hongyang Lin"
  - "Qixuan Zhang"
  - "Cheng Shi"
  - "Wei Yang"
  - "Ye Shi"
  - "Sibei Yang"
  - "Lan Xu"
  - "Jingyi Yu"
category: "Neural & Generative"
track: "Journal"
source: "arxiv"
institution:
  - "ShanghaiTech University"
  - "Huazhong University of Science and Technology"
tags:
  - "Text-to-3D"
  - "Diffusion Model"
  - "3D Face"
  - "Facial Animation"
  - "Score Distillation Sampling"
  - "Physically-Based Rendering"
links:
  paper: "https://doi.org/10.1145/3592094"
  project: "https://sites.google.com/view/dreamface"
---

## 一句话总结

DreamFace 用一段文字描述就能生成带精细几何、物理渲染纹理、且可被驱动的动画化 3D 人脸资产，且与现有 CG 生产管线完全兼容。

## 研究背景

- 领域现状：大规模视觉-语言模型（CLIP、Stable Diffusion）让"文本生成图像"乃至"文本生成 3D"变得触手可及，2D 人脸生成（StyleGAN2）也早已逼真。基于 GAN/NeRF 的 3D-aware 人脸方法能出细节，但要么难以并入工业 CG 管线，要么几何过度平滑。
- 核心痛点：现有文本驱动的 3D 生成方法（DreamFusion、Latent-NeRF 等）虽然多样，却无法产出满足人物身份特征的资产——缺乏精细几何、物理渲染纹理和细粒度动画能力，难以直接落地到影视、游戏、元宇宙的制作流程。
- 本文 idea：用一个"渐进式"框架把问题拆解为三段——几何生成、物理纹理扩散、动画赋能——让大模型的语言先验与工业级动态人脸资产有机结合，普通用户只靠文字就能定制人脸。

## 方法

整体框架：DreamFace 采用与 ICT-FaceKit 一致的网格拓扑，按顺序执行三个模块。先由文本选出并雕刻中性几何，再用双路扩散优化生成物理纹理，最后训练跨身份超网络与视频追踪器赋予动画能力。三段解耦保证了几何、外观、动画各自可控。

```mermaid
flowchart LR
  P["文本 Prompt"] --> G["几何生成: CLIP 选粗几何 + SDS 细节雕刻"]
  G --> T["物理纹理扩散: 双路 SDS 生成漫反射/高光/法线"]
  T --> A["动画赋能: 跨身份超网络 + 视频表情编码器"]
  A --> O["可驱动的物理渲染人脸资产"]
```

关键设计：

1. 由粗到细的几何生成。先从 ICT-FaceKit 形状空间随机采样大量候选，渲染多视角多光照图像后投影到 CLIP 空间，用与"the face"锚点做相对匹配的打分挑出最优粗几何。粗几何过于平滑，于是在其上优化顶点位移 $$\boldsymbol{V}_d$$ 与切空间法线图 $$\boldsymbol{N}_d$$，借助 Stable Diffusion 的 Score Distillation Sampling（SDS）损失做细节雕刻，并配合形状、拉普拉斯平滑、法线正则等约束保证合理性。

2. 纹理空间的扩散模型。通用 LDM 只懂自然图像、不懂 UV 纹理的语义排布，作者收集并统一多来源的物理纹理数据集微调出一个 texture LDM。针对数据集中光照/高光混杂的问题，提出 Prompt Tuning：学习两组连续词向量 $$\boldsymbol{C}_d$$、$$\boldsymbol{C}_u$$ 分别表示"干净"与"含光照"两个域，再叠加非人脸区域掩码条件，使模型能在全量数据上训练却只在"干净域"生成漫反射图。

3. 双路两阶段外观优化。核心创新是同时用通用 LDM（保证 prompt 多样性）和 texture LDM（保证 UV 规范一致性）做 SDS，解决单一扩散模型导致的五官漂移与 Janus 问题。优化分两阶段：先在 $$64\times64$$ 隐空间做 SDS 得到紧凑先验，再在图像空间做 SDS 并叠加细节法线与随机光照，把光照从漫反射中解耦、提升细节。最终再由额外解码器把漫反射隐码翻译成高光图、法线图，并超分到 4K。

4. 动画赋能。借助与 ICT-FaceKit 共享拓扑，资产天然支持 blendshape 驱动。为获得个性化表情，训练一个跨身份几何超网络：表情编码器把各种表情编码进统一隐空间，几何生成器（U-Net）以中性几何为条件重建带表情的网格。再训一个图像表情编码器，从单张 RGB 图/视频提取表情隐码，实现视频驱动的个性化动画。

## 实验结果

与文本驱动生成方法的主对比（CLIP 匹配分越高越好，生成耗时越短越好）：

| 方法 | CLIP score↑ | 生成耗时↓ |
|------|-------------|-----------|
| Text2Mesh | 0.2109 | 约 15 分钟 |
| AvatarCLIP | 0.2812 | 约 5 小时 |
| Stable-DreamFusion | 0.2594 | 约 2.5 小时 |
| 本文 | 0.2934 | 约 5 分钟 |

DreamFace 在文本匹配分上最高，同时生成一套高质量人脸资产只需约 5 分钟，远快于其他方法。此外纹理 LDM 的消融显示，同时使用 Prompt Tuning 与掩码条件时在"干净域"上取得最低的 KID（0.0578）；用户研究中，生成资产对名人和虚构角色的"神似度"分别达到 72.3% 和 71.6%。

## 亮点与局限

- 亮点：
  - 双路 SDS 巧妙融合"通用先验的多样性"与"纹理先验的 UV 一致性"，缓解了单扩散模型的五官漂移问题。
  - 产出的是拓扑统一、物理渲染就绪、可 blendshape/视频驱动的完整资产，能直接进现有 CG 管线，落地性强。
  - 两阶段隐空间+图像空间优化把生成压到约 5 分钟，效率显著优于同类 SDS 方法。
- 局限：
  - 几何被约束在 ICT-FaceKit 的形状空间与固定拓扑内，难以表达极端非人类形态的骨骼结构；发型也仅从 16 个艺术家预制款里挑选。
  - 依赖自建的物理纹理与表情捕捉数据集（含 Light Stage 级采集），复现门槛高。
  - 个性化动画超网络训练成本大（首阶段 5 天、次阶段 48 小时），且表情由统一隐码表示，对超出训练分布的夸张表演能力未充分验证。

## 延伸思考

DreamFace 本质是把 SDS 这套"用 2D 扩散先验监督 3D 优化"的范式，从通用物体收窄到具有强结构先验（参数化人脸模型）的领域，并额外补齐了工业管线最看重的物理纹理与动画。这提示一个通用思路：当目标域有成熟参数化模型时，把扩散先验与参数空间/UV 规范耦合，往往能显著提升生成质量与可控性。后续可追问的方向包括：能否用 3D-aware 扩散或多视一致扩散替代 CLIP 选型+SDS，进一步减少视角不一致；发型、口腔、眼球等部件能否也纳入统一的可动画生成；以及如何把该框架从人脸扩展到全头乃至全身资产。
