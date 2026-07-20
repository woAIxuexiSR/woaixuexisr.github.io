---
title: "TokenVerse: Versatile Multi-concept Personalization in Token Modulation Space"
authors:
  - "Daniel Garibi"
  - "Shahar Yadin"
  - "Roni Paiss"
  - "Omer Tov"
  - "Shiran Zada"
  - "Ariel Ephrat"
  - "Tomer Michaeli"
  - "Inbar Mosseri"
  - "Tali Dekel"
category: "Neural & Generative"
track: "Journal"
source: "arxiv"
institution:
  - "Google DeepMind"
  - "Tel Aviv University"
  - "Technion"
  - "Weizmann Institute"
tags:
  - "Diffusion Model"
  - "Text-to-Image"
  - "Personalization"
  - "Concept Disentanglement"
  - "Diffusion Transformer"
links:
  paper: "https://doi.org/10.1145/3730843"
  project: "https://token-verse.github.io/"
---

## 一句话总结

TokenVerse 发现扩散 Transformer（DiT）中的"调制空间"（modulation space）具有语义性，并提出对每个文本 token 单独学习一个调制方向，从而在无需掩码、无需改动模型权重的前提下，从单张图里解耦出多个复杂概念，并把来自不同图片的概念即插即用地组合到新图中。

## 研究背景

- 领域现状：文本到图像模型已能生成融合多种视觉概念的图像。个性化（personalization）方向主要有两条路线，一是微调生成模型权重（如 DreamBooth）让某个特殊 token 关联新概念，二是固定模型、只优化描述概念那个词的文本嵌入（textual inversion 类）。
- 核心痛点：微调路线难以把多个各自专门化的权重干净地组合；文本嵌入路线表达力又不够。两者本质都面向单概念，难以把一张图里纠缠的多个概念拆开；已有的多概念方法要么依赖分割掩码、边界框，要么解耦结果不可控，且大多只能处理物体，无法个性化姿态、材质、光照这类非物体概念。
- 本文 idea：现代 DiT（Stable Diffusion 3、Flux）里文本经两条路径影响生成——一是与图像 token 联合注意力，二是把全局文本嵌入映射为对通道做缩放和平移的调制系数。作者观察到这个调制空间（记为 $$\mathcal{M}$$）像 GAN 的隐空间一样语义丰富，但整体调制会引起非局部改动；而只改动"单个文本 token"对应的调制向量，就能把语义修改局部化到该 token 关联的概念上。由此定义每 token 调制空间 $$\mathcal{M}^+$$，并在其中为每个词学习一个定制方向来实现解耦个性化。

## 方法

整体框架：TokenVerse 基于预训练的 Flux-dev（58 个 DiT block，调制向量维度 3072）。给定一张概念图及其文字描述，用一个轻量 MLP（Concept-Mod）为描述里的每个 token 预测一个 $$\mathcal{M}^+$$ 空间的方向偏移 $$\Delta_i$$，用与原模型相同的扩散重建目标来优化这些偏移。推理时把预先学到的各个 $$\Delta$$ 加到新提示词里对应 token 的调制向量上，就能把这些概念注入新生成的图像。

```mermaid
flowchart LR
  A[概念图 + 描述文字] --> B[Concept-Mod MLP]
  B --> C["每个 token 的调制方向 Δ_i (M+ 空间)"]
  C --> D[扩散重建损失 + 概念隔离损失优化]
  D --> E[学好的各概念方向库]
  E --> F[新提示词按 token 注入 Δ]
  F --> G[组合多概念的新图像]
```

关键设计：

1. **从 $$\mathcal{M}$$ 到 $$\mathcal{M}^+$$——局部化的关键。** 在 DiT 里，调制向量由池化文本嵌入经 MLP 得到，即 $$y = \mathrm{MLP}(t, \mathrm{CLIP}(p))$$，同一组缩放和平移作用于所有 token。若用带/不带某属性的两个文本嵌入做差得到方向 $$\Delta_{\text{attribute}} = \mathrm{MLP}(t, e_{\text{attribute}}) - \mathrm{MLP}(t, e_{\text{neutral}})$$，把它以权重 $$w$$ 加到全局 $$y$$ 上（$$y + w\,\Delta_{\text{attribute}}$$），确实能改属性，但会连带改动图中其它无关概念。作者改为只对目标概念对应的那个文本 token 施加偏移，其余 token 用原始 $$y$$。由于联合注意力层把该文本 token 与相关图像 token 绑定，这种改动天然局部化，且不仅适用于物体，也适用于姿态等抽象概念。

2. **无监督解耦的概念学习。** 目标是为提示 $$p$$ 里每个 token 找一组方向 $$(\Delta_1, ..., \Delta_{\mathrm{len}(p)}) = \text{Concept-Mod}(p)$$，各自表示该 token 从"通用词义"到"图中具体呈现"之间的方向。解耦无需掩码，靠的是模型本身"文本 token 与对应图像区域"的关联；虽然所有方向联合优化，但每个方向只个性化它所修饰的视觉元素，因此可任意选取子集组合。

3. **分阶段、逐 block 优化。** 分两阶段学习：第一阶段偏重高噪声时间步（800 步优化，其中约 92% 迭代采样 $$t \in [800, 1000]$$）以抓粗粒度概念；第二阶段偏重低噪声时间步（再 600 步，92% 迭代采样 $$t \in [0, 800]$$）细化，并额外训练一个"逐 block"MLP，其输出加到 Concept-Mod 的输出上，形成"每 token 每 block"的方向，提升概念保真度。

4. **概念隔离损失（concept isolation loss）。** 同一张图里学的方向通常已较好隔离，但跨图组合时不同概念的方向可能相互干扰。为此在 50% 的训练迭代中加入隔离损失：把输入概念图与一张随机生成图拼接、并合并两者描述成一句话，只在输入提示的 token 上施加学到的方向，然后仅在"拼接进来那半张图"的区域对模型输出与基座模型输出施加 L2 损失，逼迫学到的方向不去影响本不该改动的内容。

## 实验结果

以 Flux-dev 为基座。评测遵循 DreamBench++ 协议，用多模态大模型给出概念保持（CP）与提示保真（PF）两个 0~1 分数，并报告二者乘积 CP·PF 作为综合指标（均越高越好）；对多概念图另建了含 30 张、每图 2~4 个概念的自定义数据集。下面取"完整任务"（从两张各含多个概念的图中各学一个概念，再组合成一张新图）的 DreamBench++ 结果作为主实验对比。

| 方法 | CP↑ | PF↑ | CP·PF↑ |
|------|------|------|--------|
| 本文（TokenVerse） | 0.553 | 0.822 | 0.455 |
| Break-A-Scene（joint） | 0.499 | 0.641 | 0.320 |
| DreamBooth（joint） | 0.306 | 0.591 | 0.181 |
| DreamBooth | 0.207 | 0.828 | 0.171 |
| OMG | 0.208 | 0.843 | 0.175 |
| ConceptExpress | 0.188 | 0.733 | 0.138 |

在概念保持与综合指标 CP·PF 上，TokenVerse 在完整任务、以及组合（composition）、解耦（decomposition）两个子任务上均优于所有对比方法，同时提示保真保持在有竞争力的水平。值得注意的是 OMG 提示保真很高但概念保持极低，原因是它先做无条件生成再靠分割贴回概念。作者还做了 37 人、共 3000 票的用户研究，结论一致：TokenVerse 概念保持得分最高（4.078，五分制），提示保真 4.292 也居前列。消融显示：直接把方向作用于进入 Transformer 前的文本 token（类 textual inversion）无法忠实重建概念；改到 $$\mathcal{M}^+$$ 空间表达力更强；再加逐 block 方向进一步提升保真；隔离损失则显著缓解跨图概念间的相互干扰。

## 亮点与局限

- 亮点：
  - 首个能"从多张图做解耦多概念个性化 + 即插即用组合"的方法，且对可组合的概念数量没有技术上限（论文展示了 9 个物体同图组合）。
  - 概念范围突破物体，可个性化姿态、材质、光照等抽象概念，且完全不需要分割掩码或边界框，天然支持相互重叠的概念（如人身上戴的项链）。
  - 不改动基座模型权重，保留其先验；模块化——不同概念可从不同图分别提取，无需联合训练。

- 局限：
  - 每个概念仍需按图做优化式学习（两阶段共约 1400 步），不是前馈、免调优的即时方法，规模化成本较高。
  - 解耦依赖模型内在的"文本 token 与图像区域"关联，对文字里没显式点名、或注意力关联薄弱的概念可能失效。
  - 跨图组合的干扰需要额外的隔离损失来抑制，说明 $$\mathcal{M}^+$$ 的解耦并非天然完美；对精细身份（如人脸）还需多张输入图。

## 延伸思考

- 该工作把 GAN 时代"在调制/隐空间里找语义方向"的成熟直觉迁移到了 DiT，$$\mathcal{M}^+$$ 这一"每 token 调制空间"可能成为 DiT 可解释性与可控编辑的通用抓手，值得探究它与联合注意力、与文本嵌入空间之间的关系。
- 当前是优化式学习，后续自然方向是训练一个前馈编码器直接预测 token 的 $$\Delta$$，实现免调优（tuning-free）个性化——事实上已有后续工作（如 Mod-Adapter）沿"调制空间 + 适配器"思路推进。
- 概念隔离损失本质是"约束方向的作用域"，能否推广到视频 DiT 的时序 token、或用于更强的属性解耦（把物体的外观与姿态、材质彻底分离），是值得追问的点。
