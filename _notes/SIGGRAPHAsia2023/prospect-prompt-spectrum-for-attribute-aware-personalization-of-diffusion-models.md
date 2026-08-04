---
title: "ProSpect: Prompt Spectrum for Attribute-Aware Personalization of Diffusion Models"
authors:
  - "Yuxin Zhang"
  - "Weiming Dong"
  - "Fan Tang"
  - "Nisha Huang"
  - "Haibin Huang"
  - "Chongyang Ma"
  - "Tong-Yee Lee"
  - "Oliver Deussen"
  - "Changsheng Xu"
category: "Neural & Generative"
track: "Journal"
source: "arxiv"
institution:
  - "Chinese Academy of Sciences"
  - "Kuaishou Technology"
tags:
  - "Diffusion Model"
  - "Personalization"
  - "Textual Inversion"
  - "Text-to-Image"
  - "Attribute Disentanglement"
links:
  paper: "https://doi.org/10.1145/3618342"
  code: "https://github.com/zyxElsa/ProSpect"
---

## 一句话总结

利用扩散模型"由低频到高频、按 layout→content→material/style 顺序生成图像"这一规律，把单张图像反演成一组"逐生成阶段"的文本 token 嵌入，从而在无需微调模型的前提下实现材质、风格、内容、布局等视觉属性的解耦表示与可控编辑。

## 研究背景

- 领域现状：文本到图像扩散模型配合个性化方法（Textual Inversion、DreamBooth 等）可以把一个物体或概念反演到文本条件空间，再用新句子组合出新场景。这些方法通常需要 3~5 张图、或需要微调模型，且把整张图/整个概念作为一个整体来学习。
- 核心痛点：在所有扩散步骤和 U-Net 结构上共享一个全局文本嵌入，导致难以从单张图像中解耦出材质、风格、布局这类具体的视觉属性；解耦不足带来可编辑性差，属性重组时也容易冲突或失真。
- 本文 idea：把扩散生成过程按"步"拆开看。作者通过一系列增删属性的可视化实验发现，扩散模型是按"布局→内容→材质/风格"的顺序生成的，且这一顺序对应信号频率由低到高。据此把连续步骤分成若干"生成阶段"，每个阶段配一个独立的文本条件，从而在阶段维度上自然地把不同频率的视觉属性分离开。

## 方法

整体框架：作者把扩散模型常规的 1000 步条件过程平均切成 10 个阶段，每个阶段对应一个独立的 token 嵌入 $$p_i$$，这组嵌入构成扩展的文本条件空间 Prompt Spectrum Space $$P^*$$。给定单张图，ProSpect 用一个超网络把用户输入词（如 "cup"）的初始嵌入映射成 $$n\times 1\times 768$$ 的一组阶段嵌入，只训练超网络、以逐阶段的重建损失优化。推理时，把代表不同属性的 $$p_i$$ 替换为编辑文本，即可做属性感知的生成与迁移。

```mermaid
flowchart LR
  A["单张参考图 + 初始词"] --> B["CLIP 文本编码 (1x768)"]
  B --> C["超网络 (仅此可训练)"]
  C --> D["阶段嵌入 P = [p1,...,p10]"]
  D --> E["按阶段注入去噪 U-Net 交叉注意力"]
  E --> F["重建损失优化 P"]
  F --> G["替换/组合某些 pi 实现属性编辑"]
```

关键设计：

1. 频率-阶段对应的观察。作者做了"在特定步区间增删提示词/属性"的对照实验：早期步（如 0-500）决定整体布局与颜色，中期步决定结构化外观，后期步决定纹理细节。用傅里叶谱分析进一步印证——随去噪推进，预测图像的高频成分逐渐增多。这解释了生成顺序本质上是"属性信号频率由低到高"。注意与 U-Net 分层不同，扩散各阶段隐变量尺寸相同，所以这种分层来自"步"而非感受野。

2. Prompt Spectrum Space $$P^*$$。定义为 $$P^* = \{p_1, p_2, ..., p_n\}$$（取 $$n=10$$），其中 $$p_i$$ 是第 $$i$$ 个生成阶段对应条件提示的 token 嵌入。相比只学一个全局条件的 $$P$$ 空间，$$P^*$$ 把条件按阶段展开，为按频率解耦属性提供了空间。

3. ProSpect 反演。把 Textual Inversion 扩展到 $$P^*$$。原始 TI 损失为 $$L_{TI} = \mathbb{E}_{z,t,p}\lVert \epsilon - \epsilon_\theta(z_t, t, p_\theta)\rVert_2^2$$；ProSpect 损失改为按阶段取条件 $$L_{PS} = \mathbb{E}_{z,t,p}\lVert \epsilon - \epsilon_\theta(z_t, t, p_i)\rVert_2^2$$，其中 $$p_i = P(t)$$ 是第 $$i$$ 阶段的可学习嵌入。训练约 1000~3000 次迭代，用 dropout（0.1）防过拟合。

4. 属性分组与编辑。把 $$p_i$$ 按频率归为三类——材质/风格（高频）、内容（中频）、布局（低频）。推理时保留某类阶段的嵌入、替换另一类为编辑文本即可实现对应属性的迁移/编辑；还可从不同参考图各取一类属性做多属性联合生成。

## 实验结果

在 CLIP 相似度上评估内容保真（Image Similarity）与可编辑性（Text Similarity），与 Textual Inversion、DreamBooth 对比（Reference 列为参考基准，非可比方法）：

| 指标 | ProSpect | DreamBooth | TI |
|------|----------|-----------|-----|
| Text Similarity (Avg)↑ | 0.3444 | 0.3334 | 0.3115 |
| Image Similarity (Avg)↑ | 0.7927 | 0.7987 | 0.7274 |

结论：TI 难以保持物体外观（图像相似度最低），DreamBooth 偏向过拟合参考图（图像相似度略高但可编辑性受限），ProSpect 在保真度与可编辑性之间取得更好平衡，且无需微调模型。用户研究（66 人）中，内容感知任务 ProSpect 获 51.97% 偏好（TI 10.30%、DreamBooth 37.72%），材质任务对 DreamBooth 66.36% 对 33.64%，风格任务对 InST 61.67% 对 38.33%，均显著领先。训练每张图约 20 分钟（RTX 3090），明显快于 TI 的 90 分钟以上。

## 亮点与局限

- 亮点：
  - 从"生成步/频率"这一新视角揭示扩散模型的属性生成顺序，并给出傅里叶谱证据，观察本身具有普适启发性。
  - 无需微调扩散模型、仅从单张图学习即可解耦材质/风格/内容/布局，支持图引导与文本驱动的多种属性编辑，能在语义无关的物体间迁移材质。
  - 相比 TI 训练更快，且在保真-可编辑权衡上优于 DreamBooth。
- 局限：
  - 仍慢于基于编码器的一次前向方法，因为要在随机步上分别优化多组阶段嵌入。
  - 域差异过大时属性迁移结果可能不够美观。
  - 当背景由同类、同尺度物体构成、共享相同频率信息时，属性编辑可能误作用到背景上。

## 延伸思考

- 把"阶段=频率"的划分做得更细或让分组可学习，或研究不同阶段文本条件之间的相互影响，是作者指出的方向；也可探索比均匀 10 段更自适应的切分。
- 该视角与 $$P+$$（按 U-Net 层展开）互补——一个按"层/感受野"、一个按"步/频率"解耦，两者结合可能带来更细粒度的属性控制。
- 阶段嵌入的可解释性与可组合性使其天然适合做"属性调色板"式的创作工具；如何在 SDXL、DiT 等更新架构上复现这一频率顺序规律，是值得验证的迁移性问题。
