---
title: "Face0: Instantaneously Conditioning a Text-to-Image Model on a Face"
authors:
  - "Dani Valevski"
  - "Danny Lumen"
  - "Yossi Matias"
  - "Yaniv Leviathan"
category: "Neural & Generative"
track: "Conference"
source: "arxiv"
institution: "Google"
tags:
  - "Text-to-Image"
  - "Diffusion Model"
  - "Face Personalization"
  - "Classifier-Free Guidance"
  - "Consistent Character"
links:
  paper: "https://doi.org/10.1145/3610548.3618249"
---

## 一句话总结

Face0 通过把人脸嵌入投影进 Stable Diffusion 的文本条件空间并联合微调模型，让文本到图像模型无需任何推理期优化（微调 / 反演）就能在几秒内根据单张人脸照片生成保留身份的图像。

## 研究背景

- 领域现状：扩散模型让基于自由文本的图像生成质量和多样性大幅提升，但"根据用户提供的一张照片生成某个特定人物"仍是难点。
- 核心痛点：已有个性化方法（如 DreamBooth 微调模型、Textual Inversion 反演到文本嵌入空间）都要在推理时解一个优化问题，效果虽好但极其耗时耗显存——单个身份的 DreamBooth 在 A100 上需约 15 分钟。
- 本文 idea：与其在推理时优化，不如把"人脸"当成一种新的条件模态，在训练阶段就教会模型读取人脸嵌入。这样推理时几乎与原始基础模型无异，一张照片、一句提示，几秒出图，无需逐人训练。

## 方法

整体框架：给训练集里含人脸的图像补上一份人脸嵌入，用一个小 MLP 把该嵌入投影到 CLIP 上下文空间，覆盖文本条件的最后三个 token，然后在标准扩散 MSE 损失下联合微调 U-Net 与投影网络。推理时对用户照片跑同一套人脸提取逻辑，把投影后的嵌入注入上下文即可采样。

```mermaid
flowchart LR
  A["输入图像 + 文本描述"] --> B["检测并裁剪人脸"]
  B --> C["人脸嵌入模型 (Inception ResNet V1)"]
  C --> D["4 层 MLP 投影到 CLIP 空间"]
  E["文本 CLIP 编码"] --> F["拼接: 覆盖最后 3 个 token"]
  D --> F
  F --> G["扩散模型 U-Net"]
  G --> H["MSE 去噪损失 / 采样输出"]
```

关键设计：

1. **人脸嵌入而非文本反演**：采用在 VGGFace2 上训练的 Inception ResNet V1，并丢弃最后几层。这样得到的嵌入不足以做精确身份识别，却保留了高质量生成所需的视觉细节，同时大体固定了姿态与表情。
2. **投影与 token 覆盖**：4 层前馈网络（隐层维度 768、ReLU）把人脸嵌入映射为三个 768 维向量，覆盖 Stable Diffusion CLIP 嵌入的第 75–77 个 token，仅新增约 1000 万参数。训练时以 10% 概率将投影嵌入置零，以支持无分类器引导。
3. **联合训练目标**：从 Stable Diffusion 1.4 出发，在带人脸条件的标准扩散损失下联合微调。冻结文本 CLIP 编码器和 VAE。
$$L(\theta) = E_{t,x_0,d,f,\epsilon}\left[w_t \lVert M_\theta(\alpha_t x_0 + \sigma_t \epsilon, t, d, f) - \epsilon \rVert^2\right]$$
其中 $$d$$ 是文本条件，$$f$$ 是新引入的人脸嵌入条件。
4. **三路加权的无分类器引导**：在标准 CFG（权重 $$w = 7.5$$）基础上，把条件项拆成"仅文本""仅人脸""文本+人脸组合"三路加权，用参数 $$c$$ 控制组合向量的相对权重、$$a$$ 控制仅人脸向量的权重：
$$\hat{\epsilon}_t(z_t, d, f) = c\cdot\epsilon_t(z_t, d, f) + (1-c)\cdot\left(a\cdot\epsilon_t(z_t, f) + (1-a)\cdot\epsilon_t(z_t, d)\right)$$
调节这几个权重即可在写实（$$c=1$$）与非写实风格（增大文本权重）之间平滑切换。

## 实验结果

与 DreamBooth 在 20 个合成身份（SYN）和 10 个 LFW 身份上对比：DreamBooth 用每个身份 4–5 张图训练，Face0 只吃单张照片。指标为生成图与文本提示的 CLIP 余弦相似度（文本对齐）以及与输入人脸的相似度（人脸对齐），二者之和为总分。两方法文本对齐相当，Face0 在人脸对齐上明显更优。

| 方法 | 文本对齐↑ | 人脸对齐↑ | 总分↑ |
|------|-----------|-----------|-------|
| Face0 (SYN) | 0.24 | 0.72 | 0.96 |
| DreamBooth (SYN) | 0.23 | 0.46 | 0.69 |
| Face0 (LFW) | 0.23 | 0.66 | 0.89 |
| DreamBooth (LFW) | 0.24 | 0.39 | 0.62 |

另外，取 8 个随机种子中的最佳结果时 DreamBooth 提升更大（如 SYN 从 0.69 提到 0.93），说明它稳定性不如 Face0；Face0 逐种子表现更一致。

## 亮点与局限

- 亮点：
  - 推理几乎零额外成本，与基础模型同速，无需逐人训练，秒级出图。
  - 方法极简（一个小 MLP + 联合微调），却带来新能力：文本或直接操作人脸嵌入（如线性插值）来控制生成。
  - 用固定随机人脸嵌入即可轻松实现跨图一致角色生成；用随机嵌入替代照片还可能缓解模型把面部特征与无关词汇（如"CEO""程序员"）绑定的偏见。
- 局限：
  - 并非总能完全保住身份，有时生成"神似但可辨别不同"的相像人物。
  - 仅支持单张人脸条件，多脸场景留作未来工作。
  - 采用的嵌入基本固定了姿态与表情，灵活性受限；偏见缓解仅为初步结果，仍有诸多开放问题。

## 延伸思考

Face0 把"个性化"从推理期优化彻底前移到训练期条件注入，这一思路后续被 IP-Adapter、PhotoMaker、InstantID 等一系列"零微调"人脸/图像条件方法继承与发扬。值得追问的方向包括：换用能保留更多姿态/表情信息或多脸的嵌入器、在每一步采样中用人脸模型做引导、以及把同一范式推广到人脸之外的其它条件域。此外，论文坦诚讨论的伪造与偏见风险，也提示这类即时个性化能力在开放前需要配套的检测与评测手段。
