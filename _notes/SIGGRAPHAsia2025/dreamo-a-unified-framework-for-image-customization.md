---
title: "DreamO: A Unified Framework for Image Customization"
authors:
  - Chong Mou
  - Yanze Wu
  - Wenxu Wu
  - Zinan Guo
  - Pengze Zhang
  - Yufeng Cheng
  - Yiming Luo
  - Fei Ding
  - Shiwen Zhang
  - Xinghui Li
  - Mengtian Li
  - Mingcong Liu
  - Yunsheng Jiang
  - Shaojin Wu
  - Songtao Zhao
  - Jian Zhang
  - Qian He
  - Xinglong Wu
category: "Neural & Generative"
track: "Conference"
source: arxiv
institution: "ByteDance Intelligent Creation Team; Peking University"
tags:
  - Diffusion Model
  - Image Customization
  - Diffusion Transformer
  - Feature Routing
  - Identity Preservation
  - Style Transfer
  - Virtual Try On
  - LoRA
links:
  paper: "https://doi.org/10.1145/3757377.3763956"
  project: "https://mc-e.github.io/project/DreamO"
  code: "https://github.com/bytedance/DreamO"
---

## 一句话总结

DreamO 在预训练扩散 Transformer（Flux）上只训练一小组附加参数（LoRA），用统一的序列条件格式把身份、主体、试穿、风格等多种定制条件塞进同一个模型，并通过特征路由约束、占位符策略与渐进式训练，实现高保真、可解耦、多条件协同的图像定制。

## 研究背景

大规模生成模型在图像定制（身份、主体外观、风格、背景等）上已展现出很强能力，但绝大多数方法都是为单一任务设计的，难以泛化到多类型条件的组合。如何做出一个"统一"的图像定制框架仍是开放难题。

已有工作各有局限。早期的 Composer 联合训练多条件（深度、颜色、草图）扩散模型；ControlNet、T2I-Adapter 这类方法在预训练模型上加控制模块以节省训练成本，但控制能力局限于简单的空间条件，条件之间交互僵硬、存在控制冗余。DiT 架构出现后，OminiControl 用统一序列把图像条件与扩散潜变量拼接，可做身份、颜色、布局等多种任务，但它对不同任务分开训练，难以同时处理多个条件。OmniGen 借助预训练大语言模型、UniReal 通过视频生成预训练加全模型后训练来实现通用生成控制。作者认为，高质量的多概念图像定制不能仅靠大语言模型或视频模型的通用能力，而需要面向图像定制专门设计的架构。当前社区缺少一个专为多概念、多条件场景量身打造的高效方法。

## 方法

DreamO 以 Flux-1.0-dev 为基座，目标是用单一模型、低训练成本支持风格、身份、主体外观、试穿等多种条件，并允许条件之间相互作用。

### 统一序列条件

给定 $$n$$ 张条件图像 $$C = \{C_1, ..., C_n\}$$，先复用 Flux 的 VAE 把条件图编码到与噪声潜变量相同的潜空间。条件图尺寸灵活：细节丰富的图建议用较高分辨率保清晰度，细节少的图用低分辨率即可，以降低压缩成本。随后把图像、文本、条件三类 token 沿序列维拼接后送入 Flux。

为让模型接入条件输入，在 Flux 输入端引入一个条件映射层。条件 token 的位置编码用旋转位置编码（RoPE）与噪声潜变量对齐，并借鉴 OminiControl 的非重叠位置编码，沿对角线方向扩展这些嵌入。此外引入一个可训练、按索引区分的条件嵌入（CE）$$\mathbb{R}^{10 \times c}$$ 直接加到条件 token 上。训练参数则以 LoRA 模块的形式集成进 Flux。

### 路由约束（Routing Constraint）

在条件引导框架里，条件图像与生成结果之间存在交叉注意力：

$$M = \frac{Q_{cond,i} K_{img}^{T}}{\sqrt{d}}$$

其中 $$Q_{cond,i} \in \mathbb{R}^{l_{cond,i} \times c}$$ 是第 $$i$$ 个条件图的 token，$$K_{img} \in \mathbb{R}^{l \times c}$$ 是噪声图像潜变量的 token。稠密相似度矩阵 $$M \in \mathbb{R}^{l_{cond,i} \times l}$$ 沿 $$l_{cond,i}$$ 维求平均，得到响应图 $$M \in \mathbb{R}^{l}$$，表示条件图在生成结果各位置上的全局相似度。为让图到图的注意力聚焦到特定主体，用 MSE 损失约束 DiT 内部条件图与生成结果之间的注意力：

$$L_{route} = \frac{1}{n_c \times n_l} \sum_{j=0}^{n_l-1} \sum_{i=0}^{n_c-1} \| M_i^{j} - M_{target,i} \|_2^2$$

其中 $$i$$、$$j$$ 分别是条件索引与层索引，$$n_c$$、$$n_l$$ 是条件数与层数，$$M_{target}$$ 是目标图的主体掩码。加了路由约束后，条件图的注意力清晰地聚焦到目标主体上，结果与参考图在细节一致性上明显提升；在多参考情形下，这一策略也有助于解耦。

在图到图路由约束之外，作者还设计了占位符到图像的路由约束，用来建立文本描述与条件输入之间的对应关系。具体做法是在实例名后追加一个占位符 $$[ref\#i]$$，例如"A woman from [ref#1] and a woman from [ref#2] is walking in the park."。多条件训练时，计算条件图像 token 与占位符 token 的相似度，约束 $$C_i$$ 与 $$[ref\#i]$$ 的相似度为 1、与其他配对为 0：

$$L_{holder} = \frac{1}{n_c} \sum_{i=0}^{n_c-1} \| Softmax(Q_{cond,i} \times K_{text,i}^{T}) - B_i \|_2^2$$

其中 $$K_{text,i}$$ 是 $$[ref\#i]$$ 的文本特征，$$B_i$$ 是二值矩阵，占位符与条件图匹配时为 1，否则为 0。

总损失为：

$$L = \lambda_{diff} \cdot L_{diff} + \lambda_{route} \cdot L_{route} + \lambda_{holder} \cdot L_{holder}$$

为让模型也能处理普通文本输入，训练时以 50% 概率使用不带占位符的普通文本并相应丢弃 $$L_{holder}$$。作者指出 $$L_{route}$$ 与 $$L_{holder}$$ 不会带来明显额外训练开销（2.5s/iter 对 3s/iter）。

其中扩散基础损失采用 Flow Matching。在时间步 $$t$$，潜变量定义为 $$z_t = (1-t) z_0 + t \epsilon$$，$$z_0$$ 为干净图像、$$\epsilon \in \mathcal{N}(0,1)$$ 为高斯噪声，模型回归目标速度：

$$L_{diff} = E[\| (z_0 - \epsilon) - V_{\theta}(z_t, t, y) \|^2]$$

### 训练数据构建

为实现泛化的图像定制，作者跨任务收集了大规模数据：

- 身份配对数据：高质量身份配对数据难以从网上收集，故用开源 ID 定制方法 PuLID 构建。用 PuLID-FLUX 生成同一身份的两张图互为参考；用 PuLID-SDXL 结合参考人脸与风格文本生成风格化配对。最终收集 15 万写实数据与 6 万风格化身份数据。
- 主体驱动数据：单主体用 Subject200K 数据集，并通过检索补充 10 万条与人物相关的配对数据；多主体则在 Subject200K 上拼接构造双栏图像，并采用 X2I-subject 数据集。为提升人物驱动生成，构建了类似 MovieGen 的流水线：从长视频用场景检测切短片，用 Mask2Former 生成人物掩码并做目标跟踪，跨片实例匹配用 SigLip 嵌入加聚类。
- 试穿数据：一部分来自网上直接采集的模特与服装配对图；另一部分先爬取高质量模特图作为真值，再用图像分割抽取衣物生成配对。经人工筛除低质样本后得到 50 万试穿配对。
- 风格驱动数据：处理两类风格迁移任务——带内容文本描述的风格参考控制，以及带内容参考图的风格参考。第一类用基于 SDXL 的内部风格定制模型，从两条不同提示生成同风格不同内容的图；第二类需要风格参考图、内容参考图和目标图，用 Canny 引导的 Flux 为每张风格图生成内容参考。
- 路由掩码提取：用 LISA 依据文本描述抽取物体掩码作为路由约束标签；复杂数据集中用 InternVL 生成目标物体描述。

虽然数据按任务分别构建，作者观察到跨任务能力的涌现，例如模型能定制 ID 与试穿的组合，而这种组合并不存在于训练数据中。

### 渐进式训练

实验发现直接在全部数据上训练难以收敛，主要因为 LoRA 优化容量有限，难以在复杂数据分布下捕捉任务专属能力；同时训练后输出质量会偏离 Flux 原有先验，这源于部分低质样本的影响。为此设计三阶段渐进式训练：

1. 先在主体驱动数据上优化，让模型具备初步的一致性保持能力。由于 Subject200K 由 Flux 自身生成、分布接近，收敛快；此阶段也把双栏 Subject200K 图作为一部分数据，促进多参考控制快速收敛。
2. 加入全部数据做全量微调，让模型在本文定义的所有子任务上进一步收敛。
3. 图像质量精修阶段：用 Flux 生成约 4 万训练样本，以原图为参考引导模型重建自身；为防止"复制粘贴"效应，丢弃参考图 95% 的 token。短时优化后生成质量显著提升，与 Flux 的生成先验对齐。

## 实验

实现细节：基座为 Flux-1.0-dev，LoRA 秩 128，新增参数共 478M；Adam 优化器，学习率 4e-5，8 张 A100 80G，批大小 8。三阶段迭代数分别为 20K、90K、3K。推理用 Flux-Turbo 加速，可在 10 秒内生成 1024×1024 结果；部分输入用 BEN2 去背景。

定性对比覆盖身份、主体、试穿、风格四类子任务，分别与 PhotoMaker、InstantID、PuLID（身份），MS-Diffusion、OmniGen、OminiControl（主体），IMAGDressing、MagicClothing（试穿），StyleShot、StyleAlign、InstantStyle、DeaDiff、CSGO（风格）等方法比较。DreamO 在各场景下都能高保真注入条件信息并保持文本对齐，试穿上既保衣物保真又不丢文本对齐，风格上内容侵入更弱、文本对齐与风格保真更好。

定量结果（节选）：

- 单主体（DreamBench）：CLIP-sim 0.9150、DINO-sim 0.8056、Text-sim 31.92，均优于 MS-Diffusion、OmniGen、OminiControl。
- 多主体：CLIP-sim 0.7775、DINO-sim 0.6253、Text-sim 31.46，主体一致性领先。
- 身份（Unsplash-50）：Face-sim 0.607、Text-sim 0.2570，优于 PhotoMaker、InstantID、PuLID。
- 风格：Style-sim 0.7340、Text-sim 0.2750，均为最优。
- 试穿（VITON-HD 300 件）：CLIP-sim 0.7613、Text-sim 30.47；IMAGDressing 虽 CLIP-sim 更高但只能生成白底、几乎无文本跟随能力。

用户研究：对风格、物体、身份、试穿四类任务各设 6 个样本，邀请 20 名志愿者从文本对齐、参考对齐、图像质量三方面打分（0–5），DreamO 在三项上均表现更好。

消融（多主体）：去掉条件嵌入 CE、路由约束 RC、渐进训练 PT 均导致性能下降。去掉 RC 会造成单条件保真度下降、多条件特征串扰（如两个玩具特征交叉）；去掉占位符损失后占位符难以精确控制对应图像；去掉渐进训练则复杂任务收敛不良、质量偏离 Flux 先验。完整实现 CLIP-sim 0.7775、DINO-sim 0.6253、Text-sim 31.46，显著优于各消融设置。

## 结论

DreamO 在单一预训练 DiT 架构内实现跨多种条件类型（身份、风格、主体、试穿）的泛化图像定制。通过把所有条件类型嵌入 DiT 输入序列并引入特征路由约束，模型在高保真一致性的同时有效解耦异构控制信号；渐进式训练让模型在复杂数据分布下逐步获得多样控制能力，并保持基座模型固有的图像质量。大量实验表明 DreamO 能高质量完成广泛的图像定制任务，并在多条件场景中展现出很强的灵活性。
