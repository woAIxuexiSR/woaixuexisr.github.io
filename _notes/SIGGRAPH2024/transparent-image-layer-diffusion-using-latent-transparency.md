---
title: "Transparent Image Layer Diffusion using Latent Transparency"
authors:
  - "Lvmin Zhang"
  - "Maneesh Agrawala"
category: "Image & Video"
track: "Journal"
source: "arxiv"
institution: "Stanford University"
tags:
  - "Transparent Image Generation"
  - "Latent Diffusion"
  - "Stable Diffusion"
  - "Image Layers"
  - "Alpha Matting"
  - "Image Editing"
links:
  paper: "https://doi.org/10.1145/3658150"
  project: "https://github.com/lllyasviel/LayerDiffuse"
  code: "https://github.com/lllyasviel/LayerDiffuse"
---

## 一句话总结

该工作提出「潜空间透明度（latent transparency）」，把 RGBA 图像的透明通道编码成一个受严格约束的潜空间偏移量，使任意预训练潜扩散模型（如 Stable Diffusion）都能在几乎不破坏原始潜分布的前提下，被微调成原生的透明图像 / 多图层生成器。

## 问题背景

尽管大规模图像生成模型已成为视觉与图形领域的基础设施，但对「分层内容生成」和「透明图像生成」的研究却出奇地少，这与庞大的市场需求形成鲜明反差——绝大多数视觉内容编辑软件和工作流都是基于图层的，高度依赖透明或分层元素来组合创作。

造成这一研究空白的主要原因有二：

- **训练数据稀缺**：高质量透明图像元素多由商业图库以受限且昂贵的方式托管；最大的开源透明图像数据集通常不足 5 万张（如 DIM 仅 45,500 张），而文本-图像数据集（如 LAION）已有数十亿张。
- **表示难以改动**：主流开源生成模型（如 Stable Diffusion）都是潜扩散模型，对潜空间数据表示极其敏感。即便对潜分布做微小改动，也可能严重损害推理或微调质量（例如 SD 1.5 与 SDXL 使用不同潜空间，用不匹配的潜表示微调会显著劣化输出）。

因此，如何在不破坏预训练模型潜分布的前提下为其增加透明格式支持，是核心挑战。

## 核心方法

整体框架分三步：先构造「潜空间透明度」的编解码器（3.1），再用调整后的潜空间微调扩散模型生成透明图（3.2），最后扩展到多图层联合 / 条件生成（3.3）。

**定义**：对透明图像 $$\boldsymbol{I}_t \in \mathbb{R}^{h\times w\times 4}$$，其 RGB 通道记为 $$\boldsymbol{I}_c$$、alpha 通道记为 $$\boldsymbol{I}_\alpha$$。alpha 严格为零处颜色物理上无定义，故用迭代高斯滤波把这些区域填充（称为「padded RGB image」）以避免混叠。预乘图像定义为 $$\boldsymbol{I}=\boldsymbol{I}_c * \boldsymbol{I}_\alpha$$，可被任意 RGB 网络处理。RGB 值范围 $$[-1,1]$$，alpha 值范围 $$[0,1]$$。

### 潜空间透明度（Latent Transparency）

核心思想借鉴了神经网络能把特征「藏」进扰动而不改变整体分布的现象（如 CycleGAN 把人脸藏进拉面图、可逆灰度化等）。作者把透明度信息藏进 Stable Diffusion 潜空间的一个微小偏移里。

关键在于用一个「有害性」度量来约束这个偏移。给定冻结的 SD 潜编码器 $$\mathcal{E}^{*}_{sd}$$ 与解码器 $$\mathcal{D}^{*}_{sd}$$，潜图像为 $$\boldsymbol{x}=\mathcal{E}^{*}_{sd}(\boldsymbol{I})$$。若对其施加偏移 $$\boldsymbol{x}_\epsilon$$ 得到调整后潜表示 $$\boldsymbol{x}_a=\boldsymbol{x}+\boldsymbol{x}_\epsilon$$，则用「原始冻结解码器还能否正确重建」来衡量偏移是否破坏了潜分布：

$$\mathcal{L}_{identity} = \lVert \boldsymbol{I}-\hat{\boldsymbol{I}} \rVert_2 = \lVert \boldsymbol{I}-\mathcal{D}^{*}_{sd}(\mathcal{E}^{*}_{sd}(\boldsymbol{I})+\boldsymbol{x}_\epsilon) \rVert_2$$

$$\mathcal{L}_{identity}$$ 越低，说明偏移越「无害」、越能被预训练模型继续处理。为进一步稳定，作者利用 VAE 已训练好的标准差参数构造偏移：$$\boldsymbol{x}_\epsilon=\lambda_{offset}\,\boldsymbol{x}_{std}\,\boldsymbol{x}_{offset}$$，其中 $$\boldsymbol{x}_{offset}$$ 为新增编码器输出、$$\lambda_{offset}$$ 默认取 $$10^{2}$$。

据此从头训练两个外部独立模型：

- **透明度编码器** $$\mathcal{E}(\cdot,\cdot)$$：把 RGB 与 alpha 转成潜偏移 $$\boldsymbol{x}_\epsilon=\mathcal{E}(\boldsymbol{I}_c,\boldsymbol{I}_\alpha)$$。
- **透明度解码器** $$\mathcal{D}(\cdot,\cdot)$$：从调整后潜表示与其 RGB 重建中提取透明图像 $$[\hat{\boldsymbol{I}}_c,\hat{\boldsymbol{I}}_\alpha]=\mathcal{D}(\hat{\boldsymbol{I}},\boldsymbol{x}_a)$$。

重建损失与 PatchGAN 判别损失联合成总目标：

$$\mathcal{L}_{recon}=\lVert \boldsymbol{I}_c-\hat{\boldsymbol{I}}_c \rVert_2 + \lVert \boldsymbol{I}_\alpha-\hat{\boldsymbol{I}}_\alpha \rVert_2$$

$$\mathcal{L}_{vae}=\lambda_{recon}\mathcal{L}_{recon}+\lambda_{identity}\mathcal{L}_{identity}+\lambda_{disc}\mathcal{L}_{disc}$$

默认权重 $$\lambda_{recon}=1,\ \lambda_{identity}=1,\ \lambda_{disc}=0.01$$。整个过程始终保持原始 SD 的 VAE 编解码器冻结不动，仅由外部模型承载透明度，从而保住大模型的生产级质量。

### 用潜透明度微调扩散模型

由于调整后的潜空间被显式约束对齐到原始分布，Stable Diffusion 可以直接在其上微调。给定调整后潜表示 $$\boldsymbol{x}_a$$，标准扩散目标为：

$$\mathcal{L}=\mathbb{E}_{\boldsymbol{x}_t,t,\boldsymbol{c}_t,\epsilon\sim\mathcal{N}(0,1)}\big[\lVert \epsilon-\epsilon_\theta(\boldsymbol{x}_t,t,\boldsymbol{c}_t) \rVert_2^2\big]$$

此时全部权重可训练，模型即成为原生透明图像生成器。

### 多图层生成

进一步用「注意力共享 + LoRA」扩展到多图层。前景、背景潜表示分别记为 $$\boldsymbol{x}_f,\boldsymbol{x}_b$$，训练前景 LoRA（$$\theta_f$$）与背景 LoRA（$$\theta_b$$）。若两者独立去噪则各自有目标；为得到协调一致的组合，在每个注意力层把两路激活的 {key, query, value} 拼接，合并成联合优化的大模型：

$$\mathcal{L}_{layer}=\mathbb{E}_{\boldsymbol{x}_f,\boldsymbol{x}_b,t,\boldsymbol{c}_t,\epsilon_m\sim\mathcal{N}(0,1)}\big[\lVert \epsilon_m-\epsilon_{\theta,\theta_f,\theta_g}(\boldsymbol{x}_f,\boldsymbol{x}_b,t,\boldsymbol{c}_t) \rVert_2^2\big]$$

其中 $$\epsilon_m=[\epsilon_f,\epsilon_b]$$。只需把前景噪声置零（$$\epsilon_f=0$$，使用干净前景潜表示）即变成「前景条件的背景生成器」；反之置 $$\epsilon_b=0$$ 变成「背景条件的前景生成器」，从而支持条件图层生成。

## 技术细节

**数据构建（人在回路）**：初始收集 2 万张商业授权高质量透明 PNG（来自 5 家图库）。先用这批数据训练 SDXL 的潜透明度 VAE 与扩散模型；随后重复 25 轮迭代——每轮用上一轮模型生成 1 万张样本（提示词取自 LAION-POP），人工挑选 1000 张回填训练集（新样本在后续 batch 中被赋予 2 倍采样概率）。25 轮后数据集增至 4.5 万张。之后无人工干预地生成 500 万对样本，用 LAION 美学阈值 5.5 与 CLIP 分数筛选出 100 万对，并自动剔除无透明像素 / 无可见像素的样本，最后用 LLaVA 生成详细文本提示。

**多图层数据合成**：用 GPT（先 ChatGPT 10 万次、后 LLAMA2 90 万次）生成结构化提示三元组——前景（"a cute cat"）、整图（"cat in garden"）、背景（"nothing in garden"）。前景由训练好的透明生成器产出；再用 SDXL Inpaint 对 alpha<1 的像素修补得到中间整图；最后反转 alpha 掩码、腐蚀 $$k=8$$ 像素后再修补得到背景层。重复 100 万次生成 100 万对图层。

**训练配置**：AdamW，学习率 1e-5，VAE 与扩散模型同用；LoRA 秩固定为 256。人在回路每轮 1 万次迭代、batch size 16。硬件为 4× A100 80G NVLink，整体训练约一周，实际 GPU 时约 350 A100 小时（约合 1000 美元以内），对个人 / 实验室规模友好。

## 实验结果

**定性**：单图基座模型能生成高质量的玻璃透明、毛发、以及发光、火焰、魔法等半透明效果，且泛化到多样主题（图 6）。多图层模型生成的前景 / 背景在光照、几何关系上一致，混合后自然美观（图 7）。条件图层生成能推断前后景交互（如「坐在长椅上」的几何）；迭代生成可反复调用背景条件前景模型堆叠任意数量图层（图 9）。还可与 ControlNet 等控制模块结合，保持全局结构与一致光照（如反光球反射效果）。

**人在回路的有效性**：作者用「每 100 次采样中的缺陷样本数（DSC，越低越好）」跟踪各轮 checkpoint。DSC 从第 0 轮的 61 稳步下降到第 20 轮的约 5，验证人工筛选逐步提升了数据集与模型质量。

**消融**：对比两种不改造外部编解码器的替代架构——(a) 直接给 UNet 加 4 通道编码 alpha：因扩散是迭代过程，UNet 必须同时生成并识别新增通道，导致潜分布改变、生成质量严重劣化；(b) 直接给 VAE 加通道再微调 UNet：训练极不稳定、时常崩溃。二者都印证了「必须严格约束潜分布」这一核心动机，凸显本文方案的必要性。

**与图像抠图的关系**：对比 PPMatting、Matting Anything（基于 SAM）、VitMatte（基于 ViT 的 tri-map 方法）等 SOTA 抠图方法后指出，火焰、阴影等半透明内容，以及纯白毛发对纯白背景等场景，一旦与复杂背景混合，抠图几乎无法干净分离；要得到完美干净的元素，唯有用原生透明生成器从头合成。

**用户研究**（14 名参与者，含专业内容创作者）：
- 第 1 组，与「SD + 抠图」的临时方案对比，用户在 **97.1%** 的情况下偏好本方法的原生透明结果（SD+PPMatting 2.1%、SD+Matting Anything 0.8%）。
- 第 2 组，与商业透明素材（Adobe Stock）搜索结果对比，本方法偏好率 45.3% 对商业素材 54.7%，质量接近付费商业素材（后者仅略微领先）。

此外作者还展示了：原始 RGBA 通道会用平滑「渗色」填充 RGB 以避免混叠；通过训练时随机丢弃 30% 偏移可得到更鲁棒的解码器，处理社区模型（动漫、卡通等）产生的偏移缺失 / 错配；方法能直接迁移到各类社区 LoRA / 模型 / 提示风格。

## 贡献与局限

**贡献**：
1. 提出「潜空间透明度」，使大规模预训练潜扩散模型能生成单张透明图或多张透明图层，且以「原始冻结解码器可否正确重建」作为约束显式保护潜分布。
2. 提出注意力共享机制，实现前后景一致、混合协调的多图层生成。
3. 开源了透明图像生成的预训练模型、两个多图层 LoRA，以及若干消融架构，训练成本低（约 350 A100 小时）。

**局限**：
- 方法建立在从头收集的 100 万透明图像 / 图层对之上，数据构建依赖人在回路与大量 GPT/inpaint 合成，流程较重。
- 某些社区微调模型可能无法产生理想的潜偏移，需借助鲁棒解码器（丢弃增强）兜底，属于事后补救。
- 多图层数据由模型自身与 SDXL Inpaint 合成，可能引入合成偏差；条件与迭代生成的图层一致性仍依赖模型学到的分布，缺乏显式几何约束。
