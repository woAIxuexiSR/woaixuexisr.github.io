---
title: "Harnessing Diffusion-Yielded Score Priors for Image Restoration"
authors:
  - "Xinqi Lin"
  - "Fanghua Yu"
  - "Jinfan Hu"
  - "Zhiyuan You"
  - "Wu Shi"
  - "Jimmy S. Ren"
  - "Jinjin Gu"
  - "Chao Dong"
category: "Image & Video"
track: "Journal"
source: "arxiv"
institution: "Shenzhen Institute of Advanced Technology, CAS; University of Chinese Academy of Sciences; The Chinese University of Hong Kong; SenseTime Research; Hong Kong Metropolitan University; INSAIT, Sofia University; Shenzhen University of Advanced Technology"
tags:
  - "Image Restoration"
  - "Diffusion Model"
  - "Generative Adversarial Network"
  - "Score Prior"
  - "Super Resolution"
  - "Low Level Vision"
  - "LoRA"
links:
  paper: "https://doi.org/10.1145/3763346"
  project: "https://www.hypir.org/"
  code: "https://github.com/XPixelGroup/HYPIR"
---

## 一句话总结

HYPIR 提出用预训练扩散模型的参数直接初始化一个图像复原网络，随后仅用轻量对抗训练（GAN）微调，抛弃扩散的迭代采样，从而在单次前向即可完成高质量、高保真、可控且极快的图像复原。

## 研究背景

深度图像复原的本质，是学习一个从退化图像分布到自然图像分布的映射。这里存在三个核心挑战：去除退化、生成逼真细节、保证与输入的像素级一致性。历史上出现了三大类方法，但都难以在质量、保真度与速度之间取得良好平衡。

基于像素损失（MSE）的卷积网络自 2014 年开启深度学习复原时代，在 PSNR、SSIM 等指标上表现优异，能较好解决去退化与一致性问题，但输出往往过度平滑、缺乏真实细节。为解决真实感问题，2017 年前后出现基于生成对抗网络（GAN）的复原模型，用感知损失加对抗训练让结果更贴合人眼感知，但 GAN 训练困难，容易模式崩溃、训练不稳定，通常只能生成有限的常见纹理，无法覆盖自然图像的完整多样性。近年来，基于大规模预训练文生图扩散模型的复原方法成功突破了真实感瓶颈，但扩散固有的多步迭代特性带来训练慢、推理慢、生成不稳定等严重缺陷；即便有蒸馏或少步方法，也难以从根本上兼顾质量与效率。

作者由此提出一个被长期忽视的思路：扩散模型本身已经非常接近一个理想的复原模型，只需极少量对抗微调就能把它转化为优秀的端到端复原网络。这一思路之所以被忽视，源于两种惯性认知——扩散模型几乎总与扩散式采样器绑定，以及学界普遍认为扩散一定优于 GAN、纯对抗训练看起来没有前途。

## 方法

方法的核心极其简单：用预训练扩散模型做初始化，再用对抗方式微调用于复原。

### 退化预去除编码器

设 $$I$$ 为干净自然图像，$$I_{deg}$$ 为其退化版本。沿用扩散模型的变分编码器 $$\mathcal{V}_E$$ 与解码器 $$\mathcal{V}_D$$ 在潜空间处理图像，记 $$x = \mathcal{V}_E(I)$$，并假设 $$x \sim p_{data}$$。为增强编码器对退化的鲁棒性，先微调编码器 $$\mathcal{V}_{ER}$$，最小化：

$$L_E = \| \mathcal{V}_D(\mathcal{V}_{ER}(I_{deg})) - \mathcal{V}_D(\mathcal{V}_{ER}(I_{GT})) \|_2^2$$

其中解码器 $$\mathcal{V}_D$$ 固定。微调后的编码器把退化图映射为潜观测 $$y = \mathcal{V}_{ER}(I_{deg})$$，其退化模型可写为：

$$y = k_{deg} * x + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \eta^2 I)$$

其中 $$k_{deg}$$ 为潜空间退化核，$$*$$ 为空间卷积，$$\varepsilon$$ 为加性噪声。经过退化预去除，潜变量 $$y$$ 的退化模式更简单，主要表现为平滑与加噪，类似直接施加 MSE 复原后的效果。

### 对抗训练与扩散初始化

目标是优化可微网络 $$U_\theta$$，使推前分布 $$p_\theta = U_{\theta\sharp} p_y$$ 在退化输入 $$y \sim p_y$$ 驱动下尽量逼近 $$p_{data}$$。采用对抗训练，配合判别器 $$D_\phi$$：

$$\min_\theta \max_\phi \; \mathbb{E}_{x \sim p_{data}}[\log D_\phi(x)] + \mathbb{E}_{y \sim p_y}[\log(1 - D_\phi(U_\theta(y)))]$$

利用 $$y$$ 与真值 $$x$$ 的逐像素对应关系，加入重建（内容）惩罚：

$$\min_\theta \max_\phi \; L_{adv}(\theta, \phi) + \lambda_{rec} \, \mathbb{E}_{(y,x) \sim p_{data}}[\mathrm{Recon}(U_\theta(y), x)]$$

其中重建项实践中采用 MSE 与 LPIPS 组合，兼顾像素精度与感知相似度。

关键改动在于：从训练好的扩散生成模型初始化对抗训练。设扩散模型用同一网络 $$U$$ 作为其分数函数 $$S = U_{\theta_{Diff}}$$，对抗训练直接从这些预训练参数出发，继承扩散模型的网络结构与前面微调好的退化去除编码器，构成完整的图到图复原流水线。复原训练时只微调中间的 U-Net，编码器与解码器保持固定，并用 LoRA 大幅减少可训练参数、加速训练。

### 为什么这个简单方法有效

作者从两点给出启发式论证。第一，图像复原等价于估计分数（退化图像对数密度的梯度），它指向回到自然图像分布的最快路径；而扩散模型恰恰被训练来在各噪声水平上学习这样的分数场，其内化的先验已经接近理想复原算子。从前向-反向 SDE 出发，条件采样的后验 SDE 为：

$$d x_t = \left[ -\frac{\beta(t)}{2} x_t - \beta(t) \left( \nabla_{x_t} \log p_t(x_t) + \nabla_{x_t} \log p_t(y \mid x_t) \right) \right] dt + \sqrt{\beta(t)} \, d\bar{w}$$

其中似然项 $$\nabla_{x_t} \log p_t(y \mid x_t)$$ 难以计算或代价高昂。作者选择丢弃该项，使用无条件反向扩散动力学，再通过"在中间时刻注入观测、随后继续无条件反向扩散"重新引入对 $$y$$ 的依赖。这引出一步复原公式：

$$\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}(t)}} \left( x_t + (1 - \bar{\alpha}(t)) S_\theta(x_t, t) \right) \triangleq R_\theta(x_t)$$

$$R_\theta$$ 与 $$S_\theta$$ 共享同一网络架构 $$U$$，输出近似干净图像；再用 GAN 损失微调以修正残余伪影。

第二，用扩散权重初始化复原网络，会把它置于自然图像空间附近。作者用定理刻画这种邻近性（扩散到复原的邻近定理）：设扩散网络分数误差以 $$\varepsilon_{sc}$$ 为界，退化核满足 $$\Delta_k := \| k_{deg} - k_\sigma \|_1 \ll 1$$，则推前分布与真实分布的 2-Wasserstein 距离满足：

$$W_2(p_{\theta_{Diff}}, p_{data}) \le C_1 \varepsilon_{sc} + C_2 \Delta_k =: \epsilon_0$$

由此得到三条推论：其一，初始梯度小，$$\| \nabla_\theta L_G(\theta_{Diff}) \|_2 \le \sqrt{2} L_J \epsilon_0$$，避免 NaN、梯度爆炸与震荡；其二，近乎完整的模式覆盖，任意可测分区上 $$|p_{\theta_{Diff}}(A_k) - p_{data}(A_k)| \le \frac{\sqrt{2}}{2}\epsilon_0$$，大幅降低模式崩溃风险；其三，收敛更快，在光滑且强凸的局部假设下只需对数级迭代步数即可逼近目标精度，理论估计约 $$8 \times 10^3$$ 步，与实验中约 10k 步收敛量级一致，而从零训练通常需 $$\ge 3 \times 10^5$$ 步。

## 其他实现细节

扩散模型的规模从根本上决定基线性能。作者考察了四种扩散模型：SD2（0.8B）、SDXL（2.6B）、SD3（8B）、Flux（12B）。由于本方法不需要额外控制适配器（如 ControlNet），显存占用低，能高效利用最大规模的扩散模型，这是以往方法难以做到的。

方法天然继承扩散模型的多种可控性：

- 文本提示：训练时用 LLaVA 为每张图生成文本标注，测试时同样自动预测提示，实现全自动的基于提示的控制，可补全细节、按用户指定填充不可恢复区域、并通过自然语言调节保真与风格。
- 纹理丰富度：引入基于图像拉普拉斯统计的"纹理丰富度"控制量，训练时提供真值纹理统计作指导，推理时用户可动态调节全局纹理密度。
- 生成-保真权衡：在退化预去除编码器输出的潜表示中注入噪声以部分遮蔽原始信号，测试时通过调节噪声注入在严格复原与生成自由之间取舍。
- 随机采样：上述噪声注入本身带来随机性，采样不同噪声即可获得多样复原结果，噪声尺度越大采样空间越大。

训练数据约 2000 万高质量图像块（带文本描述）外加 7 万张人脸图，退化合成沿用 Real-ESRGAN 流水线。判别器用预训练 ConvNeXt 初始化以增强特征提取。损失权重为对抗损失、LPIPS、MSE 分别取 0.5、5、1；用 AdamW 优化，初始 batch 384、学习率 $$1\times10^{-5}$$ 训 10k 步，后用梯度累积把 batch 提到 1536、学习率降到 $$5\times10^{-6}$$ 再训 10k 步；全程在 64 张 A6000 上进行，维护 EMA（衰减 0.999）。

## 实验

消融研究均以 SD2 为初始扩散模型。

初始化与训练策略对比：从零直接训练复原 GAN 会模式崩溃、训练不稳；MSE 或 DAE 预训练能改善画质但仍难解决 GAN 训练固有难题；扩散初始化画质显著更高，且质量可与"扩散预训练+扩散微调"（类 DiffBIR、SUPIR）媲美，却省去昂贵的迭代生成。直接用 BigGAN 这类生成式 GAN 做初始化困难重重；即便初始化到位，仅用 MSE 目标微调也无法充分发挥所学生成先验。

判别器设计：对比 DINO、CLIP、扩散 U-Net、ConvNeXt 四种骨干。DINO 与 CLIP 需把图缩放到 224×224，限制了对高频细节的监督并引入噪声；扩散 U-Net 本质是生成模型，不太适合直接做判别器；ConvNeXt 能以复原输出原分辨率处理、更好保留细节纹理，效果最佳。

LoRA 秩：增大 LoRA 秩扩展可训练容量、整体提升表现，但过高的秩相对计算成本收益递减。

退化预去除：不做预去除时编码器会误解退化内容、引入明显伪影；加入后清晰度与准确度明显改善。

扩散先验：更大更先进的扩散模型既能更精确逼近分数函数，也更能捕捉复杂图像结构，复原效果更好；但性能提升并非单纯来自模型变大——没有本方法，单独训练大规模 GAN 依然极难获得可比结果。

对比实验：与 Real-ESRGAN、StableSR、DiffBIR、SUPIR（SDXL）、InvSR、TSDSR、S3Diff、SeeSR、OSEDiff 等在 DIV2K 与 RealPhoto60 上比较，并额外引入百年历史照片评估真实场景表现。定性上，SD2 版 HYPIR 生成的纹理已优于对比方法，Flux 版进一步提升，能清晰复原大蒜、玻璃瓶等细粒度结构，以及面部、文字、建筑等细节，甚至能把严重退化的历史低分辨率照片重建到 4K 乃至 6K。

定量评估强调用户研究更能反映感知质量（作者指出 IQA 指标可被人为刷高、不足以评价生成式复原）。两轮用户研究共 26 张图、100 名有图像处理经验的参与者以 0–10 打分，HYPIR 在两轮中均取得最高感知分，且低分实例罕见。在 DIV2K 与 RealPhoto60 的 NIQE、MUSIQ、MANIQA、CLIP-IQA、DeQA 等无参考指标（合成集另加 PSNR、SSIM、LPIPS）上，HYPIR 也稳居前列，例如 DIV2K 上 Flux 版 LPIPS 达 0.2022、MANIQA 0.5966、CLIP-IQA 0.7555。

运行时间：生成 1024×1024 图像时，多步扩散方法耗时显著（SUPIR 约 40s、SeeSR 约 96s），而 HYPIR 即便用 Flux（12B）也只需约 1.74s，SD2 版约 1.05s，与其他单步方法相当却质量更优。

## 结论

HYPIR 用预训练扩散模型做参数初始化，再辅以轻量对抗微调，构建出无需扩散教师、无需控制适配器、无需迭代精修的端到端图像复原模型。理论与实验共同表明：扩散预训练为复原提供了近乎最优的初始化，使对抗微调具备小初始梯度、近完整模式覆盖与对数级快速收敛的优势。该方法在保真度、真实感与效率上均超越以往先进方法，仅需单次前向即可完成复原，并继承扩散模型的丰富可控性（文本引导、纹理丰富度、生成-保真权衡与随机采样），实现了高效且高质量的图像复原。
