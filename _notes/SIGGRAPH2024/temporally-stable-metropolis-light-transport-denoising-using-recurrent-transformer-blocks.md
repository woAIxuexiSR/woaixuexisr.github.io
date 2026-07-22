---
title: "Temporally Stable Metropolis Light Transport Denoising using Recurrent Transformer Blocks"
authors:
  - "Chuhao Chen"
  - "Yuze He"
  - "Tzu-Mao Li"
category: "Rendering"
track: "Journal"
source: "author-page"
institution:
  - "University of California San Diego"
  - "Tsinghua University"
tags:
  - "Monte Carlo Denoising"
  - "Metropolis Light Transport"
  - "Recurrent Neural Network"
  - "Vision Transformer"
  - "Temporal Stability"
  - "Kernel-Predicting Network"
  - "Global Illumination"
links:
  paper: "https://doi.org/10.1145/3658218"
  project: "https://czzzzh.github.io/MLTD/"
  code: "https://github.com/CzzzzH/MLTD"
---

## 一句话总结

针对 Metropolis Light Transport（MLT）渲染特有的相关性伪影与帧间闪烁，作者用"循环连接 + 视觉 Transformer"搭建了一个离线序列去噪器，配合专门为 MLT 设计的采样分解（sample decomposition）与分层核预测，在保持时序稳定的同时显著提升了含复杂光路场景的去噪质量。

## 研究背景

Monte Carlo 去噪近年进展迅速，但绝大多数工作都聚焦于路径追踪（Path Tracing, PT）渲染。Metropolis Light Transport 作为一种基于马尔可夫链蒙特卡洛（MCMC）的全局光照算法，擅长渲染焦散、复杂可见性等"难光路"场景，却长期被去噪研究忽视。原因在于 MLT 有几个和现有去噪管线格格不入的特性：

- **相关性伪影与闪烁**。MLT 通过对上一个样本做变异（mutation）来复用稀有的高贡献光路，这使得相邻样本之间高度相关。在动画渲染中，这种相关性会产生不可预测的斑块状伪影和帧间闪烁，破坏视觉一致性。
- **方差难以估计**。由于马尔可夫链样本彼此相关，MLT 很难像 PT 那样估计每像素方差，导致依赖方差引导的时空去噪方法（如 SVGF 一类）以及经典的自适应采样/重建方法都无法直接套用。
- **现有神经去噪器的错配**。交互式/实时去噪器为了性能牺牲了网络容量，难以处理 MLT 这种高难度输入；离线单帧去噪器容量够大但不考虑时序稳定性；最接近需求的 Balint 等人（2023）的序列去噪器仍基于 CNN 架构和时序滑窗/逐像素输入混合（per-pixel input blending），在 MLT 上力不从心。

作者的判断是：MLT 的相关噪声需要更深层的时序信息累积，而不是简单的逐像素线性混合；同时需要更强的空间感受野来识别复杂噪声模式。这两点分别指向循环结构（recurrent）与 Transformer。

## 核心方法

渲染要计算的是每个像素 $j$ 上对所有穿过该像素的光路 $\bar{x}$ 的路径积分：

$$I_j = \int_{\Omega} h_j(\bar{x}) f(\bar{x})\, d\bar{x}$$

其中 $h_j$ 是像素滤波器，$f$ 是测量贡献函数。MLT 用马尔可夫链生成样本序列 $\bar{x}_i$，其分布收敛到正比于目标函数 $f^*$（通常取亮度）的分布，路径积分估计为：

$$\hat{I}_j = \frac{b}{N} \sum_{i=1}^{N} \frac{h_j(\bar{x}_i) f(\bar{x}_i)}{f^*(\bar{x}_i)}$$

$b$ 是全图 $f^*$ 的平均值估计。整套去噪方法围绕三个部分展开：MLT 采样分解、循环 Transformer 块、分层核应用。整体网络是一个 U-Net 结构，编码/解码块由精心设计的循环 Transformer 块构成。

### MLT 采样分解（Sample Decomposition）

不同变异/扰动策略产生的噪声模式差异极大，把它们混在一起去噪会互相干扰。作者借鉴基于样本的 PT 去噪思路，把 MLT 辐射按变异策略拆开，分别去噪。具体采用 Mitsuba 中 path-space MLT 的四种默认策略：

- **双向变异（bidirectional mutation）**：从头重新生成子路径，接受率低。
- **透镜扰动（lens perturbation）**：主要处理直接可见或经镜面交互的直接光，噪声低、辐射大。
- **焦散扰动（caustic perturbation）**：处理焦散图案，噪声高。
- **多链扰动（multi-chain perturbation）**：处理复杂焦散，噪声高，易产生细长条状伪影。

每个分量单独预测去噪核，最后再合成。消融实验（图 4）显示：不分解时单一核无法消除多链扰动产生的细长伪影，而分解后多链分量的核学到了该噪声模式，与透镜分量的核协同重建出更好的结果。

### 循环 Transformer 块

这是方法的核心创新。传统交互式去噪用逐像素输入混合做时序累积——把对齐后的噪声帧按固定或自适应权重线性混合。这对 PT 有效，但对 MLT 的相关噪声不够用。作者改用基于注意力的循环块，包含两个注意力模块：

- **自注意力（self-attention）**：结构类似 Restormer，聚合空间与跨通道上下文，主要处理空间与低层噪声。其 query/key/value 都来自当前帧输入特征 $X_i^{(t)}$。
- **循环交叉注意力（cross-attention）**：query 来自当前帧输入，而 key/value 来自上一帧的隐藏状态 $H_i^{(t-1)}$（经主运动向量 warp 对齐）。它利用时序连续性先验累积正确特征、消除帧间不一致的相关噪声。

$$Q_c, K_c, V_c = E'_q(X_i^{(t)}),\; E'_k(\mathcal{W}_{t-1\to t}(H_i^{(t-1)})),\; E'_v(\mathcal{W}_{t-1\to t}(H_i^{(t-1)}))$$

注意力采用**通道注意力**而非经典空间注意力，这是效率关键：

$$\mathrm{Attention}(\hat{Q}, \hat{K}, \hat{V}) = \hat{V} \cdot \mathrm{Softmax}(\hat{K} \cdot \hat{Q} / \alpha)$$

通道注意力的计算复杂度为 $O(C^2 HW)$，对分辨率是线性的；而经典 ViT 注意力是 $O((HW)^2 C)$，对分辨率是二次的。这让模型能在大图上负担得起循环 Transformer 块。解码块沿用同样的块，但去掉交叉注意力，只保留自注意力（循环分支只在编码器中）。

### 分层核应用（Hierarchical Kernel Application）

解码块输出归一化核，保证不产生色偏并守恒能量。参照 Vogels 等人（2018）的多尺度管线，在每层 $l$ 上对下采样辐射滤波再合成。单层核应用同时包含空间核 $K$（从邻居像素聚集）和时序核 $\kappa$（从上一帧输出聚集），用混合参数 $\lambda$（经 sigmoid 约束到 $(0,1)$）加权：

$$\bar{L}^l_{xyt} = (1-\lambda_{xyt}) \sum_{uv} K^l_{uvxyt} L^l_{(x+u-w/2)(y+v-h/2)t} + \lambda_{xyt} \sum_{uv} \kappa^l_{uvxyt} \big(\mathcal{W}_{t-1\to t}(\bar{L}^l_{t-1})\big)_{(x+u-w/2)(y+v-h/2)}$$

各层粗去噪结果再通过渐进式尺度合成模块，从最低分辨率逐级升到全分辨率：

$$\hat{L}^l_{xyt} = \bar{L}^l_{xyt} - U(\alpha^l_{xyt}) D(\bar{L}^l_{xyt}) + U(\alpha^l_{xyt} \hat{L}^{l-1}_{xyt})$$

最后在全分辨率层把 4 个分解分量的去噪结果相加得到最终输出 $O_{xyt} = \sum_{k=1}^{4} \hat{L}^3_{xytk}$。

## 技术细节

- **数据集**：基于 OpenRooms 框架构造了 124 个含复杂可见性与焦散的场景、6300 段动画序列（相机在静态场景中运动，并加入移动聚光灯制造移动焦散），分辨率 640×480，用 Mitsuba v0.6 的 path-space MLT 渲染。114 个场景用于训练（7 帧序列，32–128 spp），10 个场景用于测试（每场景两段 60 帧序列，16/32/64/128 spp）。
- **损失函数**：单帧损失与时序损失的组合 $\mathcal{L} = \mathcal{L}_{single} + \lambda \mathcal{L}_{temporal}$（$\lambda=0.5$），二者均用 SMAPE 度量，时序项作用于相邻帧差分 $\partial_t I$。
- **运动向量与 warp**：用真值位姿和深度提取主运动向量（primary motion vectors），做前后向一致性检查以屏蔽不可见位置，用于循环块和分层核应用中不同分辨率的特征 warp（低分辨率特征用双线性重采样后再下采样）。
- **辅助缓冲**：albedo、distance、shading normal、world position 四种，均由 Mitsuba 直接生成并归一化到 $[-1,1]$；输入辐射先做 $x \mapsto \log(1+x)$ tone map 再归一化。
- **训练**：PyTorch + Adam，50 epoch 约 16 万次迭代，单张 A100-80GB 训练约 4 天；batch 2，初始学习率 $10^{-4}$，每 20 epoch 乘 0.8，256×256 裁剪并做旋转/翻转增强。模型约 20M 参数，推理约 0.18 s/帧、占用约 7.9 GB 显存。

## 实验结果

- **时序稳定性度量选择**：作者发现基于时序有限差分的 TRMAE、TPSNR 无法充分捕捉 MLT 特有的时序不稳定（源于强局部相关性），因此采用感知度量 FovVideoVDP 来评估动画时序质量，配合 PSNR、SSIM 评单帧质量。
- **对比基线**：与 5 个学习型方法比较——RAE、NTASD、KPCN、AFGSA、IDANF，全部在同一数据集上重训并适配到 MLT 动画去噪。在 20 段 60 帧测试序列上，本方法在所有采样率（16/32/64/128 spp）、所有三项指标上都稳居第一。例如 128 spp 时 PSNR 达 29.99（次优 IDANF 27.63），SSIM 0.882，FoVVDP 8.289，且在经典 Veach-Ajar、Monkey、Crystal 等场景上展现出良好泛化。
- **未见采样率泛化**：训练只用 32–128 spp，但从 1 spp 到 1024 spp 测试时本方法在全区间保持最佳，既能应付极噪输入也能处理高质量输入。
- **消融——时序累积**：对比简单循环（SR，去掉交叉注意力）、自适应混合（AB）、常数混合（CB）、无混合（NB）。循环分支显著优于所有输入混合方法，交叉注意力在低采样率下增益更明显。
- **消融——核设置**：核预测是必要的（直接重建输出质量最差）；分层去噪、时序核都有帮助；在相同核尺寸下，采样分解带来的提升与把核尺寸增大到同等输出条目规模相当，但一味增大核尺寸并不能持续改善，说明分解更高效。
- **消融——Transformer vs CNN**：把 Transformer 块换成 ConvNext 卷积块后，速度快约一倍（0.094 vs 0.180 s/帧），但质量全面下降。作者认为对复杂场景而言去噪时间相对渲染时间可忽略，值得用更慢的 Transformer 换质量。
- **推广到其他 MCMC**：额外训练了一个针对 PSSMLT 的模型（把渲染分解为大步/小步变异两个分量），同样去噪良好，证明采样分解 + 循环网络的思路可迁移到其他 MCMC 算法。

## 贡献与局限

**贡献**：

- 提出针对 MLT 动画的采样分解技术，按变异/扰动策略拆分辐射分别去噪，有效去除闪烁伪影。
- 提出结合循环连接与视觉 Transformer 的序列去噪器，用循环交叉注意力做深层时序累积，用线性复杂度的通道注意力兼顾感受野与大图效率，在保持时序稳定的同时高质量去噪 MLT。
- 构建了一个基于 OpenRooms、含大量高难场景的 MLT 去噪评测数据集。

**局限**：

- 推理时间不是瓶颈，但大规模 Transformer 模型的显存需求较高，去噪高分辨率动画在部分 GPU 上可能受限。
- 方法本身面向离线去噪（约 0.18 s/帧），优化目标是质量而非实时性。
- 依赖真值位姿与深度提取主运动向量；去噪器对特定 MLT 变体的噪声模式需要重训才能适配。

## 延伸思考

这项工作最有价值的观点，是明确区分了 MLT 与 PT 在时序累积上的本质差异：PT 的噪声较独立，逐像素线性混合就够；而 MLT 的相关噪声需要网络在多个层级上保留深层时序状态，因此循环结构重新变得必要——这与近年来"能混合就别循环"的主流实践形成有趣的反差。作者用通道注意力的线性复杂度巧妙化解了 Transformer 在高分辨率下的算力障碍，使"大网络 + 循环 + 注意力"这套组合在离线去噪场景变得可行。采样分解则再次印证了"分而治之"在渲染去噪中的普适性：把不同来源、不同统计特性的噪声解耦后交给各自的核，往往比堆大核更高效。沿着这条路，如何把该方法与去噪相关/非相关图像对的思路结合、如何降低显存以支持高分辨率、以及推广到更多 MCMC 变体，都是自然的后续方向。
