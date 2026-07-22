---
title: "ColorVideoVDP: A visual difference predictor for image, video and display distortions"
authors:
  - "Rafal K. Mantiuk"
  - "Param Hanji"
  - "Maliha Ashraf"
  - "Yuta Asano"
  - "Alexandre Chapiro"
category: "Image & Video"
track: "Journal"
source: "arxiv"
institution: "University of Cambridge"
tags:
  - "Image Quality"
  - "Video Quality"
  - "Visual Difference Predictor"
  - "Contrast Sensitivity"
  - "Contrast Masking"
  - "Color Vision"
  - "Display Modeling"
  - "XR Displays"
links:
  paper: "https://doi.org/10.1145/3658144"
  project: "https://www.cl.cam.ac.uk/research/rainbow/projects/colorvideovdp/"
  code: "https://github.com/gfxdisp/ColorVideoVDP"
---

## 一句话总结

ColorVideoVDP 是首个同时显式建模「时空视觉 + 彩色视觉 + 显示器物理特性」的全参考图像/视频质量度量，基于新的时空-彩色对比敏感度函数（castleCSF）与跨通道对比掩蔽模型，能把测试图/参考图之间的差异输出为可解释的 JOD 分数、失真热力图与 distogram，并在多个数据集（含作者新采集的 XR 显示伪影数据集 XR-DAVID）上显著超越现有度量。

## 研究背景

评估显示内容的视觉质量是计算机图形学与显示工程中的长期任务。让人眼主观打分最直接但昂贵、缓慢，且当需要在大参数空间里快速搜索最优配置时几乎不可行，因此自动化度量非常重要，常被当作优化的代价函数使用。

然而现有度量普遍忽略人眼视觉的关键侧面：

- **亮度类度量**（SSIM、FovVideoVDP 等）能建模模糊、对比度损失等消色差伪影，却无法处理色度子采样这类彩色失真。
- **色差公式**（CIEDE2000、ΔEITP 等）能算颜色差异，却不建模视觉的空间/时间特性，会忽略失真的空间分布或随时间的变化。
- **空间色差公式**（sCIELAB、FLIP）虽引入 CSF，但方式过于简化，不考虑对比敏感度随亮度变化，也不建模阈上视觉（对比掩蔽、对比恒常性）。

而现代显示技术（广色域、VR/AR/XR）恰恰带来大量时空彩色伪影：透镜像差引起的色边、光波导导致的色度不均匀等，这些低频、彩色、随时间变化的伪影正是现有度量的盲区。作者据此提出 ColorVideoVDP，目标是造一个有坚实心理物理基础、能外推到未见条件（不同帧率、分辨率、亮度）、同时输出单值质量与差异图、且高效可微的度量。

## 核心方法

ColorVideoVDP 沿用 FovVideoVDP 的整体处理流水线，最大的新贡献是把时空-彩色的近阈值与阈上对比感知模型整合进来，从而能统一量化消色差与彩色失真的可见性。测试图与参考图先经同一条流水线处理，再进入对比敏感度与掩蔽模型，最后池化并回归到 JOD 分数。

```mermaid
flowchart LR
    A["测试/参考<br/>图像或视频"] --> B["显示模型<br/>像素→CIE XYZ 光度量"]
    B --> C["对立色通道<br/>DKL: Ach / RG / YV"]
    C --> D["时间分解<br/>持续+瞬变通道"]
    D --> E["多尺度分解<br/>拉普拉斯金字塔"]
    E --> F["对比敏感度 + 跨通道掩蔽"]
    F --> G["跨波段/通道/帧池化"]
    G --> H["JOD 回归<br/>+ 热力图 / distogram"]
```

### 显示模型

显示模型承担两件事。其一，把像素坐标换算成视觉度（每度像素数 $n_\text{ppd}$）：

$$
n_\text{ppd} = \frac{\pi}{360}\,\arctan\!\left(\frac{0.5\,d_\text{width}}{r_w\,d_v}\right)^{-1}
$$

其二，把标准色彩空间的显示编码像素值转换为显示器实际发出的绝对线性光度量，考虑峰值亮度、色域、黑电平以及环境光反射：

$$
I_{\text{lin},c}(\boldsymbol{x}) = \min\!\left\{(L_\text{peak}-L_\text{black})\,E(I_{\text{de},c}(\boldsymbol{x})) + L_\text{black},\; L_\text{peak}\right\} + L_\text{refl}
$$

其中 $E(\cdot)$ 是相应编码的电光转换函数（SDR 用 sRGB，HDR 用 PQ），反射光 $L_\text{refl}=k_\text{refl}\,E_\text{amb}/\pi$。最终转成设备无关的 CIE XYZ。正是这一步让度量「知道」显示器的几何与光度特性。

### 对立色通道与时间/空间分解

XYZ 先转到 **DKL 对立色空间**（消色差 Ach、红-绿 RG、紫-黄 YV 三个基本方向），因为 castleCSF 就在该空间标定，且大量彩色对比检测数据也采集于此。

- **时间分解**：消色差通道拆成「持续（低通）」与「瞬变（带通）」两个时间通道，两个彩色通道各用一个低通时间通道（对高频彩色闪烁不敏感）。滤波器直接由 castleCSF 定义，250 ms 支撑长度即可刻画其特性。
- **空间分解**：对四个时间通道各做**拉普拉斯金字塔**分解得到多个空间频带。与 FovVideoVDP 不同，ColorVideoVDP **保留了低频与基带（base band）**，因为许多显示伪影（如波导不均匀）只在低频可见——这也让它能作为可传播低频差异的可微损失。

局部带限对比定义为拉普拉斯金字塔系数除以其上一层高斯金字塔（对应持续亮度通道的局部背景亮度）：

$$
C_{b,c,f}(\boldsymbol{x}) = \frac{\mathcal{L}_{b,c,f}(\boldsymbol{x})}{L_{\text{bkg},b,f}(\boldsymbol{x})}
$$

### 对比敏感度 castleCSF

度量的核心是自研的 **castleCSF**：能同时刻画色度调制与不同时间频率的对比敏感度，建模颜色、面积、空间/时间频率、亮度与偏心度。它把对比分解到 DKL 三个基本方向、消色差方向再分持续/瞬变，与度量的通道一一对应。castleCSF 在 19 个对比敏感度数据集（10 消色差、6 彩色、3 混合）上标定。为效率，实际以 2D 查找表（亮度 × 空间频率，每通道一张）预存。

### 跨通道对比掩蔽（关键组件）

掩蔽模型把测试图与参考图之间的物理对比差异转换成「感知差异」，同时刻画对比敏感度、掩蔽（纹理区差异更难被察觉）、跨通道掩蔽（一个通道的强对比会降低另一通道对比的可见性）以及阈上对比感知。先用 CSF 归一化对比：

$$
C'_{b,c,f}(\boldsymbol{x}) = C_{b,c,f}(\boldsymbol{x})\,S_{b,c,f}(\boldsymbol{x})
$$

作者对比了两种对比编码 × 三种掩蔽模型共六种方案，最终选定「乘性对比编码 + 互掩蔽（mutual masking）」。单波段视觉差异为：

$$
D_{b,c,f}(\boldsymbol{x}) = \frac{\left|C'^{\,\text{test}}_{b,c,f}(\boldsymbol{x}) - C'^{\,\text{ref}}_{b,c,f}(\boldsymbol{x})\right|^{p}}{1 + C^{\text{mask}}_{b,c,f}(\boldsymbol{x})}
$$

掩蔽信号先取测试/参考的互掩蔽 $C^{\text{mm}}=\min\{|C'^{\text{test}}|,|C'^{\text{ref}}|\}$，再做局部高斯空间池化，并按跨通道系数 $k_{i,c}$ 组合：

$$
C^{\text{mask}}_{b,c,f}(\boldsymbol{x}) = \sum_{i} k_{i,c}\left((C^{\text{mm}}_{b,i,f})^{q_c} * g_{\sigma_\text{sp}}\right)(\boldsymbol{x})
$$

训练得到的跨通道掩蔽权重与心理物理发现吻合：**彩色通道能强烈掩蔽消色差通道，而亮度不掩蔽颜色**（反而会易化颜色检测，这一易化效应互掩蔽模型无法建模）。此外用软钳制函数限制单波段最大对比，避免少数极大差异主导结果：

$$
\hat{D}_{B,c,f}(\boldsymbol{x}) = \frac{k_C\,D_{B,c,f}(\boldsymbol{x})}{k_C + D_{B,c,f}(\boldsymbol{x})}
$$

### 池化与 JOD 回归

差异跨空间、频带、通道、帧做多重 $p$-范数池化（空间与帧的指数取 2 表示能量求和，频带与通道取 4 表示「胜者通吃」式求和）：

$$
D_\text{pooled} = \left\| \frac{1}{F}\,\Big\| w_c \big\| \tfrac{1}{N_b}\|\hat{D}_{b,c,f}(\boldsymbol{x})\|_{\beta_x,\boldsymbol{x}} \big\|_{\beta_b,b} \Big\|_{\beta_c,c} \right\|_{\beta_f,f}
$$

再回归到可解释的 **JOD（Just-Objectionable-Difference）** 单位：

$$
Q_\text{JOD} = 10 - \alpha_\text{JOD}\,(D_\text{pooled})^{\beta_\text{JOD}}
$$

10 JOD 表示最高质量（测试图与参考图完全相同）；质量下降 1 JOD 意味着在成对比较中约 75% 的观察者会察觉到这一质量损失。图像质量则跳过时间分解，用可训练常数 $k_I$（可理解为注视时长）替代时间池化，从而统一图像与视频。

### 可视化与实现

除单值 JOD 外，度量提供两种可视化：叠加在灰度失真内容上的**逐像素热力图**，以及把失真按通道 × 空间频带 × 时间展开的 **distogram**（能看出波导不均匀集中在低频/基带、光源不均匀表现为高频闪烁等）。实现基于 PyTorch、GPU 并行、castleCSF 预存 LUT，速度与 SOTA 度量相当，且**完全可微**，可用于优化与参数标定。

## 技术细节：XR-DAVID 数据集与训练

**XR-DAVID 数据集**：为标定 XR 显示伪影，作者用 Eizo CG3146 专业参考显示器（4096×2160、300 cd/m²、sRGB、P3 色域、有效 77 ppd、下巴托固定距离）采集了新数据集。77 名通过石原色盲测试的被试做成对比较（判断哪段视频失真更小），用 ASAP 主动采样、pwcmp 缩放到 JOD 单位。数据集含 14 段参考视频 × 8 类伪影 × 3 个强度 = 336 个失真视频。八类伪影覆盖时空/彩色特性：时空抖动、光源不均匀（LSNU）、模糊/MTF 退化、对比度降低、波导不均匀（WGNU，低频且随时间在两种模式间切换）、动态校正误差（DCE，眼动追踪不准导致的时变色伪影）、色边（color fringes）、色度子采样。

**训练策略**：视频质量度量训练的难点是海量像素只对应单个质量值（60 帧 4K ≈ 5 亿像素），反向传播内存吃不消。作者用**特征式 + 端到端混合训练**：池化之后的参数（JOD 回归等）可先预计算池化特征再优化，快且能用大 batch；池化之前的参数用梯度检查点 + 随机采样 0.5 秒片段做端到端训练。两种训练交替进行（50 个特征式 epoch + 1 个端到端 epoch）。训练/测试用 XR-DAVID 与 UPIQ（含 SDR/HDR 图像），并按场景切成 7 份（5 训 2 测，场景不跨集）；LIVE HDR、LIVE VQA、KADID-10k 用作跨数据集验证。作者刻意**训练单一版本度量**并追求跨数据集泛化，而非为每个数据集单独调参。

## 实验结果

- **整体性能**：在测试数据集（LIVEVQA、LIVEHDR、KADID）与 UPIQ/XR-DAVID 测试部分上，ColorVideoVDP 平均 RMSE 约 **0.661 JOD**，显著优于第二名 VMAF（约 0.863），并领先 FSIMc、FovVideoVDP、VSI、MS-SSIM、IQT、HDR-VDP-3 等。色差公式及其空间扩展、无参考度量（NIQE、PIQE、BRISQUE）表现更差。虽然个别度量能在单一数据集上表现好（如 STRRED 在 LIVEVQA，因其正是用该集标定），但只有 ColorVideoVDP 在各种失真与内容上都稳定优秀——因为唯有它同时建模时空彩色视觉并考虑显示模型。
- **消融实验**：乘性对比编码 + 互掩蔽明显优于其它组合；**对比饱和（软钳制）对互掩蔽至关重要**；去掉 CSF 或掩蔽模型性能大幅下降。彩色通道对含色失真的数据集（XR-DAVID、UPIQ、KADID）很重要；跨通道掩蔽是微妙但有效的效应（主要利好 XR-DAVID）；时间通道的增益主要体现在含时间失真的视频数据集。用 LIVE-HDR 替代 XR-DAVID 训练会明显变差，说明 **XR-DAVID 提供的时空/彩色失真多样性对标定彩色视频度量不可或缺**。
- **合成测试**：14 组合成测试图验证边界行为。例如阈上对比匹配测试中，ColorVideoVDP 能正确预测跨颜色方向的阈上对比量级，而 HDR-FLIP 会高估红-绿、紫-黄方向的对比。

**应用**：(1) **色度子采样分析**——ColorVideoVDP 能正确预测高色度子采样率下的质量损失，而 SSIM（即便算在 RGB 上）会严重低估；还能揭示 YCbCr 空间平衡消色差/彩色失真的优势。(2) **可微优化 / 自适应色度子采样**——作为可微损失自适应地移除色度平面高频，一例中 PNG 体积减小 10% 而质量无损。(3) **显示色容差规范**——以 JOD、结合真实内容与图像结构，为显示器初级色偏（峰值、FWHM 偏移）设定可解释的容差规范；还可量化观察者同色异谱差异。

## 贡献与局限

**贡献**：
- 首个**显式同时建模人眼时空视觉与彩色视觉**、并考虑显示器几何/光度特性的全参考图像/视频质量度量。
- 整合新的时空-彩色对比敏感度函数 castleCSF 与**跨通道对比掩蔽**模型，正确处理阈上对比与低频差异，能刻画 XR 显示伪影。
- 完全可微、GPU 高效，速度与 SOTA 度量相当，可作优化损失。
- 采集并公开 **XR-DAVID** 数据集（336 个 XR 显示伪影失真视频），并展示色度子采样分析、显示色容差规范等新应用。

**局限**：
- 缺少高层的显著性/烦扰度模型，当语义内容强烈影响质量判断时精度下降。
- 未针对精确的空间失真图训练（视频领域缺此类数据），热力图仅用于解释。
- 不建模眩光（HDR-VDP 有）、注视相关视觉（FovVideoVDP 有）、眼动与双目视觉。

## 延伸思考

ColorVideoVDP 展示了「以心理物理模型为骨架」的度量相较纯手工特征或 CNN 特征度量的优势：不仅泛化性更好、能外推到未见的显示条件，配合 GPU 优化后速度也不吃亏，还天然可微、可解释。它把「颜色」这一在视频编码里常被轻视的维度重新拉回质量评估的中心，尤其契合 XR/广色域显示这类新兴场景。其显示模型 + 感知模型 + 单值 JOD + distogram 的组合，为显示工程（色容差、波导校正评估）与感知驱动优化（自适应色度子采样、可微感知损失）提供了统一工具。后续若能补上显著性/烦扰度、眩光、注视相关与双目视觉等高层与低层缺口，适用范围有望进一步扩大。
