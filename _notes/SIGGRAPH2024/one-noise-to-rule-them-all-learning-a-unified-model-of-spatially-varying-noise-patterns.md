---
title: "One Noise to Rule Them All: Learning a Unified Model of Spatially-Varying Noise Patterns"
authors:
  - "Arman Maesumi"
  - "Dylan Hu"
  - "Krishi Saripalli"
  - "Vladimir G. Kim"
  - "Matthew Fisher"
  - "Sören Pirk"
  - "Daniel Ritchie"
category: "Rendering"
track: "Journal"
source: "arxiv"
institution:
  - "Brown University"
  - "Adobe"
  - "Christian-Albrechts-Universität zu Kiel"
tags:
  - "Procedural Noise"
  - "Texture Synthesis"
  - "Diffusion Model"
  - "SPADE"
  - "CutMix"
  - "Spatially-Varying Noise"
  - "Inverse Procedural Material Design"
links:
  paper: "https://doi.org/10.1145/3658195"
  project: "https://armanmaesumi.github.io/onenoise/"
  code: "https://github.com/ArmanMaesumi/OneNoise"
---

## 一句话总结

该工作用一个统一的去噪扩散模型（DDPM）同时建模 18 种程序化噪声，通过可解释参数 + 随机种子控制生成；并借助 SPADE 空间条件模块与改进版 CutMix 数据增强，让模型在从未见过空间变化训练数据的情况下，也能生成在画布上平滑过渡的空间变化噪声（spatially-varying noise），并可用于逆向程序化材质设计。

## 问题背景

程序化噪声是计算机图形学管线的基础组件，广泛用于反照率贴图、凹凸/法线贴图、地形高度场、云与烟等体积密度场。但现有噪声（Perlin、Worley、Gabor 等）各自由独立算法生成，构成一个「噪声动物园」，设计过程中存在几个根本限制：

- 必须离散地选择噪声类型。理想噪声可能不在库中，或者期望的视觉特征落在两种噪声之间。
- 传统的类型混合手段（alpha 混合）效果差：特征会重叠、不透明度不一致、缺乏合理的过渡（见论文 Figure 2）。
- 逆向设计（从照片恢复材质节点图）时，若节点噪声类型未知，搜索空间随节点组合爆炸，且离散选择妨碍基于梯度的优化。

作者的目标是：从「只含全局均匀属性」的噪声样本中，学习一个连续的空间变化噪声空间，无需任何空间变化的训练观测。

## 核心方法

整体是一个带空间条件机制的 DDPM。给定各种参数化噪声函数采样得到的噪声纹理及其参数配置，学习一个条件生成模型，用单一「万能函数」合成多种噪声，并支持类型之间、参数之间的空间混合。

关键在于两点：如何注入空间可变的条件信号（SPADE），以及如何让局部条件真正只影响局部区域（改进版 CutMix）。

```mermaid
flowchart TD
    A[噪声参数集 + 类别标签] --> B[MLP 投影到共享特征空间 f]
    B --> C[平铺成特征网格 F: |f|×H×W]
    C --> D[SPADE 模块调制 U-Net 的 GroupNorm]
    E[初始高斯噪声 / 随机种子] --> F[U-Net 噪声预测器]
    D --> F
    G[CutMix 数据增强: 拼接多种噪声patch] --> F
    F --> H[DDIM 采样 30 步]
    H --> I[空间变化噪声纹理 / 可平铺 / 任意尺寸]
```

### 空间变化条件（SPADE）

训练时条件信号由两个向量给出：类别嵌入 $$\mathbf{f}_c$$（编码噪声类型的类别标签）和参数向量 $$\mathbf{f}_p$$（scale、distortion、warp、waves 等语义属性）。显式提供参数带来可解释的用户控制，同时把噪声「风格」（由参数决定）与「种子」（由初始高斯噪声决定的随机分量）解耦。

将类别与参数向量经一个小 MLP 映射到共享特征空间得到 $$\mathbf{f}$$，再平铺成形状 $$|\mathbf{f}|\times H\times W$$ 的特征网格 $$\mathbf{F}$$，输入 SPADE（spatially-adaptive denormalization）模块，用它调制网络 Group Normalization 层的中间特征：

$$S(\mathbf{h}, \mathbf{F}) = \gamma(\mathbf{F}) \odot \mathrm{GroupNorm}(\mathbf{h}) + \beta(\mathbf{F})$$

其中 $$\gamma,\beta$$ 是把特征网格转成逐元素缩放/平移的卷积层。训练数据里 $$\mathbf{F}$$ 的所有条目相同（全局均匀），但推理时可人为构造特征网格——例如用双线性插值在多个特征向量之间做空间混合，从而生成在类型与参数上平滑过渡的空间变化噪声。

作者还提出一个发表后发现的增强：球面类别嵌入正则化。对维度为 $$d$$、从 $$\mathbf{f}_c\sim\mathcal{N}_d(0, I_d)$$ 初始化的嵌入，目标范数取 $$d$$ 维高斯向量期望平方范数：

$$T_d^n \coloneqq \mathbb{E}_{\mathbf{f}_c\sim\mathcal{N}_d(0,I_d)}\big[\lVert\mathbf{f}_c\rVert_2^n\big] = 2^{n/2}\,\frac{\Gamma((d+n)/2)}{\Gamma(d/2)}$$

训练时惩罚嵌入范数偏离目标：

$$\mathcal{L}_{reg} = \frac{1}{|C|}\sum_{c\in C}\big(\lVert\mathbf{f}_c\rVert_2^2 - T_d^2\big)^2$$

这为嵌入施加球面结构，使类别之间可用球面线性插值（slerp）过渡，显著改善类别间的纹理混合。

### 增强局部条件：改进版 CutMix

理想情况下，局部修改条件网格 $$\mathbf{F}$$ 只应改变对应区域的噪声。但实测发现局部调整常导致全局变化——原因是 U-Net 的瓶颈形状叠加大量卷积层，使每个输出像素的感受野很宽，条件信号会向整个画布传播。

为此作者把 CutMix 数据增强引入训练：把不同数据样本的轴对齐裁剪块拼接成合成训练样本，同时拼接对应的条件特征网格。以 0.5 的概率应用 CutMix；增强时随机采样 1~4 个辅助噪声纹理，随机裁剪成矩形块并施加随机旋转 $$\theta\sim U(0,2\pi)$$（只旋转裁剪掩码，不旋转纹理本身）。所有采样块必须属于不同噪声类型，否则合成图会脱离目标分布。这既丰富了训练多样性，又赋予模型对局部条件信号正确响应的能力。

## 技术细节

- 数据集：从 Adobe Substance 3D Designer 采样约 120 万张噪声图，覆盖 18 种噪声函数（cells 1/4、voronoi、microscope view、grunge galvanic small、liquid、bnw spots 1、grunge leaky paint、grunge rust fine、grunge map 002/005、grunge damas、messy fibers 3、perlin、gaussian、clouds 1/2/3）。每种噪声用低差异 Halton 序列确定性采样 16,384 组参数，每组 4 个不同种子，得每类 65,536 张、共 1,179,648 张。采样分辨率 512×512，训练时降采样到 256×256。
- 网络：U-Net 编解码各三个空间层级，每级（含瓶颈）两个 ResNet 块，同时被扩散时间步 $$t$$ 和 SPADE 模块条件化，二者组合为 $$\gamma_2(Z)\odot(\gamma_1(t)\cdot\mathrm{GroupNorm}(\mathbf{h})+\beta_1(t))+\beta_2(Z)$$；时间步用正弦位置编码。参数 MLP 三层各 128；参数向量是数据集全部噪声参数的拼接，不适用项置零。U-Net 约 510 万参数。
- 训练目标（噪声预测器形式）：

$$\mathcal{L} = \mathbb{E}_{\boldsymbol{\epsilon}\sim\mathcal{N}(0,1),\,t\sim U(0,1)}\,\lVert\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{f}_c, \mathbf{f}_p)\rVert^2 + \lambda\mathcal{L}_{reg}$$

  使用 offset noise（把 $$\boldsymbol{\epsilon}\sim\mathcal{N}(0,1)$$ 改为 $$\mathcal{N}(0.1\delta, 1),\ \delta\sim\mathcal{N}(0,1)$$）以帮助网络解析极暗/极亮的噪声；1000 个时间步，cosine-beta 调度，$$\lambda=0.02$$。AdamW（lr $$8\times10^{-5}$$，$$\beta_1=0.9$$，$$\beta_2=0.99$$，weight decay 0.01），8 块 RTX 3090 训练约 30 万步、batch 128。
- 推理：单块 RTX 3090 在 256×256 下每秒约 80 个扩散步（随分辨率二次增长）；用 DDIM 采样，默认 30 步。把所有 conv2d 改为循环 padding（画布拓扑为环面）即可产生可平铺噪声，并支持任意尺寸合成（如 2048×2048 单次扩散生成的大马士革钢纹理）。

## 实验结果

- 定性对比：与 GAN 纹理合成器 PSGAN、非参数纹理混合方法 Image Melding 比较，后两者出现伪影与重复视觉细节，且 PSGAN 会产生数据分布中不存在的各向异性条纹；本方法能平滑混合并在整个画布合成新细节（Figure 6）。
- 定量（FID，越低越好）：对每种噪声各采样 20,000 张与 Substance 3D Designer 真值比较。本方法平均 FID 20.9、中位数 13.1；PSGAN 为 99.2 / 87.5。PSGAN 在很多噪声类型上发生模式崩塌，因数据存在大量互不相交的模式，GAN 难以刻画。
- 逆向材质图设计：接入可微材质图库 MATch，从 88 个模板图中按纹理描述子选图，用本模型替换一个噪声生成节点，暴露 $$\mathbf{f}_c$$、$$\mathbf{f}_p$$ 与扩散潜噪声 z 为可优化量（对前两者加 L1 稀疏正则）。端到端优化能恢复目标照片中的非平凡图案，在 MATch 纹理相似度与 LPIPS 两个指标上优于 MATch 基线；优化后还能方便地重着色、重采样噪声、修改条件参数进行编辑。
- 可平铺 + 尺寸无关：可生成无缝平铺噪声，也能在更大画布上合成以避免平铺常见的重复内容。
- CutMix 消融：去掉增强后模型无法对条件信号做局部响应，噪声特征「模糊」不明显（如某类别混合中 galvanic 噪声直接消失）；CutMix $$n=1$$ 与 $$n\in[1,4]$$ 结果相近，后者在部分情形下过渡更平滑。

## 贡献与局限

贡献：
1. 一个能学习生成空间变化噪声的统一生成模型，把 18 种程序化噪声纳入同一连续空间，并保留可解释参数与随机种子控制。
2. 基于改进版 CutMix 增强的训练方案，使模型在没有任何空间变化训练数据的前提下学会生成空间变化噪声。
3. 将模型用于逆向程序化材质设计，无需预先指定离散噪声类型即可获得更高保真的材质重建。

局限：
- 并非所有噪声对都能良好插值：几何特征差异过大的两种噪声，其过渡区可能模糊或视觉别扭（Figure 14）。可尝试非矩形 patch、自适应采样聚焦「困难过渡」等改进，但一定程度上不可避免，选好可插值的噪声对本身是艺术决策。
- 扩散模型对数据分布中低密度模式建模较差，某些噪声函数的低密度模式捕捉不佳，可能需要更精细的数据采样。
- Substance Designer 中还有确定性图案生成器，其确定性本质不适合去噪扩散生成，纳入统一空间需修改模型/训练流程。
- 逆向材质设计目前仅是概念验证；若能蒸馏为一次性扩散模型，则可替换材质图中每个噪声节点，进而通过连续优化求解图结构。
