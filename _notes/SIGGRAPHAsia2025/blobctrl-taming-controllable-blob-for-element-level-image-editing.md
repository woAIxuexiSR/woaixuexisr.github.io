---
title: "BlobCtrl: Taming Controllable Blob for Element-level Image Editing"
authors:
  - "Yaowei Li"
  - "Lingen Li"
  - "Zhaoyang Zhang"
  - "Xiaoyu Li"
  - "Guangzhi Wang"
  - "Hongxiang Li"
  - "Xiaodong Cun"
  - "Ying Shan"
  - "Yuexian Zou"
category: "Neural & Generative"
track: "Conference"
source: "arxiv"
institution:
  - "Peking University"
  - "The Chinese University of Hong Kong"
  - "ARC Lab, Tencent"
  - "The Hong Kong University of Science and Technology"
  - "Great Bay University"
tags:
  - "Image Editing"
  - "Element-level Editing"
  - "Blob Representation"
  - "Diffusion Model"
  - "Layout Appearance Disentanglement"
  - "Self-supervised Learning"
  - "Identity Preservation"
links:
  paper: "https://doi.org/10.1145/3757377.3763897"
  project: "https://liyaowei-stu.github.io/project/BlobCtrl/"
  code: "https://github.com/TencentARC/BlobCtrl"
---

## 一句话总结

本文提出 BlobCtrl：以概率化的"blob"（二维高斯分布 / 椭圆）作为视觉基元，把图像中某个元素的布局（位置、尺度、朝向）与外观（身份）显式解耦，配合双分支扩散架构与自监督"先解耦再重建"训练范式，在单一框架内实现元素级的添加、平移、缩放、替换、移除等细粒度可控编辑。

## 研究背景

元素级图像编辑要对已有图像中的单个视觉元素做布局与外观的细粒度修改。现有扩散生成模型合成质量高，却缺乏对单个元素的直接精细控制；传统可控生成方法（引入边缘图、边界框等空间条件，或参考图、ID 特征等身份条件）只能从头生成新图，无法修改已有图像的布局与外观，也不支持交互式、多轮、基于元素的重排操作。

近期基于优化、分割、聚类、拖拽的细粒度编辑方法有两大局限：一是编辑过程中未编辑区域会发生不期望的改变；二是依赖视频数据训练，导致编辑内容出现伪影（如移动元素后原位置的补全失败）。

作者的核心观察是：元素级视觉表示的本质在于布局与外观的灵活解耦。为此选用 blob 作为视觉基元——它既是一个 5 自由度 $$(x, y, a, b, \theta)$$ 的可操控原语，又天然带有不透明度（opacity），能精确控制位置、尺度与朝向。已有 blob 工作（BlobGAN、BlobGEN 等）主要用 blob 指定布局做从头生成，本文则进一步"驯服"blob，使其支持对已有图像的布局重排与外观替换。

## 方法

### Blob 表示与不透明度

一个 blob 有两种等价可互换的形式：几何上是椭圆 $$e_\tau = [C_x, C_y, a, b, \theta]$$，其中 $$(C_x, C_y)$$ 为中心，$$a$$、$$b$$ 为半短轴与半长轴长度，$$\theta \in [0, \pi)$$ 为朝向；统计上是二维高斯，均值 $$\mu = [\mu_x, \mu_y]$$、协方差 $$\Sigma$$ 含方向方差 $$\sigma_{xx}$$、$$\sigma_{yy}$$ 与协方差 $$\sigma_{xy}$$。

把 blob 建成高斯后，可在空间上定义不透明度。先计算网格点到中心的平方马氏距离

$$
d_M(x_{grid}, Q) = (x_{grid} - \mu)^T \Sigma^{-1} (x_{grid} - \mu)
$$

其中 $$x_{grid}$$ 是覆盖图像网格的坐标图，$$Q = (\mu, \Sigma)$$ 为高斯参数。不透明度定义为

$$
O(x_{grid}) = \mathrm{sigmoid}(-d_M)
$$

将距离映射到 $$(0, 1)$$，得到中心峰值、向边缘平滑衰减的不透明度。

Blob splatting 把第 $$i$$ 个特征向量 $$f_i$$ 投影到二维网格。按深度排序后，考虑 blob 间遮挡的合成不透明度为

$$
O_c^i = O_i \odot \prod_{j=i+1}^{m} (1 - O_j)
$$

逐 blob 的 splatting 为

$$
F_i = g_{splatting}(f_i, O_c^i) = O_c^i \otimes f_i
$$

其中 $$\odot$$ 为逐元素相乘，$$\otimes$$ 采用外积广播把不透明度图与特征向量相乘。

### 上下文双分支架构

框架把目标物体作为前景元素分割出来，并通过"双掩码"构建背景——同时移除前景在原位置与目标位置的内容。前景定义为可数的 "things"（如鸟、狗），背景为不可数的 "stuff"（如天空、草地），简化为每图一前景一背景。前景合成不透明度 $$O_c^1$$ 与背景合成不透明度 $$O_c^0$$ 分别编码各自布局。

前景分支负责身份编码，把 DINOv2 特征按前景不透明度 splatting 得到空间感知的前景语义图 $$F_1$$。其输入按通道拼接与沿宽度轴的上下文拼接构造：

$$
C_1 = \mathrm{cat}_0(z_1, O_c^1, F_1)
$$

$$
X_t^1 = \mathrm{cat}_w(C_1, \mathrm{cat}_0(z_1^t, O_c^1, F_1))
$$

其中 $$z_1$$ 为前景 VAE 隐编码，$$z_1^t$$ 为时间步 $$t$$ 的带噪前景隐编码。前景分支用去掉交叉注意力层的预训练扩散骨干处理，只专注视觉内容，并在多个分辨率层级抽取层次化特征。

背景分支负责场景保持与融合，作用于整幅图像隐编码 $$z_t$$：

$$
C_0 = \mathrm{cat}_0(z_0, O_c^0)
$$

$$
X_t^0 = \mathrm{cat}_w(C_0, \mathrm{cat}_0(z_t, O_c^0))
$$

背景分支用带交叉注意力的完整扩散骨干，通过零初始化把前景特征逐层级注入背景分支：

$$
\epsilon_\theta^{i, enhanced}(t, X_t^0, X_t^1) = \epsilon_\theta^{i, bg}(t, X_t^0) + \omega \cdot Z(\epsilon_\theta^{i, fg}(t, X_t^1))
$$

其中 $$\omega$$ 为控制融合强度的超参，$$Z$$ 为零初始化的线性层。

### 自监督"先解耦再重建"训练范式

获取真实编辑操作的元素级配对监督既难又贵，用视频代理又会引入混杂因素。作者把每张已有图像视为"编辑后"结果：将前景元素从背景中解耦，构造双掩码去除元素在假设的"编辑前"源位置与真实目标位置的内容；然后在源位置补全背景、在目标位置合成前景以保证场景和谐。具体地，图中前景 blob 作为目标（编辑后）状态，通过随机扰动其参数（中心/尺度/朝向）采样一个合成的编辑前 blob。训练用噪声预测目标：

$$
L = \mathbb{E}_{X_t^0, X_t^1, \epsilon \sim N(0, I)}\left[\|\epsilon - \epsilon_\theta^{enhanced}(t, X_t^0, X_t^1)\|_2^2\right]
$$

### 身份保持损失

为让前景分支专注元素身份、背景分支专注场景协调，对前景分支施加身份保持损失。训练时前景头在掩码区域预测噪声，推理时禁用该头：

$$
L_{id} = \mathbb{E}_{X_t^1, \epsilon \sim N(0, I)}\left[\|M_1 \odot (\epsilon - \epsilon_\theta^{fg}(t, X_t^1))\|_2^2\right]
$$

其中 $$M_1$$ 为二值前景掩码。总目标为

$$
L_{total} = L + \lambda_{id} L_{id}
$$

训练中把 $$\lambda_{id}$$ 从 1.0 衰减到 0.6，后期把重心移向场景协调同时保持身份一致。

### 定制训练策略

- 随机数据增强：对前景做颜色抖动、缩放、旋转、随机擦除、透视变换，避免模型陷入简单"复制粘贴"局部最优，并增强对残缺元素的补全鲁棒性。
- 随机丢弃：以概率 $$p_\omega$$ 令 $$\omega \leftarrow 0$$ 关闭前景-背景融合；以概率 $$p_{feat}$$、$$p_{vae}$$ 分别令 $$F_1 \leftarrow 0$$、$$z_1 \leftarrow 0$$。推理时这些超参可由用户设定，用于调节身份保持、在语义与外观之间权衡。

### 数据与实现

BlobData（186 万样本）基于带实例分割的 BrushData 构建：保留短边超 480 像素的图，掩码面积占比在 $$[0.01, 0.9]$$ 且不触边界，对每个掩码拟合椭圆并导出二维高斯，丢弃协方差病态（低于 1e-5）的样本，并用 InternVL-2.5 生成详细描述。BlobBench 含 100 张精选图，均匀覆盖五类操作（添加、平移、缩放、移除、替换），每图标注椭圆参数、前景掩码与专家撰写的详细描述，兼含真实与 AI 生成图像。

BlobCtrl 基于 Stable Diffusion v1.5，图像与标注统一到 512×512。两分支均以预训练 UNet 权重初始化；前景分支全量微调，背景分支用 LoRA（秩 64）微调。Adam 优化器，学习率 1e-5，权重衰减 0.01；在 24 张 V100 上以 batch 192 训练 7 天，丢弃概率均设 0.1，caption dropout 0.1 以支持无分类器引导。

## 实验结果

评测指标涵盖身份保持（用 CLIP-I、DINO-I 度量物体级特征相似度，移除任务用 CLIP-I*、DINO-I* 且越小表示移除越干净）、布局定位精度（用 SAM 提取生成图掩码拟合椭圆，与真值算 MSE）、生成质量（FID、PSNR、SSIM、LPIPS），另有 10 人参与的用户研究。

与通用方法对比（BlobBench）：

| 方法 | Add CLIP-I↑ | Add DINO-I↑ | Trans MSE↓ | Replace DINO-I↑ | Remove CLIP-I*↓ | Remove DINO-I*↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Anydoor | 86.7 | 81.2 | 6.8 | 80.2 | 39.5 | 13.6 |
| GliGEN | 70.7 | 57.8 | 7.1 | 60.6 | 40.2 | 15.3 |
| Magic Fixup | 83.7 | 84.5 | 6.8 | 81.2 | 37.2 | 9.7 |
| BlobCtrl（本文） | 88.3 | 86.9 | 6.3 | 86.0 | 35.3 | 8.6 |

在需要保持元素的任务（添加、平移、缩放、替换）上平均 CLIP-I 87.48、DINO-I 87.45，超越此前最佳 Magic Fixup（84.93、83.40）；移除任务的 CLIP-I*/DINO-I* 平均 21.95，优于基线 23.45；布局 MSE 平均 7.65 对 7.95，相对提升约 3.8%。

与仅平移的拖拽类方法对比（平移任务），BlobCtrl 达 CLIP-I 88.9、DINO-I 87.8、MSE 6.3，全面领先 InstantDrag、DragAnything、DiffEditor（后两者常因伪影导致物体定位失败而无法计算 MSE）。

生成质量上，通用编辑达 PSNR 32.16、SSIM 0.751、LPIPS 0.220、FID 102.8；平移任务达 PSNR 29.48、SSIM 0.975、FID 74.6，均为各组最优。

用户研究中，通用编辑的保真度、布局、和谐度偏好率分别为 79.5%、75.0%、78.5%；平移任务为 84.5%、82.5%、80.0%，大幅领先所有基线。

消融显示：融合权重 $$\omega$$、融合步比 $$t_\tau$$ 及前景输入 $$z_1$$、$$F_1$$ 可灵活权衡语义对齐与身份保持；去掉身份保持损失会使收敛更慢（全图去噪损失 0.0399 对 0.0235）且质量下降；把 blob 换成 4 自由度边界框会削弱布局灵活性，5 自由度 blob 能更好保持形状、生成更合理的编辑结果。

## 亮点与局限

亮点：
- 以 blob 作为几何/统计双重良定义、可互换的视觉基元，用 5 自由度加不透明度显式解耦布局与外观，实现精确元素级操控。
- 上下文双分支扩散架构分工明确：前景分支管身份、背景分支管场景协调与融合，通过零初始化逐层注入前景特征。
- 自监督"先解耦再重建"范式无需视频代理即可获得配对监督，配合身份保持损失、随机增强与随机丢弃，避免复制粘贴与视频训练伪影。
- 贡献了大规模训练集 BlobData 与统一评测基准 BlobBench，在多类元素级编辑任务上取得领先且保持计算高效。

局限：
- 在单次前向内只支持迭代式的单元素操作。
- 作者指出 blob 表示天然可扩展到深度感知的组合，是有前景的未来方向（暗示当前对多元素协同与深度关系的处理仍有提升空间）。

## 延伸思考

BlobCtrl 的关键在于选对了中间表示：blob 既足够简洁可被用户直接操控，又具备不透明度这种连续、可组合、可表达遮挡的属性，使"布局"与"外观"能在扩散框架内干净分离。相比拖拽点或边界框，5 自由度加不透明度的 blob 在形状保持与朝向控制上更具表达力；相比依赖视频监督的方法，自监督的解耦-重建范式绕开了配对数据难题。其把每张图当作"编辑后结果"再反向构造"编辑前"状态的思路，是一种巧妙的数据自举。作者提到的深度感知组合方向，若结合多 blob 排序不透明度所隐含的遮挡建模，有望把该框架从单元素迭代编辑推向真正的多元素、层次化场景重排。
