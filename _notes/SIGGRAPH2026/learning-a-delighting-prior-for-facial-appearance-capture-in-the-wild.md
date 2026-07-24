---
title: "Learning a Delighting Prior for Facial Appearance Capture in the Wild"
authors:
  - "Yuxuan Han"
  - "Xin Ming"
  - "Tianxiao Li"
  - "Zhuofan Shen"
  - "Qixuan Zhang"
  - "Lan Xu"
  - "Feng Xu"
category: "Reconstruction"
track: "Journal"
source: "arxiv"
institution:
  - "Tsinghua University"
  - "ShanghaiTech University"
tags:
  - "Facial Appearance Capture"
  - "Delighting"
  - "Inverse Rendering"
  - "Light Stage"
  - "Diffuse Albedo"
links:
  paper: "https://doi.org/10.1145/3811303"
  project: "https://yxuhan.github.io/OpenDelight/index.html"
---

## 一句话总结

本文提出 OpenDelight——一个在异构光照数据上训练的强力"去光照(delighting)先验",把从手机随手拍视频恢复高质量可重光照人脸反射率的难题,由不稳定的逆渲染转化为受先验约束的简单优化,并开源了模型与一个 4K 分辨率的 Light Stage 扫描数据集 NeRSemble-Scan。

## 研究背景

- 领域现状:高质量人脸外观捕获长期依赖昂贵的影棚(Light Stage)。近年出现用手机随手拍视频"在野(in-the-wild)"捕获的思路,但多为基于模型的逆渲染,受限于过度简化的光照模型。
- 核心痛点:在野场景光照未知且复杂,把干净的、可重光照的反射率(尤其漫反射 albedo)从复杂真实光照中解耦极其病态(ill-posed),优化不稳定、结果里常"烘焙"进阴影。并发工作 WildCap 用数据驱动的去光照 + 逆渲染的混合方案改善了质量,但需要人工圈出去光照产生伪影的区域,并依赖专门的区域式逆渲染,可扩展性差、纹理不连续。此外,业界最强的去光照模型(如 SwitchLight)是闭源的,开源数据集 FaceOLAT 又存在 OLAT 帧对齐不准导致的模糊、以及缺乏物理正确漫反射监督的问题。
- 本文 idea:把在野人脸外观捕获的重心转向"学一个足够强的去光照先验"。只要先验足够强,把光照解耦这一步做干净,后续逆渲染就能变简单、稳定,无需人工干预。为训练这样的先验,联合利用互补的两类数据(OLAT 数据集 + 渲染的 Light Stage 扫描),并用 Dataset Latent Modulation 解决异构数据混合训练的冲突。

## 方法

整体分三块:先训练去光照先验 OpenDelight(单图 → 漫反射 albedo);再把该先验嵌入基于模型的逆渲染,构成全自动的外观捕获流水线;最后用该流水线把多视角人脸数据集 NeRSemble 转成开源的 Light Stage 扫描数据集。

```mermaid
flowchart LR
  A["手机随手拍视频"] --> B["COLMAP 标定 + 2DGS/Wrap3D 建网格"]
  B --> C["逐视角跑 OpenDelight 预测漫反射 albedo"]
  C --> D["融合成 UV 纹理"]
  D --> E["patch 级扩散先验 + 全局 SH 光照优化"]
  E --> F["超分到 4K 反射率贴图 (albedo/specular/normal)"]
```

关键设计:

1. 互补的异构训练数据。作者把 OLAT 与渲染扫描视为互补:渲染的 Light Stage 扫描(自购 120 个 4K 扫描)提供锐利细节与物理正确的漫反射 albedo,但与真实图像有较大域差;FaceOLAT 的 OLAT 数据更接近真实图像分布,但因帧对齐不完美而模糊、且把均匀光照近似当 albedo 会烘焙高光。二者联合才能兼得锐利、物理正确与泛化。数据引擎用 PolyHaven 的 HDRI(按频率分层、提高高频光照采样比)分别渲染约 20K 图像-albedo 对,且只在皮肤区域学去光照。

2. Dataset Latent Modulation(DLM):核心贡献。朴素地把两个数据集混训会让网络在推理时不可预测地在两种数据分布间"跳变",导致多视角不一致。DLM 的思路是把知识拆成"共享机制(去光照的高层原理)"与"数据集专属风格(色彩分布、物理正确程度、锐利度)"。做法是为每个数据集学一组可训练的 source-aware token $$\{d^{*}_{i}\}_{i=1}^{K}$$($$*$$ 取 $$o$$ 或 $$r$$),把它们和图像 token 一起送入 ViT 编码器;由于这些 token 参数量相对核心网络可忽略,端到端训练会自然涌现解耦——token 吸收各数据集风格,核心网络提炼去光照原理。编码后丢弃 token,仅把图像 token 送入卷积解码器得到 $$\hat{A}^{*}$$,并用 L1 与 LPIPS 损失监督。推理时喂入渲染数据集的 token $$\{d^{r}_{i}\}_{i=1}^{K}$$,即可稳定导向锐利、物理正确且视角一致的预测。

3. 细节增强网络 $$F_{detail}$$。ViT 架构缺少 skip connection,基础网络 $$F_{base}$$ 的预测丢失细粒度细节。于是加一个 UNet,把原图 $$I$$ 与预测 $$\hat{A}$$ 拼接输入、恢复出细节增强的 $$\hat{A}_d$$;仅在渲染扫描数据集上、以随机退化(高斯噪声/模糊)构造的降质 albedo 为输入训练。

4. 先验驱动的逆渲染。逐视角用 OpenDelight 预测 albedo 并融合成 UV 纹理。由于该纹理几乎不含烘焙伪影,优化被大幅简化:无需 WildCap 的人工掩膜与区域式逆渲染,只需在 patch 级扩散先验下优化一个全局球谐(SH)光照,最后用超分网络把 1K 反射率贴图升到 4K。

## 实验结果

在 FaceOLAT 测试集与未见过的 3DRFE 扫描上评估漫反射 albedo 预测(评测前对每方法/每图做 6 参数通道级 scale+bias 颜色对齐以消除全局色偏歧义):

| 方法 | FaceOLAT PSNR↑ | FaceOLAT LPIPS↓ | 3DRFE PSNR↑ | 3DRFE LPIPS↓ |
|------|------|------|------|------|
| IC-Light | 30.78 | 0.1573 | 31.17 | 0.1282 |
| DreamLight | 27.72 | 0.1847 | 28.84 | 0.2055 |
| SwitchLight(闭源最强) | 32.98 | 0.1031 | 31.08 | 0.1032 |
| OpenDelight(本文) | 32.88 | 0.0952 | 34.27 | 0.0649 |
| OpenDelight*(全开源版) | 31.93 | 0.1072 | 35.02 | 0.0606 |

OpenDelight 在 LPIPS(感知质量)上全面领先,并在未见的 3DRFE 上 PSNR 大幅超过闭源的 SwitchLight;即便是仅用 NeRSemble-Scan + FaceOLAT 训练的全开源版 OpenDelight*,在 3DRFE 上也超过 SwitchLight。在 FFHQ 真实人脸上,SwitchLight 在硬阴影区域倾向烘焙光照,而本文得到更干净的 albedo。消融进一步表明:仅用 OLAT 会继承模糊与高光烘焙;仅用渲染扫描会因域差出现错误肤色/脏 albedo;Finetune 策略有灾难性遗忘;去掉 DLM 则出现严重的多视角不一致。相比扩散式的 GPSR 需 50 步采样,本文推理只需 ViT+UNet 两次前向。外观捕获方面,作为全自动方法,其质量与需人工干预的并发工作 WildCap 相当或更好。

## 亮点与局限

- 亮点:
  - 范式转变清晰:把在野外观捕获从"硬解病态逆渲染"转为"训练强去光照先验 + 简化优化",全自动、去掉了 WildCap 的人工掩膜与专门逆渲染。
  - DLM 用极少参数的 source-aware token 优雅解决异构数据混训的风格冲突与视角不一致,单阶段训练即可,思路可迁移到其它跨数据集训练场景。
  - 开源姿态实在:模型、数据引擎(HDRI + 渲染管线)、以及 4K 的 NeRSemble-Scan 数据集都将开放,降低高端人脸捕获门槛。
- 局限:
  - 身份信息损失:去光照训练身份有限,有时会把有色痣当作光照效果去除;外观捕获阶段因 WildCap 几何配准/补全不准,重渲染与照片不完全吻合。
  - 肤色偏差:训练数据不均衡,方法偏向较浅肤色,深肤色(如非裔)结果较弱。
  - 效率:从视频到可重光照 3D 资产约需 30 分钟;仍依赖 WildCap 的几何重建与 patch 级扩散先验,系统链路较长。

## 延伸思考

- 本质是"用数据驱动先验为病态优化提供强正则"的又一次成功范例,与近年逆渲染 + 生成先验(扩散先验、patch 级先验)的趋势一致;DLM 的解耦思想与并发的 avatar 学习工作(用可学习 latent 统一异构数据)殊途同归,但目标是解耦风格而非共享色彩空间或 3D 感知。
- 作者点出的前馈化方向值得关注:把"采样帧 → 可重光照 3D 人脸资产"做成前馈网络,有望把 30 分钟压缩到近实时。
- 肤色与身份偏差本质是数据分布问题,配合自监督重光照损失在大规模单视角人脸(如 FFHQ)上训练,或许能同时缓解身份丢失与肤色偏差,这也是提升公平性的实际抓手。
