---
title: "High-quality Surface Reconstruction using Gaussian Surfels"
authors:
  - "Pinxuan Dai"
  - "Jiamin Xu"
  - "Wenxiang Xie"
  - "Xinguo Liu"
  - "Huamin Wang"
  - "Weiwei Xu"
category: "Reconstruction"
track: "Conference"
source: "arxiv"
institution:
  - "Zhejiang University"
  - "Hangzhou Dianzi University"
  - "Style3D Research"
tags:
  - "3D Gaussian Splatting"
  - "Gaussian Surfel"
  - "Surface Reconstruction"
  - "Depth-Normal Consistency"
  - "Poisson Reconstruction"
links:
  paper: "https://doi.org/10.1145/3641519.3657441"
---

## 一句话总结

把 3D 高斯点的 z 方向尺度直接压到 0，将椭球拍扁成 2D 椭圆（Gaussian surfel），再配合自监督的深度-法向一致性损失、单目法向先验与体素裁剪，在保持 3DGS 快速优化的同时重建出高质量、贴合表面的网格。

## 研究背景

3D Gaussian Splatting（3DGS）用一组显式、无拓扑约束的高斯点表示场景，支持优化中动态增删点、并借助 GPU 光栅化实现快速训练与实时渲染。但它在高质量几何重建上表现不佳，原因有三：

- 非零厚度：3D 高斯是椭球，沿各轴都有厚度，难以贴紧真实表面。
- 法向歧义：单个 3D 高斯的法向轴在优化中可能在不同尺度方向间跳变，导致细节几何重建不准。
- 尖锐边缘建模困难：alpha 混合会让越过或远离表面边缘的高斯点混入，给重建表面边缘带来偏差。

SuGaR、NeuSG 等方法通过正则化最小化某个尺度分量来缓解厚度问题，但重建质量仍不理想。本文提出 Gaussian surfels，结合 3DGS 灵活的优化流程与 surfel 的表面对齐特性，从表示层面根治法向歧义与厚度问题。

## 方法

### 整体框架

输入一组带位姿的 RGB 图像，输出一组 Gaussian surfels，并最终抽取全局网格。流程分三步：

```mermaid
flowchart LR
    A[带位姿 RGB 图像<br/>随机初始化] --> B[Gaussian Surfels 表示<br/>z-scale = 0 的 2D 椭圆]
    B --> C[优化<br/>光度损失 + 深度法向一致性<br/>+ 单目法向先验 + 不透明度/掩膜]
    C --> D[渲染多视角深度图与法向图]
    D --> E[体素裁剪<br/>去除错误深度点]
    E --> F[屏蔽泊松重建<br/>抽取高质量网格]
```

### 关键设计

Gaussian surfel 表示。每个核由位置、四元数旋转、两个局部轴的尺度、不透明度与球谐系数构成。3D 高斯分布为：

$$G(\mathbf{x}; \mathbf{x}_i, \Sigma_i) = \exp\left\{-0.5\,(\mathbf{x}-\mathbf{x}_i)^\top \Sigma_i^{-1}(\mathbf{x}-\mathbf{x}_i)\right\}$$

将尺度设为 $\mathbf{s}_i = [s_i^x, s_i^y, 0]^\top$ 把高斯拍扁，协方差变为：

$$\Sigma_i = R(\mathbf{r}_i)\,\mathrm{Diag}\!\left[(s_i^x)^2, (s_i^y)^2, 0\right] R(\mathbf{r}_i)^\top$$

此时每个核退化为 2D 椭圆，法向可直接取旋转矩阵第三列 $\mathbf{n}_i = R(\mathbf{r}_i)[:,2]$，为优化器提供了"把局部 z 轴当法向"的明确指引，避免了需要在优化中动态判定最小尺度轴。

精确深度渲染。直接用高斯中心深度做混合会忽略椭圆斜率，导致深度与法向不一致。作者通过求光线与高斯椭圆的交点、并用局部泰勒展开近似，得到像素级深度：

$$d_i(\mathbf{u}) = d_i(\mathbf{u}_i) + (W_k R_i)[2,:]\,J_{pr}^{-1}(\mathbf{u}-\mathbf{u}_i)$$

深度图与法向图同样通过 alpha 混合得到，并用 $1/(1-T_{n+1})$ 归一化混合权重。

深度-法向一致性损失。由于协方差对局部 z 轴的导数为零，光度损失无法直接约束法向，会出现梯度消失。作者设计自监督一致性损失，让渲染法向 $\tilde{N}$ 与由渲染深度反算的法向对齐：

$$\mathcal{L}_c = 1 - \tilde{N}\cdot N(V(\tilde{D}))$$

它能双向纠正：中心深度对但法向错时用深度纠正椭圆朝向；法向对但深度错时充当法向感知的深度平滑。

其余损失。总损失为 $\mathcal{L} = \mathcal{L}_p + \mathcal{L}_n + \lambda_o \mathcal{L}_o + \lambda_c \mathcal{L}_c + \lambda_m \mathcal{L}_m$。其中光度损失 $\mathcal{L}_p$ 组合 $L_1$ 与 D-SSIM；法向先验损失 $\mathcal{L}_n$ 用 Omnidata 单目法向做正则、缓解高光区域的形状-辐射歧义；不透明度损失 $\mathcal{L}_o$ 鼓励不透明度趋近 0 或 1；掩膜损失 $\mathcal{L}_m$ 用前景掩膜做二值交叉熵。

体素裁剪与网格化。alpha 混合会在深度不连续处产生错误深度（前景高斯的 alpha 污染背景深度）。作者在 $512^3$ 体素网格上累积各高斯的加权不透明度 $G(\mathbf{x};\mathbf{x}_i,\Sigma_i)\cdot o_i$，把累积值低于阈值 $\lambda = 1$ 的体素判为未占据并裁掉其中的 3D 点，再对融合后的深度与法向做深度为 10 的屏蔽泊松重建，得到最终网格。相比对高斯中心直接泊松重建，此法显著提升点密度与细节。

## 实验结果

在 DTU 与 BlendedMVS 上以 Chamfer 距离（越低越好）与训练时间评估。DTU 15 个物体的平均结果（单位 mm / 分钟）：

| 方法 | Ours | 3DGS | SuGaR | NeuS | NeuS2 | INSR |
|------|------|------|-------|------|-------|------|
| CD 均值 | 0.88 | 2.58 | 2.05 | 0.77 | 0.70 | 1.68 |
| 时间 | 6.67 | 5.19 | 30.9 | 408 | 3.27 | 8.48 |

BlendedMVS 18 个场景的 Chamfer 距离均值：Ours 2.27，3DGS 5.83，SuGaR 8.71，NeuS 2.67，NeuS2 2.63，INSR 2.82。

本方法显著优于同为点表示的 3DGS 与 SuGaR，训练速度与 NeuS2、INSR 相当且远快于 NeuS。相比 NeuS2 在 DTU 上 Chamfer 略高（作者归因于逐视角 alpha 混合深度的固有偏差），但重建噪声更少、细节多于 NeuS。渲染质量方面，在稀疏输入（50% 图像测试）下 PSNR 达 31.70，优于 3DGS（30.08）与 NeuS2（30.66）。消融显示去掉一致性损失 $\mathcal{L}_c$ 时 CD 由 0.882 恶化到 1.243，是最关键的组件。

## 亮点与局限

亮点：

- 从表示层面（z-scale = 0）根治 3DGS 的法向歧义与厚度问题，思路简洁且给优化器提供了清晰指引。
- 自监督深度-法向一致性损失巧妙解决了拍扁后局部 z 轴梯度消失的问题。
- 体素裁剪有效去除 alpha 混合在边缘处产生的错误深度点，且能自然重建开放表面。

局限：

- 即便有单目法向先验，在强镜面反射区域仍难以保证准确重建。
- 对纹理极弱的表面，重建结果可能相对真值出现整体平移。

## 延伸思考

作者提出的未来方向是在 Gaussian surfel 上存储并优化视相关的外观特征，配合神经解码器替代球谐来处理镜面反射；对弱纹理的整体平移问题，则可借助深度传感器或更多形状先验缓解。把"拍扁高斯 + 一致性约束"这一思路与后续 2D Gaussian Splatting 等工作对照，可以看到点基表面重建正从"事后正则化"走向"表示本身即贴合表面"的演化脉络。
