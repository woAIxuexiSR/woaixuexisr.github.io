---
title: "BakedSDF: Meshing Neural SDFs for Real-Time View Synthesis"
authors:
  - "Lior Yariv"
  - "Peter Hedman"
  - "Christian Reiser"
  - "Dor Verbin"
  - "Pratul P. Srinivasan"
  - "Richard Szeliski"
  - "Jonathan T. Barron"
  - "Ben Mildenhall"
category: "Reconstruction"
track: "Conference"
source: "arxiv"
institution:
  - "Google Research"
  - "Weizmann Institute of Science"
tags:
  - "Neural Radiance Fields"
  - "Signed Distance Function"
  - "Surface Reconstruction"
  - "Novel View Synthesis"
  - "Real-Time Rendering"
links:
  paper: "https://doi.org/10.1145/3588432.3591536"
  project: "https://bakedsdf.github.io/"
---

## 一句话总结

BakedSDF 把无界真实场景重建成一张高质量三角网格，并为每个顶点配上基于球面高斯的视角相关外观模型，从而在普通消费级硬件的浏览器里实现实时、照片级的新视角合成。

## 研究背景

- 领域现状：新视角合成的最强方法大多基于 NeRF——用 MLP 表示连续体积场，渲染质量很高，但每像素要查询 MLP 上百次，速度慢。后续用体素网格、哈希编码等加速的方案仍依赖体积光线步进和大量显存，无法在普通硬件上实时跑，因为现代图形硬件是为多边形表面而非体积场设计的。
- 核心痛点：既有的可实时网格方法要么只适用于简单单物体，要么像 MobileNeRF 那样把几何初始化为轴对齐小片、优化后退化成"多边形汤"——外观还行，但几何杂乱，难以用于纹理编辑、重光照、物理仿真等常规图形应用。
- 本文 idea：把 VolSDF 那种"体积-表面混合表示"扩展到无界场景，让密度由一个行为良好的 SDF 参数化，其零等值面就是真实表面；再用 Marching Cubes 提取高分辨率网格，并烘焙成带球面高斯外观的可栅格化模型。

## 方法

整体上分三个阶段：先用 NeRF 式体积渲染优化一个基于 SDF 的表面-体积表示，再把几何"烘焙"成高分辨率三角网格，最后为网格顶点训练一套球面高斯外观模型来替换昂贵的 MLP 外观。产物是一张网格加逐顶点参数，渲染时只需栅格化并查询少量球面高斯。

```mermaid
flowchart LR
  A["输入多视角图像"] --> B["SDF 体积-表面表示<br/>收缩空间中优化"]
  B --> C["Marching Cubes<br/>烘焙高分辨率网格"]
  C --> D["逐顶点球面高斯<br/>外观模型优化"]
  D --> E["glTF 网格<br/>浏览器实时渲染"]
```

关键设计：

1. **收缩空间中的 SDF 密度**：沿用 VolSDF，把体积密度写成 SDF 的函数 $$\tau(\boldsymbol{x}) = \alpha \Psi_\beta(f(\boldsymbol{x}))$$，其中 $$\Psi_\beta$$ 是零均值拉普拉斯分布的 CDF，$$\beta \to 0$$ 时密度退化为物体内为 $$\alpha$$、空气中为 0。关键创新是把 $$f$$ 定义在 mip-NeRF 360 的收缩坐标空间而非世界空间：这样更强地正则化远景内容，也让网格在收缩空间中提取时三角形预算分布更合理（中心密、边缘疏）。训练用 Eikonal 项 $$\mathcal{L}_{\text{SDF}} = \mathbb{E}_{\boldsymbol{x}}\left[(\lVert \nabla f(\boldsymbol{x}) \rVert - 1)^2\right]$$ 约束 $$f$$ 逼近真实符号距离场。SDF 天然有良定义的法线 $$\boldsymbol{n} = \nabla f / \lVert \nabla f \rVert$$，方便引入 Ref-NeRF 的漫反射/镜面分离外观。$$\beta$$ 采用退火调度而非自由优化，训练更稳定。

2. **网格烘焙与剔除**：在 $$2048^3$$ 规则网格上查询 SDF，取 0.001 作为等值面（补偿 VolSDF 密度扩散导致的几何侵蚀）后跑 Marching Cubes。由于遮挡区和"自由空间"里 SDF 未被监督，会产生虚假表面：作者沿训练光线计算每个采样点的体渲染权重，把权重大于 0.005 的样本 splat 进网格并只在这些格子上提取表面。收缩空间的均匀采样映射回世界空间后自然形成前景小三角、远景大三角的 LOD 效果，投影到像面上各三角面积近似相等。再用区域生长（32 次迭代，在顶点邻域 $$8^3$$ 体素内重跑 Marching Cubes）补洞。

3. **球面高斯视角相关外观**：每个顶点配一个漫反射色 $$\boldsymbol{c}_d$$ 和若干球面高斯瓣，中心区 $$\lVert \boldsymbol{x} \rVert \le 1$$ 用 3 个瓣、边缘用 1 个瓣。像素颜色为 $$\boldsymbol{C} = \boldsymbol{c}_d + \sum_{i=1}^{N} \boldsymbol{c}_i \exp(\lambda_i(\boldsymbol{\mu}_i \cdot \boldsymbol{d} - 1))$$，每瓣仅 7 个参数（均值方向、颜色、宽度）。为避免直接优化逐顶点参数耗尽显存，改用 Instant NGP 式压缩哈希网格建模，训练完再烘焙回逐顶点参数；配合鲁棒损失和直通估计器（8 bit 量化），最终以 glTF + gzip 导出。

## 实验结果

在 mip-NeRF 360 的真实无界场景数据集上评测。离线版（第一阶段体渲染）质量不及部分顶尖离线方法，但实时版在三项指标上全面超过两个最强实时基线：

| 方法 | 室外 PSNR↑ | 室外 SSIM↑ | 室外 LPIPS↓ | 室内 PSNR↑ | 室内 SSIM↑ | 室内 LPIPS↓ |
|------|-----------|-----------|------------|-----------|-----------|------------|
| Deep Blending（实时） | 21.54 | 0.524 | 0.364 | 26.40 | 0.844 | 0.261 |
| MobileNeRF（实时） | 21.95 | 0.470 | 0.470 | − | − | − |
| 本文（实时） | 22.47 | 0.585 | 0.349 | 27.06 | 0.836 | 0.258 |
| 本文（离线） | 23.40 | 0.619 | 0.379 | 30.21 | 0.888 | 0.243 |

性能上，在 16 寸 MacBook Pro（Radeon 5500M）浏览器内以 1920×1080 渲染达 72 FPS、85 W，FPS/W 比 MobileNeRF 高 1.44 倍、比 Instant-NGP（RTX 3090）高约 77 倍，代价是磁盘存储略大（434.5 MB）。网格质量对比 COLMAP、MobileNeRF、mip-NeRF 360 等距面均更平滑、更保真。外观消融显示：球面高斯瓣从 3 减到 0 质量单调下降，但边缘只用 1 个瓣既省显存带宽又避免过拟合；用小 MLP 替代球面高斯反而质量更差且计算量高出约 100 倍（约 2070 FLOPS/像素 vs 21 FLOPS/像素）。

## 亮点与局限

- 亮点：
  - 首次从 NeRF 式神经体积表示中稳定提取出无界真实场景的高质量网格，几何平滑、保真，可直接用于外观编辑、重光照、物理仿真。
  - 用极轻量的球面高斯建模视角相关外观，比小 MLP 又快又好，配合标准栅格化管线在浏览器/普通硬件实现实时渲染，能效大幅领先。
  - 巧用收缩空间同时作为正则化手段和天然的 LOD 策略，让三角形预算按投影面积均衡分布。

- 局限：
  - 离线渲染质量仍不及 mip-NeRF 360 等顶尖离线方法，实时化以一定保真度为代价。
  - 存储开销高于 MobileNeRF、Instant-NGP，网格资源体积偏大。
  - 依赖遮挡/自由空间剔除与区域生长等启发式步骤来抑制虚假表面和补洞，在训练视角覆盖不足处仍可能受影响。

## 延伸思考

BakedSDF 把"神经隐式重建"与"传统图形管线"之间的鸿沟接上了：它证明只要几何足够干净、外观模型足够轻，NeRF 的成果完全可以落进人人可用的实时渲染栈。与其并行的 MERF 走的是压缩体积表示、质量更高但需要复杂渲染器且不便下游应用，二者形成"网格 vs 体积"的清晰权衡。后续 3D Gaussian Splatting 的爆发从另一个方向解决了同一个实时性问题，回看 BakedSDF 会发现"可编辑、可仿真的显式几何"这一优势至今仍有独特价值，值得关注它在重光照、材质分解等物理属性恢复方向上的延展。
