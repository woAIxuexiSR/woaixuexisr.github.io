---
title: "3Doodle: Compact Abstraction of Objects with 3D Strokes"
authors:
  - "Changwoon Choi"
  - "Jaeah Lee"
  - "Jaesik Park"
  - "Young Min Kim"
category: "Rendering"
track: "Journal"
source: "arxiv"
institution: "Seoul National University"
tags:
  - "3D Sketch"
  - "Non-Photorealistic Rendering"
  - "Differentiable Rendering"
  - "Bezier Curves"
  - "Superquadrics"
  - "Shape Abstraction"
  - "Multi-View Optimization"
links:
  paper: "https://doi.org/10.1145/3658156"
  project: "https://github.com/changwoonchoi/3Doodle"
  code: "https://github.com/changwoonchoi/3Doodle"
---


## 问题背景

手绘草图（free-hand sketch）是一种高效的视觉交流工具：只需寥寥数笔就能概括一个物体最具辨识度的结构与语义特征，因此在图形编辑、VR/AR 交互等场景中被广泛使用。然而传统草图有两个根本缺陷：

- **主观性强**：手绘线条往往偏离物体真实几何，受限于绘制技巧或艺术家的创作表达。
- **视角不一致**：从不同视角单独提取的 2D 草图线条彼此之间无法保持 3D 一致性，难以传达真实的三维形状。

已有工作要么依赖完整的 3D 模型（高分辨率网格、NeRF 体）来提取线稿，要么用数据驱动的方式从图像生成草图。但前者需要精确网格且渲染出的线条过于稠密、不够抽象；后者需要大量高质量的"图像—草图"配对数据集，且通常只能覆盖特定类别的物体，泛化性差。

本文提出 **3Doodle**：仅给定物体的多视角图像，就能生成既有描述性又保持视角一致的抽象草图。核心思想是——**一组紧凑的 3D 几何图元可以高效编码 3D 结构信息，并渲染出跨视角一致的 2D 草图线条**。

## 核心方法

3Doodle 把 2D 草图分解为两类互补的成分，并用不同的 3D 图元来表达：

$$S^{3D} = S^{3D}_{ind} \cup S^{3D}_{dep}$$

- **视角无关成分（view-independent）** $S^{3D}_{ind}$：来自纹理或尖锐边缘等特征线，这些线在 3D 空间中位置固定，用 **3D 三次贝塞尔曲线（cubic Bézier curves）** 表示。
- **视角相关成分（view-dependent）** $S^{3D}_{dep}$：来自光滑表面边界的轮廓线（contour），其在物体表面的确切位置随视角变化，用 **超二次曲面（superquadrics）的轮廓** 表示。

整个流程无需数据集训练、无需把输入图像风格化成草图、也无需重建显式的 mesh 或 NeRF 体。它构建了一条**完全可微的渲染管线**，直接对图元参数进行优化，使渲染结果在感知损失（CLIP、LPIPS）下逼近手绘草图。最终得到的 3D 笔画表示极为紧凑（小于 1.5 kB）。

## 技术细节

### 图元表示

**视角无关（3D 贝塞尔曲线）**：每条曲线由 4 个有序控制点 $p^j_i \in \mathbb{R}^3$ 参数化：

$$B^{3D}(t; p_i) = \sum_{j=0}^{3} b_j(t)\, p^j_i, \qquad b_j(t) = \binom{3}{j} t^j (1-t)^{3-j}, \quad t \in [0,1]$$

优化的就是这 $4 N_{ind}$ 个控制点的位置。

**视角相关（超二次曲面）**：每个超二次曲面在其规范坐标系下的隐式函数为

$$f(\mathbf{x}; \alpha_i, \epsilon_i) = \left( \left(\frac{x}{\alpha_{i,1}}\right)^{\frac{2}{\epsilon_{i,2}}} + \left(\frac{y}{\alpha_{i,2}}\right)^{\frac{2}{\epsilon_{i,2}}} \right)^{\frac{\epsilon_{i,2}}{\epsilon_{i,1}}} + \left(\frac{z}{\alpha_{i,3}}\right)^{\frac{2}{\epsilon_{i,1}}}$$

其中 $\epsilon_i$ 决定形状、$\alpha_i$ 决定各轴尺度；$f=1$ 时点在表面上，$f>1$ 在外部，$f<1$ 在内部。再加上刚体变换 $\{R_i, t_i\}$ 把规范坐标转到世界坐标。多个超二次曲面的并集用隐式函数取最小值实现：

$$S(\mathbf{x}; \theta) = \bigcup_{i=1}^{N_{dep}} S(\mathbf{x}; \theta_i) = \min_i S(\mathbf{x}; \theta_i)$$

### 可微渲染

**视角无关线条的渲染**：论文证明了一个关键定理——在**正交投影**假设下（相机足够远、透视畸变可忽略），3D 贝塞尔曲线的投影恰好等于由投影后控制点定义的 2D 三次贝塞尔曲线。于是渲染只需：1）把 4 个控制点投影到图像平面；2）用现成的可微光栅化器（diffvg）渲染 2D 贝塞尔曲线。整条链路都可微。

**视角相关轮廓的渲染**：借助体渲染的平滑密度场来传播图像空间梯度。先把超二次曲面的体密度收缩到表面附近，得到"表面体密度" $\sigma_{surf}$。轮廓是那些法线与视线方向垂直的点，于是用法线 $\mathbf{n}$ 与视线 $\mathbf{d}$ 的夹角对 $\sigma_{surf}$ 做衰减，定义出视角相关的轮廓密度：

$$\sigma_{contour}(\mathbf{x}, \mathbf{d}) = \left(1 - (\mathbf{n}(\mathbf{x}) \cdot \mathbf{d})^{\beta}\right)\, \sigma_{surf}$$

法线 $\mathbf{n}(\mathbf{x}) = \left[\frac{\partial S}{\partial x}, \frac{\partial S}{\partial y}, \frac{\partial S}{\partial z}\right]$ 可以解析求导得到闭式解。随后用体渲染与数值积分把轮廓画出来。

### 优化目标

考虑到真实照片与抽象草图之间存在巨大的域间隔（domain gap），损失函数结合了捕捉几何布局的 LPIPS 和保持高层语义的 CLIP 余弦距离，并用鲁棒损失 $\rho$ 稳定优化以对抗离群点：

$$\mathcal{L} = \sum_{I \in \mathcal{I}} \lambda\, \rho\!\left(\text{LPIPS}(I, R(S^{3D})), \alpha, c\right) + \text{dist}\!\left(\text{CLIP}(I), \text{CLIP}(R(S^{3D}))\right)$$

参数用 Adam 直接优化。由于目标高度非凸，**初始化**很关键：作者利用估计相机位姿时顺带得到的 SfM 点云，用最远点采样来初始化贝塞尔曲线控制点和超二次曲面中心。

## 实验结果

- **数据集**：NeRF 与 InvRender 的合成数据、作者自采的多样合成物体（Blender Cycles 渲染），以及真实场景 CO3D。用户可自行指定贝塞尔曲线数量与超二次曲面数量来控制抽象层级。
- **基线对比**：与 5 个基线比较，涵盖单图 2D 草图（Kampelmühler & Pinz、CLIPasso）、稠密线稿（ARF、Suggestive Contours）和稀疏 3D 图元（NEF）。
- **定量指标**：在 LPIPS、DINO、CLIP$_{img}$、CLIP$_{txt}$ 及表示大小上评估。3Doodle 在所有稀疏表示方法（CLIPasso、NEF）上全面领先；相比稠密表示（ARF、Suggestive Contours）在结构相似度上略低，但在 CLIP$_{txt}$（"最像草图"）上取得**全场最高分**，同时表示大小仅 < 1.5 kB（NEF 约 6 MB，稠密方法达数十 MB 到数 GB）。
- **感知问卷**：72 名参与者从"像人手绘""有效性""表现力""视角一致性"四个维度打分。3Doodle 在草图逼真度上显著优于稠密方法，视角一致性远超 CLIPasso（后者因为纯 2D 缺乏一致性得分最低）。
- **消融实验**：（1）去掉视角相关成分 $S^{3D}_{dep}$ 后，纯线条无法表达光滑表面（如球体、泰迪熊、雪人）的轮廓；（2）LPIPS 与 CLIP 损失缺一不可，边缘图损失会陷入局部极小；（3）SfM 初始化比随机初始化和线段 SfM 初始化更稳定，尤其能覆盖弱图像证据区域；（4）贝塞尔曲线数量从 16 到 48 控制抽象层级——越多细节越丰富，越少越抽象。
- **鲁棒性与扩展**：即使只有 15 个输入视角也能得到合理草图；能直接处理 CO3D 真实拍摄；还能把结果导出为矢量图并在 Illustrator 中套用不同笔刷做风格化。

## 贡献与局限

**主要贡献**：

- 提出首个从多视角图像直接生成 3D 笔画草图的方法，能概括图像观测中的重要语义信息。
- 无需数据驱动训练、无需把输入风格化、也无需重建 NeRF/mesh，直接从多视角图像找到 3D 笔画。
- 提出高度紧凑（< 1.5 kB）的 3D 笔画表示，把视角无关与视角相关成分统一起来，快速绘制多种物体的概念草图。
- 引入完全可微的渲染方法，用极少参数就能有效优化强大的感知损失。

**局限**：

- 图元数量作为输入参数固定，无法自动寻找平衡表现力与稀疏性的最优笔画数。
- 优化耗时较长（最多约 6 小时），尽管最终表示紧凑且渲染很快。
- 当前公式**忽略图元间的深度遮挡关系**，会把整个线框都画出来（例如雪人或泰迪熊的正面在背面视角下仍会显示）；由于已有超二次曲面的体表示，未来可用其体积来处理遮挡。
