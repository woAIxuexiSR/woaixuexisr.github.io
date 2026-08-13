---
title: "GS-RoadPatching: Inpainting Gaussians via 3D Searching and Placing for Driving Scenes"
authors:
  - "Guo Chen"
  - "Jiarun Liu"
  - "Sicong Du"
  - "Chenming Wu"
  - "Deqi Li"
  - "Shi-Sheng Huang"
  - "Guofeng Zhang"
  - "Sheng Yang"
category: "Reconstruction"
track: "Conference"
source: "arxiv"
institution: "Beijing Normal University"
tags:
  - "3D Gaussian Splatting"
  - "Scene Inpainting"
  - "Driving Scenes"
  - "Patch Matching"
  - "Object Removal"
  - "Novel View Synthesis"
links:
  paper: "https://doi.org/10.1145/3757377.3763892"
  project: "https://shanzhaguoo.github.io/GS-RoadPatching/"
---

## 一句话总结

GS-RoadPatching 直接在 3D Gaussian Splatting（3DGS）模态上做街景补全：借鉴 PatchMatch 的"搜索—度量—替换"思路，把驾驶场景里高度重复的路面结构用已重建好的区域块替换掉待补全区域，从而无需依赖 2D 扩散先验、也不必对高斯做大量重训，就能实现更完整、更协调的移动物体移除与场景编辑。

## 研究背景

自动驾驶的仿真与交互依赖对街景的高质量三维重建与渲染，3DGS 因其实时渲染与可编辑性成为热门表示。但在物体移除、场景补全这类编辑任务上，现有 3DGS 补全方法普遍走"2D 先验引导"的两阶段路线：先用扩散或 GAN 模型在 2D 透视图上做 inpainting 得到外观或深度先验，再回过头做第二轮重建。这条路线有两个痛点：

- **强依赖 2D 跨模态的时空一致性**：补全后的 2D 区域必须在多帧间物理一致才能重建出清晰的 3D 结果，而这本身是个极难保证的问题，容易产生伪影。
- **重训代价高**：需要重新生成训练数据并对高斯做额外一轮优化，效率低。

作者的关键洞察是：驾驶场景高度结构化、重复性强（车道、路面等），这些重复模式在 3DGS 的隐式特征空间里天然具有多模态相似性，非常适合做结构化匹配。因此可以绕开 2D 先验，直接在 3DGS 上做"替换式（substitutional）"补全——用场景内已重建好的区域块去填补待补全区域。

## 方法

整体框架由四步构成：场景重建（嵌入视觉特征的神经高斯）→ 定位目标块 → 正交流形 BEV 空间搜索与度量候选块 → 替换与融合。

```mermaid
flowchart LR
    A[驾驶序列<br/>SfM+LiDAR 初始化] --> B[视觉特征嵌入的<br/>神经高斯重建]
    B --> C[定位目标块<br/>锚点分为 missing/incomplete/intact]
    C --> D[流形 BEV 搜索<br/>沿车道纵横向找候选块]
    D --> E[块亲和度度量<br/>视觉特征余弦相似度]
    E --> F[体素级替换<br/>+ 重投影融合与边缘混合]
    F --> G[补全后的可编辑 3DGS 场景]
```

### 关键设计一：视觉特征嵌入的神经高斯

以 Scaffold-GS 的锚点结构组织神经高斯，并在每个高斯上额外嵌入视觉特征 $$f$$，作为衡量块相似度的描述子。用 SuperPoint 从输入图像提取的特征图作为监督真值 $$F_t(I)$$，渲染出特征图 $$\hat{F}_s(I)$$ 后用带动态物体掩码 $$M_i$$ 的 L1 损失训练：

$$\mathcal{L}_{Feat} = M_i \cdot \lVert F_t(I) - \hat{F}_s(I) \rVert_1$$

总重建损失联合了图像 L1、SSIM、LiDAR 深度与特征监督：

$$\mathcal{L} = (1-\lambda_{SSIM})\mathcal{L}_1 + \lambda_{SSIM}\mathcal{L}_{SSIM} + \lambda_{Depth}\mathcal{L}_{Depth} + \lambda_{Feat}\mathcal{L}_{Feat}$$

嵌入特征走独立的锚点特征与梯度回传路径，避免干扰其他高斯属性。相比只用几何或颜色，视觉特征显著提升了跨视角的块检索精度。

### 关键设计二：目标块定位与锚点三分类

把语义掩码反投影到锚点，并将不透明度低（alpha < 0.9）的锚点识别为未收敛、需补全的目标。随后把目标块内锚点分为三类：**missing（缺失，待填）**、**incomplete（重建不完整，缺监督，不作参考）**、**intact（完整，可作搜索参考）**。通过把锚点投影到前后帧、统计其落入目标语义掩码的复现率 $$\lambda_{CMP}$$ 来区分 incomplete 与 intact。最后用体素网格（体素尺寸 $$\lambda_{size}$$）与哈希编码建立锚点与体素的双向索引，提升检索效率。

### 关键设计三：流形 BEV 搜索与块亲和度

3D 空间搜索比 2D 更模糊、代价更高。作者观察到最合适的候选块通常就在目标块所处的曲面附近，而路面可建模为 2D 流形，且车道的纵向、横向方向最可能找到匹配块。于是把地面流形分解为沿道路与行驶方向的两个主方向，构造与流形正交的 BEV 搜索空间，沿 X 轴做双向路径搜索得到矩形候选区域。度量时只用 intact 锚点，把目标块与候选块的锚点中心投影到图像平面得到 2D 坐标块 $$\pi(T_i)$$、$$\pi(S_j)$$，再从渲染特征图取特征算余弦相似度：

$$A_{i,j} = \langle \hat{F}_s(I)[\pi(T_i)],\, \hat{F}_s(I')[\pi(S_j)] \rangle$$

取相似度最大者为最优候选 $$S_j^{*} \leftarrow \arg\max_j (A_{i,j})$$。

### 关键设计四：替换与重投影融合

先做体素级锚点替换：以锚点为单位迁移神经高斯，把目标区域外的高斯不透明度置零以便混合。替换后仍有视觉不一致，作者利用"动态前景物体在被遮挡前会短暂露出背景"这一时序窗口，从未遮挡帧提取干净背景块 $$\hat{B}$$ 并重投影到被遮挡视角，再与渲染图边缘做膨胀/腐蚀 + alpha 混合（带宽 10 像素，权重 $$W_\alpha=0.5$$）：

$$\hat{I}' = W_\alpha \cdot \pi(\hat{B}) + (1 - W_\alpha) \cdot \hat{I}$$

以此合成伪真值，再用 L1 融合损失做少量迭代优化：

$$\mathcal{L}_{Fusion} = \frac{1}{N}\sum^{N} \lVert \hat{I}' - \hat{I} \rVert_1$$

融合只依赖 RGB 监督，仅需极少迭代即可获得时序与多视角一致性。

## 实验结果

在 Waymo Open Dataset 上选取 23 个序列（含多动态物体、严重遮挡、上下坡、复杂天气光照等），与两阶段 GS 补全方法（GScream、InFusion）、2D 视频补全方法（ProPainter）以及 StreetGaussians+InFusion 组合对比，报告 LPIPS、FID 与平均额外处理时间（NVIDIA RTX 3090）。

| 方法 | LPIPS↓ | FID↓ | Time(s)↓ |
|------|------|------|------|
| ProPainter | 0.207 | 81.925 | 147 |
| GScream | 0.421 | 152.879 | 20 |
| InFusion | 0.396 | 126.556 | 96 |
| StreetGS + InFusion | 0.243 | 114.490 | 115 |
| Ours | 0.237 | 74.124 | 64 |

在所有 GS-based 补全方法中，本方法的 LPIPS 与 FID 均最优（FID 74.124 明显领先）。ProPainter 直接在原视频上操作故 LPIPS 略低，但会产生明显伪影。用户研究（43 名标注者）显示，88.22% 与 83.57% 的参与者分别在"整体协调"和"目标移除质量"上选择本方法为最佳。在 360-USID 数据集的通用场景补全上，本方法在无参考视图条件下 PSNR 17.917 / SSIM 0.975 / LPIPS 0.360，同样优于 SPIn-NeRF、GaussianGrouping、AuraFusion360 等方法，验证了策略的通用性。

## 亮点与局限

亮点：
- 直接在 3DGS 模态做**替换式补全**，摆脱了对 2D 扩散先验与跨模态时空一致性的依赖，也避免了耗时的高斯重训。
- 把 PatchMatch 的搜索—度量—替换范式成功迁移到 3D 高斯场景，配合正交流形 BEV 搜索，将高维 3D 搜索有效降维到路面流形上。
- 视觉特征嵌入（SuperPoint）显著提升跨视角块匹配鲁棒性，消融显示其优于 DINO 与纯 2D 颜色相似度。
- 巧用动态前景遮挡的时序窗口提取干净背景做重投影融合，效率高、仅需少量迭代。

局限：
- 方法强依赖场景的重复结构假设，非常适合结构化路面，但在缺乏可复用相似块的非重复区域可能失效。
- 视觉嵌入在稀疏监督区域会因嵌入与纹理学习不足而退化。
- 体素尺寸敏感：过小丢失目标区域整体一致性、过大放大替换噪声（250 cm 为经验最优），需按场景调参。

## 延伸思考

- 替换式补全的上限受限于"场景内是否存在合适相似块"，未来可探索与生成式补全（扩散先验）互补：结构化区域用检索替换、稀有区域用生成填补。
- 流形 BEV 搜索目前依赖路面可建模为 2D 流形的假设，如何推广到更一般的非流形三维场景（室内、复杂地形）是有意思的方向。
- 视觉特征嵌入除用于块匹配外，天然可作为 3DGS 场景的语义/结构索引，或能支撑更丰富的场景级编辑与检索应用。
