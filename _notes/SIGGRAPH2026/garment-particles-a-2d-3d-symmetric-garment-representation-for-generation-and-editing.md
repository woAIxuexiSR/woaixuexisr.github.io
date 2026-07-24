---
title: "Garment Particles: A 2D-3D Symmetric Garment Representation for Generation and Editing"
authors:
  - "Kiyohiro Nakayama"
  - "I-Chao Shen"
  - "Ruofan Liu"
  - "Yiming Wang"
  - "Gordon Wetzstein"
  - "Takeo Igarashi"
category: "Neural & Generative"
track: "Conference"
source: "arxiv"
institution:
  - "Stanford University"
  - "The University of Tokyo"
  - "Institute of Science Tokyo"
  - "ETH Zürich"
tags:
  - "Garment"
  - "Diffusion Model"
  - "Point Cloud"
  - "Sewing Pattern"
  - "Generative Model"
links:
  paper: "https://doi.org/10.1145/3799902.3811102"
  project: "https://garment-particles.github.io/"
  code: "https://huggingface.co/georgeNakayama/GarmentParticles"
---

## 一句话总结

提出 Garment Particles——一种把服装的 2D 缝纫纸样和 3D 悬垂几何联合编码为「5D 点云」的对称表示，并基于它训练 rectified flow 生成模型（GPF）与纸样恢复模型（PPF），从而用一个免训练的扩散后验采样框架统一支持从文本/图像/草图生成服装，以及在 2D 纸样和 3D 几何两个域上的多种编辑操作。

## 研究背景

- 领域现状：数字服装设计有两条主线。一是 CLO3D、Style3D 等工业软件，让用户直接编辑结构化的低层表示（拓扑一致的面板、Bézier 曲线），精确但要求很强的打版专业知识；二是生成式模型，能从文本、图像、3D 扫描等高层输入快速生成/编辑缝纫纸样。
- 核心痛点：现有框架只覆盖了问题的一半——要么从随意输入生成服装，要么直接编辑纸样，无法同时兼顾。生成式方法依赖 modality-specific 训练，为覆盖所有编辑操作就得针对每种操作单独采集数据、分配训练预算、平衡控制信号，代价高昂。更关键的是，主流纸样生成模型学到的是「3D 无关」的先验：它们只建模纸样，对悬垂后的 3D 几何一无所知，因此难以用 3D 空间定义的目标去引导生成。geometry image 表示虽捕捉了 2D-3D 对偶，但存在非对称性（3D 外观依赖面板像素不透明度），恢复 3D 需要不可微的离散化，无法直接对 3D 目标做优化。
- 本文 idea：借鉴图像生成社区，把各种服装编辑任务统一转化为「免训练的逆问题」，用扩散后验采样（DPS）求解——不同编辑应用只需换一个引导目标函数，无需重新训练。为让 DPS 能用 3D 目标引导，需要一个 2D 与 3D 对称、且完全可微的表示，这正是 Garment Particles 的动机。

## 方法

整体框架：把一件裁剪缝合的服装看成参数方程 $$\boldsymbol{r}: U \to \mathbb{R}^3$$ 的图（graph），其中定义域 $$U \subset \mathbb{R}^2$$ 就是 2D 纸样，像 $$\boldsymbol{r}(U)$$ 就是悬垂后的 3D 几何。对这个图做点采样离散化，就得到 5D 点云表示 Garment Particles，可通过简单可微的投影算子 $$\pi_D$$、$$\pi_I$$ 分别映射回 2D 纸样域和 3D 几何域。在此表示上训练两个 flow 模型：GPF 从多模态输入生成 5D 粒子，PPF 把粒子转成可仿真的曲线纸样。

```mermaid
flowchart LR
  A["文本 / 图像 / 草图"] --> B["GPF 生成 5D 粒子 X"]
  B --> C["DPS 目标引导编辑"]
  C --> B
  B --> D["PPF 粒子转纸样"]
  D --> E["可仿真缝纫纸样 / 悬垂服装"]
```

关键设计分为三块：

- 5D 粒子表示（是什么 / 为什么）。每个粒子编码一个采样点的 2D 纸样坐标（2 维）和 3D 悬垂坐标（3 维），并额外附一个边界标志 $$f_{\boldsymbol{x}}$$（该点是否落在面板边界上，供后续纸样重建用）。构造时把服装网格按面积约束重新三角化、取顶点作采样点，使点数大致正比于面板面积；再把各面板无重叠地铺进 $$\mathbb{R}^2$$，并按面板语义标签（袖、身、腰头等）迭代消除重叠以保证跨服装的语义一致性。这种表示的对称性在于：2D 与 3D 通过投影算子对等互通，且全程可微，天然适配 3D 目标引导的采样优化。

- Garment Particles Flow（GPF，怎么做）。用 rectified flow 学习粒子的生成先验 $$P_\theta(\boldsymbol{X})$$：把噪声 $$\boldsymbol{X}_0 \sim \mathcal{N}(0, \boldsymbol{I})$$ 线性插值映射到训练数据粒子 $$\boldsymbol{X}_1$$，用标准 flow matching 损失训练漂移场 $$\boldsymbol{v}_\theta$$ 去逼近 $$\boldsymbol{X}_1 - \boldsymbol{X}_0$$。骨干采用 LightningDiT-XL（28 层 DiT）；由于粒子无序，去掉了位置编码，并用 masking 支持可变点数（最多 8192），推理时点数作为输入控制生成纸样的复杂度。文本条件用 CLIP 编码后经 cross-attention 注入；图像条件在文本预训练基础上，为每个 transformer block 增加一路 cross-attention，用冻结的 DINOv2 编码图像并 fine-tune。

- 纸样恢复 PPF 与 DPS 编辑（怎么做）。生成的粒子可能带噪，PPF 用另一个 flow 模型建模条件分布 $$P_\varphi(P \mid \boldsymbol{X})$$，把粒子转成结构化纸样张量（最多 $$M_{max}$$ 个面板、每面板最多 $$E_{max}$$ 条有序参数化边，含控制点、位移、弧标志、缝合标志/标签、边界条件类型、有效性掩码等），用面板/边嵌入排序 token、cross-attention 施加条件。编辑则统一为 DPS 逆问题 $$\boldsymbol{X}^\star = \arg\min_{\boldsymbol{X} \sim P_\theta(\boldsymbol{X})} \mathcal{L}(\mathcal{A}(\boldsymbol{X}), \boldsymbol{Y})$$：在采样每一步对后验均值 $$\hat{\boldsymbol{X}}_{1\mid t}$$ 关于目标 $$\mathcal{L}$$ 做梯度优化。换不同的前向变换 $$\mathcal{A}$$ 和目标即可实现不同编辑——点云条件生成用 $$\mathcal{A} = \pi_I$$ 加 EMD，服装补全用单边 Chamfer 距离，纸样编辑改用 $$\mathcal{A} = \pi_D$$，任意视角轮廓条件生成则用 $$\mathcal{A} = P \circ \pi_I$$（投影到相机视空间）加 2D EMD。超参 stop_t、opt_n、$$T$$ 用来平衡保真度与多样性。

## 实验结果

在 GarmentCodeDatav2（GCDv2，共 124,339 个样本，9:1 划分）上评估。文本条件生成的主实验对比 5 个基线，指标覆盖 3D 分布（COV↑、MMD↓、1-NNA↓、p-FID↓）、CLIP 对齐与仿真成功率 SSR：

| 方法 | COV↑ | MMD (×10³)↓ | 1-NNA↓ | P-FID↓ | CLIP↑ |
|------|------|------|------|------|------|
| Ours | 48.4 | 4.64 | 54.6 | 3.15 | 26.42 |
| SewingLDM | 40.1 | 5.29 | 62.7 | 3.69 | 26.17 |
| Omage | 40.6 | 7.70 | 68.1 | 68.8 | 24.26 |
| D2GC | 27.6 | 6.29 | 81.2 | 18.9 | 27.15 |
| AIpparel | 24.6 | 8.21 | 87.6 | 89.4 | 24.11 |
| ChatGarment | 13.5 | 8.90 | 88.4 | 16.2 | 25.71 |

本文在全部分布指标上最优，说明生成结果既最多样、又最贴近测试集分布，作者归因于 2D-3D 对称表示捕捉了纸样与悬垂几何的关系（基线都不存 3D 信息）。ChatGarment 和 D2GC 因用 LLM 直接生成 GarmentCode，在 SSR 和文本对齐上略高，但多样性差（COV 低、1-NNA 高）。

图像条件生成上，本文在 GCDv2 和 Garment Sketches 两个数据集的 Panel Acc、Panel IOU、Stitch Acc、Chamfer Distance 上全面领先，SSR 约 90%，比 SVG 生成类方法（AIpparel、SewingLDM）高约 10 个百分点，仅略低于程序生成类（ChatGarment、D2GC）。编辑方面定性展示了服装插值（GPF 的 SLERP 插值比基线过渡更平滑、能跨拓扑变化）、点云条件生成与补全、纸样编辑、任意视角轮廓条件生成，以及多步编辑会话；还请裁缝实际缝制了两件无条件生成的服装验证可制造性。

## 亮点与局限

- 亮点：
  - 用「参数方程的图」这一简洁数学视角，把 2D 纸样与 3D 几何统一进对称、可微的 5D 点云，从表示层面解决了以往生成模型「3D 无关」的根本缺陷。
  - 编辑完全免训练：靠 DPS 换目标函数就能覆盖点云条件、补全、纸样编辑、轮廓条件等多种任务，避免了 modality-specific 的数据采集与训练开销。
  - 文本/图像生成均达到 SOTA 分布指标，并用真实缝制验证了从生成到可制造的完整闭环。

- 局限：
  - 5D 点云是对连续曲面与精确纸样边界的离散采样，受粒子分辨率限制，难以做细粒度调整（如省道大小），单阶段也难精确表达连续曲面。
  - GPF 需要把点数作为输入，缺乏自动预测合适点数的机制；迭代编辑时粒子被重采样，细节无法精确保留。
  - DPS 编辑耗时，难以支持鼠标拖拽级别的即时交互；PPF 纯数据驱动，不能保证生成纸样严格匹配粒子（无硬约束）。
  - 只在 GarmentCode 生成的服装上训练，且假设固定人体、未建模体型/姿态/面料对悬垂的影响。

## 延伸思考

这项工作把「统一表示 + 免训练逆问题求解」的范式从图像迁移到了服装设计，DPS 在 flow 模型上的应用（FlowDPS 的扩展）是关键使能技术，值得关注其在其他需要 2D-3D 一致性的领域（如可展曲面、参数化建模）的推广潜力。局限里提到的两个方向尤其有价值：一是为交互性把慢速 DPS 优化蒸馏/摊销成前馈预测，二是把 5D 扩展到「同一纸样在不同体型/姿态/面料下的多种 3D 悬垂」，这实际上是要在表示里引入物理仿真的条件变量，可能与可微布料仿真结合。此外 PPF 缺乏硬约束导致粒子与纸样不严格对应，是否可以用投影/约束优化层替代纯数据驱动的恢复，是一个直接可做的改进点。
