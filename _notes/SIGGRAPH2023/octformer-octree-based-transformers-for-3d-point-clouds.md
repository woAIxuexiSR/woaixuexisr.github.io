---
title: "OctFormer: Octree-based Transformers for 3D Point Clouds"
authors:
  - "Peng-Shuai Wang"
category: "Geometry & Modeling"
track: "Journal"
source: "arxiv"
institution: "Peking University"
tags:
  - "Point Cloud"
  - "Transformer"
  - "Octree"
  - "Semantic Segmentation"
  - "3D Object Detection"
links:
  paper: "https://doi.org/10.1145/3592131"
  project: "https://wang-ps.github.io/octformer"
---

## 一句话总结

OctFormer 用八叉树的 z-order 排序把点云切成"点数固定、形状可变"的窗口来做局部注意力，从而以线性复杂度、10 行代码的实现拿到远超以往点云 Transformer 的速度，并在 ScanNet 等分割/检测基准上取得当时最优精度。

## 研究背景

- 领域现状：Transformer 在 2D 视觉里已通过窗口注意力（如 Swin）扩展到大图像的密集预测任务；点云领域也涌现出 PCT、Point Transformer、Stratified Transformer 等一系列点云 Transformer。
- 核心痛点：注意力的复杂度是 $$O(N^2)$$，而扫描点云动辄 10 万点以上。为降复杂度，前人把点云按固定尺寸的立方体窗口切分再做局部注意力，但点云稀疏导致每个窗口里的点数差异极大（在 ScanNet 上窗口尺寸为 7 时平均只有 48 点、最多却有 343 点）。这种不均衡逼得算法要么按点数分桶批处理，要么写定制 GPU kernel，严重拖累 GPU 并行效率；另一条"在下采样特征图上做注意力"的路则会掉精度。
- 本文 idea：作者发现注意力对窗口的"具体形状"并不敏感（把预训练 ViT 的图像块随机遮掉 20% 变成不规则窗口，精度仅从 85.1% 掉到 84.2%）。于是反其道而行：不固定窗口的空间形状，而是固定每个窗口里的点数，让窗口形状随点云自由变化——这样切分可以用纯张量 reshape 实现，天然对齐 GPU 并行。

## 方法

整体框架：给定点云先归一化并构建八叉树，把非空叶节点的平均坐标、颜色、法线作为初始特征；经过一个 Embedding 模块下采样投影到高维后，交替堆叠若干 OctFormer block 与下采样模块，形成四个分辨率层级的层次化特征，最后接一个轻量 FPN 完成分割或检测。

```mermaid
flowchart LR
  A["点云 → 八叉树"] --> B["Embedding 下采样"]
  B --> C["OctFormer Block × N1"]
  C --> D["Downsampling"]
  D --> E["... 共四个 Stage"]
  E --> F["层次化特征"]
  F --> G["FPN 分割/检测头"]
```

关键设计：

1. **八叉树窗口切分（是什么/为什么/怎么做）**：八叉树同一深度的节点按 shuffled key（z-order 曲线）排序，空间上相邻的节点在内存里也连续存放。作者把非空节点特征按此顺序堆成张量 $$X \in \mathbb{R}^{N \times C}$$，补零使空间数可被每窗口点数 $$K$$（默认 32）整除后，直接 reshape 成 $$(B, K, C)$$ 就得到窗口划分，再对这 $$B$$ 个窗口并行做标准多头注意力。这样把复杂度降到 $$O(K^2 \cdot \tfrac{N}{K})$$，即关于 $$N$$ 线性。

2. **膨胀八叉树注意力（扩大感受野）**：单纯窗口注意力感受野小、窗口间不通信。引入膨胀率 $$D$$（取 1 或 4），把张量 reshape 成 $$(\tilde{B}, K, D, C)$$ 再转置为 $$(\tilde{B}, D, K, C)$$、展平前两维，即可让同一窗口内的点在空间上间隔采样，用同样的 reshape/transpose 就实现更大感受野。两个相邻 block 分别用膨胀 1 和 4。

3. **条件位置编码（CPE）**：3D 相对位置偏置需要 $$H \times (2W-1)^3$$ 个参数，窗口一大就爆炸。改用条件位置编码：在每个注意力前用一个八叉树深度可分卷积加 BatchNorm 动态生成位置信息，即 $$X = X + \mathrm{batchnorm}(\mathrm{depthwise\_conv}(X))$$，参数更少且效果更好。

4. **极简实现**：整个八叉树注意力完全省掉了以往的 KNN 邻域搜索和最远点采样，退化为等大小小组上的标准多头注意力，用 PyTorch 现成的 `MultiheadAttention` 加张量 reshape/transpose，batch 为 1 时 10 行代码即可实现；点数达 20 万时单个 block 比 Stratified Transformer / Point Transformer V2 快 17 倍以上。

## 实验结果

在 ScanNet 语义分割验证/测试集上与代表性方法对比（mIoU，越高越好；OctFormer 括号内为不投票结果）：

| 方法 | 类型 | Val. mIoU↑ | Test mIoU↑ |
|------|------|-----------|-----------|
| MinkowskiNet | 稀疏体素 CNN | 72.2 | 73.6 |
| O-CNN | 八叉树 CNN | 74.5 | 76.2 |
| Stratified Transformer | 点云 Transformer | 74.3 | 73.7 |
| Point Transformer V2 | 点云 Transformer | 75.4 | 75.2 |
| Mix3D | 稀疏体素 CNN | 73.6 | 78.2 |
| OctFormer（本文） | 点云 Transformer | 75.7 (74.5) | 76.6 |

OctFormer 在验证集上取得所有方法中最优 mIoU，比 Stratified Transformer 高 1.4、比 MinkowskiNet 高 3.5；测试集次于用了额外 mixup 增强和后处理的 Mix3D。在更难的 ScanNet200（200 类）上，OctFormer 从零训练即达 32.6 mIoU，比 MinkowskiNet 高 7.3，甚至比借助 CLIP 预训练的 LGround 还高 5.4。效率上，输入 20 万点时单 block 比两个对手快 17 倍以上。消融显示 CPE 相比无位置编码把 mIoU 从 66.5 提到 74.5，且模型越大、体素越细精度越高。

## 亮点与局限

- 亮点：
  - 抓住"注意力对窗口形状不敏感"这一反直觉观察，把"定形状"换成"定点数"，让点云窗口切分退化为纯张量 reshape，实现极简、并行友好。
  - 复用八叉树 z-order 排序天然提供的空间局部性，彻底省掉 KNN 与最远点采样，速度大幅领先且能作为分割/检测/分类的通用骨干。
  - 精度与效率同时刷新多个基准，尤其在 ScanNet200 长尾场景上从零训练就超过预训练方法。
- 局限：
  - 窗口只按 z-order 一维顺序切分，spatially 相邻但跨 z-order 边界的点可能被分到不同窗口，几何邻接关系被曲线顺序近似。
  - 每窗口点数固定需要补零并 mask，稀疏度极不均匀时仍有一定冗余计算。
  - 测试集精度仍略逊于叠加了额外数据增强/后处理的 Mix3D，说明训练技巧层面还有空间。

## 延伸思考

- "固定 token 数、放开窗口形状"的思路本质上是把点云的稀疏不规则性交给数据排序（z-order）而非注意力结构去消化，这与后续把点云统一序列化再喂给标准 Transformer 的工作（如各类 serialization-based point transformer）方向一致，OctFormer 可看作这条线较早且极简的代表。
- 八叉树同时承担了"窗口划分"和"多尺度层级"两个角色，这种把数据结构与网络结构耦合的做法很适合大规模室内扫描；值得追问的是在自动驾驶那种更稀疏、各向异性更强的 LiDAR 点云上，z-order 切分是否还能保持同样的均衡性与精度优势。
- CPE 用深度可分卷积生成位置编码，与体素卷积骨干可无缝共享算子，提示"卷积做位置编码 + 注意力做长程聚合"在 3D 上是一种低成本高收益的组合。
