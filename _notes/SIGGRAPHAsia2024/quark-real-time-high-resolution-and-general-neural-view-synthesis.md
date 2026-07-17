---
title: "Quark: Real-time, High-resolution, and General Neural View Synthesis"
authors:
  - John Flynn
  - Michael Broxton
  - Lukas Murmann
  - Lucy Chai
  - Matthew DuVall
  - Clément Godard
  - Kathryn Heal
  - Srinivas Kaza
  - Stephen Lombardi
  - Xuan Luo
  - Supreeth Achar
  - Kira Prabhu
  - Tiancheng Sun
  - Lynn Tsai
  - Ryan S. Overbeck
track: "Journal"
source: arxiv
category: "Rendering"
institution:
  - Google
tags:
  - Neural Rendering
  - Novel View Synthesis
  - Real-Time Rendering
  - Image-based Rendering
  - Layered Depth Map
  - Transformer
links:
  paper: "https://doi.org/10.1145/3687953"
---

## 一句话总结

Quark 是一个前馈式神经网络，从稀疏的多视角图像中同时完成场景重建与渲染，在单张 NVIDIA A100 上以 30fps 输出 1080p 新视角，达到实时方法中的最优质量，并在部分场景上逼近甚至超越离线方法。

## 研究背景

主流新视角合成（如 NeRF、3D Gaussian Splatting）通常分两步：先做逐场景的优化重建得到 3D 表示，再从新视角渲染。渲染这一步已能做到实时，但重建这一步始终缓慢，且往往需要密集输入图像并针对每个场景单独训练。

Quark 的目标是把重建与渲染合并做到实时，并且泛化：无需逐场景训练，仅用少量宽基线（相机间距可达 30cm）输入图像即可处理动态场景。作者围绕这一目标组合了若干关键概念，形成一个统一算法：

- 采用分层深度图（Layered Depth Map, LDM）作为场景表示，兼顾体表示的质量与表面表示的效率；
- 采用多尺度的"渲染-精修"（render-and-refine）网络核心，把大部分计算放在低分辨率完成；
- 用基于 Transformer 的跨视角融合聚合多视角信息；
- 每一帧都重新生成并丢弃 LDM，使几何针对当前新视角对齐优化，对反射/折射等视角相关材质尤其有利。

## 方法

### 整体框架

网络输入一组多视角图像（实验用 8 张），输出目标视角下的 LDM 表示并渲染成图。LDM 由 $$L$$ 个层构成，每层带三种属性：深度图 $$d$$、密度（不透明度）$$\sigma$$、以及用于混合 $$M$$ 张输入图像的混合权重 $$\beta$$。深度图形状 $$[L,H,W,1]$$，密度 $$[L,H,W,1]$$，混合权重 $$[L,H,W,M]$$。

渲染时，把输入图像反投影到各深度层，用逐图像混合权重 $$\beta_m$$ 生成每层 RGB，再连同密度做从后到前的 over 合成：

$$
c_{target} = \mathcal{O}\left(\sum_{m=1}^{M} \beta_m \cdot \mathcal{P}^{T}_{\theta}(I, d),\ \sigma\right)
$$

其中 $$\mathcal{O}$$ 是标准 over 合成算子，$$\mathcal{P}^{T}_{\theta}$$ 是反投影算子。训练用可微渲染，推理用 CUDA 优化渲染器，1080p 下约 1.3ms/帧。整个模型无需真正实例化网格。

```mermaid
flowchart LR
    A[多视角输入图像] --> B[编码为多尺度特征金字塔]
    B --> C[学习式初始化 LDM]
    C --> D[Update & Fuse x5<br/>渲染到输入视角→精修]
    D --> E[逐步升分辨率<br/>逐步减层数]
    E --> F[Upsample & Activate<br/>双线性上采样+激活]
    F --> G[混合输入图像+over合成]
    G --> H[1080p 新视角]
```

### 关键设计 1：多尺度渲染-精修核心

直接在输出分辨率 $$[H,W]$$ 求解 LDM 代价过高。Quark 先把输入图像激进下采样编码，再经 5 个 Update & Fuse 步骤迭代精修。这些步骤嵌入类 UNet 结构：早期迭代层数多但空间分辨率极低（定位表面），后期迭代空间分辨率升高、层数通过"层塌缩"（相邻层取均值走直通路、拼接走残差路）减半（贴合表面）。这样在空间维和深度维之间平衡计算。这一"渲染-精修"思想来自 Flynn 等人，类似展开的梯度下降但收敛极快（约 5 次迭代而非数千次），Quark 首次将其做到实时。

Quark 有两个配置：Quark 从 $$64\times36$$、24 层起始，输出 $$512\times288$$、6 层；Quark+ 从 $$96\times54$$、32 层起始，输出 $$768\times438$$、8 层。二者都面向 1080p，最终上采样倍数分别为 3.75× 与 2.4×。

### 关键设计 2：One-to-Many 注意力融合

跨视角信息聚合需要高效且与相机顺序无关。Quark 在每个 Fusion Block 中用一种优化的跨注意力：由单个聚合特征 $$V$$ 反复 cross-attend 到 $$M$$ 个逐视角更新特征 $$\Delta$$。标准注意力为：

$$
\text{Attention}(Q,K,Val) := \text{softmax}\left(\frac{QK^{T}}{\sqrt{C}}\right)Val
$$

作者利用注意力公式中的冗余，把对 $$\Delta$$ 的投影矩阵折叠进 $$W^{q}_i$$ 和输出投影 $$W^{O}$$，从而省去为生成 $$M$$ 个 key/value 的全部矩阵乘：

$$
head_i = \text{Attention}\left(V W^{q}_i,\ \Delta,\ \Delta\right)
$$

这样把复杂度从标准 cross-attention 的层空间 $$O(LM)$$ 次矩阵乘降到接近 $$O(1)$$，大部分逐视角计算在图像空间完成再抬升到 3D。Fusion Block 由 One-to-many 注意力与残差卷积交替组成：

$$
V' = V + \text{Conv}(\text{gelu}(\text{Conv}(\mathcal{N}(V))))
$$

### 关键设计 3：射线方向编码与后采样激活

用射线方向编码替代 Transformer 传统位置编码：先算射线与 LDM 视锥近/远平面在投影空间交点的差向量，经 tanh 后做正弦位置编码（8 个 octave）。它在输入射线与目标射线对齐时取零，精度集中在感兴趣方向，帮助网络偏向靠近目标视角的输入视图，改善反射与非朗伯表面。

深度约束在近/远平面（$$\eta$$、$$f$$）间等视差带内，围绕深度锚点 $$\delta$$ 求解：

$$
\delta = \frac{\ell - 0.5}{L}
$$

$$
d_{\ell} = \left(\left(\delta + \frac{0.5}{L}\ast\tanh(V W_d)\right)\cdot\left(\frac{1}{\eta}-\frac{1}{f}\right)+\frac{1}{f}\right)^{-1}
$$

所有激活都在双线性上采样到目标分辨率之后进行，这种后采样激活对分片光滑的深度/密度能产生清晰的遮挡边界，同时在全分辨率输入图上混合得到高分辨率 RGB。

训练用 $$10\ast L_1 + LPIPS$$ 损失，batch 16、16 卡 A100，先半分辨率训练 250K 步再全分辨率 350K 步，共 600K 步，数据集组合 Spaces、RFF、Nex-Shiny、SWORD。

## 实验结果

在 1024×768 分辨率下与可泛化方法对比（TIME 含重建+渲染），Quark 在三个数据集上全面领先且最快：

| 分辨率 | 方法 | RFF PSNR↑ | RFF LPIPS↓ | Shiny PSNR↑ | Neural3D PSNR↑ | Neural3D LPIPS↓ | TIME↓ |
|---|---|---|---|---|---|---|---|
| 1024×768 | SIMPLI | 23.35 | 0.191 | 25.42 | 28.79 | 0.169 | 4.4s |
| 1024×768 | GPNR | 25.42 | 0.218 | 25.98 | 29.80 | 0.215 | 25s |
| 1024×768 | ENeRF | 23.71 | 0.224 | 25.75 | 29.86 | 0.172 | 205ms |
| 1024×768 | Quark | 26.03 | 0.137 | 26.51 | 31.57 | 0.127 | 32.2ms |
| 1024×768 | Quark+ | 26.53 | 0.127 | 26.67 | 31.95 | 0.122 | 91.9ms |
| 2048×1536 | ENeRF | 23.62 | 0.308 | 24.21 | 29.62 | 0.246 | 811ms |
| 2048×1536 | Quark | 25.42 | 0.233 | 24.66 | 31.46 | 0.187 | 33.0ms |

Quark 是唯一能实时运行的方法，比最接近的 eNeRF 快 6× 以上；在 2048×1536 上比 eNeRF 快 20× 以上且质量更优。相比 GPNR 快 700×、相比 SIMPLI 快 100×。

在 DL3DV-10K 基准上，Quark 无需逐场景优化即超过 Zip-NeRF（PSNR 31.22）与 3DGS（29.82）：Quark 31.31 / Quark+ 31.53 PSNR。在 Mip-NeRF360 / Tanks&Temples / Deep Blending 上与逐场景优化方法竞争：Tanks&Temples 排第一，Mip-NeRF360 排第三，Deep Blending 因输入视角覆盖不足表现较弱。

消融显示：去掉渲染-精修使 PSNR 从 30.41 降到 29.71；去掉跨注意力降到 29.73；改用直接 RGB 输出（不用混合权重）降到 29.18；层塌缩以微小质量损失换来大幅提速（无层塌缩运行时间增加 115%）。

## 亮点与局限

亮点：
- 首个把"重建+渲染"合并做到实时（30fps@1080p）的可泛化神经视图合成方法，质量达实时方法 SOTA 并逼近离线方法；
- LDM 表示 + 多尺度渲染-精修 + One-to-Many 注意力的组合在质量与速度间取得极佳平衡；
- 逐帧重建并丢弃 LDM 的设计，让几何针对当前视角优化，对反射/折射等视角相关材质有天然优势；
- 网络高度可调，可在更慢（如 10fps）时换取更高质量。

局限：
- 依赖最近的 8 个输入相机渲染目标视角，当目标视角覆盖不足（如 Deep Blending 中部分区域仅被 1~2 个远相机可见）时会产生明显伪影，且因混合权重可能出现错误硬边，失效不如逐场景优化方法"优雅"（后者倾向于变模糊）；
- LDM 不显式建模反射/折射，靠逐帧重生成来缓解；
- 作为分层深度图，无法从所有方向观看（但逐帧重生成降低了这一限制的影响）。

## 延伸思考

Quark 的核心洞见是把"逐场景优化"替换为"逐帧前馈重建"，用泛化网络把慢速优化摊薄成一次前向推理。这一思路与近年可泛化 NeRF/GS 前馈化方向一致，但它通过 LDM 这种介于体与面之间的表示、以及把重计算压到低分辨率的多尺度设计，真正跨过了实时门槛。

一个值得关注的点是 One-to-Many 注意力通过折叠投影矩阵消除层空间的矩阵乘，这类"利用注意力公式冗余做等价简化"的技巧对任何需要大量跨集合聚合的实时网络都有借鉴意义。

局限也指明了后续方向：更智能（甚至可学习）的视角选择策略，以及对视角相关材质的显式建模，可能进一步提升低覆盖区域的稳健性和难材质的保真度。
