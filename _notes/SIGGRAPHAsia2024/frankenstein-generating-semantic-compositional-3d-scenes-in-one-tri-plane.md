---
title: "Frankenstein: Generating Semantic-Compositional 3D Scenes in One Tri-Plane"
authors:
  - "Han Yan"
  - "Yang Li"
  - "Zhennan Wu"
  - "Shenzhou Chen"
  - "Weixuan Sun"
  - "Taizhang Shang"
  - "Weizhe Liu"
  - "Tian Chen"
  - "Xiaqiang Dai"
  - "Chao Ma"
  - "Hongdong Li"
  - "Pan Ji"
category: "Neural & Generative"
track: "Conference"
source: "arxiv"
institution:
  - "Shanghai Jiao Tong University"
  - "Tencent"
  - "The University of Tokyo"
  - "Australian National University"
tags:
  - "3D Scene Generation"
  - "Semantic Composition"
  - "Diffusion Model"
  - "Tri Plane"
  - "SDF"
links:
  paper: "https://doi.org/10.1145/3680528.3687672"
---

## 一句话总结

Frankenstein 用一个三平面（tri-plane）张量同时编码多个语义部件的独立 SDF，通过扩散模型一次前向即可生成"语义可分解、部件形状完整"的 3D 场景（房间或数字人）。

## 研究背景

高质量 3D 资产生成在游戏、影视、AR/VR 中需求旺盛，扩散模型与 Transformer 推动了 3D 生成的进步。但主流方法通常把 3D 数据表示为单一神经场（NeRF 或 SDF），语义信息与其他属性纠缠在一起，输出的是一个统一整体形状。

然而下游应用往往需要语义可分解的形状：游戏里的车要能拆成车身与四个可滚动车轮，数字人要能分成身体、四肢、头发、服装等部件以便驱动。事后用 3D 分割工具切割网格效果不佳，常产生不完整的碎片。

生成语义组合式 3D 场景面临两个核心挑战：其一，需要一种能同时建模多个部件完整形状的通用 3D 表示；其二，部件之间的空间关系复杂，相对位置要语义合理且物理可行（例如避免相互穿插）。

## 方法

整体框架分三个训练阶段：先把训练场景拟合成三平面，再用 VAE 把三平面压缩到紧凑连续的潜空间，最后训练扩散模型逼近潜三平面的分布。推理时给定 2D 布局图，扩散模型去噪得到潜三平面，VAE 上采样到高分辨率，轻量 MLP 解码出多个语义 SDF。

```mermaid
flowchart LR
    A[组合式训练场景<br/>各部件独立网格] --> B[三平面拟合<br/>Coarse-to-Fine]
    B --> C[VAE 压缩<br/>潜三平面]
    C --> D[条件扩散<br/>布局图引导]
    D -->|推理: 2D布局| E[潜三平面]
    E --> F[VAE 解码上采样]
    F --> G[MLP 解码<br/>多个语义 SDF]
```

### 关键设计一：单三平面解码多 SDF 的组合表示

给定含 $$L$$ 类部件的房间 $$R = \{R_1, \dots, R_L\}$$（如床、柜、墙），目标是训练三平面 $$T = \{T_{xy}, T_{xz}, T_{yz}\} \in \mathbb{R}^{3 \times C \times R_h \times R_h}$$，由 MLP $$\Phi$$ 解码成 $$L$$ 个独立 SDF。三平面按 3D 位置 $$p$$ 查询得特征向量：$$f = Query(T, p)$$，即在三个平面投影坐标处双线性采样并求和。

与"输出单一 SDF 加语义场再做硬分割"（会撕裂表面、导致部件不完整）不同，本文选择直接解码多个形状：$$(d_1, \dots, d_L) = \Phi(f)$$，每个 $$d_i$$ 是第 $$i$$ 类的 SDF 值，从而为每一类保留完整形状。

### 关键设计二：Coarse-to-Fine 三平面拟合与类别专属损失

离散表示（如三平面）用有限差分算法线时存在梯度局部性问题——梯度只影响邻近网格，导致形状拟合噪声大。本文不采用 Neuralangelo 的特征金字塔（难以配合扩散模型），而是用由粗到细策略：先在低分辨率 $$R_l^2$$ 拟合大致语义与形状，再双线性上采样到 $$(2R_l)^2$$ 并细化，重复直到 $$R_h^2 = 2^{\eta} R_l$$。该策略把 500 次迭代的拟合时间从 380s 降到 279s。

拟合损失包含 Eikonal、SDF、表面、法向四项：$$L_{tri} = \lambda_1 L_{eik} + \lambda_2 L_{sdf} + \lambda_3 L_{sur} + \lambda_4 L_{nor}$$。其中表面项与法向项按类别分别计算（如 $$L_{sur} = \sum_{l=1}^{L} \frac{1}{N_l} \|\tilde{d}_l\|_1$$），配合语义感知采样（各类均衡采点 $$N_1 : N_2 : N_3 = 4:1:1$$），缓解墙面积远大于家具导致的采样不平衡，保证小表面部件也能被重建。

### 关键设计三：VAE 潜空间与布局条件扩散

直接在拟合的三平面（$$R_h = 160, C = 32$$）上训练扩散计算代价过高，且三平面空间不连续（两个三平面插值无意义）。因此借鉴 BlockFusion 用 VAE 把三平面编码到紧凑连续的潜空间 $$\mathbb{R}^{3 \times c \times r \times r}$$，并以 $$L_{tri}$$ 为主导损失确保潜三平面能忠实还原形状。

生成阶段以房间布局张量 $$F \in \mathbb{R}^{L \times r \times r}$$（各类部件正交投影到地面得到的二值图）为条件。由于 $$xz$$ 平面与地面布局对齐，只把 $$F$$ 拼接到 $$xz$$ 平面，其余平面补零：$$z_0 = \{\hat{T}_{xy} \oplus 0, \hat{T}_{xz} \oplus F, \hat{T}_{yz} \oplus 0\}$$。训练 U-Net 去噪骨干 $$\Psi$$，且预测 $$z_0$$ 而非噪声 $$\epsilon$$：$$L_{diff} = \|\Psi(z_t, \gamma(t)) - z_0\|_2^2$$。

## 实验结果

房间数据来自 3D-FRONT / 3D-FUTURE，经水密化与穿插修正后得 2558 个卧室（含墙、床、柜三类）。用户研究邀请 60 余名参与者按 1–5 分打分，指标包括房间几何质量（RGQ）、布局质量（RLQ）、布局一致性（RLC）、部件几何质量（CGQ）。

| 方法 | RGQ↑ | RLQ↑ | RLC↑ | CGQ↑ |
| --- | --- | --- | --- | --- |
| Text2Room | 2.18 | 2.18 | - | 1.05 |
| CC3D | 1.50 | 1.55 | 2.06 | - |
| CommonScenes | 3.00 | 2.57 | - | 3.72 |
| Frankenstein (Ours) | **4.33** | **4.35** | **3.83** | 3.63 |

在整体场景指标（RGQ、RLQ、RLC）上最优，部件质量（CGQ）与逐物体生成的 CommonScenes 相当。物理合理性（50 个场景中无穿插的比例）上，Frankenstein 达 84%，远高于 CommonScenes 的 40%。数字人生成方面，相比基于 SDS 优化的 TADA（271.6 分钟/个），Frankenstein 仅需 0.5 分钟且几何更自然。消融显示单个三平面可建模最多 $$L = 7$$ 类的场景。

## 亮点与局限

亮点：首个能在单三平面中一次前向生成语义组合式 3D 场景的扩散模型；把三平面张量分解扩展为解码多个 SDF，天然保证每个部件形状完整、场景可分解；不依赖 SMPL 等形状先验，房间与数字人通用；生成速度相比 SDS 优化类方法有数百倍优势，并支持部件级重贴图、物体重排、服装重定向等下游控制。

局限：作者指出三点——单三平面建模整个场景，细节受分辨率限制（可考虑结合 BlockFusion 的分块方案）；VAE 训练缓慢（约一周），需要更高效骨干；对数据集外的布局仍有瑕疵（如墙上出现孔洞），需更多训练数据缓解。

## 延伸思考

- "多 SDF 共享一个三平面"本质上是把语义分解前置到表示层面，而非事后分割。这一思路能否推广到更细粒度的部件层级（如可活动关节、可拆卸配件）或开放类别场景，是值得探索的方向。
- 布局条件只拼接在 $$xz$$ 地面平面上，天然契合室内场景的俯视布局约束；但对于没有明确"地面投影"结构的物体（如复杂机械），如何设计条件注入方式仍是开放问题。
- VAE 训练耗时约一周成为流程瓶颈，说明"三平面→潜空间"这一压缩环节的效率仍有较大优化空间，可能是该范式规模化落地的关键。
