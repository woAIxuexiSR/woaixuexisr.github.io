---
title: "PuzzleAvatar: Assembling 3D Avatars from Personal Albums"
authors:
  - "Yuliang Xiu"
  - "Yufei Ye"
  - "Zhen Liu"
  - "Dimitris Tzionas"
  - "Michael J. Black"
category: "Reconstruction"
track: "Journal"
source: "arxiv"
institution:
  - "Max Planck Institute for Intelligent Systems"
  - "Carnegie Mellon University"
  - "Université de Montréal"
  - "University of Amsterdam"
tags:
  - "3D Avatar"
  - "Text to Image Diffusion"
  - "Score Distillation Sampling"
  - "Digital Human"
  - "Subject-Driven Generation"
links:
  paper: "https://doi.org/10.1145/3687771"
  project: "https://puzzleavatar.is.tue.mpg.de/"
  code: "https://github.com/YuliangXiu/PuzzleAvatar"
---

## 一句话总结

PuzzleAvatar 把一组"当日穿搭"(OOTD)日常照片里一致的身份、服装、发型、配饰拆解成一组可互换的"拼图 token"，通过微调文本到图像扩散模型学到这些个性化先验，再用 Score Distillation Sampling 组装出一个规范姿态下的、忠实且带纹理的 3D 数字人，全程绕开对相机与人体姿态的显式估计。

## 研究背景

为普通人创建个性化全身 3D 数字人一直是难题。已有方法主要分两类：

- 文本到 3D 的数字人生成方法擅长生成名人或虚构角色，但难以还原"具体某个人"的真实外观。
- 忠实重建方法通常要求受控环境下的全身图像：干净背景、标定同步相机、简单可控姿态、无裁剪，难以应对随手拍的照片。

作者提出一个新任务 **Album2Human**：输入一组"唯一约束是身份/服装/发型/配饰保持一致，其余(姿态、相机、构图、光照、背景)任意变化"的个人相册，输出忠实的 3D 数字人。相比一般场景重建，该任务多了人体关节姿态变化这一难点；相比实验室数字化，它面对的是无约束、常被裁剪遮挡、视角未知的照片，使得相机标定与姿态规范化极其困难，直接重建难以奏效。

核心洞见是遵循"重建即条件生成"的新范式：不做显式姿态估计，而是把文本到图像模型当作个性化先验，让它对未观测区域进行"补全"并隐式完成人体规范化。

## 方法

### 整体框架

PuzzleAvatar 分两大阶段：

- **阶段一 PuzzleBooth（把人拆成拼图碎片）**：对无约束照片集自动做描述与分割，把人体拆成若干"资产"(脸、服装、配饰、发型)，每个资产绑定一个可学习的唯一 token `<asset X>`，用于微调 T2I 扩散模型。
- **阶段二 Create-3D-Avatar（把碎片拼起来）**：用组合式的完整文本提示，通过多视角 SDS 损失优化一个 T 姿态、带纹理的四面体网格数字人。

```mermaid
flowchart LR
    A[OOTD 无约束照片集] --> B[Grounded-SAM + GPT-4V<br/>资产分割 + 描述 + 粗视角]
    B --> C[资产-token 对<br/>&lt;asset k&gt;]
    C --> D[微调 PuzzleBooth<br/>Text-Encoder + UNet]
    D --> E[个性化 T2I 扩散先验]
    E --> F[多视角 SDS / NFSD 优化]
    F --> G[DMTet 几何 + 纹理<br/>规范姿态 3D 数字人]
```

### 关键设计 1：拼图式 token 与 Union-Sampling

借鉴 Break-A-Scene，把所有图像分割成 $$K$$ 个资产 $$\{[V_k]\}_{k=1}^{K}$$，每个资产带分割掩码 $$M_k$$、可学习 token $$[v_k]$$ 和类别标签 $$[c_k]$$。训练时对每张图像随机选取其中出现的 $$J < K$$ 个资产，用它们的并集训练：像素级掩码并集 $$M_{\cup} = \cup_{i=1}^{J} M_i$$，图像并集 $$I_{\cup} = I \odot M_{\cup}$$，文本提示 $$p_{\cup}$$ 由所选资产拼接而成(脸部特征在前、服装配饰在后，并附带视角词 $$[d]$$)。这种 Union-Sampling 对资产解耦至关重要。

相比最接近的工作 AvatarBooth 只用两个 token 编码身份与整体服装，PuzzleAvatar 为每个部件单独设 token，并只微调一个统一的 PuzzleBooth 而非为每个部件训练独立的 DreamBooth，因而随部件增多更高效、更可扩展，并天然支持部件互换与"仅用可见 token 在重度裁剪照片上训练"。

### 关键设计 2：两阶段个性化 + 三项损失

微调分两步以防"引导坍缩"(新 token $$[v_k]$$ 与已有类别名 $$[c_k]$$ 冲突)：先用大学习率单独优化 token 的 CLIP 嵌入 1000 步；再用小学习率同时优化文本部分与 UNet 权重 4000 步。总目标为

$$
\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_{attn}\mathcal{L}_{attn} + \mathcal{L}_{prior}, \quad \lambda_{attn} = 0.01
$$

- 掩码扩散损失(保真复现每个概念)：

$$
\mathcal{L}_{rec} = \mathbb{E}_{z,\epsilon,t}\left[\ \|[\epsilon - \epsilon_\theta(z_t, t, p_{\cup})] \odot M_{\cup}\|_2^2\ \right]
$$

- 交叉注意力损失(让每个新 token 只与目标资产关联，促进解耦)：

$$
\mathcal{L}_{attn} = \mathbb{E}_{z,j,t}\left[\ \|CA_\theta(v_j, z_t) - M_j\|_2^2\ \right]
$$

- 先验保持损失(去掉特殊 token 时仍能生成通用人像，保持泛化)：

$$
\mathcal{L}_{prior} = \mathbb{E}_{z^{pr},\epsilon,t}\left[\ \|[\epsilon - \epsilon_\theta(z^{pr}_t, t, p^{*}_{\cup})]\|_2^2\ \right]
$$

其中通用人像来自两处：SD 生成的图像，以及由 THuman2.0 渲染的合成"颜色-法线"配对(用于提升几何质量与颜色-法线一致性)。

### 关键设计 3：NFSD 驱动的 3D 组装

3D 人体用 DMTet(四面体神经表示)参数化，几何 $$\psi_g$$ 与外观 $$\psi_c$$ 均可微渲染，几何初始化为 A 姿态的 SMPL-X。蒸馏采用 Noise-Free Score Distillation(NFSD)以缓解颜色过饱和，将引导拆为两项残差：

$$
\nabla_\psi \mathcal{L}_{NFDS}(z,\psi) = w(t)(\delta_D + s\,\delta_C)\frac{\partial z}{\partial \psi}
$$

$$
\delta_C(z_t, p, t) = \epsilon_\theta(z_t; p, t) - \epsilon_\theta(z_t; \varnothing, t)
$$

$$
\delta_D(z_t, t) = \begin{cases} \epsilon_\theta(z_t; \varnothing, t), & t \le 200 \\ \epsilon_\theta(z_t; \varnothing, t) - \epsilon_\theta(z_t; p_{neg}, t), & \text{otherwise} \end{cases}
$$

引导尺度 $$s = 7.5$$。几何与颜色分两阶段各优化 10000 步：几何阶段在法线空间引导并在提示前加"a detailed sculpture of"，外观阶段在颜色空间引导。

## 实验结果

作者构建了新数据集 **PuzzleIOI**(41 个受试者、约 933 套 OOTD 配置、22 个视角，含 A 姿态真值扫描、SMPL-X 拟合与文本描述)，用 3D 指标(Chamfer、P2S、Normal，单位 cm)与 2D 指标(PSNR、SSIM、LPIPS)在全量 933 套配置上评测。PuzzleAvatar 兼容不同扩散骨干，与 SOTA 的 TeCH 及 MVDreamBooth 对比如下：

| 方法 | 骨干 | Chamfer↓ | P2S↓ | Normal↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------|----------|------|---------|-------|-------|--------|
| TeCH† | SD-2.1-base | 1.646 | 1.590 | 0.076 | 23.635 | 0.919 | 0.065 |
| **PuzzleAvatar** | SD-2.1-base | **1.617** | 1.613 | 0.077 | **24.687** | **0.930** | **0.062** |
| MVDreamBooth† | MVDream | 1.705 | 1.835 | 0.100 | 19.401 | 0.909 | 0.091 |
| **PuzzleAvatar** | MVDream | **1.697** | **1.811** | 0.101 | **21.361** | 0.906 | **0.083** |

(†表示用真值扫描的 SMPL-X 拟合初始化 DMTet 以排除姿态误差。)

结论：在 SD-2.1 骨干下，PuzzleAvatar 的 3D 精度与 TeCH 持平，但在全部 2D 纹理指标上更优——且无需 TeCH 依赖的法线预测、轮廓掩码、拉普拉斯正则、RGB Chamfer 及像素回投影等辅助手段。在 MVDream 骨干下，纹理质量大幅领先 MVDreamBooth(PSNR +10.09%、LPIPS -8.79%)。

消融要点(120 套子集)：合成"颜色-法线"先验贡献最大，去掉后 Chamfer 恶化 +38.1%、P2S +58.8%、Normal +73.3%，且法线先验比 RGB 先验更关键；去掉 NFSD 用普通 SDS 会显著劣化纹理(PSNR -17.3%);视角提示能降低法线误差(去掉后 +9.3%);过于详细的 GPT-4V 描述反而引入偏差、损害身份一致性。可扩展性上，仅用 10% 训练数据即可达到满意重建，多受试者联合训练(5 人/模型)甚至略有提升。

## 亮点与局限

亮点：

- 提出全新 **Album2Human** 任务，把无约束日常相册转成 3D 数字人，摆脱对全身照、标定相机、简单姿态的依赖。
- "拼图式 token"组合建模让身份/服装/发型/配饰可解耦、可互换，天然支持虚拟试穿与文本引导编辑，且能在重度裁剪照片上仅用可见 token 训练。
- 统一 PuzzleBooth + NFSD + DMTet 的极简管线，无需大量现成估计器与正则项即达到 SOTA 纹理质量，并配套可复现评测基准 PuzzleIOI。
- 输出为 A 姿态，便于绑定蒙皮并用 SMPL-X 运动数据(AMASS、AIST++)驱动动画。

局限：

- 基于 SDS 且不用回投影损失，幻觉不可避免：可能错判服装类型、纹理，出现"描述污染"与几何-颜色解耦不彻底(皱褶等几何细节渗入纹理)。
- DMTet 不擅长建模极细结构，SDS 雕刻时会产生尖刺状噪声伪影；有时出现体型不一致(重建偏"胖")。
- 缺少高分辨率头部照片时面部身份保持困难。
- 主要瓶颈是计算开销：训练 PuzzleBooth 加 SDS 优化约需 4 小时，难以在线或实时。
- 依赖公开/商用合成数据集，可能继承其性别、种族、年龄偏差。

## 延伸思考

- PuzzleAvatar 与 TeCH 的对比揭示了"间接重建(语义对齐的 SDS 生成损失)"与"直接重建(像素对齐的回投影损失)"的权衡：前者更鲁棒于无约束输入、失败时仍是"看似真实但身份略偏"，后者正面更贴合原图但依赖脆弱的相机/姿态/几何图估计、失败时易产生非人伪影。这提示无约束场景下"可扩展性"可能比"逐像素保真"更值得优先。
- 多受试者联合训练不降反升，暗示扩散模型能同时容纳并整合大量身份，为"联邦式"分布式训练、汇聚全球用户相册构建多样化"风格库"提供了想象空间。
- 计算复杂度是落地关键，免训练策略、更快采样(如一致性模型)、以及用自由形态 3D 表示或网格形变器替换 DMTet，都是把该范式推向实用的重要方向。
- "把外观拆成可组合 token"的思路不止用于人体，或可推广到通用物体的组合式个性化生成与编辑。
