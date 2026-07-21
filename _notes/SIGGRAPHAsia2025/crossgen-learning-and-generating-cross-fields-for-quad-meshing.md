---
title: "CrossGen: Learning and Generating Cross Fields for Quad Meshing"
authors:
  - Qiujie Dong
  - Jiepeng Wang
  - Rui Xu
  - Cheng Lin
  - Yuan Liu
  - Shiqing Xin
  - Zichun Zhong
  - Xin Li
  - Changhe Tu
  - Taku Komura
  - Leif Kobbelt
  - Scott Schaefer
  - Wenping Wang
track: "Journal"
source: arxiv
category: "Geometry & Modeling"
institution:
  - The University of Hong Kong
  - Hong Kong University of Science and Technology
  - Shandong University
  - Wayne State University
  - Texas A&M University
  - RWTH Aachen University
tags:
  - Cross Fields
  - Quad Meshing
  - Geometry Processing
  - Latent Space Representation
  - Signed Distance Field
  - Auto Encoder
  - Diffusion Model
  - Point Cloud
links:
  paper: "https://doi.org/10.1145/3763299"
  project: "https://arxiv.org/abs/2506.07020"
---

## 问题与动机

Cross field（十字场）是几何处理中的基础工具，尤其在四边形网格生成中起关键作用：它捕捉曲面的内在结构与流向，通过与主曲率方向对齐、贴合尖锐特征边和开边界，引导生成低畸变、间距均匀、贴合特征的四边形网格。

现有 cross field 生成方法主要分两类，都存在明显短板：

- 优化类方法（如 MIQ、Power Fields、PolyVectors、Instant Meshes、QuadriFlow、NeurCross）依赖逐形状优化，单个形状常需数分钟，且对新形状不具泛化能力。其中 NeurCross 质量最好但计算尤其昂贵。
- 数据驱动方法探索不足。较相关的 Dielen 等人的工作依赖全局形状编码（PointNet），仅针对单一类别（如人体）设计与测试，且假设形状处于规范姿态，难以跨类别泛化，对小的非刚性变化也很敏感。

作者提出 CrossGen，目标是：(1) 快速高保真地生成 cross field，无需逐形状迭代优化，毫秒级完成；(2) 跨广泛形状类型泛化，涵盖光滑有机曲面、常见物体、CAD 模型以及带开边界的曲面。

## 方法

CrossGen 假设输入为点云曲面（可由任意曲面表示经采样得到），每个点带有表面法向（若无则用已有方法估计）。网络同时输出形状的 SDF 与其上的 cross field，SDF 零水平集提供连续曲面供四边形网格提取。

**自编码器架构与"局部编码器 + 全局解码器"设计。** 这是泛化的核心思想：不同类别的形状全局差异大，但局部几何常相似（如椅子与桌子的局部部件）。因此：

- 编码器 E 使用浅层稀疏卷积网络（sparse CNN），每个下采样步仅用单个卷积层以保持**小感受野**，聚焦局部几何而非全局语义。输入点被量化到 $$256^3$$ 高分辨率网格，经线性投影为 16 维嵌入后逐级下采样，空间分辨率依次为 $$256^3, 128^3, 64^3, 32^3, 16^3$$，特征维度为 16、32、64、128、128。瓶颈处得到稀疏网格上的隐编码 patch 描述子集合 $$F = \{f \mid f \in \mathbb{R}^m\}$$，默认 $$m = 128$$。
- 解码器 D 同样用稀疏 CNN，逐级上采样恢复到 $$16^3, 32^3, 64^3$$，每级用多个卷积层扩大**感受野**，聚合更广的全局上下文，促进 cross field 的平滑与全局一致性，得到高分辨率特征网格。

**SDF 与 cross field 模块。** 在高分辨率特征网格上，用三线性插值取出两组 32 维特征：一组在输入点 $$P = \{p \mid p \in \mathbb{R}^3\}$$ 上用于 cross field 预测，一组在查询点 $$Q = \{q \mid q \in \mathbb{R}^3\}$$ 上用于 SDF 预测。$$Q$$ 采样自 SDF 零水平集周围的薄壳（真值 SDF 幅值低于阈值 $$\epsilon = 0.02$$），且 $$P \subset Q$$。两个 MLP 分别为 $$M_{cf}(p) \in \mathbb{R}^3$$（预测一条 cross field 方向）与 $$M_{sdf}(q) \in \mathbb{R}$$（预测 SDF 值）。

**Cross field 表示。** 网络只预测一条方向向量 $$M_{cf}(p)$$，然后投影到由点 $$p$$ 及其真值单位法向 $$n_p$$ 定义的切平面上：

$$\bar{\alpha}_p = M_{cf}(p) - (M_{cf}(p) \cdot n_p) n_p = (I - n_p^T n_p) M_{cf}(p)$$

归一化得 $$\alpha_p$$，第二条方向由叉积 $$\beta_p = \alpha_p \times n_p$$ 得到，cross field 即 $$(\alpha_p, \beta_p)$$。相比传统"预测局部坐标系旋转角"的表示，直接预测方向更易保证全局一致性。

**损失函数。** 总损失为四项加权和：

$$L = \lambda_o \cdot L_o + \lambda_{cf} \cdot L_{cf} + \lambda_{sdf} \cdot L_{sdf} + \lambda_{kl} \cdot L_{kl}$$

其中 $$\lambda_o = \lambda_{cf} = \lambda_{sdf} = 1$$，$$\lambda_{kl} = 1e{-6}$$。

- 占用损失（多分辨率体素占用监督），基于二元交叉熵 $$L_o = -\frac{1}{|V|} \sum_{v \in V} \text{BCE}(D_o(v), G_o(v))$$，其中 $$\text{BCE}(x, y) = y \log(x) + (1 - y) \log(1 - x)$$。
- Cross field 损失，利用旋转对称性，只要求 $$\alpha_p$$ 与真值 $$(\mu_p, \nu_p)$$ 平行或垂直，即 $$\|\alpha_p \cdot \mu_p\| + \|\alpha_p \cdot \nu_p\|$$ 取到最小值 1：

$$L_{cf} = \frac{1}{|P|} \sum_{p \in P} \left( \|\alpha_p \cdot \mu_p\| + \|\alpha_p \cdot \nu_p\| - 1 \right)$$

- SDF 损失 $$L_{sdf} = \frac{1}{|Q|} \sum_{q \in Q} \|M_{sdf}(q) - G_{sdf}(q)\|$$。
- 隐空间 KL 正则 $$L_{kl}$$，将隐分布拉向标准正态 $$N(0, 1)$$，稳定并平滑隐空间。

**四边形网格提取。** 由预测 SDF 经 Marching Cubes 得三角网格，查询三角面中心特征得 cross field，再用 libigl 的全局无缝参数化 + libQEx 提取四边形网格；对 CAD 与开边界曲面则用 QuadWild 提取。若输入本身是三角网格则可跳过 SDF 重建直接提取。

**生成式扩展。** 借助同一隐空间，作者设计两阶段隐空间扩散管线（受 LAS-Diffusion 与 TRELLIS 启发）：第一阶段在草图条件下用占用扩散生成粗体素占用（全局结构），第二阶段用细粒度隐扩散细化几何与 cross field 信息，最后由解码器 D 解出 cross field，实现由草图端到端合成新形状并生成其四边形网格。

## 数据集

作者构建了首个成对"高质量 SDF + cross field"的大规模 3D 形状数据集：真值 SDF 用 TSDF 计算，cross field 用 NeurCross 计算。含 1,700 个不同形状，来源为：DeformingThings4D（光滑有机形状，每类 10 个共 310 个）、ShapeNetCore（550 个）与 Thingi10K（530 个常见物体）、ABC 数据集（310 个 CAD 模型）。经随机旋转增强得到超过 10,000 个形状，全部归一化到 $$[-0.5, 0.5]^3$$，按 9:1 划分训练/测试。

## 实验与结果

**训练设置。** 8 张 NVIDIA RTX 4090（各 24GB），训练 2000 个 epoch，Adam 优化器，学习率 $$1 \times 10^{-4}$$，batch size 16，完整训练约 15 天。评价 cross field 用角度误差（AE，相对主曲率方向，$$AE \in [0, \sqrt{2} - 1]$$）与耗时；评价四边形网格用面积畸变、角度畸变、奇异点数、Chamfer 距离、Jacobian 比五项指标。

**Cross field 质量与速度。** 在测试集上，CrossGen 与 NeurCross 同为最低 AE（0.04），优于其它所有基线。得益于前馈推理，CrossGen 平均运行时间最快，在全测试集上比 NeurCross 快约 5382 倍；在超过 100,000 面的高分辨率子集上，CrossGen 运行时间保持恒定（约 0.079 秒），比 NeurCross 快约 11,550 倍，展现分辨率无关的效率。

**四边形网格质量。** CrossGen 在多项指标上仅次于 NeurCross（其为训练数据的"专家"模型，构成性能上界），优于其它所有基线。基于点云重建（CrossGen）与基于真值网格（CrossGen*）两种设置结果相当，说明 SDF 分支能产出适合下游四边形网格化的高质量几何。

**鲁棒性与泛化。** 方法对缺失区域、稀疏点云（低至 10K 点）、高斯噪声（1%–3%）均保持稳定，仅奇异点略增；对随机姿态（非刚性形变与刚性旋转）鲁棒，无需规范对齐；能泛化到训练中未见的类别与带开边界的曲面（归因于稀疏局部编码器）。与 Point2Quad 相比，CrossGen 生成更规则、更平滑、更贴合主曲率的四边形网格，奇异点更少。

**消融。** 对比两种 cross field 表示：直接预测方向（本文，AE 0.05）显著优于预测旋转角（AE 0.34），且更易保证全局一致性。

**生成能力。** 在完整的椅子与飞机类别上训练两阶段扩散管线，能由草图条件生成多样且连贯、与草图对齐的四边形网格。

## 局限

其一，训练集相对 3D AIGC 领域的大型基础模型仍较小，对罕见几何的泛化可能受限。其二，对含精细几何细节的极复杂物体仍会失败，导致四边形网格缺乏平滑性与全局一致性。其三，将隐空间用于新四边形网格生成仍是初步探索，扩展到 Objaverse 等大规模数据集受限于缺乏高质量真值四边形网格，留待未来工作。此外，四边形提取步骤（libQEx 平均约 40.81 秒、QuadWild 约 15.17 秒）尚未加速，整条从点云到四边形网格的流程约需 15–41 秒。
