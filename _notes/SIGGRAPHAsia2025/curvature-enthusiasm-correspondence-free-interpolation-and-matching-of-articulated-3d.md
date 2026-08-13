---
title: "Curvature Enthusiasm: Correspondence-Free Interpolation and Matching of Articulated 3D Shapes using Compressed Normal Cycles"
authors:
  - "Adam Hartshorne"
  - "Allen Paul"
  - "Tony Shardlow"
  - "Neill D. F. Campbell"
category: "Geometry & Modeling"
track: "Journal"
source: "author-page"
institution:
  - "University of Bath"
  - "University College London"
tags:
  - "Shape Interpolation"
  - "Shape Registration"
  - "Articulated 3D Shapes"
  - "Dense Correspondence"
  - "Diffeomorphic Transformations"
  - "Neural Ordinary Differential Equations"
  - "Normal Cycles"
  - "Geometric Measure Theory"
  - "Dual Quaternions"
  - "Constrained Optimization"
links:
  paper: "https://doi.org/10.1145/3763366"
  project: "https://curvature-enthusiasm.github.io/"
  code: "https://github.com/neural-geometric-shape-models/curvature-enthusiasm"
---

## 问题与背景

在同一身份、不同姿态的两个 3D 关节形状之间，同时求解两件事：一是预测物理上合理的插值轨迹（形状渐变动画），二是自动估计两者之间的稠密对应关系。这两个问题密切相关，却往往被割裂或串行处理。核心难点在于设计高效的正则化框架，让轨迹既物理合理，又能在目标对齐与几何特征保持之间取得平衡。

本文建立在此前的 ARC-Flow 之上。ARC-Flow 用平滑的微分同胚流场保证拓扑一致，并借助几何测度论中的 Varifold 度量在无需已知对应的前提下把变形后的源形状对齐到目标。本文识别了 ARC-Flow 的若干弱点并加以改进，在保持无监督、免对应设定的同时，提升了插值的物理合理性、数值稳定性以及稠密对应的精度。方法明确聚焦于"同一身份的姿态变化"，因为骨骼被限制为刚性变换，不处理身份/体型变化。

## 核心贡献

论文相对 ARC-Flow 提出四项关键改进：

- 用 Normal Cycles（法环）度量替换 Varifold，显著改善高曲率区域（手指、耳朵等）的稠密表面匹配；并配合一套压缩方案，虽然单次计算略贵，但用更少迭代得到更优结果，抵消了额外开销。
- 更优的骨骼（关节）运动学建模与旋转表示，用对偶四元数统一表达刚体运动，强制物理约束、提升精度与数值稳定性。
- 用约束优化（MDMM）代替软约束加权求和，自动逐骨调参，消除复杂的调度与手工超参调节，更鲁棒且结果更好。
- 更好的骨骼与软组织采样方案：基于体网格的均匀体积采样，改善物理一致性与效率，进一步减少超参、降低对初始骨骼的敏感度。

## 方法

### 几何测度论度量：从 Currents/Varifold 到 Normal Cycles

方法的匹配度量来自几何测度论，其思想是把形状嵌入到测度空间，从而在无需显式对应的前提下比较两个曲面。

Currents 的出发点是对向量场 $$\vec{v}(x): \mathbb{R}^3 \to \mathbb{R}^3$$ 在曲面 $$\mathcal{X}$$ 上做（有向）面积分：

$$\mu_{\mathcal{X}}(\vec{v}) := \int_{\mathcal{X}} \vec{v}(x)\cdot \hat{n}(x)\, dS_{\mathcal{X}}(x).$$

其中 $$\hat{n}(x)$$ 为单位法向。用再生核希尔伯特空间（RKHS）$$V$$（带空间核 $$\kappa(x,x')$$）表示向量场后，可定义对偶范数

$$\|\mu_{\mathcal{X}}\|_{V^*} := \sup_{\vec{v}\in V,\, \|\vec{v}\|\le 1} \big|\mu_{\mathcal{X}}(\vec{v})\big|,$$

进而定义形状间度量

$$d(\mathcal{X},\mathcal{Y}) := \|\mu_{\mathcal{X}}-\mu_{\mathcal{Y}}\|_{V^*}^2 = \langle\mu_{\mathcal{X}},\mu_{\mathcal{X}}\rangle_{V^*} - 2\langle\mu_{\mathcal{X}},\mu_{\mathcal{Y}}\rangle_{V^*} + \langle\mu_{\mathcal{Y}},\mu_{\mathcal{Y}}\rangle_{V^*}.$$

RKHS 结构的关键好处是无需显式求解向量场，内积即可闭式计算：

$$\langle\mu_{\mathcal{X}},\mu_{\mathcal{Y}}\rangle_{V^*} = \int_{\mathcal{X}}\int_{\mathcal{Y}} \kappa(x,y)\,\langle \hat{n}_{\mathcal{X}}(x),\hat{n}_{\mathcal{Y}}(y)\rangle\, dS_{\mathcal{X}}(x)\, dS_{\mathcal{Y}}(y).$$

Varifold 用更一般的法向核 $$\kappa_n(\hat{n}_{\mathcal{X}},\hat{n}_{\mathcal{Y}})$$ 替换上式中的欧氏内积，并与空间核 $$\kappa_x(x,x')$$ 结合。但由于法向核固定，Varifold 在曲率变化剧烈处（如尖角）表现受限。

Normal Cycles 通过把 Currents 提升到同时考虑空间位置与法向位置，并在更高阶对象上积分来突破这一限制。其内积写作

$$\langle N_{\mathcal{X}}, N_{\mathcal{Y}}\rangle_{W^*} = \int_{N_{\mathcal{X}}}\int_{N_{\mathcal{Y}}} \kappa_w(w_x, w_y)\,\langle \tau_x, \tau_y\rangle\, d\mathcal{H}^2(w_x)\, d\mathcal{H}^2(w_y),$$

其中 $$w_x := (x, \hat{n}_{\mathcal{X}}(x)) \in \mathbb{R}^3\times S^2$$ 同时编码空间位置与法向，$$\tau_x,\tau_y$$ 为微分 2-形式（其中一项是法向的导数，正是这一项引入了曲率敏感性）。积分在"法丛"（normal bundle）上进行：

$$N_{\mathcal{X}} := \{(x,\hat{n}_{\mathcal{X}}) \mid x\in\mathcal{X},\ \hat{n}_{\mathcal{X}}\in S^2,\ t\in T_x[\mathcal{X}],\ t\cdot\hat{n}_{\mathcal{X}}=0\},$$

即所有点及其与切平面正交的单位向量组成的集合。离散化时，法丛通过并集规则

$$N[A\cup B] = N[A] + N[B] - N[A\cap B]$$

拆分为顶点上的球分量与沿边/面的柱面（及面片）分量，从而把整体计算写成对网格顶点与边的加权内积之和。采用可分核 $$\kappa_w(w_x,w_y) := \kappa_x(x,y)\,\kappa_n(\hat{n}_{\mathcal{X}},\hat{n}_{\mathcal{Y}})$$ 且取常数法向核，可大幅简化计算。曲率敏感性的直觉是：当两点在空间上接近时，度量鼓励它们两侧的归一化边向量相匹配，而这些边向量恰好编码了局部是平坦还是高曲率。

### 高效计算与压缩

离散 NC 内积在两形状点数上是二次复杂度 $$O(I_{\mathcal{X}} I_{\mathcal{Y}})$$。方法通过对目标形状做基于 Ridge Leverage Score（RLS）的稀疏 Nyström 近似（带理论保证），把目标压缩为一个仅含 $$I_{\mathcal{Y}_c} \ll I_{\mathcal{Y}}$$ 个顶点及权重 $$\beta$$ 的紧凑表示。压缩后的 NC 损失为

$$\mathcal{L}_{nc}(\theta) := d_{nc}\big(\mathcal{X}(T), \mathcal{Y}_c\big),$$

其中与优化参数无关的常数项无需计算。这样即使目标是高分辨率网格，也能避免二次计算爆炸。实践中先用较便宜的 Varifold 得到粗对齐，再切换到 Normal Cycles 做细节精修；由于 NC 只需更少迭代即可达到高质量对应，总匹配时间反而下降。

### 微分同胚变形（NeuralODE）

变形被建模为由时变向量流场 $$\vec{f}(x,t): \mathbb{R}^{3\times 1}\to \mathbb{R}^3$$ 驱动的微分同胚，遵循常微分方程

$$\frac{d}{dt}x(t) = \vec{f}(x(t),t), \quad \text{s.t.}\ x(t=0)\in V_{\mathcal{X}}.$$

因为连续可微向量场的 ODE 解唯一，流线不会相交，从而保证变换光滑、可逆、保拓扑，且独立于网格分辨率与参数化。流场用神经网络参数化（即 NeuralODE），通过数值积分从 $$t=0$$ 的源演化到 $$t=T$$ 的目标：

$$x(T) = x(0) + \int_0^T \vec{f}_\theta(x(t),t)\, dt =: \mathrm{ODESolve}(f_\theta, x_0, T).$$

若需强制体积保持，则通过旋度算子参数化流场 $$\vec{f}_\theta(x,t) := \nabla\times \vec{a}_\theta(x,t)$$，得到无散度流。

### 骨骼插值与对偶四元数

给定源形状的一个简单内部骨架 $$\mathcal{S}_{\mathcal{X}} = \{b_j, e_k\}$$（关节与骨），骨骼应局部刚性运动，而周围软组织与表面做非刚性变形。相比 ARC-Flow 的标准前向运动学加逐骨插值，本文用对偶四元数（DQ）统一表示刚体运动。给定旋转四元数 $$q$$ 与平移 $$s$$，对偶四元数为

$$Q(q,s) := q_r + q_d \in \mathrm{DQ}, \quad q_r := q,\ q_d := \tfrac{1}{2}(q_r q_s),\ q_s := (0, s_x, s_y, s_z).$$

DQ 把旋转与沿轴平移的螺旋运动统一为一个紧凑对象，能在 SE(3) 上保持旋转与平移的正确耦合，比线性插值产生更自然的路径。插值采用 SCLERP：

$$\mathrm{SCLERP}(t; Q_A, Q_B) := Q_A (Q_B^* Q_B)^{t/T},$$

它沿对偶四元数空间的测地线插值，保持单位范数，避免线性混合带来的剪切，产生物理上更合理的运动（如肩、肘、膝关节的自然过渡）。骨骼的局部刚性通过一个逐骨误差度量强制，当流场变形后的骨样本与局部刚性插值一致时其值为零：

$$g_k\big(\theta,\tilde{s},\{\tilde{Q}^{(T)}_k\}\big) := \mathbb{E}_{t, p^{(0)}_k}\big[\|x_{p_k}(t) - p_k(t)\|^2\big].$$

### 软组织采样

ARC-Flow 用围绕伪骨柱面的圆柱壳估计软组织，导致采样密度不均（偏向小骨群如手指）且无法覆盖完整体积。本文改用均匀体积采样：先用快速四面体网格化填充源曲面包围的体积，对超过阈值的四面体细分至体积近似相等，用其质心作为软组织采样点。该预处理只在源形状上运行一次，仅需几秒。

### 约束优化（MDMM）

ARC-Flow 把形状匹配、骨骼、软组织、表面畸变四项损失做加权和，超参难以平衡且需依赖调度。本文改为约束优化：保证硬骨骼约束满足到给定容差，其余损失尽量减小：

$$\mathcal{L}_{match}(\Omega) := \mathcal{L}_{nc}(\theta) + \mu_{surf}\mathcal{L}_{surf}(\Omega) + \mu_{soft}\mathcal{L}_{soft}(\Omega), \quad \text{s.t.}\ g_k(\Omega) \le \varepsilon_{skel},\ \forall k.$$

联合优化变量 $$\Omega := (\theta,\phi,\tilde{s},\{\tilde{Q}^{(T)}_k\})$$，包含流场网络参数 $$\theta$$、预测共形样本旋转的网络参数 $$\phi$$、全局平移 $$\tilde{s}$$ 与目标骨架的对偶四元数。约束优化通过 Modified Differential Method of Multipliers（MDMM）转为无约束问题，修正的拉格朗日为

$$\mathcal{L}_{mdmm} := \mathcal{L}_{match}(\Omega) + \lambda^\top(G(\Omega) - \varepsilon) + \delta\|G(\Omega)-\varepsilon\|^2,$$

其中 $$\lambda \ge 0$$ 是每骨一个的拉格朗日乘子，$$\delta>0$$ 为阻尼超参。求解通过对 $$\Omega$$ 梯度下降、对对偶变量 $$\lambda$$ 梯度上升寻找鞍点，并在优化中对 $$\lambda$$ 阈值截断保持非负。对偶参数的自动平衡（动态调整每骨权重以满足约束）让方法避免了 ARC-Flow 的复杂调度。

### 实现要点

实现基于 JAX 与 Equinox，沿用 Keops（核运算）、ProbDiffEq（概率 ODE 求解器）、VectorAdam（向量优化器）。Flow-Net 是 5 层、宽度 256、GeLU 激活的简单 MLP——得益于约束优化，比 ARC-Flow 依赖的 SIREN/FINER 层更简单。每骨用 50 个样本、软组织 2000 个、表面 500 个，每个 epoch 重新随机采样。刚性区域半径一般取骨长的 10%，但 DFAUST 的手指细长需降到 1%。优化用 VectorAdam 并加入 Cautious Optimization 加速。ODE 用概率求解器（Kronecker EK0，单导数，平滑策略）。先用 Varifold 数据项初始化 500 个 epoch 做粗对齐，NC 核长度尺度每 2000 个 epoch 由粗到细（0.5、0.25、0.1）。

## 实验与结果

对比方法包括当前最优的 ARC-Flow、需训练数据的功能映射方法 SMS、另一 Varifold 方法 ESA、需对应的哈密顿插值方法 HAM、以及无散度场方法 DIV FREE。定量评测在三个标准数据集上进行：DFAUST（人体）、MANO（手）、SMAL（动物）。所有样本归一化到单位立方体，用 K-medoids 挑选多样姿态，采用 80/20 划分。定性评测还包括含拓扑合并噪声的 TOPKIDS 数据集，以及在 Blender 中手工定义骨架的 TOSCA 数据集。

评价指标三项：测地距离（点到点对应精度）、Chamfer 距离（整体重建精度）、准共形畸变（插值局部几何质量）。方法在全部三项指标上均优于对比方法，且误差条更窄，表明性能更一致、更稳健。

骨架选择：在 DFAUST 上分别用 SMPL 与 SKEL 骨架，结果几乎无差异，说明方法对骨架选择不敏感。

消融研究：逐项加入 Normal Cycles（NC）、Dual Quaternions（DQ）、MDMM 约束优化（CO）。单独把单四元数换成 DQ 影响甚微，但与 CO 结合后同时改善拟合质量与插值平滑度；先用 Varifold 初始化再用 NC 精修（完整方法）持续优于仅用 NC——对大变形，Varifold 对朝向不敏感的特性有助于在流场尚未成形时引导早期粗匹配。

定性发现：Normal Cycles 在手指、耳朵等高曲率区域给出远优于 Varifold 的拟合（这些区域 Chamfer 误差不一定高但会被 Varifold 错位）；SCLERP 比 SLERP 产生更自然的插值路径（避免不合理的肩肘旋转或膝关节屈曲突变）；约束优化保证整个插值过程中骨样本对齐在预设容差内。

压缩与效率：把 12k 源网格匹配到 120k 高分辨率目标时，压缩到约 10k 样本的结果与全分辨率视觉上难以区分，却大幅降低计算与内存开销。压缩后的相对训练时间只取决于压缩规模，与原始网格分辨率无关；而 Varifold 与原始 NC 随分辨率指数增长。整体上，不压缩时运行时间与 ARC-Flow 相当（在 NVIDIA RTX A5000 上分钟级），且因 NC 用更少 epoch 达到更高质量而在总时间上胜出；与需要每个新数据集预训练至少 24 小时的深度学习方法（如 SMS）形成鲜明对比。方法还能把已知 MANO 骨架迁移到真实手部扫描。

## 局限与未来工作

局限：与 ARC-Flow 一样假设微分同胚变形，无法修正源或目标网格的拓扑错误；基于骨架的表述明确针对同一身份的不同姿态，因骨骼限于刚性变换而不适用于身份/体型变化；Normal Cycle 框架不支持部分输入，且对噪声很大的目标形状敏感。未来工作方向是几何测度论框架下的部分形状匹配——现有工作仅限于 Varifold，用 Normal Cycles 做部分匹配尚属空白，被视为有前景的研究方向。
