---
title: "Position-Based Nonlinear Gauss-Seidel for Quasistatic Hyperelasticity"
authors:
  - "Yizhou Chen"
  - "Yushan Han"
  - "Jingyu Chen"
  - "Zhan Zhang"
  - "Alex Mcadams"
  - "Joseph Teran"
category: "Animation & Simulation"
track: "Journal"
source: "arxiv"
institution:
  - "University of California, Los Angeles"
  - "Epic Games"
  - "University of California, Davis"
tags:
  - "Position-Based Dynamics"
  - "Quasistatic Simulation"
  - "Hyperelasticity"
  - "Nonlinear Gauss-Seidel"
  - "Finite Element Method"
  - "Implicit Time Integration"
links:
  paper: "https://doi.org/10.1145/3658154"
---

## 一句话总结

针对准静态超弹性问题，作者用"基于节点（position-based）"而非"基于约束（constraint-based）"的非线性 Gauss-Seidel 迭代取代 PBD/XPBD：每次迭代更新一个节点、同时考虑该节点参与的所有约束，从而既保留 PBD 在有限算力预算下的稳定性，又能在预算增大时收敛到与牛顿法可比的解。

## 研究背景

超弹性固体的运动由弹性势能定义的连续体应力控制，在图形学中广泛用于角色肌肉、软组织等仿真。作者聚焦**准静态（quasistatic）**问题——忽略惯性项，材料运动由一系列平衡问题（能量极小化）定义：

$$
0 = \nabla_X \cdot P + f_{ext}
$$

准静态求解正变得越来越重要，因为它是生成**准静态神经网络（QNN）**训练数据的关键：这类网络能在实时性能下逼近超高分辨率的弹性平衡态，但训练需要成千上万个高分辨率平衡解。理想的求解器应当在极少的人工干预与参数调节下，用最小算力生成视觉上可信的结果。

PBD [Müller et al. 2007] 因其鲁棒性和在有限预算下产出可信结果的能力，是生成 QNN 数据的天然候选。XPBD [Macklin et al. 2016] 进一步把 PBD 关联到后向欧拉系统的总拉格朗日乘子形式的 Gauss-Seidel 近似。但 PBD/XPBD 在准静态场景下有三大局限：

- **无法直接准静态化**：XPBD 为后向欧拉设计，去掉惯性项会导致更新公式中除以零；把 PBD 看作 XPBD 无穷刚度极限时它虽近似准静态方程，却**不可逆地丢掉了外力项**。
- **本构模型受限**：PBD/XPBD 只能离散那些能写成"某种应变约束的二次型"的超弹性模型，排斥了计算力学中的许多模型。
- **约束中心迭代产生伪影**：PBD 一次只投影/求解单个约束涉及的节点，忽略相邻约束的影响。当一个节点出现在多种约束中时会产生伪影，在准静态问题中尤为严重，并显著降低收敛性（呈现依赖迭代顺序的行为）。

## 核心方法

### 从"约束中心"到"节点中心"的 Gauss-Seidel

FEM 离散后的准静态问题等价于带边界条件的能量极小化：

$$
x^{n+1} = \arg\min_{y \in W} \hat{PE}(y) - y \cdot \hat{f}_{ext}
$$

其中 $\hat{PE}$ 包含弹性势能项 $\hat{PE}_\Psi$ 和用于自碰撞/绑定的弱约束项 $\hat{PE}_{wc}$。

PBD/XPBD 在第 $k$ 个子迭代里求解"第 $k$ 个约束"涉及的节点；而 **PBNG（Position-Based Nonlinear Gauss-Seidel）在第 $k$ 个子迭代里只更新单个节点 $i_k$**，让它在其余耦合节点固定的前提下极小化势能：

$$
\Delta x_{(k+1)i_k} = \arg\min_{\Delta y \in \mathbb{R}^d} \hat{PE}(x_{(k)} + \tilde{C}_{i_k}\Delta y) - \Delta y \cdot \hat{f}_{ext,i_k}
$$

这样每次位置更新都"感知"该节点参与的**全部**约束，消除了 PBD/XPBD 逐约束处理带来的伪影。该极小化对应一个小的非线性方程组（3D 为 3 个方程，2D 为 2 个），作者用**单步修正牛顿法**近似求解（初始猜测取 0，实验发现多步并不显著提升鲁棒性或收敛）：

$$
\Delta x_{(k+1)i_k} = \left(A_{(k+1)i_k}\right)^{-1}\left(f_{i_k}(x_{(k)}) + \hat{f}_{ext,i_k}\right)
$$

其中 $A \approx -\partial f_{i_k}/\partial y_{i_k} \in \mathbb{R}^{d\times d}$ 是势能 Hessian（负力梯度）的近似。

### 免 SVD 的 Hessian 半正定投影

节点 Hessian 中最贵的是能量密度的四阶张量 $\mathcal{C}^e_{\alpha\gamma\beta\delta}$，且它可能不定，妨碍牛顿迭代收敛。以往的定性投影（Teran et al. 2005b、Smith et al. 2019）需要对变形梯度做 SVD、还要解 $3\times3$ / $2\times2$ 对称特征系统。作者提出一个廉价但有效的近似：

$$
\tilde{\mathcal{C}}^e_{\alpha\gamma\beta\delta}(y) = 2\mu\,\delta_{\alpha\beta}\delta_{\gamma\delta} + \lambda\, J F^{e-1}_{\alpha\gamma}(y)\, J F^{e-1}_{\beta\delta}(y)
$$

这里 $JF^e = \det(F^e)F^{e-T}$ 是变形梯度的**余因子矩阵（cofactor matrix）**，对任意变形梯度（奇异、反转/负行列式等）都有定义，这对大变形鲁棒性至关重要。该近似是"单位阵的正数缩放 + 余因子矩阵的秩一更新（$\lambda>0$）"，显然半正定，无需 SVD、无需特征分解。

其理论依据在 Lamé 系数分析中给出：任意各向同性超弹性模型都可写成不变量 $I_\alpha$ 的函数，若要用 Lamé 参数（由杨氏模量 $E$、泊松比 $\nu$ 设定），则 Hessian 在 $F=I$ 处应与线弹性一致。作者的近似恰好保留了 $\partial^2 I_0/\partial F^2$ 与 $\partial I_{d-1}/\partial F \otimes \partial I_{d-1}/\partial F$ 两项，从而既半正定又与任何按 Lamé 系数设定的模型一致。

### 节点染色与并行

Gauss-Seidel 的并行受限于更新间的数据依赖，需要染色。PBD 对**约束**染色（同色约束不共享节点）；PBNG 对**节点**染色（同色节点不共享任何网格单元或弱约束）。作者观察到"对节点染色"比"对约束染色"所需颜色数明显更少——例如箱体拉伸例子中 PBNG 只需 5 种颜色，而约束染色需 39 种。更少的颜色意味着无竞争条件下能并行的工作更多，扩展性和性能更好。对动态自碰撞产生的弱约束，只需对新增约束的关联节点做增量重染色。

### 加速：Chebyshev 与 SOR

与多数 Gauss-Seidel 一样，PBNG 的收敛速率会随迭代数下降。作者引入两种简单加速：**Chebyshev 半迭代法**（$\rho=0.95$，欠松弛 $\gamma=1.7$）与 **SOR**（$\omega=1.7$）。由于 PBNG 非常稳定，甚至允许使用超松弛。两者在残差下降和视觉表现上相近，都能明显提升收敛速率。

### 与 XPBD 收敛性的对比

作者指出 XPBD 的 Gauss-Seidel 更新省略了拉格朗日乘子系统中的两项：左端二阶项（省略是为了解耦位置与乘子）和右端**位置（主）方程的残差项**。省略主残差会让 XPBD 的残差在一两次迭代后就停滞——尽管次残差在下降，真实（牛顿）残差却停滞；而把该项加回来又会引入不稳定。PBNG 抛弃拉格朗日乘子形式，不存在这些问题，即便加入惯性项也能收敛。

## 实验结果

所有实验在 AMD Ryzen Threadripper PRO 3995WX（64 核 128 线程，Intel OpenMP 并行）上运行，泊松比统一取 $\nu=0.3$。

- **箱体拉伸/扭转**（32K 顶点、150K 单元）：固定预算下对比。在充足预算（1.3s/帧）时只有 PBNG 收敛到真值，PBD/XPBD 不收敛；在极小预算（170ms/帧）时牛顿法极不稳定，PBNG 仍视觉可信，PBD/XPBD-QS 出现伪影。
- **不同分辨率**：最高分辨率块达 209.7 万顶点、1024 万单元，PBNG 仅需 40 次迭代（61s/帧）即得到视觉可信结果，且结果随网格细化保持一致。
- **不同本构模型**：corotated、Neo-Hookean、stable Neo-Hookean 三种模型均可用，40 次迭代/帧都视觉可信。
- **大规模肌肉仿真**（284K 顶点、1097K 单元，含碰撞与结缔组织弱约束）：PBNG 每帧 67s，牛顿法 430s，PBNG 视觉上与牛顿法相当但**快 6–7 倍**；PBD 变得不稳定，XPBD-QS 因约束处理顺序不同要么过度拉伸单元、要么留下缝隙（顺序依赖行为）。
- **染色对比**：PBNG 每次迭代做的工作比 PBD 多，但因颜色数少、扩展性好，速度相当甚至更快（如 Res 64 箱体 PBNG 65ms vs PBD 137ms/迭代）。
- **变刚度 / 悬挂双块**：XPBD-QS 无法收敛，残差振荡并给出视觉错误结果，PBNG 快速收敛。
- **PBD 外力丢失验证**：重力下悬臂梁随迭代增加，PBD 收敛到刚性直杆（外力效应被抹去），PBNG 收敛到合理下垂形态。

## 贡献与局限

**贡献**：
- 提出用于超弹性隐式时间步进的**基于节点而非约束**的非线性 Gauss-Seidel 方法（PBNG），修复了 PBD/XPBD 在准静态问题上的外力丢失、模型受限、顺序依赖伪影等问题。
- 提出**免 SVD、免特征分解**的超弹性能量密度 Hessian 半正定投影（基于余因子矩阵），对大变形鲁棒且与按 Lamé 系数设定的模型一致。
- 提出**节点染色**方案，比约束染色颜色更少，改善并行扩展性与性能。

**局限**：即便加上 Chebyshev / SOR 加速，当算力预算被放大时，PBNG 在数值残差下降上仍不敌标准牛顿法。作者提出未来可结合多重网格（multigrid）或区域分解（domain decomposition）来弥补这一点。

## 延伸思考

PBNG 的核心洞见是把 PBD 的"约束中心"视角换成"节点中心"视角：在准静态这种非凸项主导、外力不可忽略的场景下，一次只投影单个约束会系统性丢失邻接约束信息，而以节点为单位的局部极小化恰好把这些信息重新纳入。它本质上是"用极小算力换稳定视觉效果"的定位——瞄准 QNN 数据生成这类"要量、要稳、不要极致精度"的需求，而非与牛顿法争夺高精度收敛。免 SVD 的余因子 Hessian 投影是一个可复用的独立技巧，值得在其他需要频繁定性投影的弹性求解器中借鉴；而"节点染色优于约束染色"的观察，也提示 Gauss-Seidel 类方法中并行粒度的选择会直接影响可扩展性。
