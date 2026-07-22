---
title: "Stabler Neo-Hookean Simulation: Absolute Eigenvalue Filtering for Projected Newton"
authors:
  - "Honglin Chen"
  - "Hsueh-Ti Derek Liu"
  - "David I. W. Levin"
  - "Changxi Zheng"
  - "Alec Jacobson"
category: "Animation & Simulation"
track: "Conference"
source: "arxiv"
institution:
  - "Columbia University"
  - "University of Toronto"
tags:
  - "Neo-Hookean Elasticity"
  - "Projected Newton"
  - "Eigenvalue Filtering"
  - "Finite Element Method"
  - "Hyperelasticity"
  - "Physics-Based Simulation"
links:
  paper: "https://doi.org/10.1145/3641519.3657433"
---

## 一句话总结

在投影牛顿法求解 Neo-Hookean 类保体积弹性能量时，把逐元素 Hessian 的负特征值取绝对值（而非传统的裁剪到零或小正数），只需改动一行代码，就能在高泊松比与大体积变化下显著提升优化的稳定性与收敛速度。

## 研究背景

Neo-Hookean 及其保体积变体是橡胶、软组织等近不可压缩材料（泊松比 $\nu \in [0.45, 0.5)$）的事实标准材料模型。但在近不可压缩区间，保体积项带来强非凸性，尤其当初始形变造成较大体积变化时，数值优化极易失稳。

二阶方法中最常用的是投影牛顿法。由于全局 Hessian 直接做特征分解代价过高，实践中把全局 Hessian 拆成逐元素子 Hessian 之和：

$$\mathbf{H} = \sum_i \mathbf{P}_i^\top \mathbf{H}_i \mathbf{P}_i$$

再对每个 $\mathbf{H}_i$ 做特征分解，把负特征值裁剪到零或小正数 $\epsilon$（Teran 等人 2005 的经典做法），从而保证组装后的全局 Hessian 半正定。

作者指出这种裁剪策略的缺陷：当某方向的特征值 $\lambda$ 为很大的负数（对应能量在该方向强凹），把它裁剪到接近零会在求逆时使该方向的系数爆炸（$\mathbf{d} = -\mathbf{H}^{-1}\mathbf{g}$），导致更新方向被这个本应被抑制的非凸方向主导，甚至指向能量上升方向。作者用一个二变量最小例子说明：裁剪到 $\epsilon$ 时更新方向沿坏方向爆炸，而取绝对值则给出更合理的下降方向。

## 方法

整体框架：在标准投影牛顿的逐元素 PSD 投影环节，把「裁剪负特征值」替换为「取特征值绝对值」。

传统裁剪：

$$\lambda_k^+ = \begin{cases} \epsilon & \lambda_k \le \epsilon \\ \lambda_k & \text{otherwise} \end{cases}$$

本文的绝对值投影：

$$\lambda_k^+ = \lvert \lambda_k \rvert$$

代码上仅一行改动：把 `Hi_proj = U*max(Λ,0)*U'` 换成 `Hi_proj = U*abs(Λ)*U'`，无需任何额外参数。该步骤也可解释为一种广义信赖域方法：模型取一阶泰勒展开，信赖域用 Hessian 度量定义。

关键设计——为何绝对值有效（保体积能量的特征分析）：稳定 Neo-Hookean 能量为

$$\Psi = \frac{\mu}{2}(I_C - 3) + \frac{\lambda}{2}(J - \alpha)^2, \quad \alpha = 1 + \frac{\mu}{\lambda}$$

其中 $I_C = \mathrm{tr}(\mathbf{F}^\top \mathbf{F})$，$J = \det(\mathbf{F})$。逐元素 Hessian 的六个 twist/flip 特征值形如

$$\Lambda_{6\sim 8} = \mu - \sigma(\lambda(J-1) - \mu)$$

（$\sigma$ 为形变梯度的奇异值）。当泊松比 $\nu \to 1/2$ 时，第二 Lamé 参数 $\lambda = \frac{2\nu}{1-2\nu}\mu \to \infty$（例如 $\nu = 0.495$ 时 $\lambda \approx 100\mu$）。因此只要体积变化 $(J-1)$ 较大，这些特征值就会变成很大的负数——正是裁剪策略失效、而绝对值投影仍能给出良好下降方向的情形。作者据此论证：大负特征值的出现与高泊松比、大体积变化直接相关，比想象中更常见。

## 实验结果

作者实现基于 C++、libigl 与 TinyAD 自动微分，默认使用稳定 Neo-Hookean 模型（杨氏模量 $E = 10^8$，泊松比 $\nu = 0.495$），在 MacBook Pro（Apple M2，24GB）上测试。在 TetWild Thingi10k 数据集上对 593 个闭合、亏格为零的高分辨率四面体网格施加多种形变对比收敛牛顿迭代数。

| 场景 | clamp 迭代数 | abs（本文）迭代数 |
| --- | --- | --- |
| 大形变娃娃拉扭 90° | 49 | 27 |
| 圆柱拉伸 3.0x | 53 | 25 |
| 圆柱压缩 0.5x | 30 | 26 |
| 拉伸 11x（$\nu = 0.4999$）| 未收敛（>200）| 27 |
| 保体积参数化 | 52 | 24 |

在 Thingi10k 大形变（拉伸、扭转、弯曲）上，本文方法平均获得约 2.5 倍加速，且对压缩等中等体积变化场景至少与裁剪法相当。加速比随网格分辨率、泊松比和体积变化的增大而增大。方法参数无关，而裁剪法的最优 $\epsilon$ 需逐例手工调参。与全局绝对值投影相比，本文的逐元素做法计算可行（0.25 秒/迭代 对比 全局法 10.2 分钟/迭代）；与 Longva 等人的按需投影、加单位矩阵等替代方案相比，避免了过阻尼收敛与额外 Cholesky 分解。

## 亮点与局限

亮点：改动极小（一行代码），无额外参数，每步计算成本与传统投影牛顿一致，却在稳定性和收敛速度上大幅改进；配有对非凸来源（高泊松比 + 大体积变化导致大负特征值）的清晰理论分析；可泛化到 Mooney-Rivlin、ARAP、Symmetric Dirichlet 等保体积能量，以及表面参数化、含碰撞（IPC）等任务。

局限：主要在大形变场景验证；对小形变尤其是压缩情形，本文方法有时会相对裁剪法轻微拖慢收敛。方法聚焦准静态优化，向更广泛有限元仿真（如碰撞）的适用性仍需进一步研究。

## 延伸思考

绝对值投影的本质是让非凸方向的影响与其特征值幅度成正比，而非粗暴地抹平。这一视角提示：Hessian 修正策略并非只有「投影到 PSD 锥」一条路，自适应的 PSD 投影、给弹性能量加高阶正则项、或设计不引入大负特征值的保体积项，都是值得探索的方向。此外，该方法与信赖域框架的联系，也为把优化理论中的鞍点逃逸思想引入图形学仿真提供了桥梁。
