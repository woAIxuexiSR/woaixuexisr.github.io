---
title: "A Differential Monte Carlo Solver For the Poisson Equation"
authors:
  - "Zihan Yu"
  - "Lifan Wu"
  - "Zhiqian Zhou"
  - "Shuang Zhao"
category: "Animation & Simulation"
track: "Conference"
source: "author-page"
institution: "University of California, Irvine"
tags:
  - "Monte Carlo Method"
  - "Poisson Equation"
  - "Walk On Spheres"
  - "Differentiable PDE Solver"
  - "Shape Optimization"
links:
  paper: "https://doi.org/10.1145/3641519.3657460"
  project: "https://www.shuangz.com/projects/diff-wos-sg24/"
---

## 一句话总结

本文提出一种无网格的蒙特卡洛方法，用于对泊松方程（带 Dirichlet 边界条件）的解关于任意参数（包括域的形状）求导，核心是把导数改写为一个边界积分，并配合 walk-on-spheres 过程与新的法向导数估计器，从而以更低的方差支持可微 PDE 求解与逆问题优化。

## 研究背景

泊松方程是一类椭圆型偏微分方程，广泛用于建模热传导、流体动力学、静电学等现象，在物理、工程与计算机图形学中应用众多。传统求解方式（如有限元 FEM）需要先离散化域或边界，再求解线性方程组；当域的几何非常复杂时，离散化代价极高，严重限制了实用性。

近年来，一族基于 walk-on-spheres（WoS）过程的无网格蒙特卡洛方法被引入图形学，用于求解二阶线性椭圆 PDE。它们无需离散化域或边界，因而能很好地扩展到复杂几何域。然而，这些进展主要集中在"前向"求解上，可微的无网格求解器仍然缺乏。

本文聚焦的问题是：对泊松方程的解关于任意参数（尤其是域的形状）求导。能够高效估计这些导数，是求解逆泊松问题（即根据已知解场反推源项、边界条件或域形状）的关键。此前 PDE 解的导数大多依赖离散化方法（对离散化与线性求解两步施加自动微分），难以扩展到复杂几何。

## 方法

### 整体框架

带 Dirichlet 边界条件的泊松方程写作：

$$\Delta u = -f \ \text{on}\ \Omega, \qquad u = g \ \text{on}\ \partial\Omega$$

其中 $\Omega \subset \mathbb{R}^n$（本文关注 $n=2,3$），$f$ 为源函数，$g$ 为边界函数。前向解可用 WoS 过程通过积分方程递归估计。

本文的目标是求解导数 $\partial_\theta u$。当参数 $\theta$ 只控制 $f$ 与 $g$、不改变域时，导数本身满足一个源项与边界项被微分后的新泊松方程，可用同样的 WoS 求解；难点在于当域 $\Omega(\theta)$ 随参数演化时的一般情形。直接用 Reynolds 输运定理去微分积分方程，会因积分球 $B_x$ 与 $\epsilon$-壳对 $\theta$ 的依赖而产生难以处理的不连续项。

作者的核心思路是：先用固定的参考域重参数化演化域，再推导出一个只含边界积分的导数表达式，从而避开上述不连续，并让蒙特卡洛估计器显著更高效。

### 关键设计

材料形式参数化。设参考域 $\hat{\Omega} := \Omega(0)$ 独立于参数 $\theta$，用一个一一映射 $X(\cdot,\theta)$ 把参考域变换到演化域 $\Omega(\theta)$。任意标量场经拉回算子 $X^*$ 变换回参考域：$(X^* h)(p) := h(X(p,\theta))$。在 $\theta=0$ 处，映射及拉回算子均退化为恒等映射。

边界积分公式。设 $\hat{u} := X^* u$。通过对材料导数应用链式法则并结合参考域上的表示公式，作者推导出导数最终可写成单个边界积分：

$$(\partial_\theta u)(p) = \int_{\partial\hat{\Omega}} P_{\hat{\Omega}}(p \to s)\left[\partial_\theta \hat{g}(s) - v(s)\cdot \nabla_{\hat{\Omega}}\hat{u}(s)\right]\, \mathrm{d}s$$

其中 $P_{\hat{\Omega}}$ 是参考域的泊松核，$v(p) := \partial_\theta X(p,\theta)$ 表示点随参数变化的"速度"。这一公式把体积分转化为边界积分，是全文方法效率提升的基础。

差分 walk-on-spheres 估计器。针对上式，作者设计了三步蒙特卡洛估计流程：

```mermaid
flowchart LR
    A["给定评估点 p"] --> B["S.1 用 WoS 从边界采样 s<br/>密度等于 P(p→s)"]
    B --> C["S.2 用第二个 WoS<br/>估计边界梯度 ∇u(s)"]
    C --> D["S.3 返回 ∂θ ĝ(s) − v(s)·∇u(s)<br/>ĝ 与 v 用自动微分解析求得"]
    D --> E["(∂θ u)(p) 的无偏估计"]
```

- S.1 边界点采样：泊松核 $P_{\hat{\Omega}}$ 一般无解析形式，作者用 WoS 算法模拟从 $p$ 出发的布朗运动粒子，取其首次抵达边界的位置。由 Kakutani 原理，当 $\epsilon \to 0$ 时该输出恰好服从密度 $P_{\hat{\Omega}}(p \to s)$。
- S.2 边界梯度估计：梯度分解为切向的表面梯度与法向导数，$\nabla u(s) = \nabla_{\partial\hat{\Omega}} u(s) + n(s)\,\partial_{n(s)}u$。表面梯度可由边界函数 $g$ 的梯度投影到切空间解析得到；难点是法向导数 $\partial_{n(s)}u$，作者为其设计了新的蒙特卡洛估计器。

法向导数的新估计器。作者在与边界相切于 $s$ 的球 $B_c \subset \hat{\Omega}$ 上建立法向导数的积分关系，并引入控制变量以消除泊松核与差分核在 $y=s$、$z=s$ 处的奇异性：

$$\partial_{n(s)}u = \int_{B_c}\!\big(f(y)-f(s)\big)P_{B_c}(y\to s)\,\mathrm{d}y + \frac{f(s)R}{n} + \int_{\partial B_c}\!\big(u(z)-g(s)\big)\partial_{n(s)}P_{B_c}(s\to z)\,\mathrm{d}z$$

其中 $R = \lVert c-s\rVert$ 为球半径，$n$ 为维度。控制变量使得内部项与边界项的被积函数在 $y \to s$、$z \to s$ 处收敛。作者进一步对边界项采用对偶采样（antithetic sampling），并用二分法搜索最大的相切球以降低方差。整个估计器在 GPU 上以 Dr.Jit 作为数值后端实现。

## 实验结果

作者在若干合成算例上评估方法。法向导数估计的消融实验对比了基线方法、无对偶采样版本、小球版本与完整方法，在圆盘上的 2D Laplace 问题与四叶草形域上的 2D 泊松问题中，完整方法方差最低。

完整估计器在可微 PDE 求解与逆问题上与有限差分（FD）参考、基线方法（用 Reynolds 输运定理直接微分积分方程）对比：

| 算例 | 问题类型 | 结论 |
| --- | --- | --- |
| Wrench（2D） | 2D Laplace | 结果贴合 FD 参考，等时间下显著比基线更干净 |
| Teapot（2D） | 2D Laplace | 同上 |
| Globe（2D） | 2D 泊松（正弦波源） | 同上 |
| Bunny（3D） | 3D Laplace | 同上 |

在逆问题中，作者用本方法与基线生成的梯度分别优化域形状相关参数（Wrench 与 Globe 的旋转角、Bunny 的位姿、Diffusion curve 的边界折线形状）。所有情形下本方法都能平滑收敛到真值，而基线方法因方差过高而失败。

## 亮点与局限

亮点在于把域形变下泊松方程解的导数化归为单个边界积分，绕开了 Reynolds 输运定理带来的不连续项；配套的法向导数估计器通过新形式的控制变量与对偶采样大幅降低方差，且整套方法无网格、可对任意参数（含形状）求导，天然适配复杂几何与逆问题。

局限方面：前向求解仅使用了最基础的 WoS，尚未整合双向采样、缓存或 walk-on-boundary 等更先进变体；法向导数估计中的内部与边界积分都采用均匀采样，采样策略仍有改进空间；方法目前只针对带 Dirichlet 边界条件的泊松方程，推广到更广泛的 PDE（如 screened 泊松）与其他边界条件（如 Neumann）仍是未来工作。

## 延伸思考

这项工作把差分渲染中成熟的方差缩减思想（控制变量、对偶采样、边界积分形式化）迁移到了 PDE 求导领域，提示"可微蒙特卡洛"作为一种通用方法论，可能在几何处理、物理仿真、逆设计等更多需要对形状/参数求梯度的场景中复用。将其与更高效的前向求解器、更优的重要性采样，以及 Neumann 边界、screened 泊松等更一般的 PDE 结合，或许能催生一整套无网格、可微、可扩展到高细节几何的科学计算与逆问题求解工具链。
