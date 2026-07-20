---
title: "Painless Differentiable Rotation Dynamics"
authors:
  - Magí Romanyà-Serrasolsas
  - Juan J. Casafranca
  - Miguel A. Otaduy
category: Animation & Simulation
track: Journal
source: author-page
institution: Universidad Rey Juan Carlos
tags:
  - Rigid Body Dynamics
  - Differentiable Simulation
  - Lie Theory
  - Rotation Derivatives
  - Cosserat Rods
  - Adjoint Method
  - Incremental Potential
links:
  paper: https://doi.org/10.1145/3730944
  project: https://mslab.es/projects/Painless/
  code: https://gitlab.com/mslab-urjc/mandos/-/tree/SIGG25
---

# Painless Differentiable Rotation Dynamics

## 一句话总结

用李代数（Lie algebra）的旋转导数来做刚体的前向与可微分仿真，配合把时间步更新写成"状态 + 步长（step）"而非"状态 + 速度"的关键选择，让涉及旋转的梯度和 Hessian 变得极其简洁、条件数更好、运行更快，相比流行的旋转向量（rotation vector）参数化最高可提速约 4.8×。

## 研究背景

旋转生活在特殊正交群 $SO(3)$ 上，这是一个非线性流形，与欧氏空间里的平移完全不同。常见的旋转表示都各有痛点：欧拉角有奇异性，旋转矩阵要满足正交且行列式为 1 的约束，四元数要归一化。一旦需要对旋转求导，这些麻烦会被放大——微分量本身受约束，且不能像欧氏量那样直接相加。

近年图形学里旋转向量（轴角）很流行，因为它参数化在 $\mathbb{R}^3$，无需额外约束，微分域里的量可以直接相加，再通过指数映射（Rodrigues 公式）还原成合法旋转。但 Rodrigues 公式的非线性会带来作者所称的"求导地狱"：无论是前向动力学的代表作 Rigid IPC，还是可微分动力学的代表作 ADD，都不得不在附录里堆砌复杂推导，往往依赖自动微分并用特殊的 Taylor 近似来处理不定型（indeterminate forms），代价是计算开销和数值稳定性。

本文指出，一个在图形学里长期被忽视的工具——李理论——能够优雅地处理"对旋转求导"和"旋转对某量求导"。核心洞见是：旋转的微分可以在 $SO(3)$ 局部的切空间上进行，微分量本身不是旋转，但可以通过指数映射转成旋转再与其他旋转复合。

## 方法

### 整体思路

```mermaid
flowchart TD
    A[旋转生活在 SO(3) 非线性流形] --> B[李代数 so(3): 单位元处的切空间, 同构于 R^3]
    B --> C[李导数: 在切空间上用常规向量微分]
    C --> D[前向仿真: 增量势能优化时间积分 + 牛顿法]
    C --> E[可微分仿真: 状态伴随 + 步长伴随]
    D --> F[状态与步长均用旋转矩阵参数化]
    E --> F
    F --> G[导数极简, 无不定型, 条件数好]
    G --> H[刚体动力学 / Cosserat 杆 多刚体动力学]
```

### 李导数（Lie derivatives）

李代数 $\mathfrak{so}(3)$ 是 $SO(3)$ 在单位旋转处的切空间，由反对称矩阵构成，且是一个线性向量空间，$\mathfrak{so}(3)\cong\mathbb{R}^3$。它可以用微分旋转向量 $\theta$ 参数化。配合指数映射与对数映射在群和代数之间往返：

$$\exp:\mathfrak{so}(3)\to SO(3), \qquad \log:SO(3)\to\mathfrak{so}(3)$$

对一个自变量旋转 $R$，定义微分向量 $\theta_R$，通过指数映射与复合得到新旋转 $R^+=\exp(\theta_R)\,R$。当因变量也是旋转 $Q$ 时，李导数定义为切空间之间的常规向量微分：

$$\frac{DQ}{DR}\equiv\frac{\partial\theta_Q}{\partial\theta_R}=\left.\frac{\partial\log\left(Q(\exp(\theta_R)R)\,Q(R)^T\right)}{\partial\theta_R}\right\vert _{\theta_R=0}$$

当因变量是普通向量 $u$ 时：

$$\frac{Du}{DR}\equiv\frac{\partial u}{\partial\theta_R}=\left.\frac{\partial u(\exp(\theta_R)R)}{\partial\theta_R}\right\vert _{\theta_R=0}$$

关键的计算便利：对"旋转对旋转"的李导数，指数映射和对数映射会在几步运算后相互抵消；对"向量对旋转"的李导数，只需把 $\exp(\theta_R\to0)$ 近似到所需导数的阶数即可。文中采用左乘微分旋转（在世界坐标系中表达）。

### 与旋转向量的对比

若用旋转向量 $r$ 参数化 $R=\exp(r)$，则任意函数 $f(R(r))$ 需要链式法则 $\frac{\partial f}{\partial r}=\frac{\partial f}{\partial R}\frac{\partial R}{\partial r}$，而 $\frac{\partial R}{\partial r}$ 来自 Rodrigues 公式的求导：

$$R(r)\equiv\exp(r)=I+\sigma(r)\,\mathrm{skew}(r)+\tfrac{1}{2}\sigma^2\!\left(\tfrac{r}{2}\right)\mathrm{skew}(r)^2$$

其中 $\sigma(r)=\mathrm{sinc}(\lVert r\rVert)$。对指数映射 Jacobian $\frac{\partial\,\mathrm{vec}(\exp(r))}{\partial r}$ 做奇异值分析发现：随着旋转角增大，奇异值逐渐衰减，$\sigma_2=\sigma_3$ 在 $\lVert r\rVert=2\pi$ 处甚至归零。这解释了旋转向量导数的复杂与病态。因此文中所有对比实验都把旋转向量限制在 $\lVert r\rVert\le\pi$。

### 前向仿真：状态与步长

采用基于增量势能的后向欧拉优化积分。状态记为 $q$，步长记为 $\Delta q$（相邻采样间的状态变化）。对旋转分量，步长定义为增量旋转：

$$\Delta R_k = R_k R_{k-1}^T$$

这是与主流方法的关键区别：不用速度（角速度）而用步长来离散时间，这样所有与朝向相关的运动学都能用旋转表达，从而在伴随推导中充分发挥李导数的威力。

时间积分被写成优化问题，其最优性条件给出前向动力学的非线性方程组：

$$q_k=\arg\min\Psi(q_k,q_{k-1},\Delta q_{k-1}),\qquad \frac{\partial\Psi}{\partial q_k}=0$$

对旋转分量，梯度就是李导数 $\frac{D\Psi}{DR_k}\equiv\frac{\partial\Psi}{\partial\theta_{R_k}}$，其物理意义恰是力矩。惯性项写成：

$$\Psi_R=-\frac{1}{h^2}\mathrm{tr}\!\left(R_k J\tilde R_k^T\right),\qquad \tilde R_k=\left(2I-\Delta R_{k-1}^T\right)R_{k-1}$$

用李导数求得的惯性项梯度和 Hessian 出奇地简洁：

$$\frac{D\Psi_R}{DR_k}=2\,\mathrm{skew}^{-1}(A_R)^T,\qquad \frac{D^2\Psi_R}{DR_k^2}=\mathrm{tr}(S_R)\,I-S_R$$

其中 $\frac{1}{h^2}R_kJ\tilde R_k^T=S_R+A_R$ 分解为对称部分 $S_R$ 与反对称部分 $A_R$。这远比旋转向量的对应表达式简单，且不会像 Rigid IPC 那样因参数化导致惯性 Hessian 变得不定而需要额外稳定化处理。牛顿迭代求解后，旋转的试探更新通过 $R_k^*\leftarrow\exp(\alpha\,\delta R_k)R_k^*$ 施加线搜索权重。

### 可微分仿真：状态伴随与步长伴随

可微分仿真被看作带约束的优化：目标 $g(\{q_k\},\gamma)$，参数 $\gamma$，约束为前向动力学最优性条件与步长计算。步长约束求导给出惊人简单的结果：

$$\frac{\partial f}{\partial R_k}\equiv\frac{D\Delta R_k}{DR_k}=I,\qquad \frac{\partial f}{\partial R_{k-1}}\equiv\frac{D\Delta R_k}{DR_{k-1}}=-\Delta R_k$$

作者定义状态伴随 $a_k$ 与步长伴随 $\Delta a_k$，递归反向更新：

$$a_{k-1}=z_k\frac{\partial^2\Psi}{\partial q_k\partial q_{k-1}}+\Delta a_k\frac{\partial f}{\partial q_{k-1}}+\frac{\partial g}{\partial q_{k-1}},\qquad \Delta a_{k-1}=z_k\frac{\partial^2\Psi}{\partial q_k\partial\Delta q_{k-1}}$$

其中 $z_k\frac{\partial^2\Psi}{\partial q_k^2}=-a_k-\Delta a_k$。目标梯度为 $\frac{dg}{d\gamma}=\frac{\partial g}{\partial\gamma}+\sum_k z_k\frac{\partial^2\Psi}{\partial q_k\partial\gamma}$。步长 Jacobian $\frac{\partial f}{\partial q_{k-1}}$ 有优美的几何意义：负责在不同时间步的切空间之间"搬运"反向传播的梯度。

### 推广到弹性/阻尼势能与 Cosserat 杆

对形如 $\Psi_p(p_k=R_k\bar p)$ 的一般势能（弹性、重力、增量阻尼与摩擦），李导数可由常规梯度/Hessian 推广：

$$\frac{D\Psi_p}{DR_k}=-\frac{\partial\Psi_p}{\partial p}\mathrm{skew}(p)$$

作者用 Spillmann 与 Teschner 的 Cosserat 杆模型作为多刚体离散的范例：位置采样在节点、朝向采样在边中心，拉伸/剪切/弯扭能量都用李导数求导，成功复现了 Kirchhoff 杆的螺旋反转（helical perversion）现象。完整实现开源在 GitLab 的 mandos 仓库。

## 实验结果

所有李导数都用有限差分验证过，且在可微分仿真中确认李导数、旋转向量、有限差分三者得到的目标梯度在数值精度内一致（旋转向量用 TinyAD 自动微分）。实验在 AMD Ryzen 7 6800HS（8 核）上单线程运行，优化用 SciPy 的 L-BFGS。

前向刚体仿真（复现 Ferguson 等人的摩擦接触基准）比其报告结果快约 10×，但作者说明自己用的是基于符号距离场的二次接触势能，而非屏障势能。

Cosserat 杆前向仿真性能对比（忠实于表 1）：

| 例子 | #q | h | 帧数 n | 装配/iter (μs) 旋转向量 | 装配/iter (μs) 本文 | 求解/iter (μs) 旋转向量 | 求解/iter (μs) 本文 | 总仿真 (s) 旋转向量 | 总仿真 (s) 本文 |
|---|---|---|---|---|---|---|---|---|---|
| 螺旋反转 | 969 | 0.01 | 1000 | 2910 | 608 | 195 | 191 | 81 | 13 |
| 结肠镜 | 969 | 0.001 | 2000 | 2970 | 671 | 753 | 464 | 31.1 | 7.2 |

装配（梯度 + Hessian）主导仿真开销；系统求解因矩阵呈窄带状而更便宜。每次牛顿迭代的平均装配开销提速 4.4–4.8×，总仿真开销提速 4.3–6.2×。为剥离自动微分与牛顿迭代次数的干扰、单看导数表达式复杂度的影响，作者在只含惯性项的蝴蝶螺栓例子上用解析导数测量：旋转向量导数每次装配慢 2.17×（505 ns 对 233 ns）。此外，蝴蝶螺栓验证了网球拍定理，本文每步只需 1 次牛顿迭代，旋转向量需要 2 次。

可微分仿真：
- 掷骰子控制初始角速度，使其最终掷出 6 点且 5 点面朝相机——仅 12 次迭代、565 ms 收敛。
- "SIGGRAPH" 拼字（优化直杆初速度）——总体提速 4.8×（1m 21s）。由于目标梯度与旋转向量在数值精度内一致，两种方法收敛迭代次数相同，加速完全来自前向求解更快。各字母优化时间对比（忠实于表 3，秒）：S 26.6→5.6，I 1.8→0.5，G 37.7→7.5，R 44.1→9.1，A 75.2→16.1，P 152.2→32.0，H 49.0→10.5。
- 还验证了控制力优化（拉索控制杆尖轨迹）与静止形状优化（控制杆的本征 Darboux 帧）。结肠镜例子整体提速 4.4×。

## 亮点与局限

亮点：
- 把李理论这一被图形学忽视的工具系统性地引入前向与可微分刚体动力学，惯性项、步长约束的梯度/Hessian 都"简单到令人尴尬"，无不定型、可解析、可解释。
- "状态 + 步长"取代"状态 + 速度"的看似微小却关键的选择，让状态和步长都能用旋转表达，从而在伴随计算里彻底释放李导数的威力，步长 Jacobian 还获得了"跨时间步切空间搬运梯度"的几何解释。
- 更好的数值条件数：避免了旋转向量参数化导致的病态与惯性 Hessian 伪不定问题，牛顿迭代次数减少。
- 通用性强：涵盖初始条件、控制力、静止形状等各类可微分仿真用例，并推广到 Cosserat 杆这类多刚体动力学。

局限：
- 本文只需要到一阶旋转导数（借助指数/对数映射抵消而易处理），更高阶的旋转导数可能带来额外复杂度。（对欧氏空间函数的二阶 Hessian 不受此限。）
- 现成的自动微分与数值优化工具不支持旋转的李导数（它们基于欧氏微分规则），需要先代入指数映射的近似，或将来扩展工具以识别自变量/因变量的旋转语义。
- 接触问题的优化可能收敛到远离全局最优的局部极小，根源是接触的不连续性破坏了梯度优化的假设。

## 延伸思考

这项工作的价值在于"以正确的数学结构换取工程上的简洁"。图形学社区多年来为了绕开旋转向量的 Rodrigues 求导而堆砌 Taylor 近似、稳定化技巧和自动微分开销，而本文表明只要把导数放到 $SO(3)$ 的切空间上，很多复杂度会自然消失。"状态 + 步长"的重参数化尤其值得借鉴——一个恰当的变量选择能把非线性约束的 Jacobian 变成近乎恒等的形式，这类"选对坐标"的思路在其他带流形约束的可微分仿真（如可微分渲染中的相机位姿、四元数姿态估计）里也可能有类比价值。

由于可微分多体动力学是机器人仿真学习的核心组件，这套更快、更易实现的公式对训练 AI 控制器的场景有直接吸引力。一个自然的下一步是将其与支持任意接触与优化参数的通用可微分求解器（如 Huang 等人的工作）结合，或推动主流自动微分框架原生支持李导数语义，从而让"无痛旋转求导"不再依赖手工推导。
