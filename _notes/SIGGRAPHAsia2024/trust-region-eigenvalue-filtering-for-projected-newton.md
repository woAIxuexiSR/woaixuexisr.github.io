---
title: "Trust-Region Eigenvalue Filtering for Projected Newton"
authors:
  - Honglin Chen
  - Hsueh-Ti Derek Liu
  - Alec Jacobson
  - David I. W. Levin
  - Changxi Zheng
track: "Conference"
source: arxiv
category: "Animation & Simulation"
institution:
  - Columbia University
  - Roblox
  - University of British Columbia
  - University of Toronto
  - Adobe Research
  - NVIDIA
tags:
  - Projected Newton
  - Trust Region
  - Neo-Hookean
  - Hyperelastic Simulation
  - Eigenvalue Filtering
  - Quasistatic Simulation
links:
  paper: "https://doi.org/10.1145/3680528.3687650"
---

## 一句话总结

用广义信赖域框架把"原始牛顿法、特征值 clamp、特征值取绝对值"三种投影牛顿策略统一起来，并据此在优化过程中**自适应**地逐迭代选择投影策略，仅需两行代码改动即可显著加速并稳定 Neo-Hookean 能量的准静态仿真。

## 研究背景

超弹性仿真中，体积保持项的高度非凸性会让常用的投影牛顿（Projected Newton）优化器在**高泊松比（接近 0.5）和大初始体积变化**下变得不稳定、收敛缓慢。投影牛顿依赖特征值滤波来强制 Hessian 正定以保证收敛，主流做法有两类：

- **特征值 clamp**（Teran et al. 2005）：把每个单元 Hessian 的负特征值截断到零或一个小正数 $$\epsilon$$，再重组全局 Hessian，能保证全局正定。
- **特征值取绝对值**（Chen et al. 2024）：把负特征值投影为其绝对值 $$\lambda_k^+ = \vert \lambda_k\vert $$。在近不可压场景能带来数量级加速，但在近凸场景又会沿负曲率方向过度阻尼、拖慢收敛。

问题在于：**没有单一策略在所有场景都最优**，如何选择投影策略一直是开放问题。本文正是要给出一个不需要事先选定、能在优化中自动适配的方案。

## 方法

### 整体框架

作者的核心洞见是：把逐单元的特征值滤波纳入**广义信赖域**（generalized trust region）视角。考虑上下界对称的广义信赖域子问题，取 $$b=0$$、$$A=H$$，约束写成绝对值形式：

$$\min_{u}\; f(x) + g^\top u + u^\top H u \quad \text{s.t.}\ \vert u^\top H u\vert  \le \Delta.$$

利用有限元 Hessian 可按单元求和的性质，引入引理 $$\vert x^\top A x\vert  \le x^\top \vert A_e\vert  x$$（其中 $$\vert A_e\vert =\sum_i P_i^\top \vert A_i\vert  P_i$$ 为逐单元取绝对值），把约束收紧为 $$u^\top \vert H_e\vert  u \le \Delta$$。当约束取等号并用拉格朗日乘子求解时，得到步长（把标量折进线搜索后）：

$$u = -\big((1-w)H + w\,\vert H_e\vert \big)^{-1} g,\qquad w = \frac{\lambda}{1+\lambda}\in[0,1].$$

于是不同的权重 $$w$$ 恰好对应不同的已有策略——这就是三者的统一。

```mermaid
flowchart TD
    A[广义信赖域子问题<br/>约束 |uᵀHu| ≤ Δ] --> B[拉格朗日求解<br/>u = -((1-w)H + w|Hₑ|)⁻¹g]
    B --> C0[w=0: 原始牛顿步<br/>u = -H⁻¹g]
    B --> C1[w=0.5: 特征值 clamp<br/>u = -Hₑ⁺⁻¹g]
    B --> C2[w=1: 取绝对值<br/>u = -|Hₑ|⁻¹g]
    D[信赖域比率 ρ] --> E{|ρ-1| ≤ ε ?}
    E -- 是, 拟合好 --> C1
    E -- 否, 拟合差 --> C2
```

### 关键设计一：三种策略的统一

论文证明三个特殊权重分别退化为经典方法：

- $$w=0$$（即 $$\lambda=0$$）：原始（未投影）牛顿步 $$u=-H^{-1}g$$。
- $$w=0.5$$（即 $$\lambda=1$$）：借助引理 $$u^\top H u + u^\top\vert H_e\vert u = 2\,u^\top H_e^+ u$$，退化为 clamp 步 $$u=-(H_e^+)^{-1}g$$。
- $$w=1$$（即 $$\lambda\to\infty$$）：退化为取绝对值步 $$u=-\vert H_e\vert ^{-1}g$$，等价于用一阶模型加 Hessian 度量约束。

### 关键设计二：基于信赖域比率的自适应选择

借鉴信赖域文献中"二次模型拟合好就放大信赖域、拟合差就缩小"的思路，用经典比率衡量二次模型与真实能量的吻合度：

$$\rho = \frac{f(x) - f(x+u)}{\tilde f(x) - \tilde f(x+u)}.$$

$$\rho$$ 接近 1 说明二次近似准确（可用大信赖域），远离 1（趋零或为负）说明近似差（需小信赖域）。据此离散地选权重：

$$w = \begin{cases} 0.5, & \vert \rho-1\vert \le\epsilon \\ 1, & \vert \rho-1\vert >\epsilon \end{cases}$$

其中 $$\epsilon$$ 取 0.01 到 0.1 之间的小常数。作者选离散 $$w$$ 而非精确解信赖域子问题，是为了避免每次牛顿迭代都做昂贵的迭代求解，保持与原始牛顿同样的单迭代成本。

### 关键设计三：两行代码实现

最终逐单元投影只需：

$$\Lambda^+ = \begin{cases} \max(\Lambda, 0), & \vert \rho-1\vert \le\epsilon \\ \vert \Lambda\vert , & \vert \rho-1\vert >\epsilon \end{cases}$$

在已有投影牛顿框架里，只需加一行计算信赖域比率 $$\rho$$、把原本固定的特征值投影替换为上式的三目选择即可。实现上第一次牛顿迭代总是从取绝对值策略起步，之后用 $$x_k$$ 与 $$x_{k-1}$$ 算出的 $$\rho_k$$ 决定第 $$k$$ 次迭代的策略。它在形式上仍是（正则化的）投影牛顿，正则参数模拟了信赖域子问题里的拉格朗日乘子。

## 实验结果

在 C++ + TinyAD 上实现，采用稳定 Neo-Hookean 能量（杨氏模量 $$10^8$$、不同泊松比），收敛判据为牛顿减量 $$0.5\,u^\top g < 10^{-5}$$，最多 200 次迭代。下表为 teaser 与 horse 例子中"每次牛顿迭代的平均线搜索次数"，数值越小说明下降方向越有效：

| 场景 | teaser adaptive | teaser abs | teaser clamp | horse adaptive | horse abs | horse clamp |
|------|------|------|------|------|------|------|
| 大形变 PR=0.495 | 1.0 | 1.0 | 7.5 | 1.8 | 1.0 | 7.4 |
| 大形变 PR=0.3 | 1.0 | 1.0 | 4.0 | 1.5 | 1.0 | 5.0 |
| 小形变 PR=0.495 | 1.4 | 1.0 | 4.1 | 1.0 | 1.0 | 1.0 |
| 小形变 PR=0.3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

自适应方法在各种泊松比和形变下都保持极低的线搜索次数。单迭代平均耗时上，adaptive 为 0.117 秒（其中信赖域比率仅占 6.0%），与 abs（0.107 秒）、clamp（0.126 秒）同量级，说明自适应几乎不增加单步成本。在 TetWild Thingi10k 数据集 254 个高分辨率网格上的大规模实验（拉伸/压缩/弯曲/扭转）表明：在高泊松比和大形变下相较 clamp 与 abs 有显著加速，在小形变/低泊松比下也与现有方法相当。

## 亮点与局限

**亮点：**
- 首次把原始牛顿、clamp、取绝对值三种投影策略在广义信赖域框架下统一，理论优雅。
- 自适应策略在"某一策略全程占优"时至少与最优策略相当，避免了事先选错策略的风险。
- 实现极简（两行代码），单迭代成本不变，易于集成进现有仿真管线；对网格分辨率、四面体剖分、多种超弹性变体（Mooney-Rivlin、ARAP、Symmetric Dirichlet 加体积项）均鲁棒。

**局限：**
- 主要在准静态、稳定 Neo-Hookean 能量上验证；动态仿真、碰撞（IPC）场景尚未深入，碰撞后线搜索会主导收敛。
- $$w$$ 为离散取值（0.5 或 1），非连续函数，可能不是最细粒度的最优控制。
- 第一次迭代固定从取绝对值起步，在 clamp 一直占优的简单场景会多花一次迭代。
- 需要人工选取阈值 $$\epsilon$$，尤其小压缩场景对阈值较敏感。

## 延伸思考

作者指出把 $$w$$ 推广为 $$[0,1]$$ 上的连续函数、以及对每个单元独立计算信赖域比率与投影策略，都可能进一步提升收敛。对动态仿真（小时间步下全局 Hessian 往往"意外正定"）而言，允许 $$w\in[0,0.5]$$ 或许更快——这也呼应了 Longva et al. 2023 的按需投影思路。更广地看，本文提供了一个"以二次近似质量为依据分析 Hessian 投影方案"的通用视角：把优化诊断量（信赖域比率）反馈回数值格式的选择，这种"用运行时信号自适应切换数值策略"的模式，或许能迁移到障碍函数、接触求解乃至更一般的非凸物理优化中。同时，如何为首次迭代设计更好的初始投影策略，也是一个值得深挖的小而实用的方向。
