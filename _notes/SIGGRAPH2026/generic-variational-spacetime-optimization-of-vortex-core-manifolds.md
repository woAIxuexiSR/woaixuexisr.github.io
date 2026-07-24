---
title: "Generic Variational Spacetime Optimization of Vortex Core Manifolds"
authors:
  - "Xingdi Zhang"
  - "Peter Rautek"
  - "Markus Hadwiger"
category: "Geometry & Modeling"
track: "Conference"
source: "author-page"
institution: "KAUST"
tags:
  - "Flow Visualization"
  - "Vortex Extraction"
  - "Variational Methods"
  - "Lagrangian Coherent Structures"
  - "Reference Frame Optimization"
links:
  paper: "https://doi.org/10.1145/3799902.3811230"
  project: "https://vccvisualization.org/research/genericvortexcoremanifolds/"
  code: "https://github.com/Cindy-xdZhang/GenericVariationalVortexCore"
---

## 一句话总结

提出一个通用变分框架，把 3D 非定常流动中的涡核建模为时空中的二维流形，并通过"沿流时间预积分"把原本 4D 的时空优化坍缩成单个时间步内的一维空间变分问题，从而高效、客观且拉格朗日一致地提取涡核。

## 研究背景

- 领域现状：涡结构检测是连续介质力学与流场可视化的核心任务。经典欧拉判据（如 $$Q$$、$$\lambda_2$$、平行向量算子）计算快但不满足客观性（不随参考系变化而不变），也不是拉格朗日的（涡核不随流体一起运动）；参考系优化方法（如 Generic Objective Vortices）能恢复客观性，但只关注参考系本身、不保证提取到的特征是拉格朗日的；拉格朗日相干结构（LCS）物理上一致，但通常给出的是输运壁垒（分界面）而非几何涡核线，且需要在整个时空域做昂贵的稠密粒子积分。
- 核心痛点：想同时做到"通用（支持多种涡判据）+ 客观 + 拉格朗日 + 计算高效"仍是开放问题。最大的阻力在于时空优化的复杂度——要在一段时间区间上提取一个"整体最优"的特征，直接在 4D 时空里优化流形代价极高；已有变分方法（Daßler 与 Günther 2024）只在单个时刻处理涡核，缺乏时间相干性；另一些方法强制参考系沿流线走，对 3D 流动是很强的限制。
- 本文 idea：涡核作为拉格朗日特征，其时间演化可以被完全编码进"单个时间步"上定义的变分问题的拉格朗日量 $$L$$ 中。利用底层流动及其推前（pushforward）把能量沿流轨迹预积分，就把一段时间区间上的行为投影到单个时间步上，等效地把时间维"折叠"掉。于是只需在一个时间步内、用一个空间参数求解 Euler-Lagrange 方程，得到的解就是时空上最优的涡核曲面。

## 方法

整体框架：把某时刻 $$t_0$$ 的涡核建模为空间曲线 $$s \mapsto c_{t_0}(s)$$，它随流动平流后在时空 $$\mathbb{R}^3 \times \mathbb{R}$$ 中扫出一张由 $$(s,t)$$ 参数化的二维流形（路径曲面）。方法先在候选点上优化一个刚性参考系（Killing 场），再把随时间变化的涡判据能量沿流预积分成不依赖时间的拉格朗日量，最后在单个时间步内求解一维 Euler-Lagrange 方程得到最优曲线，平流即得时空曲面。

```mermaid
flowchart LR
  A["输入非定常流场 v(x,t)"] --> B["网格候选种子点"]
  B --> C["局部最优参考系 w 与种子点交替优化"]
  C --> D["能量沿流预积分为拉格朗日量 L"]
  D --> E["单参数 Euler-Lagrange 方程 求最优曲线"]
  E --> F["按流图平流曲线 得时空二维流形"]
  F --> G["与共动参考系流线一起可视化"]
```

关键设计：

1. 涡核的时空二维流形模型。定常流中涡核线满足"流体速度平行于沿核线的切向"，即 $$\boldsymbol{v}(c(s)) = k(s)\,\tfrac{d}{ds}c(s)$$；推广到非定常流时，核线本身也会移动，粒子速度需要同时包含沿核线的切向分量和核线整体运动分量。为避免退化并统一定常/非定常情形，作者把涡核显式定义为时空中的二维曲面 $$(s,t) \mapsto (c_t(s),\,t)$$，其两个基向量 $$\boldsymbol{b}_s,\boldsymbol{b}_t$$ 因曲线正则而始终线性无关，即便涡核不随时间移动也不退化。

2. 涡核形变的客观分解。把流速分解为三部分：一个作为底层刚性运动的 Killing 场 $$\boldsymbol{w}$$、一个沿（未知）核线切向的分量 $$k_t(s)\,\tfrac{\partial}{\partial s}c_t(s)$$、以及一个与核线正交的非刚性形变场 $$\tilde{\boldsymbol{u}}$$。只测量与核线正交的形变（允许涡拉伸等沿核线方向的自由运动）。关键在于 $$\tilde{\boldsymbol{u}}$$ 可以只用已知的 $$\boldsymbol{v}$$ 和 $$\boldsymbol{w}$$ 算出、无需知道核线运动场 $$\boldsymbol{u}$$：$$\tilde{\boldsymbol{u}} = (\boldsymbol{v}-\boldsymbol{w}) - \dfrac{\langle \boldsymbol{v}-\boldsymbol{w},\, \dot{\boldsymbol{c}}\rangle}{\lVert \dot{\boldsymbol{c}} \rVert^2}\,\dot{\boldsymbol{c}}$$。由于时空方程里最后一个分量为 0 天然标记"客观向量"、为 1 标记"（非客观）速度向量"，最小化 $$\tilde{\boldsymbol{u}}$$ 的大小是客观的。

3. 局部最优参考系与种子点。在 $$t_0$$ 时刻布一个网格生成候选点，为每个候选点在随其平流的时空邻域 $$U(t)$$ 内求使"观测时间导数"最小的 Killing 场：$$\boldsymbol{w} = \arg\min_{\tilde{\boldsymbol{w}}} \int_{t_0}^{T}\!\!\int_{U(t)} \lVert \tfrac{D}{Dt}(\boldsymbol{v}-\tilde{\boldsymbol{w}}) \rVert^2\, dx\, dt$$。因参考系与种子点位置互相依赖，二者交替优化直至收敛，并用 IVD 与 Sujudi-Haimes 判据阈值过滤候选点。

4. 时间预积分与一维变分求解。核心是把能量 $$E$$ 沿流图预积分定义拉格朗日量：$$L(\boldsymbol{q},\dot{\boldsymbol{q}},s) = \int_{t_0}^{T} E\big(\phi_{t_0,t}(\boldsymbol{q}),\, \phi_{t_0,t*}(\dot{\boldsymbol{q}}),\, t\big)\, dt$$，其中 $$\boldsymbol{q}=c_{t_0}(s)$$、$$\dot{\boldsymbol{q}}\in S^2$$ 为单位切向（用弧长参数）。能量项 $$E = \lVert \tilde{\boldsymbol{u}} \rVert^2 + \mu_c C + \mu_d \lVert D \rVert^2 + \mu_r \lVert R \rVert^2$$ 由形变、通用客观涡判据 $$C$$（如取 $$C=-\mathrm{IVD}$$ 时预积分自动给出 $$-\mathrm{LAVD}$$）、观测时间导数 $$D$$ 与切向正则项 $$R$$ 组成。因 $$L$$ 不显含时间，只需在垂直于 $$\dot{\boldsymbol{q}}$$ 的平面内求解带弧长约束 $$\lVert \dot{\boldsymbol{q}} \rVert = 1$$ 的 Euler-Lagrange 方程 $$P_\perp\big(\tfrac{\partial L}{\partial \boldsymbol{q}} - \tfrac{d}{ds}\tfrac{\partial L}{\partial \dot{\boldsymbol{q}}}\big) - 2\lambda\ddot{\boldsymbol{q}} = 0$$，用四阶 Runge-Kutta 积分这个二阶 ODE 得到曲线，再按流图平流得到时空曲面。作者强调"预积分"是在需要 $$L$$ 及其导数处即时（on the fly）积分，无需在整个相空间预先计算或存储。

## 实验结果

用"流线绕核缠绕数"（winding number，越高说明核线周围旋转越强、越贴合真实涡核；对五个时间步取平均）作为定量指标，对比 VTK 平行向量算子、VDE（Günther 与 Theisel 2025）与本文方法，并在有真值的数据集上给出参考值。

| 数据集 | VTK | VDE | 本文 | 真值 |
|--------|-----|-----|------|------|
| DeltaWing3D | 0.59 | 2.13 | 2.92 | - |
| SteadyTornado3D | 2.96 | - | 2.96 | 2.96 |
| VortexRing | 1.87 | 1.88 | 1.88 | 1.88 |
| Cylinder3D | 1.99 | 2.25 | 2.23 | - |
| TrefoilKnot | 19.86 | 20.02 | 21.01 | - |
| FourCenters3D | 6.34 | 7.22 | 7.22 | 7.22 |

在有真值的三个数据集（SteadyTornado3D、VortexRing、FourCenters3D）上本文均与真值完全吻合；在最具挑战的 DeltaWing3D 上显著优于两个基线。定性方面：定常龙卷风数据上本文核线比 VTK 平滑且无需额外滤波；解析涡环上因不假设"局部直管"能准确恢复弯曲核线的旋转中心；von Kármán 涡街与 F22 战机 CFD 上能提取连续、无漂移、拓扑稳定的时空曲面；Delta Wing 上唯有本文能一直追踪到以不同速度向下游移动的涡尖（需切换到涡尖的局部参考系才能看到其旋转）。

## 亮点与局限

- 亮点：
  - 用"沿流时间预积分"把 4D 时空优化降为单时间步内的一维空间变分问题，效率提升显著，无需昂贵的 4D 全局优化。
  - 通用性强：能量中的涡判据项 $$C$$ 可插入任意客观标量判据，Euler-Lagrange 求解流程与具体判据无关；统一处理定常与非定常流，无需特判。
  - 客观性由构造保证（只要所选判据客观），且提取结果是拉格朗日一致的连续几何曲面，变分形式天然充当几何正则化，抗网格噪声。
- 局限：
  - 依赖在求解前独立确定的 Killing 场 $$\boldsymbol{w}$$，若参考系选得不好会影响结果，这是当前方法的固有限制。
  - 无法处理核线拓扑变化（分叉、合并）——此时会局部破坏拉格朗日相干性，导致 Euler-Lagrange 积分终止，且单一观测场往往无法把区域 $$U(t)$$ 变得足够定常。

## 延伸思考

这项工作延续了 KAUST 课题组"观测者相对流场分析 + 客观特征提取"的系列（观测 Killing 场、Vortex Lens、Vortex Transformer 等），把变分特征提取从单时刻推广到具备时间相干性的时空流形，思路上与"把时间维通过物理演化预积分掉"的降维技巧相通。值得追问的方向：一是能否放松对预先确定 $$\boldsymbol{w}$$ 的依赖，将参考系与核线联合变分求解；二是如何优雅处理拓扑变化（分叉/重联），或许可借鉴隐式涡丝表示与 Clebsch/covector 流体的拓扑处理；三是把该框架从涡核推广到其他随流平流的拉格朗日几何特征（脊线、输运壁垒等）的可能性。
