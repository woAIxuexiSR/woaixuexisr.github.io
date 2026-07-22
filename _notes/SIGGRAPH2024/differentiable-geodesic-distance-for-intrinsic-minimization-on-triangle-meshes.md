---
title: "Differentiable Geodesic Distance for Intrinsic Minimization on Triangle Meshes"
authors:
  - "Yue Li"
  - "Logan Numerow"
  - "Bernhard Thomaszewski"
  - "Stelian Coros"
category: "Geometry & Modeling"
track: "Journal"
source: "arxiv"
institution: "ETH Zurich"
tags:
  - "Geodesic Distance"
  - "Intrinsic Minimization"
  - "Differentiable Simulation"
  - "Triangle Mesh"
  - "Newton Method"
  - "Embedded Elasticity"
  - "Voronoi Diagram"
  - "Karcher Means"
links:
  paper: "https://doi.org/10.1145/3658122"
  code: "https://github.com/liyuesolo/DifferentiableGeodesics"
---

## 一句话总结

本文提出一种可微测地距离框架：以测地路径端点为唯一自由度，利用最短路径测地线的变分表述和隐函数定理，闭式求出测地距离的一阶、二阶导数，从而首次把牛顿型二阶优化引入三角网格上的内蕴（intrinsic）距离最小化问题，在测地弹性网络、双向耦合仿真、可微测地 Voronoi 图和 Karcher 均值等一系列任务上，收敛速度大幅超越现有一阶与拟牛顿方法。

## 研究背景

在离散曲面上计算内蕴距离是几何处理及诸多领域最小化问题的核心：曲面上的生物薄膜、包裹肌肉的筋膜、紧身衣物等，本质上都是在离散流形上最小化长度的变分问题。求解这类问题需要计算曲面上的测地距离及其导数。

难点在于：梯度虽然容易得到，但一阶方法（梯度下降）收敛性通常很差，尤其在带刚性连接的嵌入弹性问题上；L-BFGS 之类的拟牛顿方法能提供一定加速，但对刚性问题仍然收敛缓慢。要用牛顿法就必须拿到测地距离的二阶导数，而这在三角网格上极为困难——测地路径通常横跨多个三角面并与网格边相交，需要显式追踪数量不定的交点变量，非常繁琐。作者指出，此前没有任何工作能计算三角网格上测地距离的解析导数。

一个关键观察支撑了整套方法：虽然测地路径一般不是端点的连续函数，但测地距离处处至少 $C^0$ 连续，在测地线唯一时甚至无限光滑。作者进一步分析了两类梯度不连续，其中一类可以通过 mollification（磨光）解决，另一类只在距离局部极大处（cut locus，割迹）出现、在极小值附近不会发生，因此可以忽略。

## 核心方法

### 可微测地距离

方法建立在 MMP 算法 [Mitchell et al. 1987] 精确计算测地路径的基础上。测地路径端点用宿主三角网格的重心坐标 $\mathbf{w}$ 表示，是唯一的仿真自由度。路径与网格边的交点 $\mathbf{x}$ 用标量参数 $t_i$ 沿边 $e_{jk}$ 参数化：

$$\mathbf{x}_i = \mathbf{v}_j + t_i(\mathbf{v}_k - \mathbf{v}_j)$$

两点 $\mathbf{c}_0, \mathbf{c}_1$ 之间的测地距离是路径各线段长度之和：

$$g(\mathbf{c}_0, \mathbf{c}_1) = \sum_i l_i(\mathbf{c}_0, \mathbf{c}_1, \mathbf{x}(\mathbf{c}_0, \mathbf{c}_1))$$

其中端点重心坐标 $\mathbf{w}$ 与端点空间坐标 $\mathbf{c}$ 的关系是显式的（重心插值），而交点变量 $\mathbf{t}$ 与端点的关系是隐式的（由精确测地算法确定）。所有量最终只依赖 $\mathbf{w}$。

### 用最优性条件简化导数

直接对 $g(\mathbf{c}(\mathbf{w}), \mathbf{x}(\mathbf{t}(\mathbf{c}(\mathbf{w}))))$ 求一二阶导会得到冗长的链式表达式。作者的关键简化来自最短测地线是距离的局部极小这一事实，给出一阶最优性条件：

$$\frac{\mathrm{d}g}{\mathrm{d}\mathbf{t}} = \frac{\partial g}{\partial \mathbf{x}}^{\mathsf{T}} \frac{\partial \mathbf{x}}{\partial \mathbf{t}} = 0$$

直观地说：内部交点满足最优性时，对交点作扰动到一阶不改变路径长度（沿路径移动使相邻线段长度变化相互抵消，垂直路径移动只是旋转线段、到一阶不变长）。因此一阶导数被大幅简化为只涉及首末线段的长度梯度：

$$\frac{\mathrm{d}g}{\mathrm{d}\mathbf{w}} = \frac{\partial g}{\partial \mathbf{c}}^{\mathsf{T}} \frac{\partial \mathbf{c}}{\partial \mathbf{w}}$$

### 隐式微分求 Hessian

二阶导数中除 $\partial \mathbf{t} / \partial \mathbf{c}$（交点关于端点的导数）外均可代数求得。显式计算这一项需要对整个精确测地算法求导，不仅代码复杂，而且在测地线不唯一时路径导数根本不存在。作者用隐式微分绕开：对最优性条件两侧关于 $\mathbf{c}$ 求导，得到一个小型线性系统

$$\left(\frac{\partial \mathbf{x}}{\partial \mathbf{t}}^{\mathsf{T}} \frac{\partial^2 g}{\partial \mathbf{x}^2} \frac{\partial \mathbf{x}}{\partial \mathbf{t}}\right) \frac{\partial \mathbf{t}}{\partial \mathbf{c}} = -\frac{\partial \mathbf{x}}{\partial \mathbf{t}}^{\mathsf{T}} \frac{\partial^2 g}{\partial \mathbf{c}\, \partial \mathbf{x}}$$

其规模 $m \times m$ 只取决于测地线与网格的交点数 $m$。代入后 Hessian 被简化为一个紧凑表达式：

$$\frac{\partial^2 g}{\partial \mathbf{w}^2} = \frac{\partial \mathbf{c}}{\partial \mathbf{w}}^{\mathsf{T}} \frac{\partial^2 g}{\partial \mathbf{c}^2} \frac{\partial \mathbf{c}}{\partial \mathbf{w}} + \frac{\partial \mathbf{c}}{\partial \mathbf{w}}^{\mathsf{T}} \frac{\partial^2 g}{\partial \mathbf{c}\, \partial \mathbf{x}} \frac{\partial \mathbf{x}}{\partial \mathbf{t}} \frac{\partial \mathbf{t}}{\partial \mathbf{c}} \frac{\partial \mathbf{c}}{\partial \mathbf{w}}$$

### 磨光（Mollification）保证 $C^2$ 连续

当交点 $\mathbf{x}(t)$ 逼近网格顶点（$t=0$ 或 $t=1$）时，测地线会切换所交的网格边，$\mathbf{x}(t)$ 在此处无定义，最优性条件失效，导致能量非 $C^2$ 连续、阻碍收敛。作者用一个磨光函数把线性交点参数化 $\mathbf{x}(t)$ 与两段三次函数光滑混合，使导数在顶点处平滑消失：

$$\hat{t}(t) = \begin{cases} -\dfrac{t^3}{\epsilon^2} + \dfrac{2t^2}{\epsilon} & 0 \le t < \epsilon \\[4pt] t & \epsilon \le t \le 1-\epsilon \\[4pt] -\dfrac{(t-1)^3}{\epsilon^2} + \dfrac{2(t-1)^2}{\epsilon} + 1 & 1-\epsilon < t \le 1 \end{cases}$$

其中 $\epsilon = 10^{-6}$ 为磨光长度。同样的磨光也施加到端点重心坐标上。消融实验表明该磨光显著加速了牛顿求解器的收敛。

## 技术细节与应用

框架建立后，作者把它推广到多个内蕴最小化任务：

- **弹性测地网络**：由测地弹簧组成的曲线网络，能量为 $E_{\text{network}}(\mathbf{w}) = \sum_i (g_i(\mathbf{w}) - \bar{g}_i)^2$。用牛顿法加回溯线搜索最小化；当端点跨越网格边/顶点进入相邻三角面时，采用 Sharp et al. [2019] 的最直测地追踪来更新局部坐标系。

- **Karcher 均值**：作为网络的特例，一点 $\mathbf{p}$ 连接多个锚点，最小化 $E_{\text{Karcher}} = \frac{1}{2N}\sum_i g(\mathbf{p}, \mathbf{x}_i)^2$。相比只有一阶导数、只能线性收敛的 Vector Heat Method，本文的解析二阶导数支持牛顿法二次收敛。

- **弹性测地三角形**：把公式从测地边推广到测地三角形，构建有限元式膜。由于测地三角形顶点通常落在宿主曲面不同三角面里，无法直接得到统一坐标系下的边向量，作者只用三条测地边长通过 $\bar{\mathbf{e}}_{ij}^{\mathsf{T}} \mathbf{C}\, \bar{\mathbf{e}}_{ij} = g_{ij}^2$ 求解对称的 Cauchy-Green 张量 $\mathbf{C}$，再套用标准 Neo-Hookean 本构。由于只依赖边长而非路径，还避开了测地三角形面积在球面顶点附近翻转造成的不连续。

- **双向耦合**：将嵌入弹性系统与可变形宿主物体耦合，自由度 $\mathbf{q} = (\mathbf{w}, \mathbf{v})$ 同时包含嵌入系统的重心坐标和宿主曲面顶点，宿主用离散壳模型或体积有限元建模，导数同样由灵敏度分析得到。

- **可微测地 Voronoi 图（GVD）**：站点位置 $\mathbf{s}$ 为显式自由度，胞元边界顶点 $\tilde{\mathbf{x}}$ 由等距约束隐式定义，内层用牛顿法求 Voronoi 图、外层用拟牛顿法优化设计目标，构成双层优化，梯度 $\mathrm{d}\tilde{\mathbf{x}}/\mathrm{d}\mathbf{s}$ 由灵敏度分析给出。

实现上代码用 C++、Eigen、Intel TBB 并行，线性系统用 CHOLMOD 求解，精确测地用 Geometry Central 库的 MMP 算法。

## 实验结果

- **Karcher 均值**：与 Vector Heat Method [Sharp et al. 2019] 在 sphere、screwdriver、ear、protein、spiral cup 等网格上对比。本文方法二次收敛，能收敛到远更紧的容差（梯度范数达 $10^{-11}$ 量级，对方约 $10^{-4} \sim 10^{-5}$），性能相当甚至更快。与 Mancinelli & Puppo [2023] 在球面上做 100 次随机测试，对方成功率仅 17%、梯度范数 $1.4\times 10^{-2}$，本文成功率 100%、梯度范数 $6.5\times 10^{-12}$（虽慢约一个数量级但鲁棒）。

- **测地三角形膜**：将宿主环面各向异性拉伸，优化嵌入膜的重心坐标最小化弹性能量，把孤立的大畸变变为平滑分布的形变，平均 5 次牛顿迭代以内收敛。

- **双向耦合**：在充气球壳上收紧测地弹簧网络会产生真实的鼓胀效果；在体积四面体有限元建模的 Neo-Hookean 兔子上，收紧网络使兔耳下垂、背部鼓起。

- **GVD 优化**：分别优化边长一致性、胞元平面性、胞元规则性三种目标。平面性目标使顶点到拟合平面距离降低一个数量级以上；规则性目标从劣质初始图得到准各向同性胞元。

- **消融实验**：(1) 对比 GD、L-BFGS 与本文牛顿法，本文二次收敛且总时间更短，因为解析 Hessian 只占整体计算的一小部分；(2) 用欧氏距离代替测地距离会引入伪局部极小，测地距离则收敛到全局最优；(3) 无磨光时牛顿法 200 次迭代仍难收敛，加磨光后恢复二次收敛。

## 贡献与局限

**贡献**：首次给出三角网格上测地距离的闭式一阶、二阶解析导数，并用最优性条件和隐式微分把表达式大幅简化，从而将牛顿型二阶优化引入内蕴距离最小化；以测地端点为唯一自由度、连接边隐式重建，避免显式追踪不定数量的交点；配套磨光技术恢复 $C^2$ 连续；在弹性网络、膜、双向耦合、Karcher 均值、可微 Voronoi 图等多样任务上验证了通用性和显著的收敛优势。

**局限**：当测地线处于受压状态、增长长度在能量上更有利时，割迹上的局部距离极大可能导致收敛失败（好在多数嵌入弹性问题中测地线受拉）；依赖 MMP 精确测地计算，尽管并行化，它仍是主要性能瓶颈；基于边长的测地三角形在单个三角形内曲率较大时精度下降（可用更密三角化或内部积分点缓解）；目前只针对三角网格，推广到多边形网格是未来方向。
