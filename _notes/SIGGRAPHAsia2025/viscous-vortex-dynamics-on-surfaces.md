---
title: "Viscous Vortex Dynamics on Surfaces"
authors:
  - "Cuncheng Zhu"
  - "Hang Yin"
  - "Albert Chern"
category: "Animation & Simulation"
track: "Journal"
source: "author-page"
institution: "University of California San Diego"
tags:
  - "Fluid Simulation"
  - "Vorticity Method"
  - "Navier Stokes"
  - "Curved Surfaces"
  - "Gaussian Curvature"
  - "Non Orientable Surfaces"
  - "Boundary Conditions"
  - "Kutta Condition"
links:
  paper: "https://doi.org/10.1145/3763320"
  project: "https://cunchengzhu.github.io/project_pages/ViscousVortex2025.html"
  code: "https://cseweb.ucsd.edu/~alchern/projects/ViscousVortex/supplementary.zip"
---

## 一句话总结

本文给出曲面上不可压黏性 Navier–Stokes 流的涡量形式方程，指出以往方法普遍遗漏了黏性力中依赖高斯曲率的项，并证明这一曲率项对涡量方程与调和分量演化都至关重要；进而提出一套三角网格上的隐-显（IMEX）求解器，可在任意拓扑（包括不可定向曲面）与多种边界条件下正确重现曲率驱动的流动现象。

## 研究背景

- 领域现状：曲面流体仿真在图形学中由来已久，其中涡量形式尤其受青睐——涡量是标量，可以直接沿曲面平流，而基于速度的形式必须处理协变导数或平行输运。近期工作已能把不可压欧拉流的完整涡量方程（含非平凡曲面拓扑与调和分量）写清楚。
- 核心痛点：即便在单连通域上，图形学文献里也很少有方法正确纳入黏性项。一个被普遍忽视的细节是：曲面的高斯曲率 $$K$$ 会给 Navier–Stokes 的黏性力贡献一个额外项。写成涡量形式即为 $$\dfrac{\partial w}{\partial t}+\mathbf{u}\cdot\nabla w=\nu\Delta w+2\nu\,\mathrm{curl}(K\mathbf{u})$$，其中前半段是"以往模型"，后半段 $$2\nu\,\mathrm{curl}(K\mathbf{u})$$ 是被遗漏的曲率项。以往在球面、一般曲面上的诸多涡量-流函数方法都退化成 $$\dfrac{\partial w}{\partial t}+\mathbf{u}\cdot\nabla w=\nu\Delta w$$，丢掉了曲率项。
- 更进一步：在非单连通域上，曲率与黏性还会影响速度场调和分量的演化。若 $$\mathbf{h}$$ 是调和向量场、$$\mathbf{u}$$ 是速度场，系数 $$c=\langle\mathbf{h},\mathbf{u}\rangle$$ 的演化含有一个新的内部曲率项 $$2\nu(\mathbf{h}\cdot\mathbf{u})K$$ 以及一个边界黏性项。以往方法在非单连通域上要么缺失调和分量动力学，要么把黏性写错。本文声称是首个在任意拓扑（含不可定向曲面）上同时正确纳入黏性项与一般调和场动力学的算法。

## 方法

### 从应变率到黏性拉普拉斯

作者从连续介质力学出发，把黏性应力建模为与应变率成正比：$$\boldsymbol{\tau}=2\nu(\mathrm{K}\mathbf{u})^{\sharp}$$。这里的核心是 Killing 算子 $$\mathrm{K}\mathbf{u}:=\tfrac{1}{2}\mathcal{L}_{\mathbf{u}}g$$，即速度场沿度量张量的李导数的一半，几何上度量了流动引起的"非等距变形率"。满足 $$\mathrm{K}\mathbf{v}=0$$ 的场称为 Killing 场，它生成的是刚体般的等距流动、不产生内部黏性应力。黏性力则可由 Rayleigh 耗散泛函 $$\mathcal{R}[\mathbf{u}]=\iint_M \nu|\mathrm{K}\mathbf{u}|^2\,dA$$ 的负泛函梯度给出。

关键在于：平直欧氏域上"对称化梯度的散度等于向量拉普拉斯"这一恒等式在有曲率时不再成立。作者区分了三种向量拉普拉斯——Bochner（连接）拉普拉斯 $$\Delta_B$$、Hodge 拉普拉斯 $$\Delta_H$$、以及对应真实黏性耗散的黏性拉普拉斯 $$\Delta_V$$，并用 Weitzenböck 恒等式把它们联系起来：对无散度场 $$\mathbf{v}\in\mathfrak{X}_{\mathrm{div}}$$，有 $$\Delta_B\mathbf{v}=\Delta_H\mathbf{v}+K\mathbf{v}=\Delta_V\mathbf{v}-K\mathbf{v}$$。三者的核各不相同：$$\Delta_V$$ 的核是 Killing 场，$$\Delta_H$$ 的核是调和场，$$\Delta_B$$ 的核是平行场。因此正确的黏性 Navier–Stokes 方程应写作 $$\partial_t\mathbf{u}+\nabla_{\mathbf{u}}\mathbf{u}=-\mathrm{grad}\,p+\nu\Delta_V\mathbf{u}$$，等价于用 Hodge 拉普拉斯时必须补上 $$2\nu K\mathbf{u}$$。由此直接得到一个物理判据：若曲面存在 Killing 场，则速度沿该 Killing 场的分量在时间上守恒（无摩擦刚体运动应当被保持）。

### 涡量与调和分量的完整演化

对速度取旋度即得涡量方程 $$\dfrac{\partial w}{\partial t}+\mathbf{u}\cdot\nabla w=\nu\Delta w+2\nu\,\mathrm{curl}(K\mathbf{u})$$。展开曲率项 $$2\nu\,\mathrm{curl}(K\mathbf{u})=2\nu\langle -J\mathbf{u},\mathrm{grad}\,K\rangle+2\nu K w$$，可见它会产生一种类似"涡量封闭"（vorticity confinement）的效果，抵抗扩散项、甚至导致涡量场的孤子样行为。速度由涡量重构走标准路线：解流函数泊松问题 $$-\Delta\psi=w$$（边界零 Dirichlet），再加上调和分量 $$\mathbf{u}=-J\,\mathrm{grad}\,\psi+\sum_{i=1}^m c_i\mathbf{h}_i$$。调和系数按 $$\dfrac{dc_i}{dt}=\iint_M[(\mathbf{h}_i\times\mathbf{u})w+2\nu(\mathbf{h}_i\cdot\mathbf{u})K]\,dA+\oint_{\partial M}\nu h_{i\partial}w\,ds$$ 演化。

### 曲率片与涡量跳变

一个核心解析结果：当高斯曲率含奇异片 $$K=K_{\mathrm{reg}}+f\delta_\Gamma$$（沿曲线 $$\Gamma$$ 集中）时，即使存在扩散项，涡量也会在 $$\Gamma$$ 两侧维持一个跳变间断，跳变量为 $$[w]_\Gamma=2f\langle\mathbf{u},\mathbf{t}_\Gamma\rangle$$，且与黏性 $$\nu$$ 无关。这个跳变条件是后续把边界条件"几何化"的关键工具。

### 边界条件的几何统一

作者把三类常见边界条件统一到"高斯-Bonnet 曲率密度"框架里：

- 自由滑移（free-slip / Navier slip）：它是 Rayleigh 变分原理的自然边界条件 $$(\mathrm{K}\mathbf{u})[\mathbf{t},\mathbf{n}]=0$$。此时边界涡量满足 $$w_\partial=2\kappa_g u_\partial$$，其中 $$\kappa_g$$ 是边界测地曲率、$$u_\partial$$ 是切向速度。
- Navier 摩擦：条件变为 $$2\nu(\mathrm{K}\mathbf{u})[\mathbf{t},\mathbf{n}]=\alpha u_\partial$$，边界涡量为 $$w_\partial=(2\kappa_g-\tfrac{\alpha}{2\nu})u_\partial$$。
- 无滑移（no-slip）：$$u_\partial=0$$，是摩擦系数 $$\alpha/\nu\to\infty$$ 的极限，边界涡量不再显式给出、需隐式求解。

作者的巧思是引入高斯-Bonnet 曲率密度 $$\Omega=(K+\kappa_g\delta_{\partial M})\,dA$$，把边界测地曲率当成一层奇异高斯曲率片。于是只要在涡量方程里把 $$K$$ 换成 $$K_\Omega=K+\kappa_g\delta_{\partial M}$$，并设零 Dirichlet 边界 $$w|_{\partial M}=0$$，前述跳变条件就会让边界内侧的涡量自然等于 $$2\kappa_g u_\partial$$——边界条件被"吸收"进方程本身。摩擦边界则相当于把测地曲率平移 $$\tfrac{\alpha}{4\nu}$$。作者还指出一个漂亮结论：这一曲率机制在无摩擦的自由滑移边界下就能自然重现空气动力学中的 Kutta 条件（流动在机翼尖锐后缘干净分离），无需显式建模边界层。

### 不可定向曲面：双重覆盖

由于涡量函数是 Hodge 星，其符号在穿过 Möbius 带时会翻转，在不可定向曲面上不全局良定义。作者用"双重覆盖"技巧解决：把带边或不可定向的 $$M$$ 加倍成一个闭合可定向曲面 $$\tilde M$$，其上带一个对合等距映射 $$o$$。据此把场分解为偶场（$$o^*f=f$$）与奇场（$$o^*f=-f$$）。速度是偶的无散度场、涡量函数是奇函数、流函数是奇函数、调和场是偶调和场。只要初值遵循这套奇偶结构，在 $$\tilde M$$ 上解 Navier–Stokes 就等价于在 $$M$$ 上求解。界面 $$\Gamma$$ 处会集中 $$\tilde K=K+2\kappa_g\delta_\Gamma$$ 的奇异曲率，恰好又通过跳变条件自动还原自由滑移边界。

### 离散化与 IMEX 求解

空间上用三角网格：涡量 $$w$$ 与流函数 $$\psi$$ 放在顶点，速度场与调和基放在面上，离散高斯-Bonnet 曲率也定义在面上（通过重标定的角亏损构造，且在闭曲面上满足离散 Gauss–Bonnet 定理 $$\sum_f A_f K_f^\Omega=2\pi\chi(M)$$）。时间上用隐-显（IMEX）方案：把黏性项 $$\nu(\Delta w+2\,\mathrm{curl}(K\mathbf{u}))$$ 当作刚性部分用隐式后向 Euler、其余非刚性项用显式 RK4，两者交替。刚性步的隐式方程 $$(A_P+\nu\Delta t L)_{II}w_I^{\mathrm{new}}=\dots$$ 用不动点迭代求解，通常 2–3 次即收敛（容差 $$10^{-5}$$）。摩擦边界只需把面上的 $$K_\Omega$$ 换成含摩擦项的 $$K_{\Omega,\alpha/(4\nu)}$$；硬无滑移则用一个专门的不动点迭代（引入非零边界涡量 $$w_B$$ 使边界速度归零）。作者同时指出：用"高friction"近似无滑移比硬约束的不动点迭代更稳、更高效，后者在高雷诺数下易失稳，需要边界附近加密网格。

## 实验结果

实现基于 SideFX Houdini，在 Intel i9-11900K、64GB 内存的桌面机上运行。约 22K 顶点的典型网格在自由滑移/摩擦边界下用主算法约需 90 分钟（约 5 秒/帧）；16K 顶点的无滑移算例约 27 分钟（约 6 秒/帧）。核心验证围绕"曲率项与调和动力学是否必要"展开：

| 对比项 | 本文（完整模型） | 简化模型（缺曲率项 / 缺调和动力学） |
| --- | --- | --- |
| 球面刚体旋转 | 无剪切变形，旋转被无限期保持 | $$\dot w+\nabla_{\mathbf u}w=\nu\Delta w$$ 使速度迅速衰减到零 |
| 非均匀曲率旋转面上的刚体旋转 | 正确保持刚体运动与被平流纹理 | 引入非物理畸变并最终静止 |
| 环面上的 Killing 模 | 含调和演化时 Killing 分量被守恒 | 关闭调和动力学则 Killing 分量被错误阻尼 |
| Enneper 曲面（含内蕴 Killing 场） | 自由滑移下松弛到纯 Killing 场 | —— |

跳变条件 $$[w]_\Gamma=2f\langle\mathbf u,\mathbf t_\Gamma\rangle$$ 在圆柱面上得到数值验证：随网格边长 $$\epsilon\to0$$，其相对误差对各种 $$\nu$$ 都随时间快速下降；作者还用量纲分析给出松弛时间 $$\tau=O(\epsilon^2/\nu)$$。不可定向曲面上分别在 Boy 曲面（$$\mathbb{RP}^2$$，双重覆盖为球）、Möbius 带（覆盖为柱面，展示 Kelvin–Helmholtz 不稳定性）、Klein 瓶（覆盖为环面）上完成仿真。边界条件方面：环面挖孔绕障流动中，自由滑移只生成少量涡量、不足以形成 von Kármán 涡街，而高摩擦（近无滑移）边界显著增强涡脱落、形成涡街；高摩擦近似与硬无滑移（专用不动点迭代）结果相近但前者效率明显更高。观察还表明，涡量（导出量）在高雷诺数、低分辨率下会出现空间振荡型数值弥散，但对速度（积分量）和最终动画影响很小。

## 亮点与局限

亮点：

- 指出并系统论证了黏性力中被长期忽视的高斯曲率项 $$2\nu\,\mathrm{curl}(K\mathbf u)$$，用刚体旋转守恒等判据清晰证明"只有完整模型才物理正确"。
- 用 Weitzenböck 恒等式把 Bochner/Hodge/黏性三种拉普拉斯统一，给出曲面上黏性 Navier–Stokes 的正确涡量形式，并首次在任意拓扑（含不可定向面）上同时处理正确黏性项与一般调和场动力学。
- 提出高斯-Bonnet 曲率密度这一优雅工具，把自由滑移、摩擦、无滑移边界都化归为"边界测地曲率的奇异曲率片"，边界条件被自然吸收进方程；并据此为 Kutta 条件提供了新的几何视角。
- 双重覆盖 + 奇偶场分解，把仿真与计算 Hodge 分解自然推广到不可定向曲面。

局限：

- 数值方案仅条件稳定，平流步受 CFL 限制。
- 隐式步用不动点迭代：主算法收敛很快，但无滑移边界的不动点迭代效率较低、在高雷诺数下易失稳，需在边界附近加密网格；作者认为 Newton 法可能更合适，留作未来工作。
- 涡量作为导出量在高雷诺数、低分辨率下存在数值弥散（空间振荡）。

## 延伸思考

- "把边界条件几何化为奇异曲率片"这一思路很有普适性：它把原本需要额外方程约束的边界自由度，转化为方程系数（曲率）里的一个分布项，从而在求解器层面自动满足。这与"把约束下沉进算子而非后处理"的一类结构保持方法一脉相承。
- 黏性力必须用黏性拉普拉斯 $$\Delta_V$$ 而非 Hodge 拉普拉斯 $$\Delta_H$$ 的洞见，提醒我们在把平直域公式照搬到曲面时要格外小心——很多"看似等价"的向量恒等式在有曲率时失效，Killing 算子/Killing 场提供了判断黏性是否物理正确的天然标尺。
- 自由滑移边界下自然涌现 Kutta 条件，暗示无需显式边界层建模也能捕捉经典气动现象，这为几何驱动的、更轻量的空气动力学近似提供了想象空间；结合双重覆盖对不可定向曲面的统一处理，"几何 + 流体"的交叉方向仍有不少值得挖掘的结构性洞见。
