---
title: "A Free-Space Diffraction BSDF"
authors:
  - "Shlomi Steinberg"
  - "Ravi Ramamoorthi"
  - "Benedikt Bitterli"
  - "Arshiya Mollazainali"
  - "Eugene d'Eon"
  - "Matt Pharr"
category: "Rendering"
track: "Journal"
source: "author-page"
institution:
  - "University of California San Diego"
  - "NVIDIA"
  - "University of Waterloo"
tags:
  - "Wave Optics"
  - "Diffraction"
  - "BSDF"
  - "Path Tracing"
  - "Fraunhofer Diffraction"
  - "Light Transport"
  - "Importance Sampling"
  - "Signal Coverage"
links:
  paper: "https://doi.org/10.1145/3658166"
  project: "https://research.nvidia.com/labs/rtr/publication/steinberg2024diffraction/"
  code: "https://github.com/ssteinberg/fsdBSDFpaper"
---

## 问题背景

自由空间衍射（free-space diffraction）是一种波动光学现象：当电磁波遇到障碍物的几何边缘和拐角时，会发生"弯折"（bending），一部分能量散射进入被障碍物遮挡的"阴影"区域。这种效应与经典几何光学（射线光学）的预测直接矛盾。

在可见光波段，只有极少数场景（如相干激光照射极薄狭缝）才会产生可观测的衍射效应。但随着波长增大，衍射效应的重要性急剧上升：

- RADAR 辐射会绕过行人和自行车，使其从接收端"隐身"；
- WiFi 与蜂窝辐射（如城市环境中的信号覆盖仿真）会绕过室内物体和建筑边缘。

在这些长波长场景下，自由空间衍射是主导能量分布的核心物理现象。

现有主流方法是**几何衍射理论（GTD）及其扩展的一致衍射理论（UTD）**。UTD 的吸引力在于传播用简单射线追踪完成。但它有根本性缺陷：这些射线携带电磁相位并相互干涉，干涉是复现衍射的必要条件，却导致**射线不再线性叠加**，因而**无法建立线性渲染方程**。这带来严重后果：

- 衍射边缘需要以亚波长频率采样才能正确捕捉边缘几何；
- 观测目标处也需要亚波长采样；
- 相互干涉的快速振荡射线会产生高频、高强度噪声，以及场景级别的光学散斑（speckle，随机相位叠加）。

结果是，现有基于 UTD 的最新方法只能处理二维场景或计算机图形学看来"极其简单"的三维场景，通常只考虑一两次交互的浅路径，且需要高度简化的几何（例如整栋楼的一个面只用一个四边形表示）。而现实场景恰恰需要复杂几何和深路径——这正是路径追踪（path tracing）擅长而 UTD 无能为力的领域。

## 核心方法

本文提出一种全新的**自由空间衍射 BSDF**（双向散射分布函数），设计目标是能直接嵌入典型的路径追踪器，传播仍由简单射线追踪完成，几乎不需修改路径追踪器内部逻辑。方法概念可总结为：

- 传播照常通过射线追踪进行；
- 当射线入射到几何体时，查看撞击点周围的几何并识别相关边缘；
- 若未检测到产生衍射的边缘（如射线打在无开口的墙上），路径追踪不变继续；
- 否则，从这组边缘构造一个 BSDF，量化光如何绕几何传播、以及入射能量中有多少发生衍射。该 BSDF 易于求值且可重要性采样。

其理论根基是**夫琅禾费衍射（Fraunhofer diffraction）**。在夫琅禾费（远场）区域，衍射可表述为傅里叶变换：

$$\psi(\vec{\xi}) \triangleq \frac{k}{2\pi R} \int_{\mathcal{A}_\perp} d^2\vec{q}_\perp\, \varphi(\vec{q}_\perp)\, e^{-ik\vec{\xi}\cdot\vec{q}_\perp}$$

其中 $k = 2\pi/\lambda$ 是波数，$R$ 是孔径到观测平面的距离，$\mathcal{A}_\perp$ 是屏上的孔径，$\varphi$ 是入射光复振幅场，$\psi$ 是衍射场复振幅。关键的是，距离 $R$ 会在推导中约掉，因此实际计算无需知道传播距离。

### 直接项与衍射项的分解

方法的第一个核心贡献是利用**巴比涅原理（Babinet's principle）**（孔径产生的衍射图案与其互补屏产生的衍射图案相同）将衍射场分解。设 $\tilde{\varphi}$ 是无障碍时自由传播的场，$\psi$ 是孔径衍射场，$\overline{\psi}$ 是互补孔径（即障碍物 $\overline{\mathcal{A}}_\perp$）衍射场，则 $\tilde{\varphi} = \psi + \overline{\psi}$。取模平方后可将衍射强度写成**非相干和**（无干涉）：

$$|\psi|^2 = d + w$$

其中：

$$d \triangleq |\tilde{\varphi}|^2 - \mathrm{Re}\,\overline{\psi}\tilde{\varphi}^\star \quad (\text{直接项})$$

$$w \triangleq |\overline{\psi}|^2 - \mathrm{Re}\,\overline{\psi}\tilde{\varphi}^\star \quad (\text{衍射项})$$

- **直接项 $d$**：包含通过开口 $\mathcal{A}_\perp$ 的全部能量，仅在中央波瓣（central lobe）附近非零。它描述未受阻挡、自由传播的光，可完全用射线光学处理——穿过开口的射线继续无阻传播。
- **衍射项 $w$**：衍射进入中央波瓣之外方向的能量，无法用射线光学描述。关键洞察是：由巴比涅原理，在中央波瓣之外（$|\tilde{\varphi}|\approx 0$ 处），障碍物互补孔径的衍射场 $|\overline{\psi}|^2$ 与孔径衍射场描述**相同的衍射图案**。因此衍射项只需在**射线打到障碍物上**时计算。

这个分解的强大之处在于：自由空间传播完全由射线追踪描述，衍射只在射线撞击几何时按需构造。为使两项都物理，需要对能量做轻微偏置（把本应衍射但未衍射的射线能量补偿给衍射射线）。

### 边缘衍射波的闭式解

第二个核心贡献是推导出**多边形孔径夫琅禾费衍射的闭式表达式**。当射线入射几何时，将撞击点附近的三角网格投影到与入射射线正交的**虚拟屏（virtual screen）**上，形成投影障碍 $\overline{\mathcal{A}}_\perp$，其补集即孔径 $\mathcal{A}_\perp$，边界 $\delta\overline{\mathcal{A}}_\perp$ 由只属于唯一前向三角形的边组成。

入射光束建模为空间平滑变化的光束（可理解为已知空间方差的高斯光束），并在投影网格上做**分段线性近似（PLA）**。通过散度定理，将孔径面积分转化为边界的线积分：

$$\psi(\vec{\xi}) \approx \frac{i}{2\pi R \xi^2} \oint_{\delta\overline{\mathcal{A}}_\perp} ds\, (\hat{\boldsymbol{m}}\cdot\vec{\xi})\, \varphi_{PL}(\vec{q}_\perp)\, e^{-ik\vec{\xi}\cdot\vec{q}_\perp}$$

对单条边 $\vec{e}_j$ 的线积分（称为**边缘衍射波**）有闭式解：

$$\psi_j(\vec{\xi}) \triangleq \frac{e^{-ik\vec{\xi}\cdot\vec{v}_j}}{kR|\Xi_j|}\left[\frac{a_j - b_j}{\alpha_1}(\Xi_j^{-\top}\vec{\xi}) + i\frac{a_j + b_j}{2}\alpha_2(\Xi_j^{-\top}\vec{\xi})\right]$$

总衍射场就是各边之和 $\psi = \sum_j \psi_j$。其中 $a_j, b_j$ 是入射光在边顶点上的取值，$\Xi_j$ 是定义边几何的线性变换，$\alpha_1, \alpha_2$ 是两个简单的辅助函数：

$$\alpha_1(\vec{\zeta}) \triangleq \frac{\zeta_y}{2\pi\zeta^2\zeta_x}\left(\cos\frac{\zeta_x}{2} - \mathrm{sinc}\frac{\zeta_x}{2}\right)$$

$$\alpha_2(\vec{\zeta}) \triangleq \frac{\zeta_y}{2\pi\zeta^2}\,\mathrm{sinc}\frac{\zeta_x}{2}$$

关键性质：$\alpha_1, \alpha_2$ **不依赖孔径几何或光的波长**，这一点后续用于重要性采样。此外，一对相反边的衍射场恰好互相抵消（$\psi_j = -\psi_l$），这就从三角形集合积分退化到边界线积分的形式化依据——属于两个前向三角形的边被丢弃。

## 技术细节

**衍射项截断（clamping）**：衍射项 $w$ 在中央波瓣内会取非物理的负值。解决办法是从每条边缘衍射波中提取中央波瓣，得到去除中央波瓣的截断项。用高斯调制函数 $\chi$：

$$\chi(\vec{\zeta}) \triangleq \exp\left(-\frac{1}{2\sigma_\zeta^2}\zeta^2\right), \quad \sigma_\zeta = \sqrt{3}$$

$$\hat{\psi}_j(\vec{\xi}) \triangleq \sqrt{1 - \chi(\Xi_j^{-\top}\vec{\xi})}\,\psi_j(\vec{\xi})$$

截断后的衍射项 $\hat{w} = |\sum_j \hat{\psi}_j|^2$ 始终非负、物理，且在中央波瓣外与 $w$ 相同。为保证能量守恒，被截断的能量从入射到障碍物上的能量（背散射或吸收能量）中"借"过来重定向到衍射波瓣。这个偏置的直观意义相当于让所有几何略微（几个波长）缩小。

**BSDF 表述**：将截断衍射项 $\hat{w}$ 表述为 BSDF：

$$f(\hat{\boldsymbol{\omega}}_o) \triangleq \frac{R^2}{P_{\overline{\mathcal{A}}_\perp}^{(PL)}\cos\theta}\,\hat{w}(\vec{\xi})$$

其中 $\vec{\xi} = (\tan\theta_x, \tan\theta_y)^\top$，$P_{\overline{\mathcal{A}}_\perp}^{(PL)}$ 是衍射场总功率（归一化因子）。注意入射方向不是 BSDF 参数，而是在构造孔径投影时隐式捕捉（投影平面与入射射线正交）。传播距离 $R$ 被约掉，衍射后无需知道传播距离。该 BSDF 单位为 $\mathrm{sr}^{-1}$，由构造保证能量守恒且非负，但一般不满足互易性。

**重要性采样**：截断衍射项展开为边缘衍射波瓣（非负强度）与干涉因子（振荡项）之和：

$$\hat{w}(\vec{\xi}) = \underbrace{\sum_j |\hat{\psi}_j|^2}_{\text{边缘衍射波瓣}} + \underbrace{2\sum_{l>j}\mathrm{Re}\,\hat{\psi}_j\hat{\psi}_l^\star}_{\text{干涉因子}}$$

采样策略是忽略干涉因子、从边缘衍射波瓣的非相干叠加中采样（作为提议分布）。由于 $(1-\chi)\alpha_{1,2}^2$ 函数通用、行为良好、无高频细节，作者预计算了逆 CDF 查找表（每张二维表 4 MB）用于采样。

**实现**：方法实现为 Mitsuba 0.6 中的 BSDF 插件，用双向路径追踪器（BDPT）渲染。光束建模为固定空间标准差 $\sigma = 25\lambda$ 的高斯光束，搜索半径设为 $3\sigma = 75\lambda$。BSDF 构造时按需实时细分三角形，使边长足够小以保证 PLA 近似精确。唯一需要修改路径追踪内部的地方是：为避免出射射线重新与衍射网格相交，将交互点改到孔径内部的一点（通过拒绝采样）。

## 实验结果

- **精度验证（图 4、图 6）**：与数值积分的夫琅禾费衍射及精确的 Rayleigh-Sommerfeld 衍射积分对比，本文方法在单缝、双缝、圆环、三角形等孔径上产生基本精确的衍射图案，仅在中央波瓣（直接项）处有差异。所有方法在低阶条纹上一致，高阶条纹的微小偏差源于观测距离 $R$ 较短。
- **对比 UTD（图 5）**：双缝衍射中，UTD 的相互干涉射线导致严重的高频高强度噪声——8 spp 时噪声弥漫整个图案，即便简单双缝也需数千样本才能收敛出暗条纹。而本文方法单个样本即可捕捉整个孔径几何，少量 spp 就能显现条纹。收敛后两者预测的衍射波瓣相似。单样本运行时间：UTD 16 ms，本文 62 ms。
- **复杂场景（图 1b、图 7）**：模拟城市环境中蜂窝辐射（$\lambda = 10$ cm）的信号覆盖，场景含 181,000 个三角形且未针对长波长优化（含波长尺度的边和细节）。辐射绕过建筑边缘进入阴影区，信号分布明显偏离纯射线光学的预测，并展现多次反射与衍射的深路径效应——这是现有方法几乎无法处理的。相比纯射线光学渲染，城市场景开销约为其 50 倍。
- **不同孔径（图 8）**：在康奈尔盒中放置带有双缝、大卫之星、圆-菱形等不同孔径的屏，成功产生对应的衍射图案。

## 贡献与局限

**主要贡献**：

1. 与线性路径追踪器兼容：将衍射表述为作用于辐射度量的经典 BSDF，无需射线间干涉，易于在现代路径追踪器中实现；
2. 性能：计算整个孔径引起的聚合角散射，避免了干涉射线的高频噪声，无需每波长多次密集采样；
3. 表述为角散射函数（BSDF）：只依赖入射与散射方向，不像 UTD 需要预先知道每条射线从边缘到目标的传播距离；
4. 易用性：针对图形学中广泛使用的三角网格设计，无需特殊网格预处理或几何轮廓扩展；
5. 首次将自由空间衍射与图形学高效路径追踪工具连接起来，能处理现实世界的复杂几何。

**局限**：

1. **重叠几何**：虚拟屏上重叠的前向三角形可能导致检测到错误的衍射边，产生"幻觉"衍射波瓣；重叠检测代价高，本文实现未处理。
2. **中央波瓣功率**：用于归一化的衍射场总功率和边功率是精确项，但中央波瓣总功率是近似估计，用于计算需从材质"借"多少能量以守恒。
3. **介质孔径**：方法假设衍射障碍是导体（完全吸收辐射）；介质孔径的表述留待未来工作（任何衍射理论包括 UTD 都需对介质特殊处理）。
4. 此外，查找三角形代价高，在无衍射发生处（如射线打在远离边缘的墙上）构造 BSDF 浪费了大量成本，提前放弃 BSDF 构造的机制留待未来工作。
