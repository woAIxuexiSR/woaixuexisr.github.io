---
title: "The Granule-In-Cell Method for Simulating Sand–Water Mixtures"
authors:
  - Yizao Tang
  - Yuechen Zhu
  - Xingyu Ni
  - Baoquan Chen
category: Animation & Simulation
track: Journal
source: arxiv
institution: Peking University
tags:
  - Granule Fluid Simulation
  - PIC DEM Coupling
  - Multiphase Flow
  - Eulerian Lagrangian Coupling
  - Sand Water Mixture
  - Volume Conservation
links:
  paper: https://doi.org/10.1145/3763279
  arxiv: https://arxiv.org/abs/2504.00745
---

## 问题背景

沙-水混合物的模拟需要同时刻画两种截然不同的尺度：单个沙粒的随机微观运动（迁移、沉积、堵塞），以及沙粒群作为整体在连续流体介质中的统计宏观输运。真实再现这些现象要求一个统一、自洽的动力学描述，把介观尺度与宏观尺度联系起来。

已有方法大致分两类，各有缺陷：

- **拉格朗日视角（SPH / SPH-DEM）**：把沙粒和流体都用拉格朗日粒子表示，聚焦单粒子受力与运动，物理效应尺度被限制在颗粒尺寸。当沙-水相互作用显著时，粒子无法准确反映颗粒群的集体效应。SPH-DEM 虽用 DEM 建模球状刚性沙粒，但 SPH 把沙当作可穿透的密度云，导致流体既不被颗粒吸收也不被排出——漏斗实验中任意半径的沙都会穿过颈部而不堵塞。
- **欧拉视角（MPM）**：把沙和流体都当连续介质，能反映颗粒群统计特性，但在相界面附近难以捕捉更精细的颗粒运动，且丢失了颗粒的离散边界，把混合物本构简化为干湿状态间的平滑过渡，忽略了其作为多孔介质的本质特征。

作者指出，以往方法最根本的问题是**沙粒的描述依附于流体框架**，使沙的运动被流体动力学主导。

## 核心贡献

本文提出 **Granule-In-Cell (GIC)** 方法，将沙视为"介观宏观输运流"而非流体域内的固体边界，实现物理一致的双向耦合：

- 一种反映颗粒离散特性的、自洽的沙-水耦合策略；
- 一种在体积约束下守恒总体积的扩展隐式密度投影（IDP）算法；
- 由系综平均系统性导出的各类受力的离散化方案。

GIC 用 **DEM** 刻画沙粒的细尺度动力学，用 **PIC/FLIP** 提供连续的空间表示与密度投影。介观视角下沙由流体驱动；宏观视角下（受 IDP 启发）沙约束流体体积从而影响流体运动。这种双向耦合严格遵守质量守恒，实现精确的体积保持。

## 物理背景与受力

**离散视角（DEM）**：每个沙粒简化为半径 $$r$$ 的相同球体，仅作平动。对距离 $$d_{ij} = \|\boldsymbol{x}_j - \boldsymbol{x}_i\| < 2r$$ 的两粒子，法向与切向接触力为

$$\boldsymbol{F}_{i,\mathrm{n}} = k_\mathrm{n}(2r - d_{ij})\,\hat{\boldsymbol{x}}_{ij}$$

$$\boldsymbol{F}_{i,\mathrm{t}} = \min\{\|\boldsymbol{f}_{i,\mathrm{t}}\|,\ \|\boldsymbol{F}_{i,\mathrm{n}}\|\tan\varphi\}\,\hat{\boldsymbol{f}}_{i,\mathrm{t}}$$

其中 $$k_\mathrm{n}, k_\mathrm{t}$$ 为法向/切向刚度，$$\varphi$$ 为摩擦角。

**连续视角（混合物）**：流体相的质量守恒为

$$\frac{\partial \alpha_\mathrm{f}}{\partial t} + \nabla\cdot(\alpha_\mathrm{f}\boldsymbol{v}_\mathrm{f}) = 0$$

其中 $$\alpha_\mathrm{f}$$ 为流体平均体积分数。混合物中沙与水的体积分数满足约束 $$\alpha_\mathrm{s} + \alpha_\mathrm{f} \le 1$$，在流体区域取等号（流体填满颗粒间隙）。

**流体中颗粒的受力**：由 Maxey–Riley 理论，力可分解为压力梯度力、拖曳力与虚拟质量力：

$$\boldsymbol{F}_\mathrm{p} = -\tfrac{4}{3}\pi r^3 \nabla p_\mathrm{f}$$

$$\boldsymbol{F}_\mathrm{d} = 12\pi r^2 \rho_\mathrm{f}\mu\,\|\boldsymbol{v}_\mathrm{f} - \boldsymbol{v}_\mathrm{s}\|(\boldsymbol{v}_\mathrm{f} - \boldsymbol{v}_\mathrm{s})$$

$$\boldsymbol{F}_\mathrm{v} = \tfrac{2}{3}\pi r^3 \rho_\mathrm{f}\left(\frac{\mathrm{d}\boldsymbol{v}_\mathrm{f}}{\mathrm{d}t} - \frac{\mathrm{d}\boldsymbol{v}_\mathrm{s}}{\mathrm{d}t}\right)$$

系综平均还引入了一个由沙体积分数梯度 $$\nabla\alpha_\mathrm{s}$$ 驱动的**浓度梯度力** $$\boldsymbol{F}_\alpha = -\dfrac{\rho_\mathrm{s}\boldsymbol{D}_\mathrm{s}}{\alpha_\mathrm{f}\tau_\mathrm{s}}\cdot\nabla\alpha_\mathrm{s}$$，其中 $$\boldsymbol{D}_\mathrm{s}$$ 为扩散张量、$$\tau_\mathrm{s}$$ 为颗粒速度松弛时间。

## 算法框架

GIC 采用类似辛欧拉的时间积分，流体部分在沙部分之后计算。每个完整时间步内含多个沙子子步（$$N = \lceil \Delta t / \Delta t' \rceil$$）：

- **沙步**：每个子步计算流体耦合力并处理颗粒碰撞，更新沙位置，累积动量交换 $$\boldsymbol{F}_\mathrm{s}$$ 传给网格。
- **水步**：基于新的沙位置确定颗粒体积分数 $$\alpha_\mathrm{s}$$，先平流、再密度投影，施加外力与动量交换后做速度投影，最后在沙-水界面处理沙的浸湿。

### 沙作为投影目标

在混合物中无法直接用无散度条件，作者借鉴 IDP 发展了分数投影方法。在流体内每个单元强制 $$\alpha_\mathrm{s} + \alpha_\mathrm{f} = 1$$，体积分数通过映射粒子体积在网格中心计算：

$$\alpha_\mathrm{s}(\boldsymbol{x}) = \frac{1}{V}\sum_i V_{\mathrm{s}i}\,N(\boldsymbol{x}_{\mathrm{s}i} - \boldsymbol{x})$$

沙的运动使 $$\alpha_\mathrm{s}$$ 随时间变化，导致 $$\partial\alpha_\mathrm{f}/\partial t \ne 0$$。对 Navier–Stokes 做算子分裂并将时间导数用后向欧拉离散，得到一个**变系数泊松方程**：

$$\frac{\Delta t}{\rho_0}\nabla\cdot(\alpha_\mathrm{f}'\nabla p) = \frac{\alpha_\mathrm{f}' - \alpha_\mathrm{f}^*}{\Delta t} + \nabla\cdot(\alpha_\mathrm{f}'\boldsymbol{v}_\mathrm{f}^*)$$

右端拆成两项：$$p_1$$ 按常规方式更新网格速度，$$p_2$$ 用于计算粒子位置修正 $$\delta\boldsymbol{x} = -\dfrac{\Delta t^2}{\rho_0}\nabla p_2$$（只移动粒子位置、不更新速度）。为防单步过度修正，比值 $$\alpha_\mathrm{f}^*/\alpha_\mathrm{f}'$$ 限制在 $$[0.5, 1.5]$$，位移不超过一个单元宽度。$$\alpha_\mathrm{s}$$ 上限设为 3D 中 0.740、2D 中 0.907（最密球堆积），以防产生负目标分数导致空腔。密度投影把当前分数 $$\alpha_\mathrm{f}^*$$ 投向目标 $$\alpha_\mathrm{f}'$$，使流体粒子分布与沙互补，从而稳定保持混合物总体积。

### 流动中颗粒的受力计算

由于沙先于流体计算，沙子步中使用的流体信息来自上一流体时间步。压力梯度力由网格插值得到梯度后代入公式；虚拟质量力则将颗粒加速度项并入运动项，得

$$\left(m_\mathrm{s} + \tfrac{2}{3}\pi r^3\rho_0\right)\frac{\mathrm{d}\boldsymbol{v}_\mathrm{s}}{\mathrm{d}t} = \tfrac{2}{3}\pi r^3\rho_0\frac{\mathrm{d}\boldsymbol{v}_\mathrm{f}}{\mathrm{d}t} + \boldsymbol{F}_\mathrm{o}$$

利用 $$\mathrm{d}\alpha_\mathrm{f}/\mathrm{d}t = 0$$，流体速度的时间导数 $$\mathrm{d}\boldsymbol{v}_\mathrm{f}/\mathrm{d}t = \boldsymbol{b}_\mathrm{f} - \dfrac{1}{\rho_0}\nabla p + \dfrac{1}{\rho_0\alpha_\mathrm{f}}\boldsymbol{F}_\mathrm{s}$$，恰好是 FLIP 框架中网格给出的速度增量。颗粒对流体的动量交换按子步累积后投影到网格，作为外力加入流体速度：

$$\boldsymbol{F}_\mathrm{s}(\boldsymbol{x}) = -\frac{\Delta t'}{\Delta t}\sum_i\sum_j \boldsymbol{f}_{\mathrm{s}j}^i\,N(\boldsymbol{x}_{\mathrm{s}j}^i - \boldsymbol{x})$$

### 单元内颗粒的浸湿

沙因毛细作用能在多孔结构中吸持水。模型把 PIC 流体粒子转化为 DEM 颗粒内的含水量（相变），吸收水量正比于颗粒体积，比率 $$r_i$$ 上限 $$r_\text{max}$$，吸水后颗粒质量 $$m_{\mathrm{s}i} = V_{\mathrm{s}i}(\rho_\mathrm{s} + r_i\rho_0)$$。为避免遍历顺序造成吸水不均，作者把每个颗粒的湿度亏缺 $$r_\text{max} - r_i$$ 投影到网格来决定每个单元移除的流体粒子数，随机选取并投影其动量与体积，再插值回颗粒，实现均匀吸收。吸水建模为完全非弹性碰撞，动量按子步逐步并入颗粒。湿颗粒不增大实际半径，$$\alpha_\mathrm{s}$$ 上限相应提高到 3D $$0.740(1 + r_\text{max})$$、2D $$0.907(1 + r_\text{max})$$。

### 沉积浓度梯度力

浓度梯度力被离散在颗粒上。由于 $$\alpha_\mathrm{s} = 1 - \alpha_\mathrm{f}$$，改写为 $$\boldsymbol{F}_\alpha = \dfrac{\rho_\mathrm{s}\boldsymbol{D}_\mathrm{s}}{\alpha_\mathrm{f}\tau_\mathrm{s}}\cdot\nabla\alpha_\mathrm{f}$$。考虑到需计入自由水与颗粒结合水，引入修正的水分数 $$\tilde{\alpha}_{\mathrm{f}i} = r_i + \dfrac{\alpha_{\mathrm{f}i}}{1 - \alpha_{\mathrm{f}i}}$$，并以 SPH 风格重建密度场求梯度。这一力在物理上对应湿沙中的**毛细/液桥作用**，被要求满足两条性质：一是短程力，随距离超过阈值衰减到零；二是随湿度变化——干沙（湿度为 0）与饱和（湿度为 1）时为零，在最大吸水率处达到最大。

距离项采用 Israelachvili 的液桥表面能模型，取 $$\cos\theta = 1$$ 得表面张力

$$F_{ij}^\text{st} = -2\pi\sigma r\left[\left(1 + \frac{2V^*}{\pi r s_{ij}^2}\right)^{-\frac{1}{2}} - 1\right]$$

其中 $$s_{ij} = \|\boldsymbol{x}_j - \boldsymbol{x}_i\| - 2r$$ 为间隙、$$V^*$$ 为液桥体积（取颗粒体积的 0.01%），有效范围由断裂距离 $$d_\mathrm{r}$$ 界定。湿度依赖分量用 Bézier 曲线 $$\Gamma$$ 拟合，最终两颗粒间的浓度梯度力在 $$0 < s_{ij} < d_\mathrm{r}$$ 时为 $$-\Gamma(\tfrac{1}{2}(\mathrm{sr}_i + \mathrm{sr}_j))F_{ij}^\text{st}$$，否则为零。

## 实验与结果

方法基于 C++ 与 oneTBB CPU 并行实现，在 AMD EPYC 9K84（80 核）上测试。典型参数：水密度 $$10^3\ \mathrm{kg/m^3}$$、沙密度 $$2.5\times10^3\ \mathrm{kg/m^3}$$、杨氏模量 $$10^6\ \mathrm{Pa}$$、泊松比 0.3。流体时间步由 CFL 条件定，颗粒时间步由 Rayleigh 准则定，通常 $$\Delta t' \sim \Delta t/100$$。

关键实验及结论：

- **消融——隐式密度投影**：大球落水实验中，GIC 保持相对体积偏差低于 1%；无 IDP 时水与沙占据同一空间，SPH-DEM 体积振荡且回落到初始流体体积，MPM 因缺乏恢复策略在流体内形成空腔。
- **消融——虚拟质量力**：作为一种惯性使沙球整体运动变慢，而轨迹几乎不变。
- **小球落水**：不同密度球体表现出压力梯度力效应，密球堆积在杯底中央、疏球漂浮且顶部保持干燥。
- **漏斗**：较大粒径的沙球因摩擦堵住漏斗、较细的顺利通过，展示 DEM 在颗粒尺度效应上的优势（MPM 中出流受网格尺寸而非物理粒径限制）；不同湿度实验中，饱和湿沙因浓度梯度力结团、支撑自身停在漏斗口，加水后 $$\mathrm{sr}=1$$ 使该力消失、沙重新流出。
- **搅拌（茶叶悖论）**：再现旋转流体中颗粒向杯底中心聚集的二次流现象，MPM 因把沙当连续介质并用双网格而无法产生沙床上的内向流。
- **猫砂盒**：湿猫砂因浓度梯度力结团，铲起时干颗粒滑落而湿团保持形状留在铲上。
- **沙瀑（Sandfall）**：120 万颗粒倒入水中，展示大规模模拟能力，沉积后在饱和层顶部形成明显干层。
- **溃坝**：沙坝靠颗粒间摩擦抵御水冲击，随饱和度上升黏聚力减弱、坝体从中部坍塌；相比 MPM 依赖网格变形获取应力，本方法粒子间相互作用更紧密、整体更不易被冲散。

## 局限与未来工作

- **计算效率**：DEM 需大量粒子且受 Rayleigh 准则限制时间步很小，成本较高；猫砂实验中铲子的快速刚体边界采用预定义运动曲线而非物理求解，快速运动刚体边界仍是挑战。
- **旋转与角动量**：模型仅考虑球形沙粒的平动，忽略旋转效应与 Magnus 力；因沙粒缺乏角动量信息而选用 FLIP 而非 APIC。未来拟纳入颗粒旋转。
- **界面效应**：沙粒用各向同性核聚合体积信息，若用非均匀核会隐含非球形假设；最优传输的分离相边界条件不适用于非疏水的沙-水混合，需将界面效应纳入最优传输过程。
- **泡沫模拟**：低密度 DEM 粒子可漂浮于水面，显示将框架扩展到流体-泡沫相互作用的潜力。
