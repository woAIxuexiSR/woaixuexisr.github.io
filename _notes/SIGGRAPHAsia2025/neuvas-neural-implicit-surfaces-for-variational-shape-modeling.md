---
title: "NeuVAS: Neural Implicit Surfaces for Variational Shape Modeling"
authors:
  - "Pengfei Wang"
  - "Qiujie Dong"
  - "Fangtian Liang"
  - "Hao Pan"
  - "Lei Yang"
  - "Congyi Zhang"
  - "Guying Lin"
  - "Caiming Zhang"
  - "Yuanfeng Zhou"
  - "Changhe Tu"
  - "Shiqing Xin"
  - "Alla Sheffer"
  - "Xin Li"
  - "Wenping Wang"
category: "Geometry & Modeling"
track: "Journal"
source: "arxiv"
institution:
  - "Shandong University"
  - "The University of Hong Kong"
  - "Tsinghua University"
  - "University of British Columbia"
  - "Texas A&M University"
tags:
  - "Variational Shape Modeling"
  - "Neural Implicit Surface"
  - "Signed Distance Function"
  - "Curve Networks"
  - "Curve Sketches"
  - "Thin Plate Energy"
  - "Sharp Feature Curves"
  - "Surfacing"
links:
  paper: "https://doi.org/10.1145/3763331"
---

## 一句话总结

提出 NeuVAS：用神经隐式曲面（神经 SDF 的零水平集）从稀疏形状控制（无结构 3D 曲线草图、连通曲线网络、稀疏点云）出发做变分曲面建模；核心是加入基于曲面曲率的薄板光顺能量约束曲线之间的自然过渡，并用一个到特征曲线的距离平方权重让光顺能量在尖锐特征附近衰减，从而忠实保留 $$G^0$$ 尖锐特征。

## 研究背景

给一组 3D 曲线"蒙皮/放样"成曲面（lofting/skinning）是几何建模里的基础难题，包含三重挑战：一是找到准确插值给定曲线的曲面；二是控制远离曲线区域的形状，让曲线之间的插值自然、美观；三是构造分片光滑曲面以忠实表达 $$G^0$$ 尖锐特征曲线。如今曲线输入多用 AR/VR 设备手绘，往往彼此不连通且不精确，这让依赖结构化曲线网络的算法失效。

现有方法大致两类：

- 网格方法（如 Pan 等 2015）能靠把曲线拆成独立闭环、每个环定义一个面片来忠实保留尖锐特征，但要求结构化曲线网络输入，无法处理无结构草图，且受网格质量与分辨率限制。
- 隐式曲面方法（如 VIPSS）拓扑灵活、可解析计算曲率，更适合处理草图，但难以做精确形状控制，尤其难构造尖锐特征。
- 还有混合方法（Yu 等 2022）：需要一个网格代理作为输入并分片拟合隐式多项式，强依赖初始曲面质量，且多项式（阶数至多 4）表达力有限，细节恢复能力差，初始曲面不好时甚至要人工搭建，低效不可靠。

作者强调，变分形状建模与传统"过拟合稠密点集"的曲面拟合本质不同：稀疏曲线不足以唯一确定曲面，因此除了保证插值曲线，还必须加入曲面能量项来正则化曲线之间的形状。仅把曲面拟合方法套用到稀疏曲线上会得到不理想的结果。

## 方法

NeuVAS 把曲面表示为神经 SDF $$f(\boldsymbol{x};\Theta):\mathbb{R}^3\to\mathbb{R}$$ 的零水平集 $$S=\{\boldsymbol{x}\in\mathbb{R}^3\mid f(\boldsymbol{x};\Theta)=0\}$$，其中 $$f$$ 由 MLP 编码。设计有三个关键方面：插值输入曲线、曲面光顺能量、创建尖锐特征曲线。

### 关键设计 1：插值输入曲线

用三个损失项约束神经函数成为一个能穿过输入曲线的 SDF。

Eikonal 条件促使梯度模长为 1，既正则化隐式场得到更干净的零水平集，也为近表面采样提供可靠的距离与梯度：

$$\mathcal{L}_E=\frac{1}{|\mathcal{Q}|}\sum_{\boldsymbol{q}\in\mathcal{Q}}\big|1-\|\nabla f(\boldsymbol{q};\Theta)\|\big|$$

Dirichlet 条件要求曲面穿过输入曲线采样点集 $$\mathcal{P}$$：

$$\mathcal{L}_{DM}=\frac{1}{|\mathcal{P}|}\sum_{\boldsymbol{p}\in\mathcal{P}}|f(\boldsymbol{p};\Theta)|$$

同时用一个约束项防止 $$f$$ 在远离曲线处也为零（避免出现多余的零水平集点）：

$$\mathcal{L}_{DNM}=\frac{1}{|\mathcal{Q}|}\sum_{\boldsymbol{q}\in\mathcal{Q}}\exp(-\alpha|f(\boldsymbol{q};\Theta)|)$$

三者合成插值损失 $$\mathcal{L}_{\text{interp}}=\lambda_E\mathcal{L}_E+\lambda_{DM}\mathcal{L}_{DM}+\lambda_{DNM}\mathcal{L}_{DNM}$$。

### 关键设计 2：曲面光顺能量（薄板能量）

为让曲面在稀疏曲线之间取自然光滑的形状，加入基于薄板能量的光顺项（$$\kappa_1,\kappa_2$$ 为主曲率）：

$$\mathcal{L}_{\text{Smooth}}=\frac{1}{|S|}\sum_{\boldsymbol{s}\in S}\big(\kappa_1^2(\boldsymbol{s})+\kappa_2^2(\boldsymbol{s})\big)$$

用平均曲率 $$H$$ 与高斯曲率 $$K$$ 可改写为：

$$\mathcal{L}_{\text{Smooth}}=\frac{1}{|S|}\sum_{\boldsymbol{s}\in S}\big(4H^2(\boldsymbol{s})-2K(\boldsymbol{s})\big)$$

其中 $$H$$、$$K$$ 由 $$f$$ 的 Hessian 矩阵通过隐式场的通用曲率公式计算。作者特别指出：尽管网络经 Eikonal 损失训练成近似 SDF，但训练中它与真实距离场可能有明显偏差，因此选用适用于一般隐式场的通用曲率公式，而非依赖真实 SDF 特有的简化公式，以准确估计零水平集附近的曲率。

### 关键设计 3：创建尖锐特征曲线

若在整个曲面上强制光顺能量，会得到无尖锐特征的整体光滑曲面。为忠实创建 $$G^0$$ 尖锐特征（曲面连续但法向不连续），引入一个到特征曲线点集 $$\mathcal{P}_f$$ 的欧氏距离平方权重 $$d^2(\boldsymbol{s},\mathcal{P}_f)$$，修改光顺能量：

$$\mathcal{L}_{\text{Smooth}}=\frac{1}{|S|}\sum_{\boldsymbol{s}\in S}\big(4H^2(\boldsymbol{s})-2K(\boldsymbol{s})\big)\cdot d^2(\boldsymbol{s},\mathcal{P}_f)$$

几何直觉清晰：在特征曲线附近权重很小、在特征曲线上为零，于是跨特征曲线没有光顺约束，共享特征曲线的两片相邻曲面以 $$G^0$$ 连续相接；同时特征曲线上光顺能量的缺席让插值项 $$\mathcal{L}_{\text{interp}}$$ 主导，保证曲线被准确插值。

特征曲线的指定：对曲线网络，按 Pan 等 2015 通过端点法向变化自动检测特征曲线；对曲线草图（无法自动提取），把所有曲线视为特征曲线；对稀疏点云，默认所有曲线为光滑曲线；此外支持用户手动指定尖锐特征曲线。

### 实现细节

总损失为：

$$\mathcal{L}=\lambda_E\mathcal{L}_E+\lambda_{DM}\mathcal{L}_{DM}+\lambda_{DNM}\mathcal{L}_{DNM}+\tau\lambda_{\text{Smooth}}\mathcal{L}_{\text{Smooth}}$$

其中 $$\tau$$ 是余弦因子，用来调制薄板能量项的影响。由于薄板能量项与 Dirichlet 项存在冲突，SDF 零水平集难以插值曲线网络，因此把 $$\tau$$ 初始化为 1、以 1K 迭代为一个周期，先逐渐减小光顺项让零水平集贴合曲线，再增大光顺项让形状演化到合理状态；周期性退火与重启有助于恢复不同尺度的特征。

零水平集采样方面，每次迭代都精确采样开销太大，作者利用相邻迭代零水平集变化很小的特点：复用上一步的采样点并投影到当前零水平集上；同时每 100 次迭代用一次 Marching Cubes（分辨率 $$128^3$$）重新生成采样点以保证均匀、避免累积误差。零水平集网格上用 Poisson-disk 采样得到约 10K 均匀点。点 $$\boldsymbol{x}$$ 的投影为：

$$\boldsymbol{x}'=\boldsymbol{x}-\frac{\nabla f(\boldsymbol{x};\Theta)}{\|\nabla f(\boldsymbol{x};\Theta)\|}\cdot f(\boldsymbol{x};\Theta)$$

网络采用 IGR 架构：8 个隐藏层、每层 256 单元、softplus 激活、约 1.86M 参数。权重设为 $$\lambda_E=0.1$$、$$\lambda_{DM}=100$$、$$\lambda_{DNM}=10$$、推荐 $$\lambda_{\text{Smooth}}=5\times10^{-4}$$。用 Adam 优化器、学习率 $$5\times10^{-5}$$、训练 10K 次迭代（多数情况 5K 内收敛）。

## 实验结果

实验在三类输入上进行：曲线网络（连通、可分解为闭环，如 Espresso、Toothpaste）、曲线草图（无法清晰组织为闭环，更贴近 AR/VR 手绘，如 Spaceship、Bishop）、稀疏点云（散点，难以连成曲线，如 Fertility、Walrus、Torus）。

与变分曲面建模方法的对比（Pan 等 2015、Yu 等 2022、VIPSS、Xu 等 2023）主要发现：

- Pan 等 2015 在高质量曲线网络上能生成理想形状并恢复尖锐特征，但受闭环算法限制，无法处理曲线草图或稀疏点云。
- Yu 等 2022 需要与目标同拓扑的代理网格初始化，Espresso 与 Toothpaste 需人工构造代理网格才能工作；且会在 Bishop 底部丢失特征、无法捕捉 Ship 前部细节。
- VIPSS 与 Xu 等 2023 只吃点、依赖法向全局一致性，点云太稀疏时无法得到正确拓扑（Espresso、Toothpaste），也无法处理尖锐特征，并在 Fertility、Walrus 引入多余凸起等伪影；两者复杂度均为 $$O(n^3)$$，只能处理不超过约 10K 点，超出需降采样，难以表示复杂模型。

与神经隐式重建方法（IGR、SIREN、NeuralSingularHessian）的对比：这些方法虽支持多种输入，但因缺乏远离输入点的曲面建模目标，无法生成视觉上令人满意的形状，都会出现错误拓扑或表面伪影；NeuVAS 则能可靠重建带细节、正确拓扑和尖锐特征的合理几何。

运行时性能：主要开销来自在零水平集点集 $$\mathcal{Q}_{zero}$$ 上估计薄板能量。固定 $$\mathcal{P},\mathcal{Q}$$ 为 10K，不同 $$\mathcal{Q}_{zero}$$ 采样量下每次迭代耗时如下：

| 采样量 $$\mathcal{Q}_{zero}$$ | 10K | 20K | 50K | 80K |
|---|---|---|---|---|
| 时间 [ms] | 127.53 | 132.39 | 259.33 | 387.13 |

由于每 100 次迭代才做一次 Marching Cubes，其成本被摊薄，采样点增多时方法仍高效。曲线网络（Espresso/Toothpaste/Roadster，25K/28K/21K 顶点）的整体耗时对比中，Pan 等 2015 最快（约 20–24 s，但仅限曲线网络），NeuVAS 约 1340 s；VIPSS 因 $$O(n^3)$$ 需降采样到 4–6K 点、Xu 等 2023 需降采样到 5–10K 点，都无法有效处理复杂模型。基于 SGD 的 IGR/SIREN/NSH 与本方法不限制输入点数，本方法虽比它们慢，但它们无法为这些输入生成合理结果。

其它评估：

- 不同曲线类型的效果——正确标注特征曲线时形状合理；若把所有曲线都当作光滑曲线，会导致过度光滑、鼓起的结果。
- 与真值对比——用 FlowRep 提供曲线网络的真值曲面，以 Hausdorff 距离着色，结果与真值非常接近。
- 能量分布——薄板能量在空白区域分布均匀，即便使用了距离衰减策略。
- 拓扑过渡——用 torus 逐步增删曲线：仅给一个圆时生成球体（合理），沿三条不同路径逐渐加入引导曲线，在不同步骤恢复出环面形状。

## 亮点与局限

亮点：

- 提出 NeuVAS，基于神经隐式曲面的高质量变分曲面建模，能接受一般稀疏形状控制：无结构曲线草图、连通曲线网络、稀疏点。
- 用到特征曲线的距离平方权重让光顺能量在尖锐特征附近衰减，从而在输出曲面上有效保留 $$G^0$$ 尖锐特征，并让相邻面片各自独立优化、自然实现分片光滑。
- 不需要任何外部初始化方法，直接以点云为输入，利用神经网络的强表达能力学习 SDF，回避了代理网格/多项式阶数的限制。

局限：

- 假设水密流形，难以处理非流形几何；面对开放曲面时会用水密流形作先验生成闭合曲面。
- 受 SDF 表示固有限制，虽已达到较高锐度，但无法得到完美的尖锐特征。
- 欧氏距离权重在极端情形（两片非常接近、近乎平行、一片由光滑曲线定义另一片由尖锐曲线定义）下可能出现相互干扰，导致轻微不规则或光顺度下降。
- 薄板（TPS）能量趋向局部平坦，理论上排斥球面/柱面：在光滑曲线约束下仍能稳定到主曲率残差均匀分布的球面，但在柱面上不稳定、会略微内凹；曲率变化能量可收敛到 cyclide（含球面/柱面），但作为三阶泛函不稳定且开销大，留作未来工作。
- 有一定抗噪能力（噪声水平约 0.05 内仍能生成高质量曲面），但超过 0.1 阈值后失效。
