---
title: "Computational Biomimetics of Winged Seeds"
authors:
  - "Qiqin Le"
  - "Jiamu Bu"
  - "Yanke Qu"
  - "Bo Zhu"
  - "Tao Du"
category: "Animation & Simulation"
track: "Journal"
source: "author-page"
institution:
  - "Shanghai Qi Zhi Institute"
  - "Tsinghua University"
  - "Peking University"
  - "Georgia Institute of Technology"
tags:
  - "computational-design"
  - "biomimetics"
  - "fluid-solid-coupling"
  - "shape-space"
  - "gradient-free-optimization"
  - "LBM"
links:
  paper: "https://doi.org/10.1145/3687899"
---

## 一句话总结

作者构建了一套"数据采集 → 几何设计空间 → 流固耦合仿真 → 无梯度优化"的计算仿生流水线，从真实翅果的 3D 扫描出发，自动生成并优化出在下落、旋转、远距离滑翔等空气动力学任务上超越天然翅果的新设计。

## 研究背景

翅果（samara，如枫树、榆树、椿树的带翅种子）依靠自身形状与质量分布被动地与气流相互作用，产生升力实现稳定的长距离飞行，是"重于空气"的被动飞行结构的天然范本，启发了单旋翼飞行器、空投微型机器人等人造飞行器的设计。

传统仿生设计高度依赖人类专家的经验与反复试错；已有的空气动力物体计算设计流水线（多旋翼、风筝、纸飞机等）大多基于 CAD 工具里参数化形状的语法描述，难以表达翅果这类自然有机曲面。将数据驱动的形状生成方法用到翅果上又面临两大难题：

- **有机开放曲面的形状生成困难**：翅果是拓扑简单但形态各异的开放薄壳曲面，网格 VAE、SDF/UDF 学习方法难以严格保证光滑性与无自交，一旦违反会破坏流固仿真的数值稳定性。
- **空气动力目标混沌且昂贵**：空气与种子的湍流耦合对初始条件极其敏感，仿真代价高，标准的搜索或基于梯度的优化难以直接使用。

## 方法

整体流水线分为四步：3D 扫描采集数据 → 用 3D 微分同胚群上的测地坐标构建线性形状空间 → 用格子玻尔兹曼方法（LBM）做流固耦合仿真 → 用改造后的无梯度优化器搜索高性能设计。

```mermaid
flowchart LR
    A[真实翅果<br/>手持激光扫描] --> B[数据集<br/>49 个高质量网格 / 14 species]
    B --> C[几何设计空间<br/>测地坐标插值 Exp*]
    C --> D[LBM 流固耦合仿真<br/>平移网格 + cut-cell]
    D --> E[概率性能目标<br/>期望最小化]
    E --> F[无梯度优化<br/>ZeroGrads 代理模型]
    F -->|更新形状参数 s| C
```

### 关键设计 1：微分同胚群测地坐标构成的形状空间

把 $$\mathbb{R}^3$$ 上所有光滑双射（微分同胚）看作一个无穷维 Lie 群 $$G'$$，其在单位元处的 Lie 代数 $$\mathfrak{g}'$$ 是所有光滑向量场。只关心模板圆盘 $$S$$ 顶点上的形变，把向量场表示为定义在 $$N$$ 个顶点位置 $$\boldsymbol{q}_i$$ 上的动量数组：

$$\boldsymbol{u}(\cdot)=\sum_{i=1}^{N} K(\boldsymbol{q}_i,\cdot)\,\boldsymbol{p}_i$$

其中 $$K$$ 是核函数。给定初始动量后，形状沿测地线由哈密顿形式的测地方程演化得到：

$$\frac{d\boldsymbol{q}_i(t)}{dt}=\boldsymbol{u}(t,\boldsymbol{q}_i(t)),\qquad \frac{d\boldsymbol{p}_i(t)}{dt}=-\sum_{j=1}^{N}\nabla_1 K(\boldsymbol{q}_i(t),\boldsymbol{q}_j(t))\,\boldsymbol{p}_i(t)^\top \boldsymbol{p}_j(t)$$

初始动量即形状的"测地坐标"，映射 $$\mathrm{Exp}^*$$ 把测地坐标映到形状，其逆为 $$\mathrm{Log}^*$$。由于微分同胚天然是双射，生成的曲面几乎不会自交并保持光滑，光滑度还可由核函数调节；而且这一演化过程与生物形态学的生长/演化路径天然吻合。

### 关键设计 2：从扫描网格反解测地坐标与插值

扫描网格 $$T_1$$ 因噪声不一定落在 $$G$$ 内，作者通过优化求其测地坐标（LDDMM 的 shooting 方法），同时得到更光滑、无自交的修正网格：

$$\min_{\boldsymbol{p}_1(0)}\; e\big(\mathrm{Exp}^*(S),T_1\big)+\frac{1}{2}\sum_{i,j}K(\boldsymbol{q}_i(0),\boldsymbol{q}_j(0))\,\boldsymbol{p}^1_i(0)^\top \boldsymbol{p}^1_j(0)$$

由于所有测地坐标都是定义在 $$S$$ 上的 3D 向量数组，可看作 $$\mathfrak{g}^*$$ 中的向量，直接按权重线性相加即可得到插值/外推形状 $$\mathrm{Exp}^*(\boldsymbol{p}_{new}(0))$$，无需每次重解优化，因而生成新形状既快又受自然先验约束。

### 关键设计 3：LBM 流固仿真 + 概率目标的无梯度优化

仿真采用以种子为中心的**平移网格 + cut-cell** 的 LBM 实现：仅把角加速度施加到种子上，把线加速度当作惯性力施加到空气上，从而在局部区域同时捕捉准确下落轨迹和旋转，大幅降低长距离飞行的计算成本。

因目标函数混沌敏感，作者不优化确定性目标，而是最小化在随机初始配置 $$\boldsymbol{\alpha}\sim p$$ 下的期望：

$$\min_{\boldsymbol{s}}\int p(\boldsymbol{\alpha})\,f(\boldsymbol{s},\boldsymbol{\alpha})\,d\boldsymbol{\alpha}$$

并把 ZeroGrads 的代理模型扩展到概率目标，用神经网络 $$h(\boldsymbol{s},\boldsymbol{\phi})$$ 局部拟合期望目标景观：

$$l(\boldsymbol{s}_i,\boldsymbol{\phi})=\int \kappa(\boldsymbol{s}-\boldsymbol{s}_i)\Big(\int p(\boldsymbol{\alpha})f(\boldsymbol{s},\boldsymbol{\alpha})\,d\boldsymbol{\alpha}-h(\boldsymbol{s},\boldsymbol{\phi})\Big)^2 d\boldsymbol{s}$$

利用 $$\int p=1$$，可将其等价改写为便于蒙特卡洛估计的形式，得到简洁的梯度估计：

$$\frac{\partial l}{\partial \boldsymbol{\phi}}\approx \frac{\partial}{\partial \boldsymbol{\phi}}\frac{1}{N}\sum_{j}\big(h(\boldsymbol{s}_j,\boldsymbol{\phi})-f(\boldsymbol{s}_j,\boldsymbol{\alpha}_j)\big)^2$$

训练好代理模型 $$h$$ 后，再用 $$\nabla_{\boldsymbol{s}} h$$（如 Adam）更新形状参数，迭代至收敛。当 $$p$$ 为均匀分布时该方法退化为原始 ZeroGrads。

## 实验结果

作者用 14 species、49 个高质量网格构建形状空间，并设计了下落（Descent）、旋转加速（Rotational）、期望对比（远距离投掷）、回归（类回旋镖返回）等多个空气动力学任务，均能优化出优于初始/天然种子的设计。

其中一个直接验证"形状空间质量"的主实验：在同一下落损失（越低越好）下，将本文空间与两个传统设计空间（对最佳样本做 6×6×6 控制点变形笼 / CAD 螺旋桨模板）各取 1000 个随机样本比较（忠于原文数值）：

| 指标（损失，越低越好） | Ours | Deformation cage | Propeller |
| --- | --- | --- | --- |
| mean | 138.24 | 153.56 | 146.34 |
| min | 168.48 | 204.17 | 237.55 |

本文形状空间在均值与最小损失上都显著更优，说明蕴含自然演化先验的低维空间比高维参数空间更少劣质样本。

其它关键结果：旋转加速任务优化后角速度约翻倍、比初始多转约一圈半，形状趋于类螺旋桨；期望优化产生"飞镖状"形状，不仅飞得更远且在 12 个随机投掷姿态下轨迹一致性显著更好。制造验证中，选取的单翅果优化设计在仿真中旋转快约 29%，纸质实物重复五次实测旋转速度提升约 27.6%–50%，且优化形状对气流/手势扰动更不敏感。

## 亮点与局限

亮点：
- 用 3D 微分同胚群的测地坐标构建形状空间，天生保证光滑与无自交，且贴合生物形态演化，规避了学习方法难以施加几何约束的痛点。
- 将 ZeroGrads 无梯度优化扩展到概率性能目标，从概率视角统一推导联合采样，有效对抗混沌流固目标的高频噪声，并提升鲁棒性。
- 提供了 14 species、49 个高质量翅果数字模型数据集，并用纸质实物验证仿真定性一致。

局限（作者自述）：
- 缺乏对优化设计的系统物理验证——有机曲面制造困难（3D 打印/热成型多告失败），只能退而用纸模型，限制了更有机、更复杂设计的验证。
- 形状空间依赖微分同胚群，只能覆盖同一拓扑，不支持拓扑不同的翅果。
- 现有任务未探索敏捷机动等更复杂的空气动力目标。

## 延伸思考

- 用"生成即受物理先验约束"的几何表示（微分同胚保光滑无自交）替代"生成后再修补"的学习式方法，这一思路对其他需要严苛几何约束的仿真驱动设计（叶片、翼型、软体机器人外形）同样有借鉴价值。
- 把随机初始条件下的期望作为优化目标，本质上是在混沌系统里追求鲁棒最优而非脆弱的点最优，与强化学习中随机奖励的动机一致；这提示在物理设计中"可制造、可复现的鲁棒性能"往往比理想条件下的极值更有意义。
- 制造瓶颈是当前从数字设计走向实物的主要断点，若能结合更轻质高强的有机曲面制造工艺，这类流水线有望真正产出可用的被动飞行微结构。
