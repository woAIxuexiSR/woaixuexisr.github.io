---
title: "Toward Optimized VR/AR Ergonomics: Modeling and Predicting User Neck Muscle Contraction"
authors:
  - "Yunxiang Zhang"
  - "Kenneth Chen"
  - "Qi Sun"
category: "HCI & XR"
track: "Conference"
source: "arxiv"
institution: "New York University"
tags:
  - "VR"
  - "AR"
  - "Ergonomics"
  - "Electromyography"
  - "Neck Muscle"
  - "Head-Mounted Display"
links:
  paper: "https://doi.org/10.1145/3588432.3591495"
  code: "https://github.com/NYU-ICL/xr-ergonomics-neck-comfort"
---

## 一句话总结

用肌电（EMG）传感器测量 VR 用户在头部运动时的颈部肌肉收缩水平（MCL），据此学习一个生物物理启发的计算模型，既能对已完成的头动轨迹做事后估计，也能仅凭目标头姿在动作发生前预测潜在的颈部不适，进而用于优化虚拟内容的空间布局以减少疲劳。

## 研究背景

- 领域现状：VR/AR 头显解锁了大范围自然的头部转动交互，但头显本身的重量与重心偏移会改变颈部肌肉的负荷，已被证实会引起不适甚至损伤。此前研究多集中在硬件减重、或事后测量特定姿态的肌肉活动。
- 核心痛点：在部署一个 VR/AR 应用之前，几乎没有定量手段去预判它会给用户带来多少颈部人体工学负担。EMG 虽能直接反映肌肉收缩，但部署繁琐、成本高，且只能"事后测"，无法在动作发生前"事前预测"。
- 本文 idea：先做一次 VR 内的生理学研究，用 EMG 采集头动与肌肉收缩的成对数据；再用这些数据拟合一个生物物理启发的模型，把头姿与角加速度映射到 MCL；最后把模型扩展成仅凭起止头姿即可预测运动轨迹和累积 MCL，从而作为可优化的人体工学度量。

## 方法

整体思路分三步：先从力学假设出发写出一个含"未知函数"的 MCL 方程，再用采集的 EMG-运动配对数据把这些未知函数用神经网络拟合出来（事后估计），最后再补一个轨迹回归网络，让系统只需目标头姿就能事前预测。

```mermaid
flowchart LR
  A["头姿 r 与角加速度 α"] --> B["MCLNet 拟合 I, Tp, E"]
  B --> C["估计 MCL (事后)"]
  D["起止头姿 rs, re"] --> E["TrajectoryNet 回归速度曲线"]
  E --> F["积分得轨迹 r_t 与 α_t"]
  F --> B
  B --> G["预测累积 MCL (事前)"]
```

关键设计：

1. **生物物理建模（open functions）**：肌肉产生的力矩与 MCL 成正比。作者把维持头部姿态所需的力矩拆成两部分——只依赖头姿的被动力矩 $$\boldsymbol{T}_p(\boldsymbol{r})$$（源自重力与肌肉松弛）和主动生成的主动力矩 $$\boldsymbol{T}_a$$。由转动惯量 $$I$$ 联系角加速度 $$\boldsymbol{\alpha}$$，得到平衡关系

$$\boldsymbol{T}_p(\boldsymbol{r}) + \boldsymbol{T}_a(\boldsymbol{r}, \boldsymbol{\alpha}) = I \times \boldsymbol{\alpha}$$

  进而把 MCL 表示为 $$\text{MCL} = \mathcal{E}\bigl(I \times \boldsymbol{\alpha} - \boldsymbol{T}_p(\boldsymbol{r})\bigr)$$，其中 $$I$$、$$\boldsymbol{T}_p(\cdot)$$、$$\mathcal{E}(\cdot)$$ 是待定的未知量。这个结构化的物理先验让模型比纯黑盒回归更有可解释性和泛化性。

2. **MCLNet（事后估计）**：用 1D CNN 在头动-MCL 配对序列上以 $$L_2$$ 损失联合拟合上述三个未知函数。为处理 EMG 与肌肉运动之间的电机械延迟（可达 100ms），模型输入 400ms 的运动窗口、只预测中间 200ms 的 MCL，即两端各留 100ms 余量以对齐时序。

3. **TrajectoryNet（事前预测）**：仅有起止头姿时，真实轨迹未知。作者基于头动"主序列效应"和观察到速度曲线只有单个主峰的现象，用单峰高斯来近似角速度曲线

$$\boldsymbol{\omega}^i_t(\boldsymbol{r}_s, \boldsymbol{r}_e) = A^i\, e^{-\frac{(t-\mu^i)^2}{2(\sigma^i)^2}}, \quad i \in \{p, y\}$$

  用一个 MLP 由 $$\{\boldsymbol{r}_s, \boldsymbol{r}_e\}$$ 回归高斯参数 $$\{A^i, \mu^i, \sigma^i\}$$，再积分得到代表性轨迹喂给 MCLNet，对整段运动的 MCL 做积分求累积值。

4. **数据采集**：8 名被试戴 Oculus Quest 2，颈部左右 SCM 与 SC 四处贴 Delsys Trigno 无线 EMG 传感器（2000Hz），在 $$60° \times 100°$$ 视场内做目标注视任务，采集约 5 小时时间同步的运动-EMG 数据。原始 EMG 经去趋势、带通滤波、整流、跨通道平衡、归一化与积分处理成单一的归一化 MCL 值。

## 实验结果

事后估计（MCLNet）与事前预测（+TrajectoryNet）均在与训练集无重叠的头姿条件上评估，指标为归一化均方根误差（NRMSE）与归一化平均绝对误差（NMAE），越低越好；并用 Pearson/Spearman 相关系数衡量趋势一致性。

| 任务 | NRMSE↓ | NMAE↓ | 与硬件测量相关性 |
|------|--------|-------|------------------|
| 事后估计（MCLNet） | 12.39 ± 4.74% | 9.54 ± 4.14% | Pearson .62 / Spearman .60 |
| 事前预测（MCLNet + TrajectoryNet） | 16.76 ± 6.05% | 14.71 ± 5.96% | Pearson .59 / Spearman .57 |
| 轨迹回归 pitch 速度 | 3.54 ± 1.11% | 2.16 ± 0.65% | — |
| 轨迹回归 yaw 速度 | 3.45 ± 0.98% | 2.01 ± 0.51% | — |

主观验证：作者用一个 3D 打气球游戏做 13 人的用户研究，设计了 MAX / RND / MIN 三种扫描路径条件——它们总头部转动量（900°）与视场覆盖相同，但模型预测的累积 MCL 比值约为 3.48 : 1.95 : 1.00。采用两选一强迫选择（2AFC）让被试判断哪条路径更不舒服。结果 MAX/RND/MIN 分别被 86.1%/50.7%/13.1% 投票为"更不适"，主观不适排序 MAX > RND > MIN 与模型预测完全一致，且重复测量 ANOVA 显示差异显著。由于总转动角被严格控制相同，说明差异确实来自模型所刻画的肌肉负荷而非单纯转动量。

## 亮点与局限

- 亮点：
  - 首次为 VR 颈部人体工学提供了可量化、可预测的计算模型，并公开了 EMG 生理数据集。
  - 生物物理先验（主动/被动力矩分解）让模型结构可解释，而非纯黑盒。
  - 关键突破在于"事前预测"：仅凭目标头姿即可在动作发生前估计不适，直接服务于内容布局优化、UX 设计、影像编辑等应用。
  - 用户研究在总转动量严格相等的前提下验证了主观不适与模型预测一致，证据较有说服力。

- 局限：
  - 只建模了 yaw 和 pitch，未考虑 roll 维度（难以用视觉刺激精确操控），而 roll 也可能影响 MCL。
  - 只覆盖颈部 SCM/SC 四块肌肉，未纳入肩部、斜方肌等其他肌群（上斜方肌信号弱被剔除）。
  - 用单峰高斯近似速度曲线，可能无法完全刻画个体行为差异；作者建议引入概率建模来揭示跨用户统计方差。
  - 数据规模有限（采集 8 人、评估 6 人、用户研究 13 人），被试均报告颈部正常。

## 延伸思考

这项工作把"感知优化"的思路从视觉（分辨率、注视点渲染）延伸到了肌肉骨骼层面的人体工学，与同组在 VR 舒适度、显示优化方向的工作一脉相承。值得追问的方向包括：把 roll、肩部、躯干运动一并纳入，形成更完整的上半身负荷模型；用概率/生成式方法建模个体轨迹差异，替代确定性的单峰高斯；以及把该 MCL 度量做成可微目标，直接嵌入 UI 布局或场景生成的自动优化管线中，实现"人体工学感知"的内容自动排布。更长远地，这类模型有望回答"VR/AR 若取代手机与显示器用于日常，会带来多少额外人体工学负担"这类系统性问题。
