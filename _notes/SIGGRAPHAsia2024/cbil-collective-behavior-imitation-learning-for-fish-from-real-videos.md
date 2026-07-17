---
title: "CBIL: Collective Behavior Imitation Learning for Fish from Real Videos"
authors:
  - Yifan Wu
  - Zhiyang Dou
  - Yuko Ishiwaka
  - Shun Ogawa
  - Yuke Lou
  - Wenping Wang
  - Lingjie Liu
  - Taku Komura
track: "Journal"
source: arxiv
category: "Animation & Simulation"
institution:
  - The University of Hong Kong
  - University of Pennsylvania
  - SoftBank Corp.
  - Texas A&M University
tags:
  - Collective Behavior
  - Imitation Learning
  - Fish Schooling Simulation
  - Reinforcement Learning
  - Video Representation Learning
links:
  paper: "https://doi.org/10.1145/3687904"
---

## 一句话总结

CBIL 提出一套可扩展的群体行为模仿学习框架，直接从真实的 2D 鱼群视频（无需 3D 运动轨迹捕捉）学习鱼群的集体运动先验，实现圆游、对齐、聚集、觅食、追逐等多样化群体行为的实时合成。

## 研究背景

复现真实的群体行为是一个既迷人又棘手的难题。已有方法主要分两类，各有短板：

- **基于规则的方法**（如 Boids、Foids、DeepFoids）依赖手工设计的原则来驱动个体，难以捕捉鱼群运动内在的高度多样、复杂且随机的模式，真实感和多样性受限。
- **数据驱动方法**能从真实数据重现多样运动，但普遍依赖 2D/3D 的真值运动轨迹。而在鱼群场景下，由于严重的遮挡和高度相似的纹理，逐条鱼的轨迹几乎无法可靠捕捉，现有跟踪器（如 YOLOv9）会给出不一致的轨迹，导致数据稀缺且噪声大。

CBIL 的核心动机是：绕开对逐个体运动轨迹的依赖，直接从"野外" 2D 视频中学习群体运动分布，从而具备可扩展性并能跨物种迁移。

## 方法

### 整体框架

CBIL 分为三个阶段：

```mermaid
flowchart LR
    A[参考视频 + 模拟渲染视频] --> B[视觉表征学习<br/>MVAE 自监督]
    B --> C[隐式状态 z]
    C --> D[群体行为模仿学习<br/>GAIL + 隐式状态聚类]
    D --> E[策略 π]
    E --> F[群体运动合成<br/>圆游/对齐/聚集/觅食/追逐]
```

第一阶段用 Masked Video AutoEncoder（MVAE）把视频映射为紧凑的隐式状态；第二阶段在 GAIL 框架下用这些隐式状态训练策略，并引入隐式状态聚类与生物启发奖励；第三阶段用学到的运动先验配合任务奖励合成具体群体行为。

### 关键设计 1：Masked Video AutoEncoder（视觉表征学习）

所有视频帧先用 SAM 分割成二值轮廓图，剔除背景与鱼体颜色信息，增强特征解耦（合成视频则直接把 3D 形状投影为轮廓）。受 MAE 启发，随机遮挡 50% 的图像块，用基于时序 ViT 的 MVAE 自监督重建，从 10 帧窗口的视频片段提取维度为 100 的低维隐式状态。训练损失为重建损失加 KL 散度损失：

$$
\mathcal{L}_R = \frac{1}{T}\sum_{t=1}^{T}(o_t - \hat{o}_t)^2
$$

$$
D_{KL}(Q(z|X)\,\|\,P(z)) = \frac{1}{2}\sum_{j=1}^{J}\left(\mu_j^2 + \sigma_j^2 - \log(\sigma_j^2) - 1\right)
$$

$$
\mathcal{L}_{Final} = \mathcal{L}_R + \beta D_{KL}
$$

KL 项把隐空间压缩得更紧凑，有利于后续 GAIL 判别。

### 关键设计 2：基于隐式状态的对抗模仿学习

策略 $$\pi$$ 采用 GAIL 风格，MVAE 在此阶段冻结。判别器不使用显式 3D 轨迹，而是学习隐式状态转移 $$D(z, z')$$，并加入梯度惩罚以稳定训练：

$$
\min_{D} -\mathbb{E}_{d_M(z,z')}[\log D(z,z')] - \mathbb{E}_{d_\pi(z,z')}[\log(1-D(z,z'))] + w_{gp}\,\mathbb{E}_{d_M(z,z')}\left[\|\nabla_\varphi D(\varphi)|_{\varphi=(z,z')}\|_2^2\right]
$$

每条鱼共享同一策略（Multi-Instance Single Policy），输入自身状态、邻居状态和目标 $$g_t$$，输出高斯分布的动作，因此可适配任意数量的鱼。

### 关键设计 3：隐式状态聚类抑制模式坍塌

GAIL 常因模式坍塌无法捕捉参考分布的频率/熵。CBIL 用 K-Means 对 MVAE 映射出的参考隐式状态（经 t-SNE 降维）无监督聚成 $$N$$ 组，出现越频繁的运动状态被赋予越大的奖励权重，从而强化判别性运动特征、抵抗噪声：

$$
r_S(z_t, z_{t+1}, i) = -\log\left(1 - \frac{W_i^{FG}\,D(z_t, z_{t+1})}{\sum_{i=1}^{N} W_i^{FG}}\right)
$$

$$
W_i^{FG} = \frac{N_{is}}{\sum_i N_{is}}
$$

聚类簇数 $$K$$ 通过在 1~10 间用肘部法则依据 SSE 自动确定。总奖励融合风格奖励、生物规则奖励与任务奖励：

$$
r_t = W_S\, r_S(z_t, z_{t+1}, i) + W_B\, r_B(s_t, a_t, s_{t+1}) + W_H\, r_H(s_t, a_t, s_{t+1}, g_t)
$$

其中系数分别为 0.4、0.1、0.5。任务奖励 $$r_H$$ 针对圆游、对齐、聚集、追逐、觅食、内聚等不同行为分别设计。系统基于 Unity 引擎与 ML-Agents 实现，用 PPO 训练策略。

## 实验结果

在自采红鲷鱼数据集及 Trout-Salmon、Coho-Salmon、Yellowtail 与若干 YouTube 视频上评估，与 Boids、DeepFoids、AMP 比较。跨视角验证的 FID（越低越好）如下：

| 方法 | 顺时针圆游 | 逆时针圆游 | 聚集 | 对齐 |
| --- | --- | --- | --- | --- |
| Boids | 864.7 | 806.3 | 968.2 | 891.5 |
| DeepFoids | 789.3 | 745.2 | 928.9 | 875.7 |
| AMP | 621.4 | 609.6 | 685.4 | 701.1 |
| **CBIL（本文）** | **534.5** | **501.9** | **489.3** | **523.3** |

技能分布相似度（归一化 JS 散度，越低越好）上 CBIL 全面领先（如聚集 0.42、对齐 0.37，均显著低于 AMP 的 0.67 与 0.79）；归一化任务回报（越高越好）也最优（聚集 0.96、对齐 0.93）。多样性 APD 上 CBIL 高于 Boids/DeepFoids，但略低于 AMP——作者指出 AMP 虽 APD 高却常产生混乱运动，高 APD 不等于高质量。消融实验表明隐式状态聚类在 FID、JS 散度、任务回报上均带来改进。此外，用合成+真实数据训练的 YOLOv8 异常行为检测模型达到整体精确率 96.8%、召回率 91.0%，框架还成功迁移到鸟群的对齐与行走行为。

## 亮点与局限

**亮点**

- 首个直接从视频（无需 3D 轨迹捕捉）学习大规模群体运动先验的 GAIL 框架，天然规避遮挡与跟踪失败问题。
- MVAE 自监督表征 + 隐式状态聚类的组合，有效缓解对抗模仿学习的模式坍塌，捕捉多样运动风格。
- 学到的运动先验可复用与组合（如把追逐先验叠加到圆游任务），并能跨物种泛化，且可服务于异常行为检测等下游应用。

**局限**

- 鱼群密度过高时分割失效，难以从 2D 观测学到运动先验；MVAE 效果依赖参考视频对隐变量分布的充分覆盖。
- 仍基于 GAIL，存在模式坍塌风险；策略训练样本消耗大。
- 每种群体行为需单独准备参考视频并训练独立策略，缺乏可平滑切换多行为的统一模型。

## 延伸思考

方法目前只学宏观轨迹层面的群体运动，未纳入鱼体生物力学及其与流体的交互。若结合物理仿真（流体-刚体耦合），既能提升动画真实感，也可能服务于生物学研究。另一个方向是把当前"一行为一策略"的范式升级为条件化的统一模型，让同一策略在圆游、觅食、追逐等行为间自然过渡，并借助数据高效的策略训练降低样本成本。视频驱动的群体先验思路也具通用性，除鱼群、鸟群外，或可扩展到昆虫群、人群等更广义的集群动画与行为分析场景。
