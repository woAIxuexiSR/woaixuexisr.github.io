---
title: "Uni-Inter: Unifying 3D Human Motion Synthesis Across Diverse Interaction Contexts"
authors:
  - "Sheng Liu"
  - "Yuanzhi Liang"
  - "Jiepeng Wang"
  - "Sidan Du"
  - "Chi Zhang"
  - "Xuelong Li"
category: "Animation & Simulation"
track: "Conference"
source: "arxiv"
institution:
  - "Nanjing University"
  - "Institute of Artificial Intelligence, China Telecom (TeleAI)"
tags:
  - "Human Motion Synthesis"
  - "Human Object Interaction"
  - "Human Scene Interaction"
  - "Human Human Interaction"
  - "Diffusion Model"
  - "Volumetric Representation"
  - "Occupancy Grid"
links:
  paper: "https://doi.org/10.1145/3757377.3763954"
---

## 概览

Uni-Inter 面向"交互式人体动作生成"这一任务，提出一个统一的、与具体任务无关的框架，在单一架构下同时支持人-人、人-物、人-场景三类交互，并能推广到这些实体的任意组合（复合交互）。

以往方法通常把三类交互当作彼此孤立的子问题，各自设计专门的表示与训练管线：人用骨架、物体用点云、场景用占据栅格。这种异构表示与任务专用设计限制了跨任务泛化，也难以刻画真实世界里同时发生的物理约束（碰撞、支撑）、社交动态（距离、注意）与任务语义（递交、坐下）。当多种交互同时出现时，往往产生不连贯的动作。

Uni-Inter 的核心是把所有交互实体映射进一个共享的三维占据场，并把动作生成重新表述为在该体积上的逐关节空间分布预测，从而在同一结构下进行一致的关系推理。

## 关键设计

### 统一交互体积 (UIV)

作者借鉴自动驾驶中基于栅格的占据表示，把人、物体、场景统一编码进一个共享的体素化三维空间。设离散化的交互空间为 $$S \in \mathbb{R}^{H \times W \times D}$$，其中 $$H, W, D$$ 为各轴分辨率。假设所有交互都发生在这个有界体积内，UIV 是一个静态、预定义的空间，生成的人在其中运动。

在每个时间步为每个体素赋予语义类别（场景、物体、人）或标记为空。语义占据映射定义为

$$\phi_t(u) = \sum_{c \in C} \mathbb{I}_c(u,t) \cdot c$$

其中 $$C$$ 是 one-hot 实体标签空间，指示函数 $$\mathbb{I}_c(u,t)$$ 在时间 $$t$$ 体素 $$u$$ 被类型 $$c$$ 的实体占据时取 1，否则取 0。对空间中每个体素施加 $$\phi$$，得到语义体积表示 $$V \in \mathbb{R}^{H \times W \times D \times 3}$$，把随时间变化的所有交互实体聚合成统一交互体积 $$\Omega = \{V_t\}_{t=1}^{T}$$。

具体地：人体借助 SMPL 模型转成稠密网格再离散化，类别取 $$c_h = [1,0,0]$$；物体经旋转平移得到每帧顶点位置后离散化，类别取 $$c_o = [0,1,0]$$；静态与动态场景的顶点离散化，类别取 $$c_s = [0,0,1]$$。UIV 特征用金字塔结构的多尺度特征提取器学习，各尺度特征注入网络对应的下采样阶段，实现层级融合。

### UIV 对齐的正则化

与传统直接回归关节旋转、经前向运动学构造姿态的方法不同（这类方法忽视三维环境的空间上下文），Uni-Inter 把动作表示为 UIV 上的体素化关节分布。对每个关节 $$k$$ 和时间步 $$t$$，在空间 $$S$$ 上计算一个空间概率图 $$P_t^k$$，刻画该关节占据每个体素的可能性。按人体姿态估计的做法，把 $$P_t^k$$ 建模为以真值关节位置 $$j_t^k \in \mathbb{R}^3$$ 为中心的归一化各向同性高斯分布：

$$P_t^k(u) = \frac{1}{(2\pi\sigma^2)^{3/2}} \exp\left(-\frac{\lVert u - j_t^k \rVert_2^2}{2\sigma^2}\right)$$

其中 $$\sigma > 0$$ 控制空间不确定性与分布展宽。给定预测分布 $$\hat{P}_t^k$$，通过在交互体积上求一阶矩（期望）得到预测关节位置：

$$\hat{j}_t^k = \mathbb{E}_{\hat{P}_t^k}[u] = \int_{u \in S} \hat{P}_t^k(u) \cdot u \, du$$

该公式完全可微，便于端到端学习；把概率铺展到邻近体素避免了稀疏梯度，即便预测不准也能学习；在连续空间分布上求期望还缓解了离散化误差，实现亚体素精度的关节定位，让预测精度超越体素网格的原生分辨率。可视化时把预测分布转成 SMPL 参数：先由期望取关节坐标，再用 SMPLify 估计姿态参数，形状参数固定为默认值。

### 训练目标

模型采用扩散架构，直接预测干净动作 $$x_0$$。基本重建损失为

$$\mathcal{L}_{rec}(x_0, x_0^*) = \lVert x_0 - x_0^* \rVert_2^2$$

在此之上，直接监督由预测分布算出的关节位置：

$$\mathcal{L}_{pos}(\hat{j}_t^k, j_t^k) = \lVert \hat{j}_t^k - j_t^k \rVert_2^2$$

对关节速度施加约束：

$$\mathcal{L}_{vel}(\hat{j}^k, j^k) = \sum_{t=1}^{T} \left\lVert \frac{d}{dt}(\hat{j}^k(t)) - \frac{d}{dt}(j^k(t)) \right\rVert_2^2$$

对每根骨骼的长度与朝向施加显式结构一致性约束：

$$\mathcal{L}_{sk}(\hat{s}_t^n, s_t^n) = \lVert \hat{s}_t^n - s_t^n \rVert_2^2$$

其中 $$s_t^n = j_t^{parent(n)} - j_t^{child(n)}$$ 是第 $$n$$ 根骨骼，由父子关节位置作差得到。

一个关键设计是：不做初始姿态朝向归一化。许多方法把所有序列对齐到固定朝向以简化学习，却让模型无法根据环境调整动作。作者转而对首帧朝向施加显式监督：

$$\mathcal{L}_{ori}(\hat{d}_0, d_0) = 1 - \frac{\hat{d}_0 \cdot d_0}{\lVert \hat{d}_0 \rVert_2 \lVert d_0 \rVert_2}$$

其中首帧朝向向量 $$d_0$$ 由骨架几何计算，取左右髋关节到根节点的连接向量的叉积：

$$d_0 = \frac{s_0^{lhip}}{\lVert s_0^{lhip} \rVert_2} \times \frac{s_0^{rhip}}{\lVert s_0^{rhip} \rVert_2}$$

总目标聚合各项损失：

$$\mathcal{L} = \mathcal{L}_{rec} + \lambda_1 \cdot \mathcal{L}_{pos} + \lambda_2 \cdot \mathcal{L}_{vel} + \lambda_3 \cdot \mathcal{L}_{sk} + \lambda_4 \cdot \mathcal{L}_{ori}$$

### 统一训练策略

把人-人、人-物、人-场景三类交互数据都转成 UIV 表示，按任务类型以 1:1:1 混合比例在全部数据上联合训练，鼓励在共享的动作生成框架内学习一致的表示，促进跨任务知识迁移。

## 实现要点

模型以文本和 UIV 为条件输入。文本先经 CLIP 编码器提取 512 维特征 $$F_{text}$$ 再送入模型。输出为分布张量

$$P = \{P_t^k\}_{k,t} \in \mathbb{R}^{T \times H \times W \times D \times K}$$

动作序列由对输出分布求期望得到。实验中序列长度 $$T = 40$$ 帧，$$\sigma = 3$$，$$H = W = D = 48$$，对应高 2.4 米、深与宽各 4.8 米的物理空间。扩散过程配置 1000 步，推理用 DDIM 采样。损失权重 $$\lambda_1 = \lambda_2 = \lambda_3 = 0.1$$，$$\lambda_4 = 1$$。学习率 $$3 \times 10^{-5}$$ 线性衰减，最大训练 500000 步，批大小 32。

## 实验与结果

在三个数据集上评测，各对应一类交互：FullBodyManipulation（人-物，约 10 小时、15 种物体）、NTU120-AS（人-人，8118 段多视角序列、26 种动作，跨主体协议）、TRUMANS（人-场景，15 小时动捕、含动态元素的 3D 环境，7:2:1 划分）。指标包括脚滑动分 (FS)、FID、接触精确率/召回率/F1 (C-prec / C-rec / C-F1)、平均每关节位置误差 (MPJPE)、根平移误差 (T-root)、多样性、多模态性，以及场景任务的目标距离 (Goal Dist.)，距离类指标以厘米计。

- 人-物交互：在 FullBodyManipulation 上多数指标超过现有方法，接触 C-prec 0.91、C-rec 0.86、C-F1 0.86，FID 0.51；相比 CHOIS，运动轨迹误差降低约 54%。
- 人-场景交互：在 TRUMANS 上各指标均优于基线，FID 2.650、FS 0.155，与目标位置的距离较基线改善约 20.8%。
- 人-人交互：在 NTU120-AS 上相较 ReGenNet 取得更低 FID (2.216) 与更好的多样性。

消融结论：

- 去掉统一训练（仅在单一数据集训练）会在三类任务上普遍抬高 FID 与脚滑动分，说明多任务联合训练提供了显著的辅助收益。
- 用直接回归运动学参数替换空间分布输出会导致性能大幅下降，尤其 FID 恶化明显，凸显空间分布表述不可替代。
- 损失项消融显示：骨骼一致性损失 $$\mathcal{L}_{sk}$$ 在三类任务中都最关键，去掉后 FID 大幅上升；速度损失与朝向损失对接触质量和整体保真度也很重要。

定性上，人-物任务在手部精细动作与文本条件控制上优于 CHOIS；人-场景任务在语义理解上优于 Trumans（如正确区分"左手"、执行"躺下"）；并展示了在人、物、场景任意组合下的复合交互生成能力。

## 局限与展望

作者提出未来把 Uni-Inter 扩展到因果、实时设定：基于部分观测在线生成动作，支持交互环境中的动态规划，面向交互式图形、具身智能体与实时虚拟体验等应用。
