---
title: "Learning Human Motion with Temporally Conditional Mamba"
authors:
  - "Quang Nguyen"
  - "Tri Le"
  - "Baoru Huang"
  - "Minh Nhat Vu"
  - "Ngan Le"
  - "Thieu Vo"
  - "Anh Nguyen"
category: "Animation & Simulation"
track: "Conference"
source: "arxiv"
institution:
  - "FPT Software AI Center"
  - "University of Liverpool"
  - "Vienna University of Technology"
  - "University of Arkansas"
  - "National University of Singapore"
tags:
  - "Human Motion Generation"
  - "State Space Model"
  - "Mamba"
  - "Diffusion Model"
  - "Temporal Conditioning"
  - "Motion Estimation"
links:
  paper: "https://doi.org/10.1145/3757377.3763948"
  project: "https://zquang2202.github.io/TCM"
---

## 一句话总结

本文提出 Temporally Conditional Mamba（TCM），把随时间演化的条件信号直接注入 Mamba 块的循环动态（调制其 $$B$$、$$C$$ 选择矩阵），从而在扩散框架下让生成/估计出的人体运动与条件输入逐帧对齐，在音乐驱动舞蹈、第一人称视频估计、物体轨迹驱动等多任务上超过现有方法。

## 研究背景

- 领域现状：人体运动学习分为运动合成（synthesis）与运动估计（estimation）两大任务，广泛用于动画、AR/VR、人机交互。条件输入可分为静态条件（如文本描述，全程保持不变）与时序条件（如音乐、第一人称视频、时间序列信号、跟踪输入），后者携带随时间变化的细粒度动态，直接影响运动的时序动态。
- 核心痛点：处理时序条件的主流做法是在扩散框架里用 Cross-Attention 把条件与运动融合。但注意力主要捕捉全局交互，忽视循环依赖，难以维持逐步的时序对齐。作者观察到，在第一人称到运动的任务里，Cross-Attention 与朴素 Mamba 生成的头部轨迹都明显偏离真值。
- 本文 idea：受控制论中"线性参数变化状态空间模型（LPV-SSM）根据外部信号动态调整系统矩阵"的启发，作者假设条件信号应被建模为对运动流的循环影响，于是不再外挂注意力，而是把时序条件直接嵌入 Mamba 的内部动态，实现条件的自回归注入，以增强运动与条件的时序一致性与对齐。

## 方法

整体框架：以扩散模型为骨干（前向加噪、反向去噪），去噪网络由 Mamba 块构成。模型输入加噪运动 $$x$$、条件嵌入 $$m$$、时间步嵌入 $$t$$，输出去噪运动。条件 $$m$$ 与运动 $$X$$ 假设等长 $$L$$。网络由三部分组成：Temporally Conditional Mamba（在 token 级注入条件、实现时序对齐）、Spatial Mamba（建模关节间的空间依赖）、Adaptive Layer Norm（把条件与时间步嵌入统一地作用到所有 token）。训练目标为预测干净运动：

$$\hat{\theta} = \arg\min_{\theta}\; \mathbb{E}_{t, X_t}\big[\lVert X_0 - f_\theta(X_t, t, m)\rVert\big].$$

关键设计：

1. Temporally Conditional Mamba（TCM，核心贡献）。标准 Mamba 的 S6 层按 $$h_{dl} = \bar{A}_{dl}\,h_{d,l-1} + \bar{B}_{dl}\,x_{dl}$$、$$y_{dl} = C_l\,h_{dl}$$ 递推。TCM 让选择矩阵 $$B$$、$$C$$ 同时依赖运动嵌入和条件嵌入，改写为 $$h_{dl} = \bar{A}_{dl}\,h_{d,l-1} + (\Delta_{dl}\cdot \tilde{B}_l(x_l,m_l))\,x_{dl}$$、$$y_{dl} = \tilde{C}_l(x_l,m_l)\,h_{dl}$$。其中条件感知矩阵通过对原矩阵做逐元素仿射调制得到：$$\tilde{B}_l = \gamma_B(m_l)\odot B_l + \beta_B(m_l)$$、$$\tilde{C}_l = \gamma_C(m_l)\odot C_l + \beta_C(m_l)$$，四组尺度/平移参数 $$\gamma_B,\beta_B,\gamma_C,\beta_C\in\mathbb{R}^N$$ 由作用于 $$m_l$$ 的可学习映射 $$S_m:\mathbb{R}^E\to\mathbb{R}^N$$（单层线性加非线性激活）预测。这样条件在每一帧都动态调整系统的循环动态与输出，实现自回归式的时序对齐。

2. Spatial Mamba。除时序动态外，关节间空间依赖对运动生成同样关键。该块先把运动表示从时域 $$(B,L,D)$$ 重排到空间域 $$(B,D,L)$$，用标准 Mamba 块建模关节间交互，再重排回时域，从而联合刻画时空结构。

3. Adaptive Layer Norm（AdaLN）。替换标准 LayerNorm，用于全局注入条件与时间步信息。逐维尺度、平移参数由时间步嵌入与条件嵌入之和经 MLP 生成：$$\lambda_i, \rho_i = \mathrm{MLP}(\mathrm{Sum}(t, m))$$，再对运动嵌入调制 $$x' = \lambda_i\odot\mathrm{Norm}(x) + \rho_i$$（$$i\in\{S,T\}$$ 分别对应空间/时序归一化）。AdaLN 施加于 Spatial Mamba 与 TCM 块之前，与 TCM 内的 token 级条件互补，提升整段序列的对齐。

## 实验结果

对照实验（音乐到舞蹈，AIST++ 数据集）：在只更换条件融合方式的公平设置下，TCM 相比朴素 Mamba 与 Cross-Attention 在保真度、多样性、节拍对齐（BAS）上均更优，且参数量与推理速度接近朴素 Mamba（TCM 26.84M / 1.11s，Cross-Attention 41.28M / 1.42s）。

| 方法 | 参数(M) | 推理(s) | FID_k↓ | FID_g↓ | Div_k↑ | Div_g↑ | BAS↑ |
|------|--------|--------|--------|--------|--------|--------|------|
| Ground Truth | - | - | 17.10 | 10.60 | 8.19 | 7.45 | 0.2374 |
| Vanilla Mamba | 26.58 | 1.04 | 25.61 | 16.35 | 7.52 | 6.05 | 0.2434 |
| Cross-Attention | 41.28 | 1.42 | 23.43 | 12.86 | 7.87 | 6.48 | 0.2411 |
| TCM (ours) | 26.84 | 1.11 | 20.66 | 9.75 | 8.98 | 7.24 | 0.2761 |

消融：去掉 TCM 块（退化为朴素 Mamba）性能显著下降，去掉 AdaLN、去掉 $$\gamma$$ 或 $$\beta$$ 均有下滑；其中去掉 $$\gamma$$ 的损失大于去掉 $$\beta$$，说明尺度参数比平移参数更关键。序列长度分析显示，随序列增长到 5/15/25 秒，TCM 的 FID 上升比 Cross-Attention 更缓慢（25 秒时 23.82 对 31.36），长程生成能力更强。对 $$\gamma_{B,C}$$、$$\beta_{B,C}$$ 做 t-SNE 可视化，不同音乐流派形成清晰聚类，说明 TCM 会依条件调整内部动态。

四个 SOTA 对比任务：

- 音乐到舞蹈（AIST++）：TCM 的 BAS 达 0.2761，优于 Bailando、EDGE、Lodge 等，运动质量与多样性也领先。
- 第一人称视频到运动（ARES 数据集）：将 EgoEgo 的 Transformer 骨干换成 TCM，在头部朝向/平移误差、MPJPE、加速度、脚滑等指标上全面超过 PoseReg、Kinpoly-OF、AvatarPoser、EgoEgo（如 Ohead 0.15 对 0.20，Thead 112.6 对 148.0）。
- 第一人称视频 + 音乐的多模态运动估计（EgoExo4D）：用 Jukebox 提音乐特征、ResNet-50 提视觉特征，经时序对比损失对齐后融合。TCM 在 MPJPE、加速度、脚滑等指标上优于各单模态基线与多模态基线 EMM。
- 物体轨迹到运动（OMOMO 数据集）：采用两阶段扩散（先预测接触手位、再生成全身姿态），两阶段均以 TCM 为骨干，在 Hand JPE、MPJPE、MPVPE 等指标上优于 GOAL、OMOMO、CHOIS。

## 亮点与局限

- 亮点：
  - 提出把时序条件注入 Mamba 内部循环动态（调制 $$B$$、$$C$$ 矩阵）的新机制，替代外挂 Cross-Attention，实现条件的自回归逐帧对齐。
  - 通用性强：同一 TCM 块可插入扩散框架，覆盖运动合成与运动估计、单模态与多模态、多种时序条件的四类任务，均取得 SOTA。
  - 高效：性能优于 Cross-Attention 的同时，参数量与推理速度接近朴素 Mamba；长序列下质量退化更慢。
  - 可解释性：t-SNE 显示学到的调制参数能区分不同条件（音乐流派）。

- 局限：
  - TCM 专为时间相关条件设计，对文本、场景描述等静态输入（与运动不共享时间分辨率/长度）可能并非最优。
  - 第一人称到运动任务中，极端或突然的头部运动（急停、甩动）仍难处理，可能导致估计不准。
  - 生成运动仍可能出现可见的脚滑与抖动伪影，需引入速度/加速度约束或显式脚部接触建模来改善。

## 延伸思考

- 把"外部信号调制状态空间模型系统矩阵"（源自控制论 LPV-SSM）引入 Mamba，是一种比注意力更贴合"逐步时序影响"直觉的条件注入范式，可能对其他强时序对齐需求的生成任务（语音、生理信号、时间序列预测）有借鉴意义。作者也在 broader impact 中指出 TCM 可推广到语音与生理信号建模。
- 相比 Cross-Attention 的全局交互，TCM 强调循环依赖与逐帧对齐，这提示在"条件与输出等长且强时序耦合"的问题上，循环式条件注入或许比注意力更合适；反之在静态/变长条件场景下，两者可能需要结合。
- 该方法通过调制选择矩阵实现条件感知，尺度参数 $$\gamma$$ 比平移 $$\beta$$ 更重要的发现，暗示"重参数化条件如何缩放状态更新"是提升对齐的关键杠杆，值得在更多条件生成模型中验证。
