---
title: "Continuous Control of Editing Models via Adaptive-Origin Guidance"
authors:
  - "Alon Wolf"
  - "Chen Katzir"
  - "Kfir Aberman"
  - "Or Patashnik"
category: "Neural & Generative"
track: "Conference"
source: "arxiv"
institution:
  - "Tel Aviv University"
  - "Decart.ai"
tags:
  - "Diffusion Model"
  - "Image Editing"
  - "Video Editing"
  - "Classifier-Free Guidance"
  - "Continuous Control"
links:
  paper: "https://doi.org/10.1145/3799902.3811077"
  project: "https://adaor-paper.github.io/"
---

## 一句话总结

针对"指令式扩散编辑模型无法平滑调节编辑强度"的问题，本文提出 Adaptive-Origin Guidance（AdaOr）：把 Classifier-Free Guidance 的"引导原点"从任意编辑的无条件预测替换为可随强度插值的"恒等预测"，从而让编辑强度像滑块一样从原图连续过渡到完整编辑结果。

## 研究背景

- 领域现状：指令驱动的扩散编辑模型（输入图像/视频 + 文本指令 → 编辑结果）已成为语义级图像与视频操控的主流工具，而 Classifier-Free Guidance（CFG）是保证生成质量与指令对齐的关键机制。
- 核心痛点：用户不仅想指定"改什么"，还想控制"改多少"（例如从没胡子到浅胡茬再到络腮胡的连续过渡）。现有的强度控制方法要么依赖逐编辑类型的优化/方向构造，要么需要采集专门的多强度数据集，泛化性差且几乎没有覆盖视频。一个自然的想法是直接调低 CFG 的引导尺度 $$w$$ 来减弱编辑，但本文发现这行不通。
- 本文 idea：当 $$w \to 0$$ 时，预测被"引导原点"（即无条件/空指令预测）主导。而在编辑模型里，空指令并不代表"忠实重建输入"，而是对应"任意一种编辑"的边缘分布，会把输入投影到一个通用的编辑流形上，产生随机的、非输入相关的改动。作者据此提出：显式学习一个恒等指令 $$\langle \text{id} \rangle$$，并按编辑强度把引导原点在"恒等预测"和"标准空预测"之间插值，即可获得从原图到目标编辑的连续可控过渡。

## 方法

整体框架：在标准 CFG 的几何解释中，一步去噪由两部分构成——把潜变量从噪声流形拉向下一层流形的"原点项"，以及在流形上把轨迹偏向条件分布的"引导方向"。AdaOr 不改引导方向，只把原点项做成随编辑强度 $$\alpha$$ 变化的自适应原点，并配套引入一个可训练的恒等指令 token。

```mermaid
flowchart LR
  A["输入图像/视频 c_I + 指令 c_T"] --> B["三路噪声预测: 空指令 / 条件 / id 恒等"]
  B --> C["自适应原点 O(alpha): 在 id 预测与空预测间按 s(alpha) 插值"]
  C --> D["加上 alpha·w·(条件 - 空) 引导方向"]
  D --> E["按强度 alpha 平滑过渡: 原图 → 完整编辑"]
```

关键设计：

1. 恒等指令 $$\langle \text{id} \rangle$$（是什么/为什么/怎么做）。作者向文本编码器词表新增一个 token $$\langle \text{id} \rangle$$，让它对应"恒等变换"即原样重建输入。之所以需要它，是因为空指令 $$\varnothing$$ 在编辑模型里语义上是"任意编辑"而非"不编辑"，无法充当低强度下的可靠原点。训练时用标准 flow matching 目标，并在数据里混入恒等样本：源图与目标图相同、指令设为 $$\langle \text{id} \rangle$$，教会模型把该 token 关联到忠实重建。

2. 自适应原点与最终引导公式。定义编辑强度 $$\alpha \in [0,1]$$，自适应原点为 $$O(\alpha) = s(\alpha)\,\epsilon(z_t; c_I, \varnothing, t) + (1-s(\alpha))\,\epsilon(z_t; c_I, \langle \text{id} \rangle, t)$$，其中 $$s$$ 是单调调度器且 $$s(0)=0, s(1)=1$$。最终预测为 $$\epsilon_{w,\alpha}(z_t; c_I, c_T, t) = O(\alpha) + \alpha \cdot w\,(\epsilon(z_t; c_I, c_T, t) - \epsilon(z_t; c_I, \varnothing, t))$$。两个边界条件保证了可控性：$$\alpha = 0$$ 时退化为恒等预测，输入原样保留；$$\alpha = 1$$ 时恰好还原标准 CFG，恢复模型默认编辑能力。

3. 为什么不能直接用 $$\langle \text{id} \rangle$$ 全面替换空预测。作者分析了"把原点和引导方向里的空预测都换成 id 预测"（记为 CFG-id）的失稳问题：在末端去噪步，id 条件分布坍缩到以输入 $$c_I$$ 为中心，噪声预测近似 $$\epsilon(z_t; c_I, \langle \text{id} \rangle, t) \approx (z_t - c_I)/\sigma_t$$。当编辑强度增大、$$z_t$$ 远离输入时分子非零，而 $$\sigma_t \to 0$$ 会让该项发散到无穷，再被引导尺度放大，导致强编辑在去噪末端爆炸。AdaOr 通过按强度把原点从 id 预测过渡到行为良好的空预测，规避了这种发散。

4. 调度器选择。作者取平方根调度器 $$s(\alpha) = \sqrt{\alpha}$$（在低强度侧给恒等项更大权重），并在实验中与线性调度器对比。骨干网络采用 Lucy-Edit，把图像当作单帧视频统一处理；训练用 10% 概率丢弃文本条件、10% 概率对齐 $$\langle \text{id} \rangle$$ 恒等映射、其余 80% 用标准编辑三元组，仅微调 3000 步。

## 实验结果

作者在图像域对比四个连续编辑基线（FreeMorph、Kontinuous Kontext、Concept Sliders、SAEdit），用 $$N=6$$ 个均匀强度评估四类指标：二阶平滑度 $$\delta_{\text{smooth}}$$、归一化 CLIP 方向一致性、DreamSim 感知轨迹一致性，以及衡量编辑步长均匀性的 Linearity。下表为 PIE-Bench 上与前作及消融的对比：

| 方法 | δsmooth ↓ | Norm. CLIP-Dir ↑ | DreamSim Align ↑ | Linearity ↓ |
|------|-----------|------------------|------------------|-------------|
| FreeMorph | 0.26 | 1.71 | 0.23 | 0.10 |
| Kontinuous Kontext | 0.12 | 1.75 | 0.32 | 0.08 |
| CFG（消融） | 0.61 | 1.48 | 0.27 | 0.12 |
| CFG-id（消融） | 0.27 | 1.65 | 0.30 | 0.05 |
| 线性调度器（消融） | 0.14 | 1.99 | 0.36 | 0.07 |
| AdaOr（本文） | 0.12 | 1.89 | 0.36 | 0.07 |

AdaOr 在各项指标上整体领先：与 Kontinuous Kontext 平滑度相当，但文本对齐一致性和感知轨迹一致性更优。在 SAEdit 提供的人像基准上，AdaOr 相对 Concept Sliders、SAEdit 同样取得更好的平滑度与一致性。消融印证了理论分析：标准 CFG 在低尺度生成任意内容；CFG-id 虽 Linearity 高但 $$\delta_{\text{smooth}}$$ 差、高强度处发散；线性调度器编辑步长不均匀。36 人的用户研究显示，AdaOr 在过渡平滑度上明显胜过两种 FreeMorph 变体，并与 Kontinuous Kontext 在中间帧质量和总体偏好上相当、平滑度更佳。方法还展示了对视频编辑的可扩展性，这是现有图像基线未覆盖的。

## 亮点与局限

- 亮点：
  - 用"替换引导原点"这一简洁视角解释了 CFG 无法连续控制编辑强度的根因，并给出边界条件严格（$$\alpha=0$$ 恒等、$$\alpha=1$$ 还原 CFG）的插值公式。
  - 不依赖专门的多强度数据集，也无需逐编辑类型的优化，天然继承骨干模型支持的多样编辑类型，并首次把连续强度控制扩展到视频。
  - 对 CFG-id 的发散性给出了清晰的理论推导，说明为何要用自适应原点而非直接替换。

- 局限：
  - 因只用了骨干训练数据的子集与更短的训练日程，模型表达范围受限，无法做超出骨干能力的编辑，也会继承其在特定编辑类型上的失败倾向（如"变出租车""让狗侧躺"未成功）。
  - 推理需要三次噪声预测（空/条件/id），比标准 CFG 的两次略增计算开销。

## 延伸思考

- 论文把 $$\langle \text{id} \rangle$$ 视为在预测空间做"算术"的一种手段，暗示这类可学习指令 token 或许能承载更多可组合的语义操作，值得推广到更一般的生成控制。
- 作者提出可把连续编辑序列当作数据生成引擎，为下游的强度条件编辑模型合成训练数据，形成"用连续控制反哺数据采集"的闭环。
- 一个值得追问的点：平方根调度器是经验选择，是否存在与骨干噪声调度耦合的更优（甚至自适应学习的）$$s(\alpha)$$，以进一步兼顾平滑度与线性度。
