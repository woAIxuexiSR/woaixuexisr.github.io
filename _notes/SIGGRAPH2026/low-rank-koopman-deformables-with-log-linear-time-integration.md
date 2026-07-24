---
title: "Low-Rank Koopman Deformables with Log-Linear Time Integration"
authors:
  - "Yue Chang"
  - "Peter Yichen Chen"
  - "Eitan Grinspun"
  - "Maurizio M. Chiaramonte"
category: "Animation & Simulation"
track: "Journal"
source: "arxiv"
institution:
  - "University of Toronto"
  - "University of British Columbia"
  - "Meta"
tags:
  - "Deformable Simulation"
  - "Koopman Operator"
  - "Dynamic Mode Decomposition"
  - "Reduced-Order Modeling"
  - "Neural Field"
links:
  paper: "https://doi.org/10.1145/3811273"
---

## 一句话总结

把可变形体降阶模拟的"时间积分"也降阶：用 Koopman 算子（以 DMD 参数化）把时间推进变成对一个小矩阵做指数运算，从而以 $$O(\log N)$$ 的复杂度一步跳过长时间跨度，并用神经场扩展到跨网格分辨率与跨形状的泛化。

## 研究背景

- 领域现状：降阶模拟（ROM）是可变形体实时仿真的主流加速手段，从早期线性子空间方法到近期的神经降阶模型，核心思路都是把高维空间自由度投影到低维流形上。
- 核心痛点：现有 ROM 只压缩了**空间**复杂度，**时间积分**仍然逐步进行——每步都要做隐式 Euler 的 Newton 求解，总成本随步数 $$N$$ 线性增长。此外，经典 DMD 类方法绑定单一网格与离散化，无法跨形状使用，限制了它在几何变化任务（如形状优化）中的价值。
- 本文 idea：在降阶空间里学一个线性的 Koopman 时间推进算子。既然时间演化是线性的，推进 $$N$$ 步就等于把算子的特征值做 $$N$$ 次幂，只需对角矩阵取幂，复杂度降到 $$O(\log N)$$；再用神经场表示基函数与特征值，使单个模型跨分辨率、跨形状共享动力学。

## 方法

整体框架：先用全空间模拟采集时序快照，构造"位移 + 动量"的联合状态；用 DMD/SVD 拟合一个低秩线性算子 $$K$$ 近似 Koopman 算子；时间推进就是对 $$K$$ 的特征值取幂。神经版本进一步用 MLP 神经场表示复数基函数 $$\boldsymbol{\Phi}$$ 与特征值 $$\boldsymbol{\Lambda}$$，并以几何编码 $$\gamma$$ 为条件，实现跨离散化与跨形状泛化。

```mermaid
flowchart LR
  A["全空间模拟快照"] --> B["联合状态 位移+动量"]
  B --> C["DMD / 截断SVD"]
  C --> D["低秩 Koopman 算子 K"]
  D --> E["特征分解 得到 Phi, Lambda"]
  E --> F["时间推进: Lambda 的 N 次幂"]
  F --> G["提升回全空间 下一状态"]
  H["几何编码 gamma"] -.-> E
```

关键设计：

1. **联合状态保证马尔可夫闭合**。可变形体的运动方程是二阶的，只用位移 $$\boldsymbol{U}_t$$ 做观测量无法闭合成一步线性映射，实验中会导致发散或非物理行为。作者把状态扩为 $$\boldsymbol{X}_t = [\boldsymbol{U}_t,\ \boldsymbol{U}_t - \boldsymbol{U}_{t-1}]^\top \in \mathbb{R}^{6n}$$，把速度分量并入观测量，才能让单个线性算子 $$\boldsymbol{X}_{t+1} \approx K\boldsymbol{X}_t$$ 稳定地捕捉弹性动力学。

2. **对数线性时间推进 + 步长自由缩放**。推进 $$N$$ 步写成 $$\boldsymbol{X}_{t+N} = \boldsymbol{\Phi}\boldsymbol{\Lambda}^{N}\boldsymbol{\Phi}^{*}\boldsymbol{X}_t$$，因为 $$\boldsymbol{\Lambda}$$ 是对角阵，取幂只需对角元求幂，复杂度 $$O(\log N)$$，且矩阵乘法都在 $$r$$ 维降阶空间里完成，把每步成本从 $$O(n)$$ 降到 $$O(r)$$。更妙的是改变时间步长只需按 $$\boldsymbol{\Lambda}(h') = \exp\!\big(\tfrac{h'}{h}\log\boldsymbol{\Lambda}(h)\big)$$ 重缩放特征值，动力学在不同步长下几乎不变——而隐式 Euler 在大步长下会出现强数值阻尼。外力则作为对提升状态速度分量的瞬时脉冲注入（等价于 DMDc 的受控更新）。

3. **离散化无关的神经扩展**。把每个基函数写成参考域上的连续复值映射 $$\boldsymbol{\phi}_i:\Omega_\gamma \to \mathbb{C}^3$$，并以几何编码 $$\gamma$$ 为条件；特征值 $$\boldsymbol{\Lambda}^\gamma$$ 只依赖几何（连续算子的谱是域的内在属性，与网格分辨率无关）。用两个 MLP 分别参数化基与特征值，训练损失包含重建损失 $$L_{\text{recon}}$$（约束基能张成变形子空间）与单步时间推进损失 $$L_{\text{step}}$$，并对特征值施加 $$\lvert \lambda_i^\gamma \rvert \le 1$$ 的约束防止非物理指数增长。

4. **实数化投影抑制虚部漂移**。DMD 的基与特征值是复数，单步应用无碍，但长跨度取幂时微小虚部会累积。作者把复算子改写成实数分块形式 $$\boldsymbol{\Phi}_R, \boldsymbol{\Lambda}_R$$，并用投影 $$P$$ 在物理空间抹掉虚部，构造低维实算子 $$\boldsymbol{K}_{\text{real}} = \boldsymbol{\Phi}_R^{*} P\, \boldsymbol{\Phi}_R \boldsymbol{\Lambda}_R$$，对其取幂即可获得稳定的实值长跨度推进。

## 实验结果

主实验为运行时间对比：在多个例子上比较本方法与全空间模拟的每步耗时与整段序列耗时。得益于时间维降阶，本方法能一步跳到终态，带来数量级加速。

| 例子 | 每步(ours) | 整段(ours) | 每步(full) | 整段(full) |
|------|-----------|-----------|-----------|-----------|
| 示例 A | 4 ms | 11 ms | 711 ms | 7.6 min |
| 示例 B | 8 ms | 10 ms | 1493 ms | 5.5 min |
| 示例 C | 6 ms | 18 ms | 1408 ms | 6.9 min |
| 示例 D | 3 ms | 13 ms | 1520 ms | 6.2 min |
| 示例 E | 2 ms | 15 ms | 524 ms | 13.1 min |

其余结论用文字补充：力的泛化上，模型在单一力幅训练后可外推到 $$0.5\times$$、$$2\times$$（及 $$0.67\times$$/$$1.33\times$$）与未见的力方向，重建 MSE 多在 0.5% 以内、非线性大变形下也低；离散化泛化上，单个神经模型在 2.5k、10k、500k 顶点网格上产生一致行为；跨形状上，一个模型能覆盖参数化的青蛙气室与软体机器人手指等整族几何。下游应用包括实时交互、通过缩放特征值做免重训的阻尼编辑、气动手指的交互式控制力反解（多数帧误差 <2%、全帧 <6%），以及在降阶空间里做快速形状优化和从真实视频光流轨迹学习动力学。

## 亮点与局限

- 亮点：
  - 把"时间积分"本身也纳入降阶，用算子取幂实现 $$O(\log N)$$ 长跨度推进，思路清晰且对控制/优化这类"只关心终态"的任务收益极大。
  - 步长可任意缩放且动力学近似不变，避免了隐式积分器在大步长下的数值阻尼。
  - 神经场 + 几何编码把 DMD 从单网格单形状解放出来，首次让 DMD 类模型支持跨离散化、跨形状与快速形状优化。
- 局限：
  - 纯数据驱动，质量与稳定性依赖训练数据覆盖面，无法可靠外推到未见动力学范围；相比经典物理 ROM，运行时不能解析地改材料参数、边界条件或积分器。
  - 强非线性（大旋转、复杂变形）下泛化会退化，如 $$1.33\times$$ 力或 $$2\times$$ 旋转时出现明显偏差。
  - 截断 SVD 对重建最优、但对求伪逆并非最优，基数增大会引入病态与误差放大，精度非单调；神经版基数过大又难收敛。
  - 未处理接触力。

## 延伸思考

这项工作与作者团队此前的连续降阶模型（LiCROM、CROM、Shape Space Spectra）以及 Chen 等把 DMD 用于流体的工作一脉相承，可视为把"时间感知基/线性传播"从流体速度场迁移到弹性体的位移-动量联合态。未来把能量/动量守恒等物理约束或混合训练目标引入学习过程，或许能改善长期稳定性与外推能力；把接触作为 Eq.(19) 式的外部脉冲输入接进来，以及扩展到多体系统与真实传感数据，都是自然的下一步。对做可微仿真与软体机器人设计的人来说，"一步跳到终态且可微"的降阶前向模型是很有吸引力的优化基座。
