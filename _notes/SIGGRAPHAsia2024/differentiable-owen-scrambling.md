---
title: "Differentiable Owen Scrambling"
authors:
  - Bastien Doignies
  - David Coeurjolly
  - Nicolas Bonneel
  - Julie Digne
  - Jean-Claude Iehl
  - Victor Ostromoukhov
track: "Journal"
source: author-page
category: "Rendering"
institution:
  - Université Claude Bernard Lyon 1
  - CNRS
tags:
  - Quasi Monte Carlo
  - Owen Scrambling
  - Low Discrepancy Sequence
  - Sampling
  - Automatic Differentiation
  - Optimal Transport
links:
  paper: "https://doi.org/10.1145/3687764"
  code: "https://github.com/liris-origami/DifferentiableOwenScrambling"
---

## 一句话总结

本文把 Quasi-Monte Carlo 中经典的 Owen scrambling（置换树）改写成一个可微形式，从而能用梯度下降在保持点集低差异性（严格保持 $$t$$ 值）的前提下，进一步优化最优传输均匀性、蓝噪声谱、积分误差等任意可微目标。

## 研究背景

Monte Carlo 积分是渲染的核心，误差与估计量方差直接相关。Quasi-Monte Carlo（QMC）通过构造低差异点集，让样本尽量均匀覆盖积分域，从而获得比纯随机采样更快的收敛率。低差异性由 $$(t,m,s)$$-net 刻画：在基 $$b$$（通常 $$b=2$$）下，$$n=b^m$$ 个样本落入每个体积为 $$b^{t-m}$$ 的层中恰好有 $$b^t$$ 个点，$$t$$ 越小越均匀。

为了在渲染中给每个像素提供不同的点集（避免走样），需要对点集做随机化（scrambling）。其中 Owen scrambling（嵌套均匀置换）是主流方法：它对每一维独立地递归置换半区间，用一棵布尔标志树决定是否交换，能严格保持 $$t$$ 值，并且可能进一步改善平滑被积函数的收敛率。

问题在于：Owen scrambling 依赖离散的置换树（本质是异或/比特翻转），不可微，因此无法接入平滑优化框架。已有的优化尝试只能靠对巨大搜索空间的随机盲搜，且随点数指数级膨胀。本文的目标就是让 Owen 树变得可微，使其能被自动微分与梯度下降驱动。

## 方法

### 整体框架

核心思路是把 Owen 树中"翻转比特"这一离散操作替换为一个平滑的"可微二值翻转"（DBF），使得比特可以取 $$[0,1]$$ 内的连续中间值，从而整棵置换树对树节点参数可微。优化时对整棵树的节点值做（随机）梯度下降以最小化各类损失，优化结束后再把每个节点参数四舍五入回 $$\{0,1\}$$，得到一棵合法的 Owen 树——由于仍是 Owen 置换，$$t$$ 值天然被保持。

```mermaid
flowchart LR
    A[低差异输入点集<br/>Sobol'] --> B[初始随机 Owen 树]
    B --> C[可微二值翻转 DBF<br/>连续松弛]
    C --> D[评估可微损失<br/>W2 / 高斯核 / 积分 / PCF]
    D --> E[梯度下降优化树节点参数 θ]
    E -->|迭代| C
    E --> F[参数四舍五入到 0/1]
    F --> G[优化后的 Owen 树<br/>严格保持 t 值]
```

### 关键设计 1：可微二值翻转 DBF

原始 1D Owen scrambling 中，点的定点二进制表示为

$$x=\sum_{i=1}^{q} a_i 2^{-i}$$

置换后

$$x'=\sum_{i=1}^{q} e_i 2^{-i},\quad e_i=\pi_{a_1 a_2 \dots a_{i-1}}(a_i)$$

在基 2 下 $$\pi$$ 只有两种（$$0\leftrightarrow 1$$），即比特翻转，可用异或实现。

作者定义可微二值翻转：取一个可微、双射、严格递增、且 $$f(0)=0,\,f(1)=1$$ 的函数 $$f$$，令

$$DBF_f(\beta,\theta):=(1-\beta)f(\theta)+\beta(1-f(\theta))$$

其中 $$\beta$$ 是比特值，$$\theta$$ 是树节点参数控制翻转程度。它满足 $$DBF_f(0,0)=DBF_f(1,1)=0$$、$$DBF_f(0,1)=DBF_f(1,0)=1$$，因此在 $$\theta\in\{0,1\}$$ 时退化为真正的比特翻转；而中间值让操作可微。由于 $$f$$ 是双射，$$DBF_f$$ 对 $$\theta$$ 无局部极小。实践中取

$$f(\theta)=\tfrac{1}{2}\big(\tanh(\alpha(\theta-0.5))+1\big),\quad \alpha=5$$

算法层面只需把经典 Owen 遍历中的 `bits[i] ← bits[i] xor tree[select]` 替换为 `bits[i] ← DBF_f(bits[i], θ[select])`，其余的二叉堆式树遍历不变。

### 关键设计 2：显式梯度与稀疏 Jacobian

作者对整棵树（固定深度 $$q'=16$$）显式存储并优化全部节点值。为省内存不用 Adam，而是用显式导数

$$f'(\theta)=\tfrac{\alpha}{2}\big(1-\tanh^2(\alpha(\theta-0.5))\big)$$

并按链式法则计算损失梯度

$$\frac{\partial \mathcal{L}(x(\theta))}{\partial\theta}=\Big[\frac{\partial x_i}{\partial\theta_j}\Big]_{ij}\nabla\mathcal{L}(x)$$

Jacobian 稀疏，第 $$j$$ 行非零比例为 $$n/2^{\ell}$$，$$\ell=\lfloor\log_2(j+1)\rfloor$$（即树第 $$\ell$$ 层影响的点数）。由于定点算术下低位比特梯度迅速衰减，而中等位（$$i\approx\log_2 N$$）最影响能量，作者随点数线性增大学习率。

### 关键设计 3：多种可微损失

框架可优化任意可微目标，本文给出四类并推导了各自梯度：

- 半离散最优传输（到均匀分布，二次代价）：

$$\mathcal{W}_2(X)=\inf_{T}\sum_i\int_{T^{-1}(x_i)}\lVert x-x_i\rVert^2\,\mathrm{d}x$$

- 高斯核蓝噪声能量：

$$\mathcal{G}(X)=\frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{n} e^{-\lVert x_i-x_j\rVert^2/(2\sigma'^2)}$$

- 对一组随机高斯的积分误差：

$$\mathcal{I}(X)=\frac{1}{K}\sum_{k=1}^{K}\Big(\int g_k(x)\,\mathrm{d}x-\frac{1}{n}\sum_{i=1}^{n} g_k(x_i)\Big)^2$$

- 成对相关函数（PCF）到参考谱的 $$\ell_2$$ 距离。

还可对多个 2D 投影或多个样本数做损失加权求和，得到"投影优化"或"多尺度（渐进）"版本。

## 实验结果

在保持 $$t$$ 值不变的前提下，方法在各自目标损失上与最优/受低差异约束的最优竞争者比较（2D/3D/…/8D）。以下为若干代表性结论（数值忠于原文文字描述）：

| 维度 / 目标 | 结论 |
| --- | --- |
| 2D，$$\mathcal{W}_2$$ | 达到与 BNOT 相近的 $$\mathcal{W}_2$$，同时保持低差异与 $$t$$ 值，并优于 LDBN |
| 3D，$$\mathcal{W}_2$$ | 自由度受限，未达 SOT 水平，但优于 GBN 且较初始化明显下降 |
| 高斯核能量 $$\mathcal{G}$$ | 与多数方法相当，较未优化 Owen 有适度提升且视觉可见 |
| 积分误差 $$\mathcal{I}$$（2D/4D/6D/8D） | 全部维度均表现最好；2D 最接近的竞争者是 Sobol'，高维中最优传输类方法次之并超过 Sobol' |
| 差异性 | 所有优化后的树与 Sobol'+随机 Owen 的差异性一致（曲线重合），验证 $$t$$ 值被保持 |

其他要点：等时间对比中，平滑优化在多数情况下远胜"随机生成 $$N$$ 棵树取最优"的暴力搜索；静态 Owen 树采样吞吐约 63.34M 样本/秒（对比 ART16 的 18.59M/秒）；6D 渲染（像素+直接光+景深/间接光）中，平滑设置下优化点集降低 $$\ell_1$$ 渲染误差，不连续设置下则与未优化持平。典型 2D、1k 点、深度 16 的优化耗时从高斯核能量约 2 秒到积分误差约 30 秒，需约 1GB 内存存树与 Jacobian。

## 亮点与局限

亮点：
- 首次把 Owen scrambling 写成可微形式，DBF 构造简洁且在 $$\theta\in\{0,1\}$$ 时严格退化为真置换，因此**天然保持 $$t$$ 值/低差异性**——这是相对于 Cranley-Patterson 旋转等随机化的关键优势。
- 通用性强：任意可微损失（最优传输、蓝噪声谱、积分误差、PCF、及其加权组合与多尺度）都能接入，且适用于任意维度，突破了很多竞品只能做 2D 的限制。
- 显式梯度 + 稀疏 Jacobian，避免了自动微分工具的额外内存开销。

局限：
- 优化后的树需要 $$O(n)$$ 存储，无法像原生 Owen 那样"按需即时计算、零存储"；优化过程还需 $$O(n^2)$$ 存储 Jacobian（作者认为可工程优化到 $$O(n)$$）。
- 目前只实现了基 $$b=2$$ 的二叉 Owen 树；扩展到高基需在 $$(b-1)$$ 维单纯形上插值。
- 难以扩展到更复杂的 Owen 式置换（如 Ahmed & Wonka 的 2D 方法），因涉及维度置换的非平凡选择。
- 渐进性只能通过对若干样本数求和损失来近似，存在渐进性与单一样本数性能之间的权衡；连续值取整为 $$\{0,1\}$$ 也会略微偏离最优。

## 延伸思考

- 可微化的本质是"把离散决策松弛为连续，再取整回收"，这一思路（配合严格保持某种离散不变量的结构）可迁移到其他离散采样/组合结构的优化上，例如格点规则或数字网的构造。
- 树的 $$O(n)$$ 存储限制了大点集应用，若能结合作者提到的树压缩或按层渐进优化，可能得到既可微又"接近零存储"的采样器。
- 论文观察到大部分比特翻转集中在中间层（约第 4~8 位，对应 $$2^8$$ 个样本），说明真正决定均匀性的是"中等尺度"结构，这对设计更高效的多尺度/分层优化目标有启发。
- 将该框架与可学习滤波器或神经谱设计结合，或许能在保持低差异性的同时，直接面向渲染管线的下游误差端到端优化 scrambling。
