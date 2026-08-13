---
title: "Marching Neurons: Accurate Surface Extraction for Neural Implicit Shapes"
authors:
  - "Christian Stippel"
  - "Felix Mujkanovic"
  - "Thomas Leimkühler"
  - "Pedro Hermosilla"
category: "Geometry & Modeling"
track: "Journal"
source: "arxiv"
institution:
  - "TU Wien"
  - "Max-Planck-Institute for Informatics"
tags:
  - "Neural Implicit Surface"
  - "Surface Extraction"
  - "Signed Distance Function"
  - "ReLU Networks"
  - "Piecewise Linear Regions"
  - "Analytic Meshing"
  - "Range Analysis"
  - "Mesh Generation"
links:
  paper: "https://doi.org/10.1145/3763328"
---

## 一句话总结

提出 Marching Neurons：一种从 ReLU 神经隐式函数中"解析式"提取零等值面的算法。它不依赖网格采样，而是逐神经元遍历网络在输入域上诱导的分片线性区域(cell)，用区间/仿射范围分析剪掉不含表面的区域，直接得到精确捕获网络全部几何细节的多边形网格；算法天然并行、易实现，在多样形状与网络结构上达到远超现有方法的精度，同时保持有竞争力的速度。

## 研究背景

三维视觉计算中，几何表示分为显式(如多边形网格、点)与隐式(如符号距离函数 SDF、水平集)两大类，二者各有所长，故二者之间的高效转换日益重要。把隐式表示转为显式表面是研究已久的问题，主流做法是空间分解+采样：把空间切成规则单元，在每个含表面的单元内用插值离散样本来构造多边形。最常用的 Marching Cubes 就是在规则网格上采样隐式函数并线性插值近似目标水平集。这类方法简单、与隐式函数形式无关，但精度受采样分辨率与插值方案限制。后续工作用层次结构剪枝、四面体网格、优化样本位置或更高级插值来改进，但一个根本问题始终存在：从有限样本重建复杂几何必然引入误差。

近十年神经场(neural fields)兴起，用基于坐标的网络表示连续信号(如 SDF)成为主流范式。本文的关键洞见是：这类神经表示本身让"无需临时采样的解析式表面提取"成为可能。对采用分片线性激活(尤其是 ReLU)的网络，网络是分片线性函数的复合，因而整体也分片线性——每一层用超平面把已有区域进一步细分，形成若干凸线性区域(cell)，在每个 cell 内网络退化为线性函数。既有的解析提取方法(Analytic Marching、Edge Subdivision 等)要么难以处理多个不连通部件、要么缺乏空区域过滤而在复杂结构上产生过量多边形且速度慢。本文据此提出一种可并行、可过滤空区域、能处理多连通形状且速度有竞争力的解析提取算法。

## 方法

### 整体思路

给定 ReLU 激活的神经隐式表示 $$f_\theta:\Omega\to\mathbb{R}$$，$$\Omega\subset\mathbb{R}^3$$，目标是把其零水平集 $$\mathcal{S}=\{x\in\Omega\mid f_\theta(x)=0\}$$ 解析地提取为多边形网格。核心是自适应地细分输入域并跟踪激活模式，把网络在每个 cell 内约化为显式线性函数；再用范围分析尽早丢弃不含零水平集的 cell，大幅提速。作者先在 2D 上讲清基本结构，再直接推广到 3D。

### 单元(cell)表示

每个 cell 在第 $$l$$ 层表示为三元组 $$\mathcal{C}^{(l)}=\{\tilde V;\tilde W,\tilde b\}$$：$$\tilde V=[v_1,\dots,v_m]$$ 是构成该多边形/多面体的顶点坐标；$$\tilde W,\tilde b$$ 定义 cell 内部的线性函数

$$\tilde p(x)=\tilde W x+\tilde b$$

它把截至当前层的所有线性贡献折叠在一起，对应当前层的预激活响应。对隐藏层，$$\tilde p(x)$$ 输出 $$d_l$$ 维(每个分量对应一个神经元)；到最后一层 $$L$$ 时输出标量，即该 cell 内完整 $$f_\theta$$ 的线性表达。初始化为覆盖整个矩形域的单个单元，$$\tilde W=\mathbb{I}$$，$$\tilde b=0$$。随后逐层遍历，反复剪枝/分裂并更新参数。

### 三步网络遍历

```mermaid
flowchart LR
    A[初始化: 整域单个 cell] --> B[Step1 剪枝 range analysis]
    B --> C[Step2 分裂 关键神经元处切割]
    C --> D[Step3 层折叠 更新 W,b]
    D --> B
    D --> E[到达最后一层]
    E --> F[每个 cell 解析提取线性水平集]
    F --> G[拼合为多边形网格 三角化]
```

- Step 1 单元剪枝(Cell Pruning)：由于线性区域数随层数指数增长而绝大多数不含零水平集，此步至关重要。采用范围分析(仿射算术)计算 $$f_\theta$$ 在每个 cell 内的保守上下界，若

$$\min_{x\in\mathcal{C}^{(l)}_i} f_\theta(x)>0 \quad\text{或}\quad \max_{x\in\mathcal{C}^{(l)}_i} f_\theta(x)<0$$

则该 cell 不与零水平集相交，直接丢弃。实践中用 cell 的轴对齐包围盒查询界限，对退化(细长)cell 可能高估范围，但足以保证对大网络的可扩展性。

- Step 2 单元分裂(Cell Splitting)：找出当前层中会把 cell 一分为二的"关键神经元(critical)"——其预激活在多边形内变号，导致后续 ReLU 的非线性折点落在多边形内部。因函数是线性的，极值出现在顶点上，故只需在顶点处评估 $$\tilde p$$：神经元 $$i$$ 为关键当且仅当

$$\min_j (\tilde p(v_j))_i<0 \quad\text{且}\quad \max_j (\tilde p(v_j))_i>0$$

对关键神经元，用 Sutherland–Hodgman 裁剪算法沿其零水平集把多边形切成两个子多边形，在多边形边与切割平面交点处插入新顶点；无需分裂的多边形保持不变。

- Step 3 层折叠(Layer Collapsing)：转入下一层时更新线性参数。定义二值掩码 $$m$$ 表示当前层各神经元在该 cell 内是否激活(只有激活神经元贡献其线性函数)，据此更新

$$\tilde W:=W^{(l+1)}\,\mathrm{diag}(m)\,\tilde W,\qquad \tilde b:=W^{(l+1)}\,(m\odot\tilde b)+b^{(l+1)}$$

其中 $$\odot$$ 为 Hadamard 积。更新后的 cell 再回到 Step 1。

### 遍历策略与水平集提取

三步只规定了层间如何更新 cell，全局遍历方式可选。广度优先需一次性保存整个前沿，最坏空间复杂度 $$O(2^d)$$($$d$$ 为神经元数)，超出常规 GPU 显存。因此采用深度优先(后进先出),只保存当前分裂树分支，空间复杂度降为 $$O(b\,d)$$($$b$$ 为并行处理的 cell 数);同时调度足量 cell 并行以充分利用 GPU。实际平均复杂度更低：并非每个神经元都分裂每个 cell，且范围分析剪枝提前移除大量 cell。当所有 cell 到达最后一层后，每个 cell 内 $$\tilde p$$ 为线性，可解析提取其零水平集 $$\{x\mid \tilde p(x)=0\}$$，所有 cell 的结果拼合即得显式表面(2D 为线带，3D 为多边形网格)。

### 3D 扩展与实现细节

推广到 3D 很直接：顶点变为 $$\tilde V\in\mathbb{R}^{3\times m}$$，$$\tilde W\in\mathbb{R}^{d_l\times 3}$$，关键神经元沿平面切割多面体；到最后一层解析提取零水平集得到多边形网格，再三角化以兼容标准管线。实现用 JAX，充分利用天然并行设计：所有活跃 cell 存于共享缓冲区并为各操作维护索引栈。深度递归细分易产生数值误差，作者发现混合精度是关键——剪枝用 32 位即可，分裂与层折叠用 64 位；这不限制输入网络 $$f_\theta$$ 的位宽。

## 实验结果

评测指标为面向 SDF 本身的两个"软"指标：Soft-Precision(SP，在重建网格表面采样点并取 SDF 绝对值均值，衡量重建面离零水平集多远)与 Soft-Recall(SR，在原始网格采点并梯度下降到 SDF 零水平集后，测其到重建网格的平均距离，衡量是否漏掉大片表面);此外报告运行时间与三角形数量。数据集为来自 Thingi10K、ABC、ShapeNet、FAUST 与 Stanford 3D Scanning Repository 的 84 个水密形状，每个形状用两种 ReLU MLP 架构(d4_w128、d4_w256)拟合 SDF。基线包括近似方法(Marching Cubes、Dual Contouring、Hierarchical Marching Cubes、Reach for the Arcs，网格分辨率 64/128/256/512)与解析方法(Analytic Marching、Edge Subdivision)。

主要结论：

- 近似方法即便用 512 的最细网格，重建仍显著偏离 SDF；Reach for the Arcs 因耗时极长只能处理低分辨率。
- 解析方法 SP 近乎完美，但 Analytic Marching 因依赖种子点迭代重建而漏掉不连通部件，导致 SR 偏高;Edge Subdivision 的 SR 低但重建大网络耗时长、产生更密的网格。
- 本方法在两种架构上都给出最准确的结果(SP 与 SR 都极低)，三角形数量比其他解析方法更少，运行时间与近似方法相当。以 d4_w256(SP/SR ×$$10^6$$)为例，本方法 SP=0.02、SR=0.03，而 Marching Cubes(512³)为 60.33/101.94，Analytic Marching 为 0.02/188.80。

可扩展性：近似方法运行时间随深度、宽度略增;解析方法增长更陡。Edge Subdivision 对神经元数尤其敏感，在宽度 512 时失败;Analytic Marching 与本方法随神经元数增长更平缓,本方法随层数增长的表现较差,但整体仍能在合理时间内处理大范围架构且精度显著更高。

网格质量与后处理：解析方法输出的三角形质量(最小/最大角、等角偏度、边长比)低于近似方法。作者比较了 fan0、centroid、strip 三种多边形三角化策略,centroid 质量略优但三角形更多,strip 最差。用二次误差度量(QEM)简化后,即便只保留原始 10% 的三角形,误差仍小于所有近似方法,且简化会提升三角形质量,缩小与近似方法的差距。

消融与扩展：关闭范围分析过滤后,d4_w256 因线性区域指数增长在两小时内无法完成单个网格;d4_w128 上开启过滤带来平均 ×12.2 加速,凸显该步骤的重要性。方法为 ReLU 设计,可直接推广到 leaky ReLU(存斜率与截距代替二值掩码);对 sine/cosine 等连续激活(如位置编码),需用分片线性代理近似,作者在 2D 圆 SDF 上验证可行,误差随分段数增加而减小。

## 亮点与局限

亮点：
- 把神经网络当作可解析遍历的分片线性结构而非黑箱,直接提取网络编码的全部几何细节,精度远超采样类方法,尤其能准确重建尖锐边缘。
- 三步遍历(剪枝-分裂-折叠)配合范围分析过滤,既天然并行、易实现,又能高效跳过绝大多数空区域;深度优先遍历把最坏 $$O(2^d)$$ 显存降到 $$O(b\,d)$$。
- 能同时提取多个不连通部件,克服了 Analytic Marching 依赖种子点的缺陷;三角形数量少于其他解析方法,运行时间与近似方法相当。
- 混合精度设计解决了深递归细分的数值稳定性问题,且不限制输入网络位宽。

局限：
- 依赖范围分析剪枝;若界限过于保守,或 SDF 编码复杂、空区域少的形状,需保留大量 cell,导致显存占用高、重建时间长。
- 随网络层数增长的可扩展性不如随宽度增长;解析提取出的三角形质量低于近似方法,需三角化策略或后处理简化来改善。
- 对 sine/cosine 等连续激活需分片线性近似,会引入随分段数而定的额外误差。

## 延伸思考

- 递归细分方案天然适合层次化的细节层级(LOD)生成,为后续需要低分辨率网格的处理步骤提供灵活性。
- 把该网格提取器嵌入端到端可微管线,可用基于网格的监督反过来优化神经 SDF,从而将该方法拓展到更广的任务上。
- "逐神经元遍历+范围分析过滤"的思路,与安全验证、鲁棒性分析中枚举 ReLU 多面体复形的研究相通,或可双向借鉴以缓解组合爆炸。
