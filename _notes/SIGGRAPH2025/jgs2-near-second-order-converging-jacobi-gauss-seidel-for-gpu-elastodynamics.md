---
title: "JGS2: Near Second-order Converging Jacobi/Gauss-Seidel for GPU Elastodynamics"
authors:
  - "Lei Lan"
  - "Zixuan Lu"
  - "Chun Yuan"
  - "Weiwei Xu"
  - "Hao Su"
  - "Huamin Wang"
  - "Chenfanfu Jiang"
  - "Yin Yang"
category: "Animation & Simulation"
track: "Journal"
source: "arxiv"
institution:
  - "University of Utah"
  - "Zhejiang University"
  - "UCSD"
  - "Style3D Research"
  - "UCLA"
tags:
  - "GPU Simulation"
  - "Elastodynamics"
  - "Second-Order Convergence"
  - "Gauss-Seidel Method"
  - "Incremental Potential Contact"
links:
  paper: "https://doi.org/10.1145/3731183"
---

## 一句话总结

针对并行弹性体仿真中"局部求解过猛反而拖慢全局收敛"的 overshoot 问题，本文给出一个二阶最优的局部子空间修正，并把最贵的部分做成可预计算的形式，使得 GPU 上 Jacobi/Gauss-Seidel 这类高度并行的迭代能达到接近牛顿法的（近似二阶）收敛速度，比现有 GPU 方法快 50 到 100 倍。

## 研究背景

- 领域现状：物理仿真计算量大，为上 GPU，主流做法是把牛顿法这种"一步直接解大系统"的方案，换成 Jacobi、Gauss-Seidel（GS）这类可并行的迭代松弛方法——把全局能量拆成许多共享自由度的小子问题，各线程独立或分组并行求解。
- 核心痛点：并行性与收敛性长期被视为一对矛盾。要更强的并行，就得让子问题更小、耦合更弱，结果是全局收敛更慢；反之加大子问题、增加重叠能提升收敛，却牺牲并行度。业界普遍认为二者不可兼得，尤其在刚性材料上并行方法收敛极慢甚至发散。
- 本文 idea：作者指出慢收敛的元凶是被长期忽视的 overshoot——局部求解只看局部能量、不知道全局情况，把局部能量降到最低，反而抬高了模型其他区域的能量，使全局目标不降反升。若能让局部求解"具备全局意识"，就能在保持并行的同时逼近牛顿法的收敛。

## 方法

整体框架：把弹性动力学每个时间步写成变分优化 $$\arg\min_{\boldsymbol{x}} E(\boldsymbol{x})$$，其中 $$E = I + \Psi$$ 由惯性项与弹性势能组成。并行化把 $$E$$ 拆成许多小子问题 $$E_i(\boldsymbol{x}_i)$$。本文不改子问题规模，而是给每个子问题的局部求解补上一项"全局余能"的感知，使局部更新 $$\delta\boldsymbol{x}_i$$ 逼近全局牛顿解的对应分量 $$\boldsymbol{S}_i\delta\boldsymbol{x}^\star$$；再通过共旋（co-rotated）近似与 Cubature 采样，把这项修正压缩成可预计算、可在单个 GPU 线程上廉价求解的小系统。

```mermaid
flowchart LR
  A[全局能量 E 拆成子问题 Ei] --> B[局部求解易 overshoot]
  B --> C[引入余能 ECi 感知全局]
  C --> D[局部扰动子空间 φi 二阶最优]
  D --> E[共旋近似 使 φi 可预计算]
  E --> F[Cubature 采样 稀疏近似约化 Hessian/梯度]
  F --> G[全坐标预计算 复用同一次分解]
  G --> H[GPU 上 Jacobi/GS 近二阶收敛]
```

关键设计：

1. **overshoot 与二阶最优目标**。局部只解 $$E_i$$ 时 $$\boldsymbol{x}_i^\star \neq \boldsymbol{S}_i\boldsymbol{x}^\star$$，把局部能量压得越狠越可能抬高别处能量（overshoot）；反之降不够则是 undershoot。理想是让局部更新等于全局牛顿更新的对应分量。作者把子问题改写为"局部能量 + 余能 $$E_{Ci}=E-E_i$$"的联合最小化，关键是引入一个映射 $$\delta\boldsymbol{x}=\phi_i(\delta\boldsymbol{x}_i)$$，描述局部扰动如何牵动全局形变。

2. **局部扰动子空间 $$\phi_i$$ 的最优性**。对每个局部自由度施加单位扰动、固定其余局部自由度，解余下部分的平衡方程得到 $$\boldsymbol{U}_{Ci}=-\bar{\boldsymbol{H}}_{Ci,Ci}^{-1}\bar{\boldsymbol{H}}_{i,Ci}^{\top}$$，这些列张成一个"扰动子空间"。作者证明用这个 $$\phi_i$$ 做局部求解，数学上等价于全局牛顿解，即 $$\delta\boldsymbol{x}_i=\boldsymbol{S}_i\delta\boldsymbol{x}^\star$$，因此是二阶最优的。直觉上余能的约化 Hessian 充当"阻尼器"，阻止局部解冲过头。

3. **共旋近似让 $$\phi_i$$ 可预计算**。$$\phi_i$$ 依赖当前位形的 Hessian，逐帧重建不现实。作者观察到真正重要的是让 $$E_{Ci}(\phi_i)$$ 估得准，而非 $$\phi_i$$ 本身精确。于是在每个顶点嵌入共旋局部标架，用极分解得到旋转 $$\boldsymbol{R}$$，把顶点转回静止朝向（旋转不改变弹性能），从而把静止形状下预计算好的 $$\bar{\boldsymbol{U}}_i$$ 旋回当前位形：$$\tilde{\phi}_i^{k}=\boldsymbol{R}^{k}\bar{\boldsymbol{U}}_i\boldsymbol{R}_i^{k\top}\delta\boldsymbol{x}_i^{k}$$，最贵的求逆项因而可离线算好。

4. **Cubature 采样 + 全坐标预计算**。约化 Hessian/梯度的精确装配复杂度为 $$O(N\cdot N_i^2)$$，代价高。借助 Cubature 只挑少量样本元素（每个子问题约四到六个，残差小于 1%）加权近似即可。此外，逐子问题分解不同的 $$\bar{\boldsymbol{H}}_{Ci,Ci}$$ 需要数天；作者改用带拉格朗日乘子的全坐标约束形式，使左上块 $$\bar{\boldsymbol{H}}$$ 对所有子问题不变，只需分解一次并复用，预计算从数天降到数十分钟（约三个数量级）。方法与 IPC（incremental potential contact）等隐式接触势也能无缝结合。

## 实验结果

主实验为与最相关的两个 GPU 竞品 VBD（vertex block descent，逐顶点子问题）和 2nd SD（second-order stencil descent，逐单元子问题）在 Armadillo 模型（1M 顶点、3.4M 单元，$$h=1/100$$，stable Neo-Hookean 材料，收敛条件 $$\|\Delta\boldsymbol{x}\|<10^{-4}$$）上比较每个时间步平均迭代次数与相对本文的加速比，并分常规刚度与提高 20 倍刚度两档。数字忠实原文：

| 方法 | 常规刚度 迭代次数 | 加刚 20 倍 迭代次数 | 相对本文加速 |
|------|------|------|------|
| 本文 | 38 | 64 | 基准（约 11 ms/迭代） |
| 2nd SD | 74 | 142 | 常规约 30 倍；加刚 34 倍 |
| VBD | 2264 | 10000+ 仍不收敛 | 常规 40 倍；加刚 137 倍 |
| 投影牛顿（参考） | 34 | 58 | 每迭代约 150 s 做 Cholesky |

其余实验用文字概述：与投影牛顿法对比（六个 Armadillo 落入容器，6M 单元）本文每步 883 ms，约快 8000 倍；"纸牌屋"用 IPC 处理 155 张卡片的摩擦接触与高速冲击，本文 31 ms/帧，比 CPU 版 Newton IPC 快 1000 倍以上，而 VBD 不收敛；100K 单元的龙可在 120 FPS 以上实时交互；"SIGGRAPH"软硬字母、海盗船（比 VBD 快 153 倍）、南瓜灯（快 40 倍）、被挤压的刺球（快 173 倍）、动物玩具（比 GPU 投影动力学快 70 倍、比 GPU-IPC 快 136 倍）、覆盖直升机的桌布（2M 三角形、3M 自由度，比协维 C-IPC 快约三个数量级）等大规模场景均验证了方法对材料软硬变化的鲁棒性——软硬字母分别只需 27 与 34 次迭代，而 VBD、2nd SD 对硬字母都不收敛。

## 亮点与局限

- 亮点：
  - 抓住 overshoot 这个被忽视的收敛瓶颈，并给出有严格推导、可证明二阶最优的局部修正，理论清晰。
  - 用共旋近似 + Cubature + 全坐标形式把昂贵项转成可预计算，运行时开销仅略高于 Jacobi，却拿到接近牛顿法的收敛。
  - 对材料刚度不敏感，在传统并行方法最吃力的刚性、强耦合、强接触场景优势尤为明显，且与 IPC/惩罚法兼容，可做布料、薄壳等协维仿真。

- 局限：
  - 预计算（局部子空间构建 + 共旋基的 Cubature 训练）较慢，带来实用上的不便。
  - 二阶收敛依赖全局能量的二次近似；当出现 IPC 障碍这类高度非线性项、$$\|\delta\boldsymbol{x}^k\|^3$$ 不可忽略时不再二次收敛，需要线搜索（好在可按子问题并行）。
  - 仿真加速后，碰撞检测反而成为新的性能瓶颈。

## 延伸思考

方法本质是一个通用的并行优化框架，核心难点在于为不同问题找到"可预计算、又能有效估计全局能量变化"的映射 $$\phi$$。作者提出用数据驱动、深度学习的思路为不同计算问题定制这个映射，是自然的下一步。此外，方法框架并不限于有限元弹性体，向布料、杆、MPM、流体等其他仿真问题迁移值得探索；与作者同期的 3DGS2（把类似"近二阶收敛"思路用到 3D Gaussian Splatting 优化）相互呼应，提示这套"局部求解全局感知"的思想可能在更广的并行优化场景中复用。碰撞检测成为新瓶颈也意味着后续在高分辨率实时仿真上，检测与求解需要协同优化。
