---
title: "TensoSDF: Roughness-aware Tensorial Representation for Robust Geometry and Material Reconstruction"
authors:
  - "Jia Li"
  - "Lu Wang"
  - "Lei Zhang"
  - "Beibei Wang"
category: "Reconstruction"
track: "Journal"
source: "arxiv"
institution:
  - "Shandong University"
  - "Hong Kong Polytechnic University"
  - "Suzhou University of Science and Technology"
  - "Nanjing University"
tags:
  - "Inverse Rendering"
  - "Signed Distance Field"
  - "Tensorial Representation"
  - "Material Estimation"
  - "Relighting"
  - "Multi-View Reconstruction"
links:
  paper: "https://doi.org/10.1145/3658211"
  project: "https://github.com/Riga2/TensoSDF"
  code: "https://github.com/Riga2/TensoSDF"
---

## 一句话总结

用张量编码的 SDF（TensoSDF）表达几何，并以"每点粗糙度"作为权重自适应地融合辐射场与反射场，让任意反射材质（漫反射到强镜面）的物体都能被稳健重建，再用"网格-SDF 融合"策略高质量估计材质，在几何精度、重光照质量与训练/推理速度上全面超越前作 NeRO。

## 研究背景

从多视角图像重建带真实材质的物体，是图形学与视觉里一个高度病态（ill-posed）的问题：几何与材质纠缠在一起难以解耦。NeRF 以来的神经渲染方法带来了突破，但按颜色函数可分成两类，各有软肋：

- 辐射场（radiance field）方法：把颜色隐式编码为视角与几何的函数。密度场类（如 TensoIR）在新视角合成与材质估计上表现好，但缺乏表面约束，重建的表面噪声大、法向不准，尤其在镜面上失效。SDF 类（NeuS、VolSDF 等）表面质量更好，但依赖辐射场时无法处理强反射物体。
- 反射场（reflectance field）方法：显式地把渲染方程写进重建（如 NeRO、NeILF++）。它们对强镜面物体的几何重建质量很高，但由于着色模型与光传输引入了大量假设，优化不稳定、容易陷入局部最优，导致几何错误；且纯 MLP 表示几何过度平滑、训练耗时长。

核心矛盾在于：没有任何现有方法能对"任意反射材质"（不含半透明）的物体都给出令人信服的结果。本文在 NeRO 的框架之上，提出把辐射场与反射场深度融合，并换用高容量的张量表示，解决这一鲁棒性难题。

## 核心方法

整体沿用 NeRO 的两阶段管线：先做几何重建，再做材质估计。三个关键创新点分别对应三个贡献。

### TensoSDF：张量表示 + SDF

用 TensoRF 的向量-矩阵（Vector-Matrix）分解作为张量编码器来表达 SDF，替代原本的纯 MLP 编码。由于 SDF 是连续函数，纯显式表示训练不稳，因此在张量编码后接一个小 MLP 解码器。与 TensoRF 用两套独立张量网格分别编码几何和外观不同，本文共享同一个张量编码器与 MLP 解码器，以增强几何与外观之间的相关性：

$$V_p = \mathbf{v}^X_k \circ \mathbf{M}^{YZ}_k \oplus \mathbf{v}^Y_k \circ \mathbf{M}^{XZ}_k \oplus \mathbf{v}^Z_k \circ \mathbf{M}^{XY}_k$$

$$\{s, v_f\} = \Theta(V_p, p)$$

其中 $V_p$ 是张量编码器输出的隐向量，再与位置 $p$ 一起经小 MLP $\Theta$ 解码出 SDF 值 $s$ 与外观特征 $v_f$。

张量表示天生缺乏全局关联，容易产生噪声表面。为此引入两条平滑先验：

训练时的高斯平滑损失，增强张量网格的局部一致性：

$$\ell_g = \sum_{k=1}^{K} \left\lVert G(\mathbf{M}_k \mid k_g, \sigma_g) - \mathbf{M}_k \right\rVert_2^2 + \left\lVert G(\mathbf{v}_k \mid k_g, \sigma_g) - \mathbf{v}_k \right\rVert_2^2$$

推理时的 mipmap 混合：训练后对张量网格做双线性插值构造两层 mipmap，用 Marching Cube 提网格时把基础层 SDF 值 $s$ 与顶层 $s'$ 按权重 $\alpha$ 混合，使提取的网格更平滑又保留细节：

$$\hat{s} = (1 - \alpha)\cdot s + \alpha \cdot s'$$

### 粗糙度感知的辐射场与反射场融合

这是全文最核心的思想。作者观察到：对于角向效果平滑（低频、偏漫反射）的表面，隐式辐射场因为不受着色/光传输假设束缚，能更好地拟合多视角观测；而对于强镜面（高频角向效果），显式反射场表达得更好。于是提出把两者都用上，关键在于"怎么融合"。

辐射场沿用 NeuS 建模：

$$c_{rad} = \Theta_{rad}(p, n, v_f, d)$$

反射场沿用 NeRO 的基于 split-sum 近似的渲染方程，材质 MLP 输出反照率 $a$、金属度 $m$、粗糙度 $r$，再结合直接光、间接光与遮挡：

$$\{a, m, r\} = \Theta_{mat}(p, v_f)$$

$$c_{ref} = \rho_{diff}\cdot L_d + \rho_{spec}\cdot\big((1-u)\cdot L_d + u\cdot L_{ind}\big)$$

两个场各自渲染出像素颜色 $C_{rad}$、$C_{ref}$，各自与真值算损失 $\ell_{rad}$、$\ell_{ref}$。融合的巧思在于——直接用固定权重效果差，改用每点估计出的粗糙度 $r$ 作为自适应平衡因子：

$$\ell_c = r\cdot\ell_{rad} + (1-r)\cdot\ell_{ref}$$

粗糙度大（偏漫反射）时更依赖辐射场，粗糙度小（偏镜面）时更依赖反射场。$r$ 的梯度在此处被 detach 以稳定优化。这样模型就能按表面粗糙度自适应地在两个场之间切换权重，实现对任意反射材质的稳健重建。

### 网格-SDF 融合的材质估计

几何重建后材质还很粗。若像 NeRO 那样直接在提取的显式网格上做蒙特卡洛重要性采样，网格效率高但会因几何退化带来材质偏差；而纯用隐式 SDF 场做体渲染又太慢（间接光计算与显存指数增长）。

折中方案：先用显式网格做快速光线-网格求交得到一个粗略命中点，再只在命中面附近（表面内外 $4u$ 距离内，$u$ 为张量网格单元尺寸）采样 $m$ 个点做体渲染，得到精确命中点 $\hat{p}$ 与法向 $\hat{n}$：

$$\hat{t} = \sum_{i=0}^{m} w_i t_i, \qquad \hat{n} = \sum_{i=0}^{m} w_i n_i, \qquad \hat{p} = o + \hat{t}\,d$$

兼顾了网格求交的效率与隐式 SDF 的几何精度。

## 技术细节

- 网络结构：TensoSDF 分辨率 $512\times512$，特征通道 36；解码器为两层 128 宽 MLP；辐射场与反射场的 MLP 均为三层 128 宽。
- 训练损失：几何阶段总损失包含颜色损失 $\ell_c$、Eikonal 损失 $\ell_e$、高斯平滑损失 $\ell_g$、以及依据是否有掩膜选用 mask loss 或 Hessian loss $\ell_h$，外加 TV 损失、遮挡损失、稳定化损失；材质阶段用颜色损失加材质正则损失。
- 训练策略：coarse-to-fine，网格分辨率从 $128\times128$ 逐步增到 $512\times512$；前 20k 步只训练反射场以得到粗形状和初始粗糙度，之后才启用双场的粗糙度加权融合。
- 数据集：作者自建含 6 个场景的合成数据集（漫反射/光泽/镜面各类材质，Blender Cycles 渲染，5 组环境光重光照），并在 TensoIR、NeRO 合成数据集，以及真实数据集 NeILF-HDR 与 Stanford-ORB 上评测。
- 算力：单张 RTX 4090，几何阶段 180k 步约 4 小时，材质阶段 100k 步约 1.5 小时。

## 实验结果

几何重建（自建合成数据集，法向 MAE↓）：本文平均 MAE 2.96，优于 NeuS(3.39)、TensoIR(3.54)、NeRO(6.18)、NeILF++(3.85)。训练时间 4 小时，与 TensoIR 持平，明显快于 NeRO(8h) 和 NeILF++(12h)。相比 NeRO，TensoSDF 把训练时间砍到约 50%。

| 方法 | Avg. MAE↓ | 平均训练时间 |
|------|-----------|--------------|
| NeuS | 3.39 | 6 hrs |
| TensoIR | 3.54 | 4 hrs |
| NeRO | 6.18 | 8 hrs |
| NeILF++ | 3.85 | 12 hrs |
| 本文 | **2.96** | 4 hrs |

重光照质量（自建合成数据集，PSNR/SSIM）：本文平均 25.97 / 0.932，显著优于 TensoIR(23.38/0.903)、NeRO(23.20/0.904)、NeILF++(22.41/0.895)。NeRO 主要输在几何不准，TensoIR/NeILF++ 则因球面高斯近似导致高频镜面丢失、材质偏漫。

真实数据 Stanford-ORB：五个场景上几何（CD↓）与重光照（PSNR/SSIM/LPIPS）全面胜过 NeRO，平均 CD 从 2.165 降到 1.035，PSNR 从 27.35 提升到 29.68。在 NeILF++ 真实数据上，本文能处理 NeILF++ 失败的强高光表面，也比 NeRO 保留更多几何细节。

消融实验：

- 从 NeRO 出发换成 TensoSDF，MAE 改善 1.65 且训练时间减半，但表面仍有凹陷。
- 加入双场固定权重(0.5)融合后，表面伪影缓解。
- 再把固定权重换成粗糙度自适应权重，质量最佳（在不同固定权重 $r=1,0.8,0.5,0.2$ 上做扫描，越接近真实粗糙度越好，用粗糙度本身达到最优 MAE 2.04）。
- 网格-SDF 融合能减少材质偏差、提升重光照 PSNR。
- 两条平滑先验（mipmap 混合 + 高斯平滑损失）逐步抑制平坦表面的噪声。

## 贡献与局限

贡献：

- 提出粗糙度感知的辐射场与反射场融合，用每点粗糙度作平衡因子，实现对任意反射物体的稳健几何重建。
- 提出 TensoSDF——把张量表示与 SDF 结合的新几何表示，几何细节更丰富且训练时间减半。
- 设计网格-SDF 融合的材质估计策略，兼顾网格求交效率与隐式 SDF 几何精度，进一步提升材质质量。

局限：

- 细节与噪声的权衡：显式表示易局部过拟合产生噪声，平滑先验虽能抑噪却会损失细节。
- 质量与存储的权衡：张量表示比体表示紧凑，但存储仍高于纯 MLP（512 分辨率 109 MB，200 分辨率 17.9 MB 仍可用），作者建议未来用剪枝缓解。
- 仅处理单物体、多视角输入，不支持半透明材质；扩展到多物体、少视角、或半透明（需更高维建模）留待未来工作。

## 延伸思考

这篇工作最有启发的一点，是把"用什么颜色模型"从二选一变成了按物理量（粗糙度）自适应加权的连续融合——辐射场和反射场各自的适用区间（低频漫反射 vs 高频镜面）恰好被粗糙度这一物理参数自然划分，于是一个本该由人工设计取舍的建模选择，被交还给数据在每个空间点上自动决定。这种"让物理先验来路由不同表示"的思路，与前作把延迟着色引入高斯溅射的做法一样，都指向一个共同方向：重建/渲染框架中"表示的组合方式本身"是可设计、可微、可优化的对象。作者也在展望里指出，把 TensoSDF 与 3D Gaussian Splatting 结合、或在第二阶段联合更新几何与材质，都是自然的下一步。
