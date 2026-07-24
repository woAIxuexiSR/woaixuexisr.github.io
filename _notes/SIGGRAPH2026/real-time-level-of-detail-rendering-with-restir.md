---
title: "Real-Time Level-of-Detail Rendering with ReSTIR"
authors:
  - "Yu-Chen Wang"
  - "Markus Kettunen"
  - "Daqi Lin"
  - "Chris Wyman"
  - "Lifan Wu"
  - "Shuang Zhao"
category: "Rendering"
track: "Conference"
source: "author-page"
institution:
  - "NVIDIA"
  - "University of California, Irvine"
  - "University of Illinois Urbana-Champaign"
tags:
  - "ReSTIR"
  - "Real-Time Rendering"
  - "Level Of Detail"
  - "Path Tracing"
  - "Sample Reuse"
  - "Light Transport"
links:
  paper: "https://doi.org/10.1145/3799902.3811100"
  project: "https://research.nvidia.com/labs/rtr/publication/wang2026levelofdetail/"
---

## 一句话总结

本文提出一种基于 UV 纹理空间的"顶点映射"，让 ReSTIR 能在几何细节层级（LoD）切换、网格拓扑发生改变的相邻帧之间继续复用光路样本，从而消除 LoD 切换时的噪声暴涨。

## 研究背景

- 领域现状：实时渲染中，几何 LoD（用预滤波的低模替换远处高频几何）是控制开销、抑制噪声与走样的常用手段；而 ReSTIR 通过在时间和空间上复用光样本，大幅提升了实时路径追踪的质量。二者结合本应互补——用对了 LoD 能削掉子像素级高频，让 ReSTIR 更容易复用。
- 核心痛点：现有 ReSTIR 方法（GRIS、reservoir splatting 等）依赖 shift mapping 在帧间搬运光路，而 shift mapping 的顶点对应是靠"三角形索引 + 重心坐标"建立的，这要求相邻帧网格拓扑一致。一旦物体切换 LoD，网格拓扑改变，顶点映射直接失效，ReSTIR 丢失时间历史、退回独立采样，导致 LoD 切换瞬间及其后若干帧噪声骤增。结果是：本该降噪的 LoD，反而在切换时刻拖累了 ReSTIR。
- 本文 idea：不再假设拓扑一致，改为假设不同 LoD 之间有一致的 UV 参数化。借助 UV 贴图这一共享的"标准参数空间"，在不同拓扑的表面之间构造一个（部分）双射的顶点映射 $$\tau$$，替换掉原来依赖三角形索引的对应方式，使 shift mapping 能跨越 LoD 切换继续工作。

## 方法

整体框架：在 reservoir splatting 版 ReSTIR 路径追踪的基础上，把帧间搬运顶点用的"三角形索引 + 重心坐标"对应，替换成一个经过 UV 空间中转的顶点映射 $$\tau$$。检测到某物体发生 LoD 变化时，就用 $$\tau$$ 把上一帧的主命中点和重连接顶点映射到当前帧的新拓扑表面上，其余流程沿用已有 shift mapping。

```mermaid
flowchart LR
  A["上一帧光路顶点 x"] --> B["查 UV 坐标 v = 前一层 UV 映射"]
  B --> C["在目标 LoD 表面反查所有同 UV 点集"]
  C --> D["世界空间贪心匹配消歧"]
  D --> E["得到映射点 y = tau(x)"]
  E --> F["代入 shift mapping 并乘雅可比补偿"]
  F --> G["跨 LoD 的时间复用成功"]
```

关键设计：

1. 非重叠 UV 下的顶点映射。设基路径表面 $$S_1$$、偏移路径表面 $$S_2$$ 各自带 UV 映射 $$\Phi_1, \Phi_2$$。当两个 UV 映射都是单射（不重叠）时，只需让 $$x$$ 与 $$\tau(x)$$ 拥有相同 UV 坐标，于是 $$\tau(x) = \Phi_2^{-1}(\Phi_1(x))$$。这把"拓扑相同"的强假设换成了"UV 一致"的弱假设——不同 LoD 只要共享一套 UV，就能对应起来。

2. 一般 UV 下的匹配消歧。真实资产里 UV 常是多对一（镜像、面片共享同一块贴图）。此时同一个 UV 坐标 $$v$$ 会对应两个表面上的多个世界空间点，记为点集 $$X_{1,v}$$ 与 $$X_{2,v}$$。为保证 $$\tau$$ 可逆（shift mapping 必须可逆，否则引入偏差），作者用一个贪心算法：反复取两集合中世界空间距离 $$\lVert x - y \rVert$$ 最近的一对未匹配点配对并移除，直到某一集合为空。这样在 UV 歧义处也能确定一一对应；找不到配对时就判定 $$\tau$$ 在该点不存在、shift 失败（这在实践中很罕见）。为支持实例化与 UV 反查，系统为每个 LoD 物体在 UV 空间额外建一棵 BVH，用硬件光追遍历找出所有候选点。

3. 雅可比行列式。$$\tau$$ 改变了表面点密度，必须在 shift mapping 的雅可比里补偿。给定 UV 处两个表面的切向量 $$\bar{u}_i, \bar{v}_i$$（即 UV 单位变化对应的世界空间位移），有 $$\lvert \partial\tau / \partial x \rvert = \lvert \bar{u}_2 \times \bar{v}_2 \rvert / \lvert \bar{u}_1 \times \bar{v}_1 \rvert$$，几何意义是两表面上对应同一 UV 微元的面积之比。每个经 $$\tau$$ 复用的顶点都乘上这一项。

4. LoD-aware ReSTIR 路径追踪。以 reservoir splatting 与 hybrid shift 为骨架，在 reservoir 中额外记录物体 ID 与主命中点、重连接顶点的纹理坐标。帧间搬运时对两个关键顶点都用 $$\tau$$：一是 LoD-aware 主命中点重投影（$$y_1 = \tau(x_1)$$，再投回相机做 splatting），二是 LoD-aware hybrid shift（重连接顶点 $$y_k = \tau(x_k)$$）。实验证实两个顶点缺一不可，只对其中一个用 $$\tau$$ 会立刻在 LoD 切换处引入额外噪声。当某物体 LoD 未变化时，退回用三角形索引 + 重心坐标的老办法做优化，避免不必要的开销。

## 实验结果

在 Falcor 框架、RTX 3090 上以 1920×1080 渲染，与两个基线做等时比较：B.1 无顶点映射的 Base ReSTIR（reservoir splatting），B.2 给更高采样数的路径追踪。下表取 LoD 切换处的等时对比主实验（RelMSE 越低越好），路径追踪用 3 spp 换取等时：

| 场景 | 本文 RelMSE↓ | Base ReSTIR RelMSE↓ | Path Tracing RelMSE↓ |
|------|------|------|------|
| Rocks | 0.276 | 0.725 | 0.331 |
| Sponza | 0.541 | 0.667 | 2.921 |
| Terrain | 0.094 | 0.134 | 0.193 |
| Chess | 0.127 | 0.358 | 0.642 |

四个场景中本文方法误差均最低，且与 Base ReSTIR 的时间开销几乎相同（如 Rocks 18.4ms 对 18.0ms）。路径追踪因缺乏复用普遍偏噪，Base ReSTIR 则在 LoD 切换处丢失时间历史、退化为独立采样。

其余实验用文字补充：顶点选择消融（Chess）显示同时对主命中点和重连接顶点用 $$\tau$$ 时 RelMSE 0.127，只用其一分别为 0.225 / 0.287，都不用则 0.358；匹配消歧实验在 UV 重叠场景（Plane、Cube）中，带匹配的 Cube RelMSE 0.061、去掉匹配则恶化到 0.261，且匹配带来的额外耗时很小（14.4ms 对 12.9ms）；Rock 的连续帧序列显示 LoD 切换后基线噪声骤增并需多帧恢复，而本方法仅有轻微质量损失。

## 亮点与局限

- 亮点：
  - 抓住了 ReSTIR 与几何 LoD 结合的真正障碍——拓扑变化导致顶点对应失效，并给出简洁到位的解法：把对应关系放到跨 LoD 一致的 UV 空间去建立。
  - 用贪心的世界空间最近点匹配干净地处理了 UV 多对一（镜像、贴图复用）带来的可逆性难题，代价很低。
  - 几乎不增加运行开销就恢复了 LoD 切换处的时间复用，可直接嵌入现有 reservoir splatting / hybrid shift 管线。

- 局限：
  - 强依赖"不同 LoD 之间 UV 映射一致"这一前提；无纹理映射、或各 LoD 间 UV 不一致的物体不被直接支持。
  - 高度扭曲或退化的 UV 会削弱表面对应精度、降低成像质量。
  - 当前实现要求每帧内每个物体使用单一 LoD（虽然理论表述不需要此限制），尚不支持按光线选择 LoD。

## 延伸思考

作者点出一个更普适的观点：ReSTIR 要做好时空复用，往往需要一个显式的顶点映射，而不能只靠复用相同的三角形索引与重心坐标。LoD 切换只是"表面表示跨帧/跨像素变化"的一个突出实例，类似问题也出现在流体模拟等表面不断重建的场景中——为这些情形设计鲁棒的顶点映射是值得追问的方向。此外，把方法扩展到按光线粒度选择 LoD（而非每帧每物体单一 LoD），能与基于光线足迹的连续 LoD 更自然地耦合。
