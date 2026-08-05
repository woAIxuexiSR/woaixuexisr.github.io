---
title: "An Extensible, Data-Oriented Architecture for High-Performance, Many-World Simulation"
authors:
  - "Brennan Shacklett"
  - "Luc Guy Rosenzweig"
  - "Zhiqiang Xie"
  - "Bidipta Sarkar"
  - "Andrew Szot"
  - "Erik Wijmans"
  - "Vladlen Koltun"
  - "Dhruv Batra"
  - "Kayvon Fatahalian"
category: "Animation & Simulation"
track: "Journal"
source: "author-page"
institution:
  - "Stanford University"
  - "Georgia Institute of Technology"
tags:
  - "Batch Simulation"
  - "Entity Component System"
  - "GPU"
  - "Reinforcement Learning"
  - "Data-Oriented Design"
links:
  paper: "https://doi.org/10.1145/3592427"
  project: "https://madrona-engine.github.io/"
  code: "https://github.com/shacklettbp/madrona"
---

## 一句话总结

把游戏引擎里流行的实体-组件-系统（ECS）架构完整搬到 GPU 上，做出一个可编程、可扩展的批量仿真引擎 Madrona，让开发者能像写普通游戏逻辑一样编写新的 AI 训练环境，同时获得比 CPU 基线快两到三个数量级的吞吐。

## 研究背景

- 领域现状：训练 AI 智能体（强化学习）需要数百万到数十亿步经验，为了提速，最快的仿真器采用"批量仿真"（batch simulation）思路——用一个引擎同时并行推进成千上万个环境实例，从而在 GPU 上榨取跨环境的指令与数据一致性。
- 核心痛点：写一个高效的批量仿真器既要懂任务领域、又要懂并行编程，门槛极高。因此现有批量仿真器都只支持一套固定功能（某种 3D 物理、某类导航、Atari 游戏等）。想做一个新任务的批量仿真器，要么改别人引擎的内部，要么从零重写；这和 Unity / Unreal 那种"写脚本即可扩展、无需懂引擎内部"的体验完全相反。
- 本文 idea：作者观察到，多核 CPU 上用来组织游戏逻辑、天然利于并行的 ECS 设计模式，同样非常适合表达能在 GPU 大规模细粒度并行上高效运行的批量仿真器。于是他们把 ECS 完整映射到 GPU，做出 Madrona，兼顾可编程、高性能、高生产力三个目标。

## 方法

Madrona 的整体思路：开发者用 ECS 的三个抽象（组件 component、原型 archetype、系统 system）描述环境的状态与逻辑，写的是"每个世界 / 每个智能体 / 每个物体一个函数"这样无需关心并行的普通代码；Madrona 运行时则利用这些抽象带来的结构信息，接管内存管理、数据流与并行调度，把整张"系统计算图"编译成一个 GPU 大核（megakernel）来执行。

```mermaid
flowchart LR
  A["ECS 描述: 组件 / 原型 / 系统"] --> B["系统计算图 (DAG)"]
  B --> C["列存表: 每原型一张表, 跨所有环境"]
  C --> D["编译为单个 GPU megakernel"]
  D --> E["每步一次 kernel 启动, 持久线程执行整张图"]
  E --> F["组件数据以 tensor 零拷贝导出给学习框架"]
```

关键设计：

1. **用 ECS 抽象表达仿真逻辑。** 环境状态被组织成实体（entity）的集合，实体的状态由若干组件（如位置、速度、队伍、动作）决定；拥有相同组件集的实体共享一个原型。逻辑写成"系统"：一个查询（query）选出含指定组件的实体，再对每个实体调用一个函数。整个仿真步是一张"系统计算图"（DAG），节点是系统、边是依赖。这让环境生成、时间推进、观测/奖励计算三部分逻辑都能用无并行概念的直白函数表达，且模块化可组合。

2. **集中式列存表管理组件存储。** 每个原型在 GPU 上建一张表，跨所有环境的同类实体的组件数据连续存放（列存）。每张表隐式加一列 EnvID，把实体映射回所属环境。单表设计有三个好处：相邻 GPU 线程处理连续实体时即便来自不同环境也保持访存一致；跨环境摊薄了过量分配与元数据开销，降低显存碎片；组件数据能以 tensor 形式零拷贝别名给学习框架，省掉数据搬运。

3. **动态实体的创建与回收。** 建实体就是在表尾加一行；启动时预留大段虚拟内存、按需分页物理内存，避免自定义 GPU 分配器又保持地址连续。删除只需把 EnvID 置为哨兵值 $$-1$$，无需线程间同步；随后用并行基数排序按 EnvID 把 $$-1$$ 行排到表尾再截断回收——排序顺带让同环境实体相邻、访存更一致。

4. **megakernel + 持久线程执行计算图。** 若每个系统单独启动一个 CUDA kernel，会被启动开销和"动态系统需要 CPU 回读确定线程数"拖垮。Madrona 改为把整张图编译进一个大核，CPU 每步只启动一次，用持久线程（persistent threads）风格让所有线程协同推进同一节点再进下一节点。引擎因此始终掌握 GPU 执行状态，能安全地在节点间做内存重映射、压表等全局操作。

5. **用前置 profiling 做 PGO 优化寄存器分配。** megakernel 里所有系统共享同一套每线程寄存器分配，复杂系统（如窄相碰撞）需要多寄存器，会拖累简单系统的占用率。Madrona 可选地编译多份不同寄存器分配的 megakernel，先跑少量步做 profiling 测每个系统耗时，再据此把计算图拆成几段分别启动，在额外 kernel 启动成本与延迟隐藏收益间权衡。

## 实验结果

在 Intel i9-13900K（32 线程）+ RTX 4090 上，用四个环境（HideSeek、Overcooked、Hanabi、Cartpole）评测。核心对比是 Madrona 的 GPU 批量后端（BATCH-ECS-GPU）与三个基线：ECS-CPU（强力 CPU 基线，多线程但非批量）、ECS-GPU（一环境一线程、非批量）、REF-CPU（开源参考实现）。下表为各配置在峰值批量下的吞吐（环境步/秒）：

| 配置 | HideSeek | Overcooked | Hanabi | Cartpole |
|------|----------|------------|--------|----------|
| BATCH-ECS-GPU | 1.9×10⁶ | 4.0×10⁷ | 2.1×10⁷ | 3.4×10⁹ |
| ECS-GPU | 4.5×10⁴ | 1.1×10⁶ | 4.1×10⁶ | 1.4×10⁸ |
| ECS-CPU | 1.6×10⁵ | 1.2×10⁶ | 4.1×10⁶ | 2.8×10⁷ |
| REF-CPU | 6.9×10³ | 9.7×10³ | 1.3×10⁴ | 1.4×10⁵ |

相比强力 CPU 基线 ECS-CPU，BATCH-ECS-GPU 在 HideSeek 上快 11×、Overcooked 上快 33×；相比开源参考实现快两到三个数量级。HideSeek 达到 190 万步/秒，作者估算 RTX 4090 单卡约 1.5 天即可跑完观察到全部涌现工具使用行为所需的 1150 亿步（同样里程碑在 32 线程 CPU 上需约 40 天）。非批量的 ECS-GPU 因显存碎片限制并发环境数、且线程执行不同系统导致 warp 发散与访存不一致，只用到 GPU 峰值算力/带宽的约 1.9%，比批量版慢 40 倍以上；批量版则把算力与带宽利用率提到 33% 和 69%。PGO 进一步带来约 35% 吞吐提升。

## 亮点与局限

- 亮点：
  - 首个完整映射到 GPU 的 ECS 架构，把成熟的游戏编程范式扩展到批量仿真，兼顾可编程性与极高吞吐。
  - 集中式列存 + 隐式 EnvID + megakernel + PGO 一套设计层层递进，充分利用了跨环境与环境内两种并行；zero-copy tensor 导出与主流学习框架无缝衔接。
  - 用消融式的四个后端对照（批量/非批量、GPU/CPU）清晰量化了"批量化"本身而非单纯上 GPU 才是性能关键。

- 局限：
  - megakernel"整卡协同处理一个节点"的调度在缺乏并行度的系统（如每环境仅一次的约束求解、速度过滤）上会让部分 SM 空转，依赖足够大的批量来填满。
  - 光线投射目前是 ECS 内的软件实现，无法用上 RTX 硬件光追——因为计算模式的 GPU API 不允许 CUDA 核发起 ray query，BVH 构建也只能在主机端调用。
  - 逻辑主要用 C++ 编写，暴露 warp 级/共享内存等高级并行仍需手工优化；作者也坦言仅探索了部分调度优化，尚属早期阶段。

## 延伸思考

- 论文明确指出这套全 GPU 的 ECS 不仅服务于 RL 训练，也可作为传统单环境游戏逻辑做 GPU 加速的设计参考；甚至数据中心里的多人游戏 headless server、云游戏这类"同机共置大量环境实例"的场景，也可能受益于换一种吞吐/延迟权衡的批量仿真设计。
- 光追硬件不可从 compute kernel 直接访问是当前一大瓶颈，随着越来越多完整应用"以 GPU 为主体"而非把 GPU 当协处理器，平台是否开放从设备端发起 BVH 构建与 ray query，会直接决定这类引擎的上限。
- 值得追问：ECS 抽象对"复杂控制流 / 不规则数据结构"的表达上限在哪？相比 JAX / PyTorch 这类数组式编程，Madrona 在可表达性上占优，但当环境逻辑进一步变复杂、智能体行为更异质时，megakernel 的寄存器压力与调度公平性会如何演化，是后续可深入的方向。
