---
title: "EgoHDM: A Real-time Egocentric-Inertial Human Motion Capture, Localization, and Dense Mapping System"
authors:
  - Handi Yin
  - Bonan Liu
  - Manuel Kaufmann
  - Jinhao He
  - Sammy Christen
  - Jie Song
  - Pan Hui
track: "Journal"
source: arxiv
category: "Reconstruction"
institution:
  - HKUST(GZ)
  - ETH Zürich
tags:
  - Human Motion Capture
  - Inertial Sensors
  - SLAM
  - Dense Mapping
  - Egocentric
  - Physics-Based Correction
links:
  paper: "https://doi.org/10.1145/3687907"
  project: "https://handiyin.github.io/EgoHDM/"
---

## 一句话总结

EgoHDM 用 6 个身上佩戴的 IMU 加一台头戴 RGB 相机，在近实时下同时完成人体动作捕捉、场景定位与稠密三维建图，并首次在惯性动捕与单目视觉 SLAM 之间"闭环互馈"，通过基于物理的足-地接触修正显著改善了非平坦地形下的表现。

## 研究背景

在无约束的真实环境中做人体动作捕捉一直很难，现有技术各有短板：

- 外部相机系统精度高，但需要固定采集空间、繁琐标定，且容易被遮挡。
- 身上佩戴的 IMU 便携、无视线约束，但存在严重的全局漂移，定位不可靠,而且大多完全忽略了对环境的重建。

把两者结合起来很有吸引力:RGB 的 SLAM 能提供去漂移的全局轨迹,而惯性传感器能提供前视相机难以捕捉的身体姿态。但真正做到互利并不容易——如果惯性、身体、相机坐标系没有对齐,身体的运动约束反而会破坏地图更新。

已有工作没有把两者的优势都吃透:HPS、HSC4D 需要预扫描或离线地图;唯一在线的 EgoLocate 也没有闭环(修正后的姿态没有反馈回去更新地图),而且只保留稀疏地图、假设地面是平的,导致穿模/漂浮、难以适应非平坦地形。EgoHDM 要解决的正是这些问题。

## 方法

### 整体框架

系统输入是同步的 6 路 IMU 信号(加速度与旋转)和头戴单目 RGB 图像,输出 SMPL 姿态参数 $$\boldsymbol{\theta}\in\mathbb{R}^{72}$$、平移 $$\mathbf{t}\in\mathbb{R}^{3}$$ 以及全局稠密点云地图 $$\mathbf{P}_G$$。整个流水线由五个环节串成一个闭环:

```mermaid
flowchart LR
    A[6 IMU + 头戴RGB] --> B[VIM 初始化<br/>对齐惯性/相机坐标系]
    B --> C[MDBA<br/>动捕感知稠密BA]
    C --> D[建图 + 回环闭合<br/>协方差引导体素融合]
    D --> E[身体中心局部高程图]
    E --> F[地图感知惯性动捕<br/>物理接触修正]
    F -- 修正姿态反馈 --> C
```

关键在于最后一步"修正后的姿态反馈回 MDBA",从而真正闭合了惯性动捕与 SLAM 建图之间的回路。

### 关键设计 1:引入身体形状约束的 VIM 初始化

初始化的目标是对齐 SMPL 坐标系与场景(相机)坐标系。由于相机刚性地固定在头部,这等价于求一个从 SMPL 头关节到相机的相似变换 $$T_{hc}=[s\cdot R_{hc}\,|\,t_{hc}]\in \mathrm{Sim}(3)$$,满足

$$G = T_{hc}\,G_h$$

仅靠关键帧上的相对变换匹配只能粗略估计尺度 $$s$$。作者的创新是加入身体形状约束:假设人站在平地上,在 SMPL 脚底构造一个虚拟平面 $$P_f$$,同时用稠密 SLAM 初始点云做语义分割拟合出地面平面 $$P_c$$,通过最小化两平面距离来精确求尺度:

$$\arg\min_{T_{hc}}\ \alpha\cdot d(P_c, T_{hc}P_f) + \sum_{t\in K}\beta\cdot \lVert \Delta G(t)\ominus \Delta(T_{hc}G_h(t))\rVert$$

其中 $$\alpha=0.9,\ \beta=0.1$$。这一设计让初始化比 EgoLocate 更快(< 3 秒)也更鲁棒,无需冗长的运动轨迹。

### 关键设计 2:动捕感知的稠密 Bundle Adjustment(MDBA)

MDBA 在 Droid-SLAM 的光流重投影框架上增加了一个惯性项,联合优化相机位姿与关键帧深度:

$$E_{total}=E_{repr}+\lambda\cdot E_{inert}$$

重投影误差沿用 Droid-SLAM:

$$E_{repr}=\sum_{(i,j)\in E}\lVert u^{*}_{ij}-\Pi_c(G_{ij}\circ \Pi_c^{-1}(u_i,d_i))\rVert^2_{\Sigma_{ij}}$$

惯性动捕误差把动捕模块给出的头关节相对平移/旋转先验约束进相机帧:

$$E_{inert}=\sum \lVert(t_{i,i-1}-\tilde t_{i,i-1})\rVert^2_{\Sigma_t}+\sum \lVert \log(\tilde R_{i,i-1}^{T}R_{i,i-1})^{\vee}\rVert^2_{\Sigma_R}$$

优化时借助 Hessian 的稀疏结构提取逐像素逆深度的边缘协方差 $$\Sigma_d$$,用它过滤低置信深度,为全局地图更新和局部高程图打基础。建图阶段采用基于哈希的 TSDF 体素表示,用 $$\Sigma_d$$ 加权融合以抑制稠密深度的噪声;检测到回环时在并行线程里跑一次仅优化位姿的 MDBA 以保证在线性能。

### 关键设计 3:身体中心局部高程图 + 地形感知接触 PD 控制器

在检测到关键帧时,以身体中心为原点构造 2m×2m、分辨率 $$M\times M$$($$M=100$$)的局部高程图,把裁剪后的全局地图沿重力方向投影到每个网格,取落入格内点的最大 $$z$$ 作为高度(空格用最近邻插值)。

随后在地图感知动捕模块中,先用 PIP 的预训练网络从 6 路 IMU 得到初始 SMPL 姿态与足部接触概率,再用物理修正模块结合高程图搜索人-场景接触。作者提出一个地形感知的接触 PD 控制器,在重力方向上约束接触关节的高度:

$$\dot r_c = J_c \dot q$$

$$\ddot r_{c\downarrow}=k_{pc}(h-r_{c\downarrow})-k_{dc}\dot r_{c\downarrow}$$

其中 $$h$$ 是高程图给出的接触高度,$$\downarrow$$ 表示重力方向分量。它与关节旋转/位置 PD 控制器配合,产出既物理合理又贴合地形的动作,有效减少漂浮与穿模;修正后的姿态再反馈回 MDBA 闭合回路。

## 实验结果

在合成数据集 TotalCapture 和真实数据集 HPS 上评估,指标为全局人体根关节绝对位置误差(米)。EgoHDM 相比此前 SOTA EgoLocate 在人体定位、相机定位、建图误差上分别下降约 41%、71%、46%。

全局动捕根位置误差(米,越低越好):

| 方法 | TotalCapture 平均 | HPS 平均 |
|---|---|---|
| TIP | 0.45 | 3.00 |
| PIP | 0.37 | 2.75 |
| EgoLocate | 0.22 | 1.70 |
| Ours | **0.13** | **1.50** |

相机定位误差方面,EgoHDM 在 TotalCapture 上平均 0.07m(EgoLocate 0.24m),在 HPS 上 1.49m(EgoLocate 1.69m);建图精度(点到点距离)平均 0.39m,优于 EgoLocate(0.66m)与离线 Droid-SLAM(0.72m)。

消融实验显示:去掉 VIM 初始化会使相机定位误差从 0.07m 飙升到 1.26m;去掉动捕约束升到 0.44m;去掉 SLAM 升到 0.37m。在自采非平坦地形数据上,完整系统建图平均误差 5.36cm,而缺少足-地约束或缺少动捕约束时分别升到 26.33cm 与 24.45cm。初始化耗时上,EgoHDM 约 3~4 秒,而 EgoLocate 需要 17~34 秒。

## 亮点与局限

亮点:

- 首个在惯性动捕与单目 RGB SLAM 之间"完全闭环"的近实时系统,且仅需 6 个 IMU 加一台商用头戴相机,证明了两个任务的紧耦合是相互增益的。
- 身体形状约束的 VIM 初始化又快又稳,把启动时间从几十秒压到几秒。
- 基于局部高程图的物理修正让系统能自然处理台阶、非平坦地形,显著减少穿模与漂浮。

局限(作者自述):

- 局部姿态沿用 PIP 预训练网络,存在动作被"软化"的问题(如上楼梯时膝盖不完全弯曲),源于训练数据大多忽略非平坦环境。
- 非平坦地形缺乏带真值姿态的数据集,相关评估只能给定性结果。
- 快速运动带来的运动模糊仍会降低建图与姿态精度,并产生更多关键帧,增加显存占用与回环闭合时间。

## 延伸思考

这项工作最有启发的是"闭环互馈"的系统观:惯性动捕为视觉 SLAM 提供运动先验以对抗纹理缺失和快速运动,而 SLAM 的稠密地图又反过来通过高程图为动捕提供物理接触约束,两者不是简单串联而是双向纠偏。这种设计思路对任何多模态感知系统都有参考价值——关键前提是坐标系对齐要足够准,否则约束会互相破坏,这也解释了为何 VIM 初始化在消融中影响最大。

另一个值得关注的方向是把高程图这类"局部环境表示"引入动作生成/修正,它比全局网格更轻量、更适合在线场景。未来若能用覆盖非平坦地形的训练数据替换被"软化"的局部姿态网络,或引入更抗运动模糊的视觉前端(如事件相机),EgoHDM 这类系统有望进一步走向真正开放世界的人-场景理解与混合现实应用。
