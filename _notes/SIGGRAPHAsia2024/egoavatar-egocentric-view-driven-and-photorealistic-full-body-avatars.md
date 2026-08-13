---
title: "EgoAvatar: Egocentric View-Driven and Photorealistic Full-body Avatars"
authors:
  - "Jianchun Chen"
  - "Jian Wang"
  - "Yinda Zhang"
  - "Rohit Pandey"
  - "Thabo Beeler"
  - "Marc Habermann"
  - "Christian Theobalt"
category: "Rendering"
track: "Conference"
source: "arxiv"
institution:
  - "MPI for Informatics"
  - "Google"
tags:
  - "Egocentric Capture"
  - "Full Body Avatar"
  - "3D Gaussian Splatting"
  - "Telepresence"
  - "Performance Capture"
links:
  paper: "https://doi.org/10.1145/3680528.3687631"
---

## 一句话总结

EgoAvatar 首次实现了仅用一路头戴式下视 RGB 相机的第一人称视频，就能驱动并自由视角渲染出照片级真实的全身可动化身，打通了从"自我视角追踪"到"逼真全身呈现"的完整链路。

## 研究背景

沉浸式 VR 远程临场（telepresence）的理想状态，是能与和真人难以区分、且精确复现其行为的数字化身交互。要实现这一点有两大技术难题：一是把真人转化为可动、全身、照片级真实的数字化身；二是仅凭头戴设备等轻量、低功耗的自我中心（egocentric）传感器就能追踪真人动作。

已有工作往往只解决其中一环：有的专注于自我中心动作捕捉却忽略渲染，有的只建模头部/面部化身，有的从多视角采集构建化身但推理时仍需固定多相机装置，把用户束缚在有限的采集空间内。唯一尝试从自我视角驱动全身化身的 EgoRenderer 因基于 SMPL 表示与不准的骨骼预测，画质远非照片级且存在明显的时序抖动。

全身自我中心化身之所以远难于头部化身，是因为身体关节带来大幅姿态变化、服装多样、材质差异大（布料 vs 皮肤反射），以及严重的自遮挡与鱼眼畸变。本文提出 EgoAvatar，第一个仅凭单路自我中心 RGB 相机驱动并渲染照片级全身化身的方案，可在无约束环境中运行并保持高视觉保真度。这里的"全身"指整个着衣人体，不含手势与面部表情。

## 方法

整体框架分两大阶段：先离线用多视角视频学到一个"可由骨骼运动驱动"的照片级化身，再在推理时仅凭自我中心视频恢复骨骼运动、驱动几何、精化对齐并预测外观。

```mermaid
flowchart LR
    A[自我中心 RGB 视频] --> B[EgoPoseDetector<br/>3D 关键点检测]
    B --> C[IKSolver<br/>逆运动学解关节角]
    C --> D[MotionDeformer<br/>运动驱动网格形变]
    D --> E[EgoDeformer<br/>测试时网格精化]
    E --> F[GaussianPredictor<br/>UV 空间动态高斯]
    F --> G[自由视角渲染]
```

**角色模型（混合表示）。** 遵循 DDC，用可动的显式网格表示着衣人体：模板网格 $$\boldsymbol{T}_0 \in \mathbb{R}^{4890\times3}$$，骨骼由 54 个自由度 $$\boldsymbol{\theta} \in \mathbb{R}^{54}$$ 控制（关节旋转、全局旋转与平移）。规范形状经由 489 节点的嵌入图做非刚性形变（逐节点旋转 $$\boldsymbol{\alpha}$$、平移 $$\boldsymbol{t}$$ 与逐顶点偏移 $$\boldsymbol{d}$$），再经前向运动学与线性混合蒙皮（LBS）$$W(\cdot)$$ 得到姿态空间网格：

$$\boldsymbol{M} = W\big(T(\boldsymbol{T}_0, \boldsymbol{\alpha}, \boldsymbol{t}, \boldsymbol{d}),\ \mathcal{J}(\boldsymbol{\theta}),\ \mathcal{W}\big)$$

关键之处在于外观不用 DDC 的动态纹理图，而改用建在网格表面的 3D 高斯溅射，大幅提升渲染质量。

**关键设计 1：个性化自我中心追踪（EgoPoseDetector + IKSolver）。** 在通用鱼眼 ViT 姿态检测器基础上，用受试者专属的"配对自我中心-多视角数据"微调，得到定位 25 个关节的个性化检测器 $$\boldsymbol{J}^{\tau}_{\text{local}} = \text{FViT}(\boldsymbol{I}^{\tau})$$。随后 IKSolver 用由粗到细的优化，从含噪、不完整的 3D 关键点恢复稳定的关节角，能量为数据项、时序平滑项、自由度限制项与正则项之和：

$$\arg\min_{\boldsymbol{\theta}^{\tau}}\ E_{\text{Data}} + E_{\text{Temporal}} + E_{\text{DoFLimit}} + E_{\text{Reg}}$$

其中正则项 $$E_{\text{Reg}} = \sum_{d=0}^{53}\|\boldsymbol{\theta}_d - \bar{\boldsymbol{\theta}}_d\|^2$$ 把优化姿态拉向训练动作的均值姿态，避免出现虽然吻合关键点但关节扭转不合理的姿态。

**关键设计 2：运动驱动几何 + 自我视角精化（MotionDeformer + EgoDeformer）。** MotionDeformer 用两个结构感知图神经网络，从归一化运动 $$\hat{\boldsymbol{\theta}}^{\tau}$$ 依次回归低频嵌入图参数 $$(\boldsymbol{\alpha}^{\tau}, \boldsymbol{t}^{\tau})$$ 与高频顶点偏移 $$\boldsymbol{d}^{\tau}$$，作为超越纯蒙皮的强形变先验。但细粒度布料运动并非只由骨骼决定，因此推理时再用 EgoDeformer 做测试时优化：用坐标 MLP 表示平滑非刚性形变，最终顶点为 $$\boldsymbol{v} + \text{F}_{\text{MLP},\Psi}(\boldsymbol{v})$$，优化目标含轮廓项、拉普拉斯平滑项与分部位 ARAP 项：

$$\arg\min_{\Psi}\ E_{\text{Sil}}(\Psi) + E_{\text{Lap}}(\Psi) + E_{\text{Arap}}(\Psi)$$

轮廓项 $$E_{\text{Sil}}(\Psi) = \|\boldsymbol{I}_{\text{Sil}} - \Pi(\boldsymbol{M}')\|^2$$ 让形变网格投影对齐自我视角分割轮廓；作者刻意不在测试时重建像素级褶皱，而把褶皱交给动态高斯生成。

**关键设计 3：网格驱动的动态高斯外观（GaussianPredictor）。** 在网格 UV 空间均匀采样 3D 高斯，每个被三角形覆盖的纹素代表一个高斯，用重心插值获取初始位置/旋转/法向。一个 C-UNet 以法向图与骨骼根位置为输入，预测高斯参数与偏移：

$$\{\Delta\boldsymbol{x}_i, \Delta\boldsymbol{\phi}_i, \boldsymbol{c}_i, \boldsymbol{s}_i, \boldsymbol{o}_i\} = \text{F}_{\text{C-UNet}}(\boldsymbol{n}_i, \boldsymbol{x}_0)$$

再经可微渲染得到 $$\boldsymbol{I}_r = R(\Delta\boldsymbol{x}_i + \boldsymbol{x}_i,\ \langle\Delta\boldsymbol{\phi}_i, \boldsymbol{\phi}_i\rangle,\ \boldsymbol{c}_i, \boldsymbol{s}_i, \boldsymbol{o}_i)$$。训练用 4K 多视角图像以 L1、SSIM 与 ID-MRF 损失监督：$$L = L_{\text{L1}} + L_{\text{SSIM}} + L_{\text{ID-MRF}}$$。头部与身体高斯分开训练，最后拼装做全身渲染。

## 实验结果

作者构建了首个"配对 120 相机多视角 + 单路自我中心"的 4K 全身数据集，含三名受试者（分别对应富纹理、素纹理、多褶皱服装）。在测试序列上评估新视角合成质量，与可动化身法 DDC 及稀疏视角驱动法 DVA、HPC 对比（PSNR 越高越好，LPIPS/FID 越低越好）：

| 方法 | S1 PSNR↑ | S1 LPIPS↓ | S1 FID↓ | S2 PSNR↑ | S2 LPIPS↓ | S2 FID↓ | S3 PSNR↑ | S3 LPIPS↓ | S3 FID↓ |
|------|------|------|------|------|------|------|------|------|------|
| DDC [2021] | 20.81 | 45.89 | 37.30 | 23.35 | 46.56 | 39.28 | 21.40 | 42.92 | 28.63 |
| DVA [2022] | 21.09 | 46.83 | 84.56 | 22.86 | 48.11 | 71.92 | 20.99 | 43.50 | 61.63 |
| HPC [2023] | 19.80 | 49.76 | 32.58 | 21.14 | 54.98 | 87.19 | 20.82 | 46.56 | 34.20 |
| **Ours** | 20.92 | **42.53** | **19.28** | 23.34 | **44.22** | **36.26** | 21.52 | **40.08** | 26.17 |

在感知类指标 LPIPS 与 FID 上，本方法对所有基线均有明显优势；PSNR 有时仅居第二，作者指出 PSNR 非感知指标、偏好模糊结果而非略有错位但更锐利的结果。消融显示：姿态检测器个性化微调把 MPJPE 从约 4~5cm 降到 2.6~3.7cm；IKSolver 的 $$E_{\text{Reg}}$$、MotionDeformer 与 EgoDeformer 各自都对追踪与渲染质量有一致提升（例如去掉 MotionDeformer 时 LPIPS 从 42.53 恶化到 47.37）。

## 亮点与局限

亮点：
- 首次实现单路自我中心 RGB 相机驱动的照片级全身化身，摆脱推理期对固定多相机装置的依赖，用户可在无约束环境自由活动。
- 混合表示（可形变网格 + 表面高斯溅射）兼顾几何可控性与高分辨率外观，配合"运动先验 + 测试时自我视角精化"的两级几何，抗自遮挡与追踪误差。
- 贡献了首个配对多视角与自我中心的 4K 全身数据集，为该方向的评测提供基准。

局限：
- 属于个人专属（person-specific）方案，需要每位受试者在 120 相机影棚采集多段序列并训练，泛化到任意新人成本高。
- 不建模手势与面部表情，"全身"仅指着衣躯体。
- 依赖头戴相机在 3D 空间被准确追踪的假设；室外测试因无法获得真值头部位姿，只能假设静态头位并给出根对齐的定性结果。

## 延伸思考

该工作把"自我中心动捕"和"照片级化身渲染"这两条此前平行的研究线首次合流，说明混合表示（显式网格提供可控几何、高斯溅射提供高频外观）在遮挡严重、姿态多变的全身场景下是有效折中。值得延伸的方向包括：把个人专属流程推向少样本或可泛化建模以降低采集门槛；引入手部与面部以支持更完整的社交临场；以及在真实室外光照下解决头位追踪与重光照问题，使其真正走出影棚。此外，测试时把褶皱"交给动态高斯生成而非几何重建"的思路，提示在病态单视角问题中，将高频细节从几何解耦到外观模块可能是更稳健的工程选择。
