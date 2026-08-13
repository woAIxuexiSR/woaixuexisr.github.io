---
title: "Spectral-GS: Taming 3D Gaussian Splatting with Spectral Entropy"
authors:
  - "Letian Huang"
  - "Jie Guo"
  - "Jialin Dan"
  - "Ruoyu Fu"
  - "Shujie Wang"
  - "Yuanqi Li"
  - "Yanwen Guo"
category: "Rendering"
track: "Conference"
source: "arxiv"
institution: "Nanjing University"
tags:
  - "3D Gaussian Splatting"
  - "Novel View Synthesis"
  - "Spectral Analysis"
  - "Spectral Entropy"
  - "Condition Number"
  - "Anti-Aliasing"
  - "Densification"
links:
  paper: "https://doi.org/10.1145/3757377.3763907"
  project: "https://letianhuang.github.io/"
---

## 一句话总结

Spectral-GS 从协方差矩阵的谱分析出发，指出 3D-GS 及其变体因缺乏形状感知而产生"针状高斯"伪影，进而提出基于谱熵的 3D 形状感知分裂与保持谱熵一致的 2D 视角一致滤波，在几乎不改动原框架的前提下显著抑制针状伪影、增强高频细节表达。

## 研究背景

3D Gaussian Splatting（3D-GS）用一组各向异性 3D 高斯显式表示场景，实现了高保真、实时的新视角合成。但它容易优化出细长的"针状高斯"（needle-like Gaussians），在提高采样率（放大、相机靠近物体）时表现为明显的针状伪影。

已有改进各有局限：

- **Mip-Splatting** 用 3D 平滑滤波做频率约束、用 2D Mip 滤波近似超采样来缓解走样，但常导致过度模糊，且针状高斯有时仍然残留。
- **Analytic-Splatting** 用像素内高斯积分的闭式表达替换 2D Mip 滤波，同样只是部分缓解针状伪影。

作者用谱分析揭示了根因：这些方法的稠密化（densification）仍只依据谱半径和视角位置梯度决定是否分裂，缺乏对高斯形状的感知；同时它们所用滤波在放大过程中会降低谱熵、增大条件数，带来视角不一致，使针状伪影在放大时更严重。

## 谱分析：三个核心观察

### 高斯的谱度量

3D 高斯的协方差矩阵可特征分解为

$$\Sigma = R S S^{\top} R^{\top} = R \operatorname{diag}\left(s_1^2, s_2^2, s_3^2\right) R^{-1}$$

其中 $$R$$ 为旋转（正交）矩阵，$$s_1^2, s_2^2, s_3^2$$ 是协方差的特征值（谱）。由此定义三个度量：

谱半径（衡量尺度）：

$$\rho(\Sigma) = \max\left(s_1^2, s_2^2, s_3^2\right)$$

条件数（衡量各向异性/尖锐程度）：

$$\kappa(\Sigma) = \frac{\rho(\Sigma)}{\rho_{\min}(\Sigma)} = \frac{\max\left(s_1^2, s_2^2, s_3^2\right)}{\min\left(s_1^2, s_2^2, s_3^2\right)}$$

谱熵（衡量形状是否接近各向同性）：

$$H(\Sigma) = -\sum_{i=1}^{3} \frac{s_i^2}{\operatorname{tr}(\Sigma)} \ln \frac{s_i^2}{\operatorname{tr}(\Sigma)}$$

当 $$s_1 = s_2 = s_3$$ 时条件数最小、谱熵最大（最接近各向同性）。2D 高斯所描述椭圆的离心率满足

$$e = \sqrt{1 - \frac{1}{\kappa(\Sigma)}}$$

条件数越大，高斯越尖锐。**针状伪影对应低谱熵、高条件数的高斯**。

### 观察一：稠密化的损失敏感与形状无感

3D-GS 仅在视角位置梯度 $$\nabla_{\mu_{\text{proj}}} \mathcal{L}$$ 超过阈值 $$\tau_{\text{loss}}$$ 时才触发克隆/分裂。当细长的低谱熵高斯能在训练视角下以很小的损失拟合高频纹理时，稠密化根本不被触发，导致伪影或过度模糊。更关键的是，即使触发分裂，3D-GS 的各向同性缩放（尺度缩为 $$\frac{1}{k}$$，$$k=1.6$$）并不改变形状：

$$\rho(\Sigma_{\text{split}}) = \frac{1}{k^2}\rho(\Sigma), \quad \kappa(\Sigma_{\text{split}}) = \kappa(\Sigma), \quad H(\Sigma_{\text{split}}) = H(\Sigma)$$

条件数和谱熵与分裂前一致，对缓解针状伪影几乎无帮助。

### 观察二：滤波的视角不一致

3D-GS 的 EWA 滤波与 Mip-Splatting 的 2D Mip 滤波使用相同的协方差 $$\Sigma_{\text{filter}} = \Sigma_{\text{proj}} + sI$$，只在不透明度项上不同。放大时雅可比矩阵 $$J$$ 变化导致投影协方差 $$\Sigma_{\text{proj}}$$ 改变，而滤波核 $$sI$$ 保持恒定，于是训练视角与测试视角下 2D 高斯的条件数不再一致：

$$\kappa(\Sigma_{\text{train}}) = \frac{\rho\left(J_{\text{train}}\Sigma' J_{\text{train}}^{\top}\right) + s}{\rho_{\min}\left(J_{\text{train}}\Sigma' J_{\text{train}}^{\top}\right) + s} \neq \kappa(\Sigma_{\text{test}})$$

其中 $$\Sigma' = W\Sigma W^{\top}$$ 是相机空间协方差。该条件数随焦距/深度比增大而上升，因此放大或相机靠近时针状伪影更明显。

## 方法

Spectral-GS 针对上述两类问题分别提出 3D 形状感知分裂与 2D 视角一致滤波，只需对原框架做少量改动。

### 3D 形状感知分裂

把分裂条件从"谱半径 + 位置梯度"改为基于 3D 高斯的谱熵 $$H(\Sigma)$$：当谱熵超过阈值 $$\tau_{\text{spectral}}$$、可能出现视觉针状伪影时，按旧高斯的概率密度采样 $$K$$ 个点，用新的高斯混合分布去逼近旧分布，在保留高频的同时提升谱熵。关键在于缩放是**各向异性**的，用贪心策略最大程度压缩谱半径方向：

$$\Sigma_{\text{split}} = R \operatorname{diag}\left(\frac{1}{k_1^2}s_1^2, \frac{1}{k_2^2}s_2^2, \frac{1}{k_3^2}s_3^2\right) R^{\top}$$

$$k_i = k \cdot \mathbb{1}\left\{s_i^2 = \rho(\Sigma)\right\} + k_0$$

其中 $$\mathbb{1}\{\cdot\}$$ 为指示函数，$$k > 0$$，$$k_0 \ge 1$$。为保证分裂后条件数不超过分裂前，需满足

$$k < -k_0 + \frac{k_0\, \rho^{\frac{3}{2}}(\Sigma)}{\sqrt{|\Sigma|}}$$

由此分裂后的高斯谱熵提高、条件数下降，且不依赖损失梯度，能主动增强高频细节表达。

### 2D 视角一致滤波

先用 2D 高斯滤波近似投影高斯在每个像素窗口内的积分（近似盒式滤波/超采样）：

$$G_k^{2D}(u)_{\text{Box}} \approx o\sqrt{\frac{|\Sigma_{\text{proj}}|}{|\Sigma_{\text{proj}} + sI|}}\, e^{-\frac{1}{2}(u-\mu_{\text{proj}})^{\top}(\Sigma_{\text{proj}} + sI)^{-1}(u-\mu_{\text{proj}})}$$

再引入一个视角自适应的高斯模糊（近似插值）以维持条件数的视角一致性，其核为

$$\Sigma_{\text{blur}} = \left(J_{\text{test}}J_{\text{train}}^{-1}\right) sI \left(J_{\text{test}}J_{\text{train}}^{-1}\right)^{\top} - sI$$

其中 $$J_{\text{train}}^{-1}$$ 为非满秩矩阵 $$J_{\text{train}}$$ 的左/右逆。两高斯卷积仍为高斯（方差相加），最终合成的滤波结果为

$$\Sigma_{\text{filter}} = \Sigma_{\text{proj}} + \left(J_{\text{test}}J_{\text{train}}^{-1}\right) sI \left(J_{\text{test}}J_{\text{train}}^{-1}\right)^{\top}$$

$$o_{\text{filter}} = o\sqrt{\frac{|\Sigma_{\text{proj}}|}{\left|\Sigma_{\text{proj}} + \left(J_{\text{test}}J_{\text{train}}^{-1}\right) sI \left(J_{\text{test}}J_{\text{train}}^{-1}\right)^{\top}\right|}}$$

与 2D Mip 滤波的恒定核不同，这里使用视角自适应核，从而在放大时保持谱熵/条件数一致。该核可近似为 $$s(\text{focal length}, \text{depth}) = s_0 \frac{\text{focal length}^2}{\text{depth}^2}$$。

## 实验结果

基于 PyTorch 在 3D-GS 框架上实现，沿用 3D-GS 默认参数。方法超参：$$\tau_{\text{spectral}} = 0.5$$，$$k = 0.6$$，$$k_0 = 1$$，$$K = 2$$，$$s_0 = 0.1$$。

数据集共 12 个场景：6 个合成场景（HOTDOG、CHAIR、SHIP、LEGO、MATERIALS 来自 Blender，BALL 来自 Shiny Blender 但改了纹理），6 个真实场景（TRUCK、PLAYROOM、FLOWERS 分别来自 Tanks&Temples、Deep Blending、Mip-NeRF360，以及作者自采的高频谱数据集 TRIPOD、STONE、PILLOW）。评测在 1×、2×、4×、8× 多焦距下模拟放大效果，指标含 PSNR、SSIM、LPIPS，并额外报告谱熵。

- **定量**：在全部 12 个场景上，Spectral-GS 在 PSNR、SSIM、LPIPS 上普遍优于 3D-GS、Mip-Splatting、Analytic-Splatting 及 Analytic-Splatting+3D Filter。同时其优化后高斯的谱熵显著更高（多数场景接近 0.9–0.99），验证了"谱熵越高、合成质量越好"的相关性。
- **定性**：在放大时能生成更真实的细节、针状伪影更少；对比方法在高频区域仍出现针状或过度模糊。
- **消融**（BALL 场景）：针状伪影分为场景固有与渲染算法产生两部分。3D 分裂主要处理场景固有伪影并增强高频表达，2D 滤波主要通过维持条件数一致性处理渲染产生的伪影，二者结合的完整版本取得最佳指标；直接在损失里加形状正则项 $$\mathcal{L}_{\Sigma}$$ 的朴素做法效果不及基于分裂的方案。

## 亮点与局限

亮点：
- 用谱半径/条件数/谱熵三个度量把"针状伪影"这一直观现象量化，并据此统一解释了 3D-GS/Mip-Splatting/Analytic-Splatting 的失败原因（损失敏感、形状无感、滤波视角不一致）。
- 3D 形状感知分裂采用各向异性缩放并给出保证条件数下降的约束条件，从根本上改变了原分裂"形状不变"的缺陷。
- 2D 视角一致滤波用视角自适应核替换恒定核，直接针对放大时的条件数漂移，且对原框架改动很小、易于实现。

局限：
- 引入谱熵阈值、$$k$$、$$k_0$$、$$K$$、$$s_0$$ 等超参，需经验设定，跨场景鲁棒性依赖调参。
- 视角自适应滤波依赖测试视角雅可比 $$J_{\text{test}}$$，对未知视角下的近似质量与合成效果存在不确定性。
- 评测集中在合成与中小规模真实场景，方法在大规模无界场景与动态场景上的表现尚待验证。

## 延伸思考

- 以谱熵/条件数作为形状先验的思路，是否可迁移到 2D Gaussian Splatting、动态 GS、表面重建类 GS 的稠密化与滤波中。
- 分裂条件目前用谱熵阈值触发，若能与渲染时可观测的走样度量或频域能量结合，可能获得更贴近感知的自适应分裂策略。
- 各向异性分裂与"减少高斯数量"类压缩方法是否正交，能否在抑制针状伪影的同时兼顾存储与速度。
