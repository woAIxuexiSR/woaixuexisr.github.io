---
title: "DSCombiner: Double Shrinkage for Combining Biased and Unbiased Monte Carlo Renderings"
authors:
  - Chenxi Zhou
  - Keheng Xu
  - Mufan Guo
  - Xianhao Yu
  - Zhimin Fan
  - Guihuan Feng
  - Yanwen Guo
  - Jie Guo
track: "Journal"
source: author-page
category: Rendering
institution:
  - State Key Lab for Novel Software Technology, Nanjing University
tags:
  - Monte Carlo Rendering
  - Shrinkage Estimator
  - Denoising
  - Bias Variance Tradeoff
  - Path Tracing
  - Photon Mapping
  - Bayesian Framework
links:
  paper: "https://doi.org/10.1145/3763315"
  project: "https://zhiminfan.work/"
---

## 一句话总结

把有偏与无偏两种蒙特卡洛渲染结果在屏幕空间融合：不用单次 James-Stein 收缩，而是做"两步收缩"——先按逆 MSE 凸组合无偏/有偏估计，再朝一个平滑的辐射先验二次收缩，收缩因子由分层贝叶斯框架推出最优解并用 CNN 预测，从而在细节保留与降噪之间取得比现有屏幕空间/路径空间融合器更好的权衡。

## 研究背景

- 领域现状：蒙特卡洛渲染长期面临偏差-方差两难。无偏算法（如路径追踪 PT）无统计偏差但噪声大、收敛慢；有偏算法（如随机渐进式光子映射 SPPM、或 PT 接去噪器）能压噪但引入偏差。没有哪个积分器能通吃所有场景。
- 主流做法：把无偏渲染与其有偏/去噪版本做后校正融合。屏幕空间常用 James-Stein（JS）收缩估计量，用一个收缩因子把估计朝整体均值收缩，能在 MSE 意义上优于极大似然估计。代表工作 NeuralJS 把 JS 收缩用于去噪器输入-输出对的融合。
- 核心痛点：JS 收缩依赖"局部块内方差同质"假设，用块级平均收缩因子。但块内方差常有强空间变化，导致方差突变的边界（阴影、水池边缘）残留噪声；当有偏、无偏由差异很大的两种积分器（如 PT 与 SPPM）生成时，两者都可能高方差且空间剧烈变化，单次收缩明显次优。
- 本文 idea：改用像素级而非块级组合以更好控制细节；但单次收缩的像素级组合最多只能把 MSE 减半、无法去掉残留噪声，因此提出双收缩框架，借助一个平滑辐射先验完成第二步收缩。

## 方法

整体思路：给定同一像素的无偏估计 $$X_u \sim \mathcal{N}(X,\sigma_u^2)$$ 与有偏估计 $$X_b \sim \mathcal{N}(X+\xi,\sigma_b^2)$$，目标是估计真值 $$X$$。在贝叶斯框架下最大化后验 $$\hat{X}=\arg\max_X p(X_u\mid X)\,p(X_b\mid X)\,p(X)$$，其中先验 $$p(X)$$ 是去噪的关键。

关键设计：

1. **理论最优双收缩（核心贡献）**。把 $$X$$ 的先验建为 $$\mathcal{N}(X_p,\eta^2)$$、偏差 $$\xi$$ 的先验建为 $$\mathcal{N}(0,\gamma^2)$$。利用共轭先验，后验仍为联合正态，令指数项导数为零解出闭式估计：

$$\hat{X}=\kappa_2 X_p+(1-\kappa_2)\big(\kappa_1 X_b+(1-\kappa_1)X_u\big)$$

其中

$$\kappa_1=\frac{\sigma_u^2}{\gamma^2+\sigma_b^2+\sigma_u^2}, \qquad \kappa_2=\frac{\sigma_u^2(\gamma^2+\sigma_b^2)}{\sigma_u^2(\gamma^2+\sigma_b^2)+\eta^2(\gamma^2+\sigma_b^2+\sigma_u^2)}$$

第一步用类似"逆 MSE"的机制融合 $$X_u$$ 与 $$X_b$$，能有效压制萤火虫噪点但可能残留低频噪声；第二步把第一步结果朝平滑先验 $$X_p$$ 收缩，收缩强度由 $$\eta^2$$ 控制。这补上了单次收缩"当 $$X_u$$、$$X_b$$ 都含噪时无法去噪"的短板，同时 $$\eta^2$$ 防止过度平滑。

2. **辐射先验 $$X_p$$ 的构造**。强调 $$X_p$$ 是"辅助色彩缓冲"而非高质量去噪结果，偏向平滑、放松细节保留。在每个像素的 $$51\times 51$$ 局部块内用一阶线性模型对 G-Buffer 特征做加权回归，权重按有偏/无偏颜色差与方差构成的指数核给出：

$$w_u^{i,j}=\exp\!\left(-\frac{\lVert X_b^i-X_b^j\rVert^2}{\sigma_{b,i}^2+\sigma_{b,j}^2+\epsilon_1}\right)$$

回归化为最小二乘 $$\arg\min_X \lVert A_u X-Y_u\rVert^2+\lVert A_b X-Y_b\rVert^2$$，闭式解 $$X=(A_u^\top A_u+A_b^\top A_b)^{-1}(A_u^\top Y_u+A_b^\top Y_b)$$，用 Cholesky 分解求逆并在对角加 epsilon 保正定。该回归比此前工作的"优化有偏渲染"更平滑，能消除 SPPM 等未收敛有偏结果的伪影。

3. **数据驱动的收缩因子**。$$\eta^2$$ 与回归得到的 $$X_p$$ 高度相关，难以解析。改用 U-Net（每层 3×3 卷积+ReLU，末接 1×1 投影层）预测。网络输入辐射先验、无偏/有偏颜色及各自 MSE 估计；每像素输出 339 个参数：两个 13×13 的滤波核（Softmax 归一，用于滤波有偏/无偏 MSE）以及一个取指数保正的标量作为 $$\eta^2$$。再用预测的 $$\eta^2$$ 与滤波后 MSE 代入闭式收缩因子。网络能识别先验过平滑区域、给这些像素更高 $$\eta^2$$ 以从原始输入恢复细节。

4. **训练与误差估计**。损失用 SMAPE。数据集含 20 个场景、2548 对图（512×512，Mitsuba 渲染，PT 出无偏、SPPM 出有偏、BDPT 出参考），训练 2000 轮、Adam、学习率 1e-4、随机裁 128×128 块，约一天。无偏方差用样本方差估计；SPPM 有偏渲染的 MSE 用 Hachisuka 等的误差估计框架获得，并按既有做法滤波精修。

## 实验结果

平台 i7-13700K + RTX 3080 Ti，默认 1280×720，后处理耗时占比 <5%，等时比较用 relMSE。

- 对比屏幕空间融合（DeepCombiner、NeuralJS，输入为 PT+SPPM）：在 Spotlight、Kitchen、Swimming Pool、Bookshelf 四个光照复杂场景中，DSCombiner 全面领先。示例 relMSE（括号内为相对 NeuralJS 的加速）：Spotlight 160s 下本文 0.003887（2.1x）对 NeuralJS 0.008053；Swimming Pool 320s 下本文 0.002991（1.6x）对 0.004780；Kitchen 1280s 下 0.000417（1.5x）；Bookshelf 160s 下 0.001245（1.5x）。DeepCombiner 擅长压相关低频噪声但丢细节（如焦散），NeuralJS 能保高频细节但在方差剧变区重新引入噪声。
- 用本文数据集重训 NeuralJS：低采样率下反而略逊于官方预训练模型（本文数据集有偏图噪声更强，阻碍收敛），高采样率下趋于一致——说明数据集不是本文提升的关键因素。
- 对比路径空间融合（VCM，本文用 BDPT+SPPM）：Swimming Pool 2560s 下本文 0.001489（4.9x）对 VCM 0.007246；Dining Room 640s 下本文 0.000332（12.2x）。既更灵活又避免路径空间昂贵计算。
- 单积分器设定（NeuralJS 用 PT+OIDN）：数值仅小幅领先，但在去噪器难处理的焦散、阴影、线状高光等区域细节与降噪更好。
- 不同有偏来源比较：SPPM 作有偏输入优于 OIDN 去噪结果——因为 DSCombiner 从辐射先验获取平滑性，更希望有偏渲染提供无偏渲染给不出的细节，故建议用积分器而非去噪器生成有偏图。
- 消融：用本文辐射先验替换 Gu 等的优化有偏渲染，显著降低有偏渲染引入的噪声图案；像素级收缩控制使恢复细节更稳健、比 NeuralJS 重新引入的噪声更少。

## 亮点与局限

- 亮点：
  - 首个双收缩框架，能灵活高效融合一大类无偏/有偏积分器，不限于"去噪器输入-输出对"。
  - 在分层贝叶斯框架下推出两步收缩因子的闭式最优解，理论清晰；再用 CNN 预测 $$\eta^2$$ 与 MSE 滤波核落地。
  - 像素级收缩取代块级平均，专门解决方差空间剧变区的残留噪声；辐射先验加速收敛且抑制未收敛有偏结果的伪影。
  - 数值与视觉质量同时超越屏幕空间（DeepCombiner、NeuralJS）与路径空间（VCM）融合基线。
- 局限：
  - 由平滑先验与含噪输入恢复细节时，可能重新引入高频或低频噪声；作者建议用多级组合压低频、双缓冲组合压高频。
  - 当前两积分器等时预算分配，未做自适应；按区域/像素自适应分配样本有望进一步提升。
  - 先验固定为正态分布，且辐射先验依赖有参考的监督训练；可探索其他分布与自监督方式预测更高质量先验。

## 延伸思考

- 这篇把"融合两种积分器"从路径空间 MIS（O(n^2)、需解析 PDF）挪到屏幕空间的收缩估计，绕开了 MIS 的适用限制（如流形采样无解析 PDF），是很实用的工程-理论结合点。
- 双收缩的本质是"逆 MSE 组合 + 朝平滑先验的正则化收缩"，$$\eta^2$$ 作为逐像素置信度把细节保留与降噪的权衡显式参数化，比单一全局收缩因子更符合真实图像的非均质方差。
- 建议"用积分器而非去噪器当有偏源"是有意思的反直觉结论：因为平滑性已由辐射先验提供，有偏分支的价值在于补充无偏分支缺失的细节（焦散、阴影），这为如何搭配互补积分器提供了设计指引。
