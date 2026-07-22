---
title: "Semantic Gesticulator: Semantics-Aware Co-Speech Gesture Synthesis"
authors:
  - "Zeyi Zhang"
  - "Tenglong Ao"
  - "Yuyao Zhang"
  - "Qingzhe Gao"
  - "Chuan Lin"
  - "Baoquan Chen"
  - "Libin Liu"
category: "Animation & Simulation"
track: "Journal"
source: "arxiv"
institution:
  - "Peking University"
  - "Renmin University of China"
  - "Shandong University"
  - "Beijing Academy of Artificial Intelligence"
tags:
  - "Co-Speech Gesture Synthesis"
  - "Retrieval Augmentation"
  - "Large Language Model"
  - "Residual VQ-VAE"
  - "GPT"
  - "Motion Generation"
  - "Semantic Gestures"
  - "Multimodality"
  - "Character Animation"
links:
  paper: "https://doi.org/10.1145/3658134"
  project: "https://lumen-ze.github.io/"
  code: "https://github.com/LuMen-ze/Semantic-Gesticulator-Official"
---

## 问题背景

伴随语言的手势（co-speech gesture）是人类非语言交流的重要组成部分。手势大致可分为两类：

- **节奏手势（beat gestures）**：与语音节奏同步的挥动动作，出现频繁。
- **语义手势（semantic gestures）**：具有明确交流功能的动作，如指示（deictic）、比喻（metaphoric）、象形（iconic）以及象征特定词语的 emblem。

语义手势才是"让手势有意义"的关键，但它们的出现频率远低于节奏手势，位于自然人类动作分布的**长尾**部分。这带来一个核心难题：基于深度学习、在中等规模数据集上训练的系统，很难捕捉动作与语音语义之间的对应关系——因为转录文本中大多数词并不伴随有意义的手势，学习过程往往把这些稀疏的语义手势当作"噪声"忽略掉。

以往方法尝试通过聚类重采样平衡数据、用对比学习挖掘隐式对应、或训练分类器作为额外监督，但都依赖训练数据中足够的"语义手势—语音"配对覆盖度。鉴于此类数据的稀疏性，这种覆盖度很难获得，导致系统在实际中难以在正确的时刻可靠地产出恰当的语义手势，尤其面对多样化的语义手势类别时。

作者的核心思路是引入**检索增强（retrieval augmentation）**：从外部数据库检索有用条目来增强生成。当检索由强大的模型（如大语言模型）执行时，外部数据库反过来被模型中蕴含的知识所增强,从而突破被检索数据本身的局限。

## 核心方法

作者提出 **Semantic Gesticulator**，一个语义感知的伴随语音手势合成系统，由三个关键模块组成：

1. **协同语音手势 GPT 模型（Co-Speech Gesture GPT）**：端到端神经生成器，处理多样的语音音频输入，生成与语音节奏匹配的手势动画。
2. **基于 LLM 的生成式语义手势检索（Generative Semantic Gesture Retrieval）**：理解转录文本上下文，从覆盖常用手势的高质量动作库中检索合适的语义手势，并确定其出现时机。
3. **语义感知对齐机制（Semantics-Aware Alignment）**：在潜空间层面将检索到的语义手势与节奏生成的动作融合，使最终手势既有意义又与节奏协调。

系统以音频与语音转录文本为输入，输出包含手指动作在内的全身手势。

## 技术细节

### 手势 Tokenizer：身体部位感知的残差 VQ-VAE

系统构建在离散潜动作空间之上，用残差 VQ-VAE（Residual VQ-VAE, RVQ）将手势序列 tokenize 成分层紧凑的动作 token。作者在标准 VQ-VAE 上做了三处增强：

1. 将动作表示拆分为**身体（body）**与**手部（hands）**两部分，用两个独立的 RVQ 网络分别建模，各自作为"专家"处理对应部位的复杂性；
2. 编码器 $E_{VQ}$ 与解码器 $D_{VQ}$ 采用 1D 卷积层 + Transformer 层构建，表达能力更强；
3. 通过增加多个残差量化层扩展量化模块容量。

编码器先计算动作特征序列：

$$Z = E_{VQ}(M)$$

随后由一系列残差量化层迭代量化。第 $i$ 层 RVQ 处理来自上一层量化的残差 $r^i_l$，通过查找对应码本 $C_i$ 量化为离散特征：

$$\hat{z}^i_l = \arg\min_{\hat{z}' \in C_i} \|\hat{z}' - r^i_l\|_2$$

新的残差按下式更新（当 $i < R$ 时）：

$$r^{i+1}_l = r^i_l - \hat{z}^i_l$$

其中第一层 RVQ-1 相当于标准 VQ 层（前置残差 $r^1_l = z_l$）。随着层数增加，RVQ 容量呈指数级扩张，显著提升表达能力。重建动作为 $M^* = D_{VQ}(\hat{Z})$。损失函数同时约束重建、一阶（速度）和二阶（加速度）导数以及码本承诺项：

$$L_{RVQ} = w_1\|M - M^*\|_1 + w_2\|\dot{M} - \dot{M}^*\|_1 + w_2\|\ddot{M} - \ddot{M}^*\|_1 + w_3\|E_{VQ}(M) - \mathrm{sg}(\textstyle\sum_{i=1}^{R}\hat{z}^i_l)\|_2^2 + w_4\|\mathrm{sg}(E_{VQ}(M)) - \textstyle\sum_{i=1}^{R}\hat{z}^i_l\|_2^2$$

### 手势生成器 G

生成器基于 GPT-2，以自回归方式在给定前序动作 token 和同步音频特征 $A$ 条件下预测未来手势 token：

$$\hat{z}^*_{hand,L+1}, \hat{z}^*_{body,L+1} = G(A, [\hat{z}_{hand,l}]_{l=1}^{L}, [\hat{z}_{body,l}]_{l=1}^{L})$$

音频特征 $A$ 由 Librosa 提取，包含 MFCC、MFCC delta、constant-Q chromagram、onset 和 tempogram。生成器使用因果注意力层保证时序因果性，用标准分类交叉熵损失训练。

### 生成式语义手势检索

传统的"索引—检索—排序"范式只支持整句检索，无法自动确定检索项在句中的出现时间戳。作者借鉴 GENRE，将检索建模为**基于 prompt 的自回归生成**：给定转录文本作为 prompt，模型复述输入并根据上下文在恰当位置插入检索到的语义手势（如 `1 ARM FLEX`，其中数字为手势索引）。

- **SeG 数据集**：作者综合语言学与人类行为学研究，编纂了一份常用语义手势清单，用专业动捕设备（一男一女两位表演者）采集包含身体、手部、手指的高质量动作。数据集含 200 多种（官方仓库为 208 种）语义手势，共 1.5 小时数据，每个手势平均有 5.7 种不同演绎方式。每条记录包含索引、标签、描述、上下文含义和例句等元信息。
- **LLM 检索模型**：收集二十段 10~20 分钟的 TED 演讲转录并人工标注，构建指令数据集微调 GPT-3.5-turbo。一个关键技巧是把索引信息显式写入指令，要求模型同时预测手势名称与索引，这能显著缓解幻觉（生成不存在的手势或错误组合身体部位）。
- **语义感知索引标识符**：将手势的标签、描述、上下文含义送入 sentence-T5-base 生成嵌入向量，再用带约束 K-means 的层次聚类对这些向量分层，用聚类结果构造分层的语义数字标识符。例如 `FINGERS RUB`（0100）与 `PALM UP`（0101）共享前缀 "010"，仅在第 4 位不同，表明描述或含义相近。这种标识符隐式编码了手势间的层次语义关系，同时保持可解释性。

### 语义手势对齐

对齐模块需解决两个问题：

- **何时融合（When to Merge）**：直接用触发词时间戳作为融合时机会打断正在进行的节奏手势。作者改为先检测音频节拍（beats，通常对应手势最富表现力的 stroke 阶段），再选择离触发词时间戳最近的节拍作为最终融合时机 $l$。
- **如何融合（How to Merge）**：先从 SeG 中提取每个语义手势的 stroke 部分裁成 1 秒片段，用 $E_{VQ}$ 编码成离散 token 构建 SeG token 库。融合时用匹配的语义手势 token 替换原始动作，并将时机 $l$ 对齐到匹配 token 序列的 3/4 处——让语义手势略微提前于对应语义内容出现，反映人类手势约 0.4 秒的规划期。为避免硬替换造成的突兀过渡，借鉴 TM2D 采用加权融合：替换前用半余弦曲线将语义手势 token 权重 $w_s$ 从 0.3 增至 0.7，替换后反向递减，原始动作权重 $w_r = 1 - w_s$。

此外，预训练生成器 G 只见过配对的语音—手势数据，未接触过 SeG token。融合前需通过监督微调（SFT）策略，用带语义标注的转录文本 + TTS 合成音频 + 语义手势 token 及其对齐时间戳构造指令数据集微调 G，只在融合区间对目标语义手势 token 优化交叉熵损失，从而隐式对齐生成器与语义手势分布。

## 实验结果

系统在两个高质量语音—手势数据集 **ZEGGS**（2 小时，19 种风格）与 **BEAT**（约 76 小时，选用英语说话人 2/4/6/8）上训练评估，以 60 fps 生成手势。RVQ 设 $R=4$、下采样率 $d=8$、码本 512；生成器为 12 层、宽 768 的 transformer。训练用四张 NVIDIA 3090Ti，ZEGGS 约 2 天、BEAT 约 5 天。

**用户研究（主观评价）**：采用成对比较，评估人类相似度（Human Likeness）、节奏匹配（Beat Matching）、语义准确度（Semantic Accuracy）三项。

- 在 ZEGGS 与 BEAT 上，本方法在人类相似度和节奏匹配上与 GT 及消融版本相当，均显著优于基线 GestureDiffuCLIP / CaMN。
- 在**语义准确度**上，完整系统显著超越所有基线（$p < 0.001$）；去掉语义对齐模块后语义准确度大幅下降（BEAT 上从 0.41 跌到 -0.58），凸显该模块的关键作用。

**定量评价**：用 Fréchet Gesture Distance（FGD，越低越好）与 Semantic Score（SC，越高越好）。

- 本方法在两项指标上均超越所有基线。ZEGGS 上 SC 达 0.38（GestureDiffuCLIP 为 0.21），BEAT 上 SC 达 0.45（CaMN 为 0.21）。
- 去掉语义对齐会让 SC 大幅下降，但 FGD 基本不变，说明融合操作不损害动作质量。
- 用朴素索引替换语义感知索引会使 SC 下降，验证了语义感知索引标识符的有效性。

**消融研究**证实：Transformer 层缓解重建抖动、残差量化层保留手指等细节；音频节拍决定融合时机对节奏和谐至关重要；加权融合避免不自然过渡；微调 LLM 相比 zero-shot / few-shot 能更鲁棒地理解手势语义。

**手势编辑应用**：作者设计了数据增强框架，用 GPT-4V 从野外 2D 视频中根据语义手势锚点的元信息进行高层筛选，再用 RVQ 潜空间的欧氏距离做低层动作匹配，从而扩充手势库多样性并支持通过自定义 2D 视频库灵活编辑输出风格。

## 贡献与局限

**主要贡献**：

- 提出语义感知的伴随语音手势合成系统，GPT 生成器 + 语义感知对齐机制兼顾动作质量与跨音频输入的泛化能力；
- 开发基于 LLM 的生成式语义手势检索框架，能从手势库高效检索语义手势并确定时机；
- 编纂常用语义手势清单并采集高质量 SeG 数据集，承诺向学术界开源。

**局限**：

- 检索模型只考虑手势文本信息与转录文本，忽略语音音频本身的韵律，可能在韵律平淡处检索出冗余手势、或在韵律显著处漏检；且倾向于检索过多手势，与用户偏好不符（部分源于标注者倾向尽可能多地标注）。
- 融合策略只将 stroke 阶段对齐到音频节拍，更 phase-informed 的策略可能更好保留动作细节。
- 动作质量仍有不足，如脚部滑动、上半身动作过度，可通过训练约束、IK 后处理和更精细的数据集缓解。
- 目前面向单人独白，扩展到双人乃至多人对话场景是值得探索的方向。未来可用 RLHF、DPO 等方法进一步对齐生成器与语义偏好或人类价值。
