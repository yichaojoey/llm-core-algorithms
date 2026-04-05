# 百万年薪大模型架构师：全栈通关面试路线图 (Interview Roadmap)

这份库不仅仅是一个代码仓库，而是为你量身定做的 **大模型 (LLM) 架构与基础设施 (Infra) 面试火力网**。
为了打破“背诵八股文立马被面试官问死”的魔咒，请严格按照以下 **由浅入深、由单体扩展到集群** 的顺序进行代码研读和白板推演。

---

## 🗺️ 登神长阶：五步通关法

### 🟢 阶段一：筑基期 (预训练底层原子结构)
**目标岗**：算法研究员 (Researcher) / 模型训练工程师 (Pretraining Engineer)
**学习重点**：熟悉当代主流架构（以 Qwen 2.5 / DeepSeek V3/R1 为蓝图），撕碎“写死前向网络”的面具，掌握真正的张量数学艺术。

- **第一天：基础注意力与 KV Cache 降维**
  - **What**: 研读 [mha.py](./Pretraining/Attention/mha.py) 和 [gqa.py](./Pretraining/Attention/gqa.py)。
  - **Why**: 几乎 100% 的底层面试起手式。面试官不看你会不会写 $QKV$ 相乘，而是看你懂不懂 KV Cache 的恐怖显存占比。你需要通过讲解 GQA 中的 `unsqueeze` 和 `expand` 张量广播机制，证明你懂得了如何在保持注意力精度的同时实现硬核的显存降级原理。

- **第二天：架构巅峰压缩术 (DeepSeek MLA)**
  - **What**: 硬啃 [mla_deepseek.py](./Pretraining/Attention/mla_deepseek.py) 及其 [README.md](./Pretraining/Attention/README.md)。
  - **Why**: 25 年大厂面试必杀！当别人还在背诵 GQA，你抛出 DeepSeek 单卡 128k 的秘密。你需要向面试官证明：放弃存储庞大 KV，转而将上下文极度压缩为一个极小的潜变量（Latent Vector $c_{KV}$），并在前推瞬间实时释放。并且要解释为什么在这种压榨下，旋转位置编码（RoPE）必须要脱离出来运算（Decoupled RoPE）。

- **第三天：绝对与相对空间的结合 (RoPE)**
  - **What**: 研读 [rope.py](./Pretraining/RoPE/rope.py)。
  - **Why**: 面试官经常挖坑问：“为什么现在的模型不用传统的 Sin/Cos 绝对位置编码了？” 你需要结合代码在白板上给出绝杀解答：因为 RoPE 通过复数空间的奇偶翻转后互相交错点乘，在保有绝对坐标的同时，能够自然推导携带“词与词物理距离的相对衰减性”。这种极其优雅的即插即用让长文外推成为可能。

- **第四天：突破硬件访存天花板 (FlashAttention)**
  - **What**: 研读 [flashattention.py](./Pretraining/Attention/flashattention.py)。
  - **Why**: 面试官套话：“FA 让计算变少了对吧？” 你必须反驳！FA 的 FLOPs 其实增加了，它伟大的地方在于通过分块 (Tiling) 算法避开了恐怖的 HBM（全局极慢显存）读写等待。你需要手推代码里的 `Online Softmax` （每当找到新的局部最大值时，如何利用指数差值暴击更新旧分母的数学纠偏法则），这证明了你是极致的物理底层优化专家！

- **第五天：大厂基石方阵与设计哲学冲突 (Decoder Block Zoo)**
  - **What**: 组装工厂横向拉通 [llama3_1_block.py](./Pretraining/Decoder_Block_Zoo/llama3_1_block.py)、[qwen2_5_block.py](./Pretraining/Decoder_Block_Zoo/qwen2_5_block.py) 和 [deepseek_v3_block.py](./Pretraining/Decoder_Block_Zoo/deepseek_v3_block.py)。
  - **Why**: 极有高度的顶峰提问！你需要像背家谱一样当面白板拆穿三家设计的哲学分水岭：LLaMA 为什么打死不用 QKV bias 偏置（死保尺度平移不变性和后期完美量化），Qwen 为什么违背常理死死加上偏置（给位置提供特定微量常数锚点解决局部乱码），以及 DeepSeek 到底是个什么怪物组合（上半身切装 MLA 压碎长程 KV 缓存显存，下半身切换 MoE 避开高密度庞大计算）的完整拼图！

---

### 🟡 阶段二：打磨期 (微调对齐与平民卡极限利用)
**目标岗**：微调工程师 (SFT Engineer) / 开源模型魔改熟手
**学习重点**：搞懂低显存魔法运转的极限数学压缩法则，而不是做只能敲启动脚本的“调包侠”。

- **第六天：因果序列惩罚过滤 (SFT Loss Masking)**
  - **What**: 研读 [sft_loss.py](./SFT/SFT_Loss/sft_loss.py)。
  - **Why**: 面试如果连大模型自回归（右移预测错位匹配）都写不清晰直接淘汰！更深入的一步：你要回答在多轮对话拼图里，我们如何通过将不关心的引导提示语标定为特权码 `-100`，强制切断这边的反向传播，只惩罚模型答错的部分，做到算力和效果极度聚焦。

- **第七天：降维分解重构流派 (LoRA Family)**
  - **What**: 拉通对比 [lora.py](./SFT/LoRA/lora.py)、[adalora.py](./SFT/AdaLoRA/adalora.py) 和 [qlora.py](./SFT/QLoRA/qlora.py)，熟读专属对标大课 [LORA-COMPARISON.md](./SFT/LORA-COMPARISON.md)。
  - **Why**: 面试官必问：“秩 $r$ 和缩放因子 $\alpha$ 是怎么作用权重的？” 接着你需要回答为什么我们要抛弃静态 LoRA，转而在 AdaLoRA 中通过 SVD 分解引入重要性惩罚（动态裁剪低价值维度分得算力）；最终解释 QLoRA 为了实现单卡练百亿，付出了极高额的前向计算代价（Double Quantization 流转解压负担）的工程妥协。

---

### 🟠 阶段三：升华期 (对齐玄学与强化学习大阵)
**目标岗**：RLHF/对齐工程师 (Alignment Researcher) —— 也是目前市面上最难招、最值钱的工种之一。
**学习重点**：抛弃旧时代的死板训练，理解如何用反馈奖励机制去“调教”概率空间分布。

- **第八天：探索马尔可夫未来传递 (REINFORCE & PPO)**
  - **What**: 研究 [reinforce.py](./RL/REINFORCE/reinforce.py) 并死磕王权算法 [ppo.py](./RL/PPO/ppo.py)。
  - **Why**: 当面试问到 RLHF 的本质不稳定问题。你要能写出倒推时间轴算广义优势估计（GAE）的魔法结构！它让当前步的评分不仅看眼下，还能够通过极其复杂的折现率融合几十步以后的因果反馈。证明你不是改改参数的人，而是玩弄策略期望的神。

- **第九天：现代革命性重构卸甲 (DPO vs GRPO)**
  - **What**: 对比钻研 [dpo.py](./RL/DPO/dpo.py) 与极简大杀器 [grpo.py](./GRPO/grpo.py)，熟读极品面霸长文 [RL-COMPARISON.md](./RL/RL-COMPARISON.md)。
  - **Why**: 这是你面试要高薪的最后防线。“当你公司没钱跑巨型 4 个模型构成的 PPO 集群怎么办？” 直接把 DPO（通过数学 Log-Sum-Exp 巧劲完全免除 Reward 模型计算）和 DeepSeek 的 GRPO（彻底裁决庞大的 Critic 预估网络，让生成同一个问题的多份不同答案直接在小集群组内互相“内卷攀比评分”归一化）甩给面试官，展示你极强的降级部署实操方案。

---

### 🔴 阶段四：基建与炼钢 (推理之王与分布式大杀器)
**目标岗**：大模型架构师 (Infra Engineer / HPC Engineer / Serving Architect)
**学习重点**：解决能跑得起来跟“如何飞起来并不崩盘”之间的质变。

- **第十天：操作系统的显存移植降维 (PagedAttention)**
  - **What**: 死磕 [paged_attention.py](./Inference_Serving/PagedAttention/paged_attention.py) 与基础对照 [kv_cache_generation.py](./Inference_Serving/PagedAttention/kv_cache_generation.py)。
  - **Why**: 推理侧最强问题：“如何解决极度漫长的不同长短的用户文本，在生成时产生的 GPU 外部碎片化坑点死机？” 你需要将操作系统的页表机制（Page Table）手绘出来，向他们证明怎么把毫无连贯性的各个断层显存小块（Block），强制在逻辑层粘合成顺滑长片的（vLLM 的霸权地位核心思想）。

- **第十一天：突破内存天堑 (Speculative Decoding)**
  - **What**: 拆解推敲极大提速魔法 [speculative_decoding.py](./Inference_Serving/Speculative_Decoding/speculative_decoding.py)。
  - **Why**: “大模型跑得慢是算力不足吗？” 你的回答必须斩钉截铁：大错特错，是被读权重的延迟卡死了（Memory Bandwidth Bound）。你需要讲解这个无懈可击的双簧打法：小模型疯狂盲猜产出草稿 -> 巨型模型在一次并行前向传播（时间等于只输出 1 词）里极其霸道地全部核验校验，如果不被降怒驳回，直接白嫖几倍的速度爆发！

- **第十二天：集群斩裂封神阵 (Distributed Scaleouts)**
  - **What**: 领略分布式巨头的神话合集 [Distributed_Parallel/](./Distributed_Parallel/) 中的 DDP/ZeRO/Megatron，以及终极归纳档 [Distributed_Parallel/README.md](./Distributed_Parallel/README.md)。
  - **Why**: 面对真大体量公司的最后一问。你要会把 Megatron Tensor Parallel 的 MLP 切分矩阵画在黑板上，证明就算把巨型计算硬生生横竖切给两张显卡，结果利用 `All-Reduce` 把它们并加起来居然在数学上毫无破损！你要证明 ZeRO-3 是怎么把让几十甚至上百 GB 臃肿的 Adam 优化器被极其残忍地均摊粉碎发配给每一个显卡的，真正攻破系统墙。

---

### 🟣 阶段五：造物主 (AI 应用与智能大反攻)
**目标岗**：多智能体研发架构师 (AI Web Agent/SWE-Agent DevOps)
**学习重点**：不玩文字游戏，让大模型长出能够读取文件执行系统动作手和脚的控制论。

- **第十三天：反击傻瓜式 Prompt 与迷失陷阱 (2025 Context)**
  - **What**: 脱离玄学写死字符串，看 [dspy_programmatic.py](./Application_Evaluation/Context_Engineering/dspy_programmatic.py) 和破局算法 [graph_rag_retrieval.py](./Application_Evaluation/Context_Engineering/graph_rag_retrieval.py)。
  - **Why**: 因为人类手写的 Prompt 没法迭代优化。你要阐明为何采用 DSPy 编译器思想将 Prompt 作为“张量矩阵”进行迭代优化调参。还要说明在面对大海捞名流的问题中，传统向量 RAG 会把实体信息淹死在中间导致提取失败，为什么 GraphRAG 那种极度费劲抓取聚变节点的社区检测能够一击致命，从而提供唯一的全局宏伟视点解答。

- **第十四天（决战前夜）：让 LLM 干实体的无上限自演化地狱 (Agentic Harness)**
  - **What**: 参透 [Application_Evaluation/Agentic_Coding_Harness/](./Application_Evaluation/Agentic_Coding_Harness/) 下面的全部代码，包括严苛沙盒路由 [tool_orchestration_loop.py](./Application_Evaluation/Agentic_Coding_Harness/tool_orchestration_loop.py) 与视口 AST 修改 [viewport_file_editor.py](./Application_Evaluation/Agentic_Coding_Harness/viewport_file_editor.py)，以及极度硬核的自循环纠错 [bash_feedback_loop.py](./Application_Evaluation/Agentic_Coding_Harness/bash_feedback_loop.py)。
  - **Why**: 如果你面前的老板要搞的是 SWE 智能体框架，你不能答你用过 Langchain，那太浅。你需要演示为了防止极度漫长的项目导致上下文耗费 OOM，你怎么强制模型对一个庞然代码库做 AST 局部差异抓取；你需要讲述，当工具执行 bash 失控爆出红线（Traceback）时，你的框架系统是怎么无休止地剥离这些报错塞回给模型强逼他自动流转、修补直至终端绿灯的。这就是这个世界的终极自动化形态！
