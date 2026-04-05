# 百万年薪大模型架构师：全栈通关面试路线图 (Interview Roadmap)

这份库不仅仅是一个代码仓库，而是为你量身定做的 **大模型 (LLM) 架构与基础设施 (Infra) 面试火力网**。
为了打破“背诵八股文立马被面试官问死”的魔咒，请严格按照以下 **由浅入深、由单体扩展到集群** 的顺序进行代码研读和白板推演。

---

## 🗺️ 登神长阶：五步通关法

### 🟢 阶段一：筑基期 (预训练底层原子结构)
**目标岗**：算法研究员 (Researcher) / 模型训练工程师 (Pretraining Engineer)
**学习重点**：熟悉当代主流架构（以 Qwen 2.5 / DeepSeek V3/R1 为蓝图），撕碎“写死前向网络”的面具，掌握真正的张量数学艺术。
- **第一天**：研读 `Pretraining/Attention/mha.py` 和 `gqa.py`。**[白板硬装]** 面试必考：GQA 如何通过 `expand` 和 `reshape` 分发 KV Cache？写出它怎么极其恐怖地节约显存！
- **第二天**：硬啃 `Pretraining/Attention/mla_deepseek.py`。**[地狱必考题]** 25年大厂面试必杀：为什么 DeepSeek 能单卡跑到 128k？它的潜变量 ($c_{KV}$) 压缩怎么在节省 90% 显存的情况下，还要把 RoPE （旋转位置编码）给脱钩（Decouple）出来算的？
- **第三天**：研读 `Pretraining/RoPE/`。**[白板硬装]** 放弃晦涩难懂的虚数公式，只看代码里是怎么用 `sin` 和 `cos` 进行奇偶翻转然后逐元素相乘的！
- **第四天**：硬啃 `Pretraining/Attention/flashattention.py`。**[灵魂提问]** 为什么它不减反增了计算浮点数，但是速度极其逆天？掌握代码里的 `Online Softmax` （每当找到新的局部 Max 就通过指数差值暴击之前算出老的分母的老底）。
- **第五天**：组装工厂 `Pretraining/Qwen2_5_Builder/`。了解什么是 `Pre-norm` 为什么一定要把 `norm` 放在残差相加之前？回答为什么 Qwen 要变态地保留 `bias=True`！

### 🟡 阶段二：打磨期 (微调对齐与平民卡极限利用)
**目标岗**：微调工程师 (SFT Engineer) / 开源模型魔改熟手
**学习重点**：搞懂你在调 LLaMA-Factory 或者是 vLLM 背后，那些低显存魔法是怎么运转的。
- **第五天**：看 `SFT/SFT_Loss/`。**[白板硬装]** 大模型自回归是怎么向右平移一格去算 Cross-Entropy 的？大模型是怎么通过标 `LABEL=-100` 让用户提问的那一部分不产生回传梯度的？
- **第六天**：拉通对比 `SFT/LoRA`、`AdaLoRA` 和 `QLoRA` 的 `README.md` 面经。**[灵魂提问]** QLoRA 为什么要挂着一个微小精密的解压网络？AdaLoRA 的 SVD 分解怎么避免无效维度计算的？

### 🟠 阶段三：升华期 (对齐玄学与强化学习大阵)
**目标岗**：RLHF/对齐工程师 (Alignment Researcher) —— 也是目前市面上最难招、最值钱的工种之一。
**学习重点**：抛弃旧时代的死板训练，理解如何用反馈奖励机制去“调教”概率空间分布。
- **第七天**：彻底推翻没有 PPO 的理解，看 `RL/PPO/ppo.py`。死死记住什么是 `GAE (广义优势估计)` 的倒序公式（极管马尔可夫决策连续性的核心）！
- **第八天**：看 `RL/DPO/` 与 `RL/GRPO/`。如果面试官问：“你服务器资源极其紧张，塞不下 PPO 那冗余拖沓的 4 个臃肿的大模型怎么办？” 直接把 DPO（不需要独立的打分模型和庞大生成） 和 GRPO（DeepSeek 核心：取消了庞大独立的评价 Critic 网络，通过群体相对内卷评价代替标杆）砸在面试官脸上。

### 🔴 阶段四：基建与炼钢 (推理之王与分布式大杀器)
**目标岗**：大模型架构师 (Infra Engineer / HPC Engineer / Serving Architect)
**学习重点**：解决能跑得起来跟“如何飞起来并不崩盘”之间的质变。
- **第十天**：死磕 `Inference_Serving/PagedAttention/`。**[灵魂提问]** 什么是 External Fragmentation 外部碎片化？页表（Page Table）是怎么把离散的显存放进去的？
- **第十一天**：研究 `Inference_Serving/Speculative_Decoding/`。**[绝杀提问]** 被大厂问到 “在极低的算力利用率下，如何打破 Memory Bandwidth 内存墙提升出词字数？” 拿出投机采样双模型 Draft-Verify 策略，解释为何并行校验 5 个词的时间等于只输出 1 个词的时间。
- **第十二天**：领略分布式的变态神话 `Distributed_Parallel/` (ZeRO / TP / Pipeline / DDP)。**[白板硬装]** 默写 Megatron 的 Tensor Parallel 是怎么切 Column 切 Row，证明它们相乘不串台？ZeRO-1、2、3 到底剥削了哪一层的 120GB 的巨额 Adam 优化器参数？

### 🟣 阶段五：造物主 (SWE 智能体与未来系统操控)
**目标岗**：多智能体研发架构师 (AI Web Agent/SWE-Agent DevOps)
**学习重点**：让大模型长出手脚！
- **第十一天**：脱离玄学写死 Prompt，看 `Application_Evaluation/Context_Engineering/dspy_programmatic.py`。
- **最后一天（决战前夜）**：参透 `Agentic_Coding_Harness/`。搞明白大模型是如何被截断包裹的（Tool Orchestration），如何通过抽象语法树（AST）的分页区间打补丁代替重写，如何接收致命致命终端错误红字自愈循环（bash_feedback_loop），这就是未来 5 年不会失业的核心命门。

---

## 🛡️ 面试必杀使用指南

1. **别看太多外部库**：千万不要去背烂大街的 `Huggingface transformers` 或者是 `TRL` (大模型强化学习库) 里面繁复的接口源码！面试是**白板写伪代码**，这个代码库里的纯净 `torch` 演算，就是你写在白板上向面试官直刺本源的尖刀。
2. **看透 `demo.py`**：在面试前，你必须能完整跑通并且解释每一级的 `demo.py` 在输出什么！比如 GRPO 的群体内卷、PagedAttention 怎么实现用户隔离，跑通这些等于证明了你拥有完整的脑内推演能力。
3. **主攻目录面经**：在通勤路上或等号等待期间，疯狂刷看每个目录下的 `README.md` —— 这不是普通说明文档，这全是**大厂考核真题中的连环“陷阱（Trap）”以及“一招致命”的回答**！

祝你通天梯阶，直通金牌架构师宝座！
