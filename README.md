# LLM Core Algorithms

A clean, minimalist, and interview-oriented repository for the core algorithms powering modern Large Language Models (LLMs). This repository focuses on bridging the gap between theoretical math and actual PyTorch implementations, specifically designed for LLM and RLHF interview preparation.

## 🚀 Features

- **Zero Dependency Core Logic**: Most algorithms are implemented using pure `torch` to perfectly demonstrate the mathematical operations (Masks, Broadcasting, Clamping, etc.) without relying on thick abstractions like `trl` or `gymnasium`.
- **Standalone Demos**: Every RL algorithm includes a `demo.py` showcasing a simulated interaction loop so you can run it out of the box and print the gradient/distribution changes locally in seconds.
- **Unit Tested**: Pure mathematical functions (Advantage calculation, Distribution clamping, Return discounting) are validated with straightforward `pytest` logic tests.
- **Interview Guides**: Each directory comes with its own `README.md` containing the theoretical derivations, concise standard code routines (perfect for whiteboard/Colab interviews), and answers to high-frequency follow-up questions.

## 🧠 Covered Algorithms

### [Pretraining & Fundamental Architectures]
1. **[RoPE (Rotary Position Embedding)](./Pretraining/RoPE/)**: The cornerstone positional encoding mechanism (LLaMA, Qwen, Mistral) showing relative geometric rotation offsets.
2. **[Tokenizer Subword Rules](./Pretraining/Tokenizer/)**: Text mapping methodologies defining intelligence limits early on.
   - **[BPE](./Pretraining/Tokenizer/BPE/)**: Absolute-Frequency bounded bindings establishing GPT sub-word behaviors.
   - **[WordPiece](./Pretraining/Tokenizer/WordPiece/)**: Mutual-Information Likelihood fractionations preserving tightest word-pairings backing BERT designs.
3. **[Attention Variants](./Pretraining/Attention/)**:
   - **[Multi-Head Attention (MHA)](./Pretraining/Attention/mha.py)**: The baseline classic implementation showing shape transformations and pure $QK^T/ \sqrt{d}$ cross-dependencies.
   - **[Grouped-Query Attention (GQA)](./Pretraining/Attention/gqa.py)**: The LLaMA 3 standard lowering KV Cache footprints by enforcing head broadcasting boundaries.
   - **[DeepSeek MLA (Multi-head Latent Attention)](./Pretraining/Attention/mla_deepseek.py)**: The 2025 DeepSeek-V3/R1 core mechanism projecting massive KV memory footprints down to a narrow Latent vector ($c_{KV}$) and decoupling RoPE coordinates, breaking the 128K context bounds.
   - **[FlashAttention Loop](./Pretraining/Attention/flashattention.py)**: Simulates the Memory Tiling and Online Softmax logic preventing O(N^2) memory block allocations avoiding the GPU memory wall.
4. **[MoE (Mixture of Experts)](./Pretraining/MoE/)**: Sparse computation routing matrices featuring crucial Aux-Loss penalties averting load collapse.
5. **[Modern Layers](./Pretraining/Modern_Layers/)**:
   - **[RMSNorm](./Pretraining/Modern_Layers/rmsnorm.py)**: Zero-mean shifting optimization.
   - **[SwiGLU](./Pretraining/Modern_Layers/swiglu.py)**: SiLU-gated bifurcated activation MLPs.
6. **[Qwen 2.5 Architecture Builder](./Pretraining/Qwen2_5_Builder/)**: 
   - **[Qwen Decoder Block](./Pretraining/Qwen2_5_Builder/qwen_block.py)**: Assembling RMSNorm, SwiGLU, and distinctive Bias-enabled GQA arrays depicting a perfect modern Pre-Norm setup separating it from LLaMA baselines.

### [Supervised Fine-Tuning & PEFT]
1. **[SFT_Loss](./SFT/SFT_Loss/)**: Causal LM offset targeting and Negative-100 logic.
2. **[LoRA](./SFT/LoRA/)**: Core Low-Rank Adaptation theory with matrix $B$ zero-initialization.
3. **[AdaLoRA](./SFT/AdaLoRA/)**: SVD-based adaptive importance evaluation with zero-gradient deadlock evasion.
4. **[QLoRA](./SFT/QLoRA/)**: Block-wise quantization memory advantages and on-the-fly computational limits.
5. **[LORA-COMPARISON Guide](./SFT/LORA-COMPARISON.md)**: Deep dive code walkthrough for the LoRA variants.

### [Reinforcement Learning & Alignment]
#### 1. [REINFORCE (Vanilla Policy Gradient)](./RL/REINFORCE/)
The grandparent of all probabilistic reinforcement alignment methods. Demonstrates reverse Markov discounted returns and the essential implementation of a Baseline value for mathematical variance stability.

#### 2. [PPO (Proximal Policy Optimization)](./RL/PPO/)
The foundational RLHF algorithm behind ChatGPT/InstructGPT. Focuses on Generalized Advantage Estimation (GAE) backwards recursion and the probability clipping mechanism (`Trust Region/Surrogate`) paired with a separate Value network (Critic).

#### 3. [DPO (Direct Preference Optimization)](./RL/DPO/)
The breakthrough offline alignment algorithm eliminating the need for a separate reward model. Focuses entirely on optimizing the difference (`log(sigmoid)`) between chosen and rejected response log-probabilities paired with implicit KL bounding against a reference model.

#### 4. [GRPO (Group Relative Policy Optimization)](./RL/GRPO/)
The group-relative policy optimization algorithm popularized by DeepSeekMath. Massively lowers VRAM dependencies by removing the Critic network parameter load and substituting an in-group standard normalization per query.

#### 5. [GSPO (Group Sequence Policy Optimization)](./RL/GSPO/)
The sequence-level RLHF advancement introduced by the Qwen team. Shifts policy importance ratios from localized Token-level constraints to Sequence-level geometrically averaged restraints, preventing noise scaling and instability for ultra-long reasoning and CoT (Chain-of-Thought) sequence tuning.

### [Inference & Serving]
1. **[PagedAttention / Virtual Memory](./Inference_Serving/PagedAttention/)**: OS-level Virtual memory mechanics porting contiguous logical blocks into scattered physical Page Tables maximizing KV Cache GPU utilization, eliminating internal/external memory fragmentation.
2. **[Speculative Decoding / Draft-Verify](./Inference_Serving/Speculative_Decoding/)**: The ultimate speed acceleration avoiding Memory Bandwidth bound limits. Small local models draft $N$ tokens ahead, while large arrays compute huge batched verification matrices in a single forward pass scaling Output Speed.

### [Distributed & Parallel Clusters]
1. **[ZeRO & DeepSpeed](./Distributed_Parallel/ZeRO_DeepSpeed/)**: Stage-1/2/3 parameter optimization splits breaking memory barriers.
2. **[Tensor Parallelism (Megatron)](./Distributed_Parallel/Tensor_Parallel_Megatron/)**: Row/Column MLP matrix splits verifying synchronous associativity equations.
3. **[Pipeline Parallelism](./Distributed_Parallel/Pipeline_Parallel_Megatron/)**: The 1F1B (One Forward One Backward) dense interleaving logic avoiding GPU Bubbles.
4. **[Data Parallelism (DDP)](./Distributed_Parallel/Data_Parallel_DDP/)**: Overcoming DP Master bottlenecks via Ring All-Reduce multi-process vector synchronizations.
5. **[Distributed Comparison Hub](./Distributed_Parallel/README.md)**: Master interview guide connecting the architectural scaleout.

### [Application & Evaluation (2025 SOTA)]
1. **[Context Engineering](./Application_Evaluation/Context_Engineering/)**:
   - **[DSPy Prompts](./Application_Evaluation/Context_Engineering/dspy_programmatic.py)**: Compiling and optimizing modules programmatically escaping manual prompt heuristic limits.
   - **[Agentic GraphRAG](./Application_Evaluation/Context_Engineering/graph_rag_retrieval.py)**: Macro-perspective Graph traversals avoiding standard 'Lost in the Middle' top-K Vector chunk drops.
2. **[Training Context Management](./Application_Evaluation/Context_Management_In_Training/)**:
   - **[Sequence Packing](./Application_Evaluation/Context_Management_In_Training/sequence_packing.py)**: Diagonal Masking and ID-overwrites avoiding catastrophic Padding wastes on batches.
   - **[Trajectory Chunking](./Application_Evaluation/Context_Management_In_Training/trajectory_chunking_replay.py)**: Safe segmentation algorithms splitting deep CoT sequences while preserving RL Markov global bounds (GAE).
3. **[Agentic Coding Harness](./Application_Evaluation/Agentic_Coding_Harness/)**:
   - **[Tool Orchestration](./Application_Evaluation/Agentic_Coding_Harness/tool_orchestration_loop.py)**: Sandbox encapsulation and rigorous JSON/XML parameter executions blocking model chat escapes.
   - **[Viewport & AST Editing](./Application_Evaluation/Agentic_Coding_Harness/viewport_file_editor.py)**: Line-range partial reading and Diff editing avoiding global repository context crashes.
   - **[Bash Feedback Matrix](./Application_Evaluation/Agentic_Coding_Harness/bash_feedback_loop.py)**: Unsupervised Agent Self-Healing looping terminals tracebacks triggering cyclic autonomous code recoveries.

## 📖 Deep Dives

For an ultra-detailed, code-level breakdown on the structural differences and dimensional changes when putting these probabilistic LLM algorithms into practice, read the comprehensive [RL-COMPARISON Guide](./RL/RL-COMPARISON.md).

---

*Prepared for advanced LLM Training & Deep Learning engineering alignment interviews.*
