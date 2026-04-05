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

### [Supervised Fine-Tuning & PEFT]
1. **[SFT_Loss](./SFT/SFT_Loss/)**: Causal LM offset targeting and Negative-100 logic.
2. **[LoRA](./SFT/LoRA/)**: Core Low-Rank Adaptation theory with matrix $B$ zero-initialization.
3. **[AdaLoRA](./SFT/AdaLoRA/)**: SVD-based adaptive importance evaluation with zero-gradient deadlock evasion.
4. **[QLoRA](./SFT/QLoRA/)**: Block-wise quantization memory advantages and on-the-fly computational limits.
5. **[LORA-COMPARISON Guide](./SFT/LORA-COMPARISON.md)**: Deep dive code walkthrough for the LoRA variants.

### [Reinforcement Learning & Alignment]


### 2. [REINFORCE (Vanilla Policy Gradient)](./RL/REINFORCE/)
The grandparent of all probabilistic reinforcement alignment methods. Demonstrates reverse Markov discounted returns and the essential implementation of a Baseline value for mathematical variance stability.

### 3. [PPO (Proximal Policy Optimization)](./RL/PPO/)
The foundational RLHF algorithm behind ChatGPT/InstructGPT. Focuses on Generalized Advantage Estimation (GAE) backwards recursion and the probability clipping mechanism (`Trust Region/Surrogate`) paired with a separate Value network (Critic).

### 4. [DPO (Direct Preference Optimization)](./RL/DPO/)
The breakthrough offline alignment algorithm eliminating the need for a separate reward model. Focuses entirely on optimizing the difference (`log(sigmoid)`) between chosen and rejected response log-probabilities paired with implicit KL bounding against a reference model.

### 5. [GRPO (Group Relative Policy Optimization)](./RL/GRPO/)
The group-relative policy optimization algorithm popularized by DeepSeekMath. Massively lowers VRAM dependencies by removing the Critic network parameter load and substituting an in-group standard normalization per query.

### 6. [GSPO (Group Sequence Policy Optimization)](./RL/GSPO/)
The sequence-level RLHF advancement introduced by the Qwen team. Shifts policy importance ratios from localized Token-level constraints to Sequence-level geometrically averaged restraints, preventing noise scaling and instability for ultra-long reasoning and CoT (Chain-of-Thought) sequence tuning.

## 📖 Deep Dives

For an ultra-detailed, code-level breakdown on the structural differences and dimensional changes when putting these probabilistic LLM algorithms into practice, read the comprehensive [RL-COMPARISON Guide](./RL/RL-COMPARISON.md).

---

*Prepared for advanced LLM Training & Deep Learning engineering alignment interviews.*
