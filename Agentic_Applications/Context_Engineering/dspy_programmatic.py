"""
DSPy 范式模拟：告别玄学黑盒，将 Prompt 视作“网络参数”进行程序化编译 (Compile)
========================================================================
【理论揭秘】：
在 2024年以前，人类专家每天像炼金术士一样天天盯着屏幕敲：“You are a helpful assistant... you must think step by step”。换个模型，Prompt 就得全重写，极度玄学无脑。
现在 (DSPy 范式)：我们写代码时只定义 **Signatures（签名）** (例如：输入 `Question` -> 输出 `Answer`) 和 **Modules（模块）**。
然后，靠着提供一份正确的小测试集 (`TrainSet`) 和评分标准 (`Metric`)，让 **Teleprompter（编译器/优化器）** 像反向传播改权重一样，自动在暗中通过大量的对话去筛选出最优质的 few-shot examples（好用示例）塞进 Prompt，甚至让模型自动用 LM 修改它自己 Prompt 前面的指引前缀词！
**Prompt 不再是玄学，而是可训练的参数！**
"""

class MockDSPyModule:
    def __init__(self, signature_name: str):
        self.signature = signature_name
        # ⚠️核心：你的 Prompt 相当于一个随时会被优化器（Compiler）悄悄篡改和强化的张量参数！
        self.learned_examples = []           # 动态塞进去的少样本示例
        self.learned_instructions = "Please answer the user." # 会被大模型自己优化的指示语
        
    def forward(self, input_data: str):
        print(f"\n[模型底层组装]: 将隐藏的指导语: '{self.learned_instructions}'")
        print(f"[模型底层组装]: 挂载 {len(self.learned_examples)} 条编译器千辛万苦为你提炼的最佳 Few-shot 示范...")
        print(f"最终大模型看到的组装 Prompt -> {input_data} ?")
        return "模拟的极其精准的回答"


def dspy_compiler_simulation():
    print("=" * 60)
    print("DSPy Compiler 模拟演练：机器自动打磨 Prompt 时代")
    print("=" * 60)
    
    # 1. 工程师只负责极其极简地定义输入输出约束（Signature）
    pipeline = MockDSPyModule(signature_name="question -> answer")
    
    print("\n--- 【第一阶段】 未经编译器处理（Zero-Shot 裸体） ---")
    pipeline.forward(input_data="为什么天空是蓝色的？")
    
    print("\n--- 【第二阶段】 启动 Optimizer (Teleprompter) 自动炼丹黑魔法 ---")
    print("【系统日志】: DSPy Optimizer 正在提取你的 300 条 QA 验证集...")
    print("【系统日志】: DSPy 调用背后的 Teacher LM 开始进行盲跑...")
    print("【系统日志】: 根据你的精确度打分法则 (Metric)，它发现有 3 条数据的推理链极其精彩！截留！")
    print("【系统日志】: 自动生成了更长的系统指导语！")
    
    # 模拟编译器修改了模块内部的隐性参数 (Prompt 结构)
    pipeline.learned_examples = [
        {"Q": "苹果什么颜色", "A": "思考：... 红色"},
        {"Q": "海什么颜色", "A": "思考：... 蓝色"}
    ]
    pipeline.learned_instructions = "You are a logical reasoner. Always output '思考：' before answering."
    
    print("\n--- 【第三阶段】 编译完成（Compiled），部署至线上 ---")
    pipeline.forward(input_data="为什么天空是蓝色的？")
    
    print("\n✅ 面试核心亮点：如果这个管线在 Llama3 上很完美，换到 Qwen 2.5 也许就崩了。")
    print("在以前你只能痛哭流涕人工手动去改几个月的 Prompt。")
    print("但在 DSPy 里，你只要换个模型接口代码，重新调用 `compile(trainset)` 一键运行！机器会自动去找最适合 Qwen 2.5 脾气的 prompt！这就叫程序化！")

if __name__ == "__main__":
    dspy_compiler_simulation()
