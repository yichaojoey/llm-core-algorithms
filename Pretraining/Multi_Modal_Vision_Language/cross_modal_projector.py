"""
燃爆多模态的引线：LLaVA 跨模态投影仪 (Vision-Language Projector)
========================================================================
【理论揭秘】：
上一课完成了切图，我们拿到了一条包含 1024 个图片块（每个图块 1024 维）的序列 $V_{tokens}$。
这个时候用户也提了个问题：“图里这只狗是什么品种？”，文本也被切成了 10 个词（每个词 4096 维）的序列 $T_{tokens}$。

极其绝命的死结来了：**图块的维度是 1024（因为视觉编码器 CLIP 它身子板小），大语言模型 LLaMA 甚至 Qwen 的维度是极其深邃的 4096 维！**
这就像插头根本插不进插板！维度不一致导致它们根本不可能融合在同一个矩阵里！

【在 2023 年 LLaVA 论文出现之前】：
大家做大模型搞了极其极其复杂的“交叉注意力管道 (Cross-Attention)”，强行写一堆非常烧计算量的怪异大网络层去把图片引渡过来。开发极其困难！

【LLaVA 开天辟地的暴击方案】：
大道至简！！它说：不就是维度不够拼不到一起吗吗？
既然 LLaMA 听不懂夹杂着 1024维 外语的词。我直接找个 **两层的普通全连接网络 (两层 MLP)** 充当**翻译官**！！
把那 1024 维的视觉词，强行暴力乘两下，把它“撑死扩大”拉伸到极其精确的 4096 维！
接着，把它毫无违和感地直接 “拼接 (Concat)” 到文字 $T_{tokens}$ 序列的旁边！大模型就可以照单全收了！！！
就是这么一层极其暴躁且粗暴的线性网络，开启了目前全世界无孔不入的大模型视觉爆发元年。
"""

import torch
import torch.nn as nn

class LLaVAMLPProjector(nn.Module):
    """
    大名鼎鼎的 LLaVA Projector。
    你看得没看错，这个改变当年整个多模态走向的核心骨干架构，它在代码里只有短短几行！！
    它甚至不用复杂注意力！它是极其极度古典的两层线性感知机 + 激活门！
    """
    def __init__(self, vision_hidden_size=1024, text_hidden_size=4096):
        super().__init__()
        
        # 就是这样！把 1024 撑爆成 4096，供大模型直接吸纳！
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, vision_tokens):
        # 非常干脆的翻译拉伸过程
        x = self.linear_1(vision_tokens)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x

def mm_vlm_concatenation_demo():
    print("=" * 60)
    print("世界级解结：大模型多模态投影同化拼接 (MLP Cross-Modal Projection)")
    print("=" * 60)
    
    vision_dim = 1024
    text_dim = 4096
    
    # 我们拥有上一课切碎的一整张长达 196 个方块的图流 (比如来自于 CLIP ViT-LEncoder)
    vit_output_tokens = torch.randn(1, 196, vision_dim)
    print(f"[视觉传感器抓取]: 截获到异星信号序列，维度十分瘦弱: {vit_output_tokens.shape}")
    
    # 语言模型（LLaMA-3）极其庞大的本体上下文空间
    user_prompt_tokens = torch.randn(1, 10, text_dim)
    print(f"[主脑文字提示词]: 用户问道 '这图是啥？'，它的维度非常深厚: {user_prompt_tokens.shape}")
    
    # 核心动作 1：启动翻译官！强制升维！！
    projector = LLaVAMLPProjector(vision_dim, text_dim)
    translated_vision_tokens = projector(vit_output_tokens)
    print(f"\n[启动 LLaVA 万能翻译器]: 异星视觉信号瞬间被强行撑大至与主脑对齐: {translated_vision_tokens.shape}")
    
    # 核心动作 2：极其简单的拼接术 (Concatenation)
    # LLaVA 的绝杀就在这里，连做都不做区分了，直接把这两段连在一起。你中有我我中有你！
    final_input_to_llm = torch.cat([translated_vision_tokens, user_prompt_tokens], dim=1)
    
    print("\n[世界大融合]:")
    print(f"抛进语言大脑最终的神奇输入: 序列拥有 {(196 + 10)} 个词块，维度统一都是 {text_dim} 维 -> {final_input_to_llm.shape}")
    
    print("\n✅ 面试核心亮点：就是这层极其短小的两层全连接网，配合冻结 (Freeze) CLIP模型 和 LLaMA 语言模型的权值，仅靠单独训练这个 几兆 大小的小投射器几个小时，生生把大象和狮子接上了骨血。这一开创性的避实就虚思想确立了当年所有大语言模型走向眼睛的统一范式！")

if __name__ == "__main__":
    mm_vlm_concatenation_demo()
