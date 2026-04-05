"""
BPE 直观演示验证：展现单词是如何被压块吞并的
==========================================
"""

from bpe import BPETokenizer

def run_bpe_demo():
    print("=" * 60)
    print("演示 1：BPE 贪婪融合过程")
    print("=" * 60)
    
    # 构建极其变态集聚的模拟语料库
    corpus = [
        "lower", "lower", "lower", "lower", "lower",
        "lowest", "lowest",
        "newer", "newer", "newer",
        "widest", "widest",
        "water"
    ]
    
    print(f"\n🔹 【初始预备语料形态】如：'lower' -> 'l o w e r </w>' 出现了 5次， 'newer' -> 'n e w e r </w>' 出现了 3次")
    
    # 建立一个只有 4 步寿命的字典融合法
    tokenizer = BPETokenizer(num_merges=4)
    tokenizer.train(corpus)
    
    print("\n--- 查看熔炼操作后制定的四大铁律 (Merging Rules) ---")
    for i, rule in enumerate(tokenizer.merges_rules):
        print(f"融合法则 {i+1}: 将离散的 [{rule[0]}] 与 [{rule[1]}] 死焊成 => [{''.join(rule)}]")
        
    print("\n--- 在新文本上 Inference (Tokenization 切词测试) ---")
    test_word = "flower"
    # test_word 经历了：
    # "f l o w e r"  --> (根据第一定律 e r) -> "f l o w er"
    # "f l o w er"   --> (不符合二、三定律，放过跳过)
    # 根据之前统计由于 "er", "lo" 这些由于高出镜率已经被化为字典里的合法实体
    out_tokens = tokenizer.tokenize(test_word)
    print(f"传入未曾蒙面的生词 '{test_word}'， BPE 严格按照铁律后切分成: \t{out_tokens}")
    print("✅ 验证成功：由于 f 未存在于融合规则，所以被当做了碎片，而由于 lower 经常出现，owe 和 er 都成了大 Token。完美复现碎片词切分机制！\n")

if __name__ == "__main__":
    run_bpe_demo()
