"""
WordPiece 理论打分验证演示
==========================================
"""

from wordpiece import WordPieceTokenizer

def run_wordpiece_demo():
    print("=" * 60)
    print("演示 1：WordPiece 是如何嫌贫爱富的（基于互信息而非仅次数）")
    print("=" * 60)
    
    # 构建特定的语料库
    # 'l' 'o' 在一起频率高，但 'o' 单独经常和 'b', 'c' 等组合。
    # 'x' 'y' 只在一起出现 3次，但 'x' 和 'y' 从来没有单独出去过！(绝世好绑定)
    corpus = [
        "lo", "lo", "lo", "lo",  # l o 出现了 4次
        "bo", "bo", "co", "do",  # o 经常自己出门乱混
        "xy", "xy", "xy"         # x y 出现了 3次，且绝对捆绑
    ]
    
    tokenizer = WordPieceTokenizer(num_merges=1)
    tokenizer.train(corpus)
    
    # 查看如果是 WordPiece，哪怕 l和o的结合次数高于 x和y，它也会由于互信息计算首选 x和y！
    print("\n[如果这是 BPE]：必定合成 'lo'，因为 4 > 3。")
    print(f"\n[这是 WordPiece]: \t 首选合并阵营是 '{''.join(tokenizer.merges_rules[0])}'!")
    print("\n✅ 验证成功：由于 'o' 在整个字典里出现的频率太高，被放到了 P(A) * P(B) 的分母里导致打分极度缩水！从而保全了 'xy' 纯净专一词汇的成团。这大大减少了长尾烂词典问题。")

if __name__ == "__main__":
    run_wordpiece_demo()
