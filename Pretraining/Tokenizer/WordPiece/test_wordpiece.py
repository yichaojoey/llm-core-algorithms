def test_wordpiece_mutual_information():
    """测试 WordPiece 打分逻辑能否超越 BPE 的绝对数量打分"""
    from wordpiece import WordPieceTokenizer
    
    # 构造极度不对等的依存逻辑
    corpus = ["c a b", "c a b", "k c", "g a"] 
    # c在很多地方，a在很多地方。
    # 但 x y z 只捆绑在一起
    corpus.extend(["x y z", "x y z"])
    
    tokenizer = WordPieceTokenizer(num_merges=1)
    tokenizer.train(corpus)
    
    # BPE 会选 'c a' 或 'a b' 因为出现两次。
    # WordPiece 会立刻锁准 'x y' 因为单独频率分母影响低。
    best_merge = tokenizer.merges_rules[0]
    
    assert best_merge == ('x', 'y') or best_merge == ('y', 'z')
