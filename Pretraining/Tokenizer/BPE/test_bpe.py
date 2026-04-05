import pytest
from bpe import get_stats, merge_vocab

def test_bpe_count_stats():
    """测试频率抽取正确性。是否正确捞出了每一对拼合兄弟？"""
    vocab = {"l o w </w>": 5, "l o w e s t </w>": 2}
    pairs = get_stats(vocab)
    
    # 'l' 连着 'o' 的总频次必须是 7 
    assert pairs[('l', 'o')] == 7
    # 'e' 连着 's' 只有 lowest 出现了两次
    assert pairs[('e', 's')] == 2
    # 'w' 连着 '</w>' 只有 low 出现了 5 次 (lowest由于接的是e所以不在其中)
    assert pairs[('w', '</w>')] == 5

def test_bpe_merge():
    """测试将指定一对兄弟融合成为统一体替换时，未涉及地区是否有发生误伤"""
    vocab = {"l o w e r </w>": 1, "s n o w </w>": 1, "f a w n </w>": 1}
    # 强行下令把 'w' 和 'e' 相连的情况锁成一体
    new_vocab = merge_vocab(('o', 'w'), vocab)
    
    # 原本带分割带空格的句子发生了突变挤压现象
    assert "l ow e r </w>" in new_vocab
    assert "s n ow </w>" in new_vocab
    assert "f a w n </w>" in new_vocab  # 由于他俩中间是a w n，o w 并不挂靠，没受伤，不受影响！
    
    assert new_vocab["l ow e r </w>"] == 1
