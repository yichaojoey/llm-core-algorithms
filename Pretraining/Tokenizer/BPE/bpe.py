"""
BPE (Byte Pair Encoding) 核心面试级模拟实现
==========================================
【理论揭秘】：GPT 系列的灵魂 Tokenizer（也是目前几乎所有大模型的基座标准）。
大模型本身根本不知道什么是中文什么是英文，它只认一组组的号码（Tokens）。
BPE 的核心机制纯粹是建立在穷举压缩上的：
1. 初始把所有待练单词硬核打碎为单字母（甚至字节 Byte）。
2. 在庞大的字典里统计相邻两个零碎符号结合出现的绝对频次。
3. 把出现频次最大的最高频 Pair 无情碾压合并成一个新的大 Token 词汇。
4. 递归循环这个合并动作，直到你的词表塞满为止！
"""

import collections

def get_stats(vocab: dict):
    """面试考点 1：如何扫描出字典里的配对频次？"""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        # 拆分解刨现在单词目前能切分出来的零碎符号阵列
        symbols = word.split()
        for i in range(len(symbols)-1):
            # 将相邻哥杀对子挂勾当做 Key 增加它的出现频数
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair: tuple, v_in: dict):
    """面试考点 2：如何执行合并更新词表？"""
    v_out = {}
    # 原本带着松散空格的两个符号 ('h', 'e') -> 'h e'
    bigram = ' '.join(pair)
    # 即将被无情铁腕焊死的连体新词 -> 'he'
    replacement = ''.join(pair)
    for word_in in v_in:
        # 【陷阱点】：利用 Python 自带的 string.replace 从左到右把离散符号抹掉合并
        # 注意必须基于空格的分隔防止错误的子串内鬼修改
        word_out = word_in.replace(bigram, replacement)
        v_out[word_out] = v_in[word_in]
    return v_out

class BPETokenizer:
    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.merges_rules = []
        self.vocab = {}
        
    def train(self, corpus: list):
        """面试官考点 3：BPE 的 Train 过程。其实就是无限循环统计合并！"""
        vocab = collections.defaultdict(int)
        
        # 建立初始碎字典：比如 "lower" 初始变成 "l o w e r </w>"
        for word in corpus:
            spaced = " ".join(list(word)) + " </w>"
            vocab[spaced] += 1
            
        # 展开残酷的吞噬同化：
        for i in range(self.num_merges):
            pairs = get_stats(vocab)
            if not pairs:
                break
            
            # 【理论揭秘】：每次取出字典里凭绝对实力称王的频次最高者
            best_pair = max(pairs, key=pairs.get)
            self.merges_rules.append(best_pair)
            
            # 缩减句子长度，无底线增加 Vocab 大小上限！
            vocab = merge_vocab(best_pair, vocab)
            
        self.vocab = vocab
        
    def tokenize(self, text: str):
        """
        面试推断 Inference (用户实际运用阶段)：
        你必须根据以前定下的融合死亡日历表 (merges_rules) 照猫画虎顺序套在别人身上！
        绝不能在推理时现场捏造新词，只能原模原样按曾经的最强老规矩套！
        """
        word = " ".join(list(text)) + " </w>"
        for root_pair in self.merges_rules:
            bigram = ' '.join(root_pair)
            replacement = ''.join(root_pair)
            word = word.replace(bigram, replacement)
            
        return word.split()
