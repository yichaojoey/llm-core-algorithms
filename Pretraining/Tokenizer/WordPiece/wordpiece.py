"""
WordPiece (BERT御用分词器) 核心面试级模拟实现
==========================================
【理论揭秘】：你会在极大可能下面临这道灵魂拷问："为什么 BERT 在设计时绝不沿用 GPT 原创好端端的 BPE，而非要闭门造车另立一个 WordPiece？"
答案在于其【融合评判指标的内蕴价值观】的深层分歧：

- **BPE**：彻底的纯粹拜金主义只论“绝对频率（Frequency）”。只要你们俩结伴出现在库里的次数最高，立马就把你们绑起来融合。它极其喜欢合并连同那些“烂大街但是单独也能成事”的废话常见长词。
- **WordPiece**：一个讲求概率论（Mutual Information / Likelihood）的有识之士。它的融合打分 Score = P(AB) / (P(A) * P(B))。
  也就是说：就算你们俩偶尔碰巧总出现很多次，但只要你 A 作为渣男在别的地方也经常出去单独鬼混（即 P(A) 概率也很巨大被放进了分母！），导致你们俩的分数就会被严重拉低，从而制止合并！
  这意味着 WordPiece 特别极其钟爱去合并那些：**“由于一旦出现了 A，它身边就只能是死死绑定 B，他俩绝对从不分家单独使用！”**的极度纯粹强偶合词干！
"""

import collections

class WordPieceTokenizer:
    def __init__(self, num_merges=10):
        self.num_merges = num_merges
        self.merges_rules = []
        self.vocab = {}
        
    def train(self, corpus: list):
        vocab = collections.defaultdict(int)
        
        for word in corpus:
            spaced = " ".join(list(word)) + " </w>"
            vocab[spaced] += 1
            
        for i in range(self.num_merges):
            
            # 统计双核（Pairs）发生频次的绝对数量字典 (等价于计算大分子 P(AB))
            pairs = collections.defaultdict(int)
            # 必须额外独立花时间统计出单身字母单核存活频次字典 (等价于计算分母 P(A))
            single_freqs = collections.defaultdict(int)
            
            for word, freq in vocab.items():
                symbols = word.split()
                # 数清每一个词在这个世界里的体量
                for sym in symbols:
                    single_freqs[sym] += freq
                # 数清他俩在一起相会的时候有多少次
                for idx in range(len(symbols)-1):
                    pairs[symbols[idx], symbols[idx+1]] += freq
            
            if not pairs:
                break
                
            # ==========================================
            # 【核心面试点】：基于语言模型最大似然差的替换判决
            # ==========================================
            best_pair = None
            max_score = -1.0
            
            for pair, count_ab in pairs.items():
                
                # 面试手撕代码必写：Score = count(AB) / (count(A) * count(B))
                # 这种以空间相互牵扯为分母打压的机制极大阻隔了垃圾标点符号成为连带体的可能
                score = count_ab / (single_freqs[pair[0]] * single_freqs[pair[1]])
                
                if score > max_score:
                    max_score = score
                    best_pair = pair
            
            if best_pair is None:
                break
                
            self.merges_rules.append(best_pair)
            
            # 使用该组天作之合 pair 进行物理隔离合并即可（同 BPE 的 merge_vocab 用法）
            v_out = {}
            bigram = ' '.join(best_pair)
            replacement = ''.join(best_pair)
            for word_in in vocab:
                word_out = word_in.replace(bigram, replacement)
                v_out[word_out] = vocab[word_in]
            vocab = v_out
            
        self.vocab = vocab
        
    def tokenize(self, text: str):
        # 此处省略复杂的贪婪前缀匹配，仅演示逻辑核心
        pass
