# 预训练基座主引擎：Attention 变体 (MHA vs GQA) 面试指南

> 📖 **一句话解释**：大厂预训练模型面试考 Attention 的时候在考什么？不是默写那个 $Q*K/\sqrt{d}$ 的小公式，而是让你说穿 **KV Cache (键值缓存) 在推理阶段是如何吃掉几百 G 显存的？** 为了保住显卡不挂掉，现在的业界龙头（LLaMa, Qwen, Mistral, DeepSeek 等）为何全票投诚并全盘抄袭了 Grouped-Query Attention (GQA) 取代原本封神的 MHA，甚至进一步演进出 MLA？

---

## 目录

- [一. Multi-Head Attention (MHA): 标准巨像](#一-multi-head-attention-mha-标准巨像)
- [二. Grouped-Query Attention (GQA): LLaMA 3 的瘦身魔法](#二-grouped-query-attention-gqa-llama-3-的瘦身魔法)
- [三. FlashAttention: 突破 I/O 硬件屏障](#三-flashattention-突破-io-硬件屏障)
- [四. Multi-head Latent Attention (MLA): DeepSeek-V3/R1 颠覆一切的显存神降](#四-multi-head-latent-attention-mla-deepseek-v3r1-颠覆一切的显存神降)
- [面试必考的 Broadcast 代码段](#面试必考张量变阵代码)

---

## 一. Multi-Head Attention (MHA): 标准巨像
大家知道，普通 Transformer 一层一层套，为了预测下一个词，历史产生的 K (Key) 和 V (Value) 会常驻在内存不能被删掉（这叫 KV Cache）。
在原汁原味的 **MHA (Multi-Head Attention)** 中，如果你配置了 32 个 Query 头捕捉不同的语义维度，那你必须为他们配发 32 个 Key 头和 32个 Value头 也就是做一对一的配置。
*   **噩梦到来**：批次一旦增大，文本只要推到几千字长。这堆一对一生成的 K 和 V 长长的大冰糖葫芦就必须存在极快的 GPU 显存内。一张 A100 立马报错 OutOfMemory！
*   换言之：**推理的吞吐量天花板根本不是算力墙，而是被这堆 32 个头维护的数据引发的内存容量墙 / 带宽 IO墙 堵死了**。

## 二. Grouped-Query Attention (GQA): LLaMA 3 的瘦身魔法
**GQA (Grouped-Query Attention)** 在 2023 年统一了江湖。
既然给每一个 Query 大少爷配发专属的 K / V 保镖太烧钱了。那我们能不能扣门一点？8 个 Query 并成一组，共享一套 K/V 保镖！
于是，Query 的维度还是 32头（用于极致发散思维找问题），但 K 和 V 被极限缩小为 4头！
*   **极简内存**：常驻内存要保存的数据直接干掉了 8 倍！！！(从32头降到了4头)从而把极其宝贵的 GPU 显存压榨到了极点，直接可以把 Batch推理放大很多倍。
*   **不掉效果**：由于 Q 依然多头保持自由，虽然 K 共享，但神奇的大模型证实并吸纳了这点缺陷。GQA 效果无限接近 1:1 的 MHA！并完胜所有的全卡只用1个公用头的 MQA。

## 三. FlashAttention: 突破 I/O 硬件屏障
**FlashAttention (FA)** 几乎是目前但凡做微调和算子必被拷问的一环。

**FA 核心理念根本不是降低算力 FLOPs！其实 FLOPs (浮点乘加指令量)它甚至反过来还多做了一丢丢额外的计算！它降低的是致命恶魔内存的极致大拖累 —— IO Read/Write！**

它洞若观火地指出：芯片上离计算晶体管最近的高速缓存 **SRAM** 其实跑得如同闪电飞一样快（算力不是瓶颈），但是它极度小（比如几十兆）。而我们在外面插的大显存 **HBM** 动不动 80 GB 超庞大但是它太外重内干导致读数据极其拉跨慢长（此即 Memory Wall 内存墙）。
为了不把以前极其长的一条队伍：$\text{Softmax}(Q K^T)$ 形成高达 $O(N^2)$ 的黑洞般大小的数字群全部扔去外部很慢的 HBM 然后再一点一点读进来。

**神级 Tiling 分块大卸八块法 + Online Softmax 逻辑锁**：
我们直接一点点抽取 Query 切块和对应的所有 KV 小切块，全部缩在小小的晶体管极少范围内做内积碰撞擦出火花立刻生成出一点局部特征。并且极其讨巧在数学推导上演变出了所谓的 `Online Softmax` (详见代码内解析如何进行老大哥被推翻时的分母退位降级乘法修正法！)。直接让最终的缩影张量一直保持最多也不超过 $O(N \cdot d)$ 而非极其恐怖的平方 $O(N^2)$。

---

## 面试必考张量变阵代码 (GQA 扩写维度操作)

既然这 8 个独立脾气各不相同的 Q 兄弟要跟仅有 1 个干瘪的共享 K 头作相乘打分。那这个干瘪的 K 头怎么复制裂变（Broadcast 广播）变成 8 份去应对大家呢？
千万不要傻傻的写 For 循环低效地一遍遍打。必须掌握高级广播：**`unsqueeze(2)` 加 `expand` 加 `reshape` 大法！**

```python
# [Batch, KV_Heads数量, Seq_Len, 脑袋维度]
def repeat_kv(self, x, n_rep): # n_rep 即为一个K需要裂变分给多少个 Q 兄弟

    B, num_kv_heads, T, HeadDim = x.shape
    
    # 🌟 破局点 1：在其原本头序列之后强行凿开一道虚拟空间
    x = x.unsqueeze(2)                    # [B, 4, 1, T, HeadDim]
    
    # 🌟 破局点 2：在凿开的这个空口处不占内存地虚拟膨胀！
    x = x.expand(-1, -1, n_rep, -1, -1)   # [B, 4, 8, T, HeadDim]
    
    # 🌟 破局点 3：用极其坚硬暴力的 Reshape 压缩打扁。使得 4组x8份 -> 真正连续相靠的 32 队列！
    x = x.reshape(B, num_kv_heads * n_rep, T, HeadDim) 
    
    return x
```

在本地直接执行 `python demo.py`，你将直接看到这种降维分包打法究竟在代码的 Params 参数占坑量和形体维系下呈现如此完美的克制缩水表象！
