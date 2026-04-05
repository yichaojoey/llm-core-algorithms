"""
PagedAttention 面试级超硬核机制模拟实现 (vLLM 物理寻址内核灵魂)
==========================================================
【理论揭秘】：
即便大厂们都学会了 KV Cache。但极速大潮下为什么只有 vLLM 统治了全世界的商用部署接口？
答案就是极其牛逼的：PagedAttention (虚拟显存分页图谱)！

【极度硬核面试考点】：什么是显存碎片化？为什么 KV Cache 会浪费 50% 显存？
普通缓存法（如上方 `kv_cache_generation.py` 所见）：如果你请求了 1024 长度的接龙，显卡必须给你去硬盘里找一段极其平整、连续不可断的 1024 片空地。
但因为每个用户的回答长度不可测（有的半路 50 就停了，有的说到 1024），这就导致显卡内存出现东一块西一块极其细碎的空隙（外部碎片）。最后即使总显存还剩 20GB，但因为拼不出一张大平床，你的新 Batch 直接报错 OOM。

**vLLM (PagedAttention) 的破局法 (模仿底层操作系统)**:
1. 建立庞大的碎块公寓（物理块 Physical Blocks）。
2. 使用 页表 (Page Table / Block Table) 把用户的每个逻辑记忆分片（Logical Blocks）。
3. 允许不连续的物理空间！哪怕你的显存被切得稀巴烂，只要页表能顺藤摸瓜在一块块不相邻的地方给你把特征取出来接合，即可完美杜绝一切外部碎片化。
"""

import torch

class SimulatorPagedAttentionObj:
    def __init__(self, block_size=4, max_physical_blocks=10):
        # 每个块(Page)存放几个Token的厚度信息
        self.block_size = block_size      
        self.max_physical_blocks = max_physical_blocks
        
        # 【物理内存大公寓】：它就是一段在 GPU 极其死板雷打不动的空荡荡碎片房间
        # 这里只存放乱序、碎块化真实的 Key 向量和 Value 向量 (我们用 None 代表房间空虚)
        self.physical_kv_cache = {i: None for i in range(max_physical_blocks)}
        
        # 记录目前哪些物理房号被占的数组
        self.free_blocks = list(range(max_physical_blocks))
        # 颠覆常识的秘密武器：【页表映射追踪器 Page Table】
        # 记录每一句逻辑话语它的先后碎块分别寄存在哪些八竿子打不着的物理房号里
        self.block_table = {}
        
    def allocate_new_block(self):
        """寻找一个极其犄角旮旯的碎片空房住下"""
        if not self.free_blocks:
            raise RuntimeError("显存彻底吃光爆栈了！")
        return self.free_blocks.pop(0)

    def decode_step_paged(self, request_id: str, new_token_kv: torch.Tensor):
        """
        核心考点！当吐出一个新词需要缓存下来时，它的调度系统是如何不占用大段连续显存的？
        request_id: 比如 "User_Chat_张三"
        """
        if request_id not in self.block_table:
            # 第一句话开张，立马先申请一个块的房间挂牌登记！
            self.block_table[request_id] = [self.allocate_new_block()]
            
        # 提取这个人逻辑上的最后一个存放房间，看有没有塞满
        current_logical_block_idx = len(self.block_table[request_id]) - 1
        current_physical_addr = self.block_table[request_id][current_logical_block_idx]
        
        room_content = self.physical_kv_cache[current_physical_addr]
        
        # 给房间硬塞数据
        if room_content is None:
            # 房间一开始是空的，放进去
            self.physical_kv_cache[current_physical_addr] = new_token_kv
        elif room_content.size(1) < self.block_size:
            # 房间还没满 (还可以加塞床位 Token)
            self.physical_kv_cache[current_physical_addr] = torch.cat([room_content, new_token_kv], dim=1)
        else:
            # ✅ 神级调度机制触发：当前房间终于满员了！
            # 以前普通大模型这下完了要把老数组全部挪到一块更大的空地上扩容！
            # 现在：没关系，原地不动！本管家再打发你去远在天边随便哪个没人住的小破角落领个新房源
            new_addr = self.allocate_new_block()
            # 把这毫无连续性的扯淡房号登记到他的名下，逻辑上的关系就被页表硬生生缝合上了！
            self.block_table[request_id].append(new_addr)
            # 再塞进去！完事！(绝对 O(1) 的开销极其牛逼极度省显存)
            self.physical_kv_cache[new_addr] = new_token_kv

    def show_memory_status(self):
        print("\n------- [底层管家的页面分配表追踪] -------")
        for req, addrs in self.block_table.items():
            print(f"用户 {req} 的长记忆碎片，散落地被藏匿在以下天南地北的物理房号内: {addrs}")
        print("------- [而整个系统没有挪动过任何一次大块拷贝] -------\n")

def run_paged_attention_demo():
    print("=" * 60)
    print("极其炸裂的 PagedAttention OS级虚拟显存调度模拟实验")
    print("=" * 60)
    
    # 假设每个 Block 块只能挤下 2 个单词
    engine = SimulatorPagedAttentionObj(block_size=2, max_physical_blocks=20)
    
    # 模拟用户 A 问了一个极度冗长的问题 (长度为 5 个单词)
    for _ in range(5):
        mock_k = torch.randn(1, 1, 16) # 一个单词量级的特征向量
        engine.decode_step_paged(request_id="User_A", new_token_kv=mock_k)
        
    engine.show_memory_status()
    
    print("\n--- 此时！用户 B 插队进来了，要求算力 ---")
    for _ in range(1):
        engine.decode_step_paged(request_id="User_B", new_token_kv=torch.randn(1, 1, 16))
    
    engine.show_memory_status()
    
    print("\n--- 此时！用户 A 突然继续发难，又输出了一堆更长的废话 ---")
    # 如果是不用 PagedAttention，此时 A 之前申请的内存已经被 B 堵住或者不连续了，必须极度大动干戈全盘重排。
    # 我们看引擎是怎么做的
    for _ in range(3):
        engine.decode_step_paged(request_id="User_A", new_token_kv=torch.randn(1, 1, 16))
    
    engine.show_memory_status()
    print("✅ 完美通过！系统甚至让 UserA 的尾巴词汇散逸跨越了缝隙。无视了物理地址的不连贯性。从而达成了目前业界面包块吞吐量霸主的地位。")

if __name__ == "__main__":
    run_paged_attention_demo()
