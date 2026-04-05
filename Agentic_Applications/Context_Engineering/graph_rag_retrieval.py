"""
Agentic GraphRAG (图知识网络 RAG) 面试级仿真
=========================================
【理论揭秘】：
普通 RAG (Vector RAG / Top-K RAG) 怎么做的？把你提的 Question 用 embedding 打成向量（比如 768 维），去干脆面数据库里搜最接近的 5个段落。
**直接坠毁情景**：“这本 10 万字小说里的所有主要人物都和那个大反派有什么联系？” 
普通 RAG：不知所措。因为它只能顺藤摸瓜找局部极度相似的词（比如包含主角名字的一两句话），**根本没有大局观视野 (Global Perspective)**。

**GraphRAG (微软主推架构) 屠龙术**:
1. 先把底库扔给超庞大的 Agent，它像苦力一样，不是切段落，而是从段落里疯狂提取。也就是 【实体 Entity】(人名、地点) 和 【关系 Relationship】(A是B仇人)。
2. 把这些抽出来，结成一张巨大的蜘蛛网 (Knowledge Graph)。
3. 使用网络图上对实体进行**聚类 (Clustering，比如鲁镇村的所有人形成一个簇/Community)**。
4. 当问全局问题时：直接不搜向量了！系统用大模型对这几百个聚集的“簇”各自生成一段全局大摘要汇总，直接拿到终极完整答案！这就是对抗 "Lost in the Middle" 最强的宏观战法。
"""

class MockKnowledgeGraph:
    def __init__(self):
        # 智能体苦工在后台提点提取出来的网络图
        self.entities = ["张三", "李四", "江湖客栈", "大反派"]
        self.relationships = [
            ("张三", "居住在", "江湖客栈"),
            ("张三", "击溃过", "大反派"),
            ("李四", "是被雇佣来刺杀某人的", "大反派"),
            ("李四", "居住在", "江湖客栈")
        ]
        
    def perform_community_clustering(self):
        print("[GraphRAG 底层架构]: 正在应用社区侦测算法 (如 Leiden 算法)...")
        # 聚类发现 张三、李四 和 客栈 高频绑定
        return {"客栈势力簇": ["张三", "李四", "江湖客栈"], "反派势力簇": ["大反派"]}
        
    def map_reduce_global_summary(self, clusters):
        print("\n[GraphRAG 底层架构]: 抛弃传统的 Vector Search 搜索。")
        print("[GraphRAG 底层架构]: 开启 Map-Reduce 狂暴算力模式！直接给大模型传入浓缩的图节点要求它瞎编...不对，要求它总结全局！")
        
        # 极度节省 token 但获得了上帝视角大局观
        summary = "【上帝视角总结】：客栈势力（包含杀手与大侠）呈现明显的反抗趋势，且与反派势力有直接且矛盾的物理性对抗连接。"
        return summary

def graph_rag_demo():
    print("=" * 60)
    print("GraphRAG 核心面试：用节点对抗无脑向量提取带来的管窥蠡测")
    print("=" * 60)
    
    question = "请用全局视角，解释《江湖》里的势力动态和反派困境"
    print(f"用户询问极其宏大甚至没有特定具体词汇的问题: {question}\n")
    
    graph = MockKnowledgeGraph()
    clustered_communities = graph.perform_community_clustering()
    answer = graph.map_reduce_global_summary(clustered_communities)
    
    print("\n✅ 【终极全局答复】:", answer)
    print("（如果是普通的 Vector RAG，它大概率只能搜出来一堆含有'江湖'两个字的不相干日常对话废代码）。")

if __name__ == "__main__":
    graph_rag_demo()
