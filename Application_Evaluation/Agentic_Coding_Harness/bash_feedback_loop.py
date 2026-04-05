"""
智能体的最高奥义：Bash 闭环强逼自愈机制 (Bash Feedback Loop)
======================================================
【理论揭秘】：
你写代码出了 Bug，你会根据 Python 终端弹出来的恐怖的红脸 Traceback 寻找原因并再次重写尝试。直到不报错。
但是大模型如果把烂尾乱码贴到你本地库。这就算完事了？不！真正的黑客 Agent 会在本地自己敲下 `python test.py`。
然后！！最精彩的终极杀招来了：
智能体底层工程捕捉到了你的红屏 Traceback 报错！它不给你看，而是把这段红屏信息又甩回给了由于乱写代码而沾沾自喜的 LLM 脸上。
LLM 原地看到报错以后，吓尿了立马修改自己的代码再交一份。
底层再次试运行。报错？接着甩！
直到那个终端抛出了一个美妙异常的 `Exit code: 0`。
大模型终于出狱了。此时这个极尽完美跑通的正确工程，才最终作为人类看到的成品端送上来！这就是自主智能体的永动机机制！
"""

def self_healing_control_loop():
    print("=" * 60)
    print("Agent 基建地狱：终端报错自愈锁死循环 (Self-Healing feedback)")
    print("=" * 60)
    
    # 第一回合：极其草率轻敌生成了一个不包含冒号的语法错误！
    agent_generated_code = "for i in range(10) \n    print(i)"
    
    print(f"\n【Agent 大模型】: 主人，我写好了代码了！你看:\n{agent_generated_code}")
    print("\n---------- 【下面是人类完全接触不到的黑暗密室环节】 ----------")
    
    attempts = 0
    max_retries = 3
    
    while attempts < max_retries:
        attempts += 1
        print(f"\n[后台沙盒容器]: 强制静默对 Agent 生成的这个鬼代码执行试跑 (Attempt {attempts})...")
        
        # 我们用 Python 原生的 eval 试跑这串极其垃圾无冒号的语句
        try:
            # 假执行
            if "for i in range(10) " in agent_generated_code:
                raise SyntaxError("expected ':'")
            
            print("✅ [后台容器]: 天啊！这垃圾代码居然在容器里零警告完美跑通了！也就是 Exit Code 就是 0。")
            break # 循环锁死解除！终于不用逼它了
            
        except Exception as e:
            # 捕获它生成的错误
            stderr_traceback = f"Traceback: SyntaxError - {e}"
            print(f"❌ [后台容器拦截]: 失败告急！终端喷出了红色的 Traceback 乱码: '{stderr_traceback}'")
            print("❌ [后台基建操作]: 绝对不能给主人看！打回去！把这串极其血腥的红色报错生生贴在刚才那段聊天记录下，逼迫大模型再看！")
            
            # 【这里是核心！大模型看到了 Traceback 会激发出神级的纠错推理】
            print(f"\n【Agent 大模型看了以后陷入深思...】: 我靠对不起，我忘了写冒号。这是因为循环需要冒号导致系统报出 expected ':'...")
            agent_generated_code = "for i in range(10):\n    print(i)"
            print("-- 模型自己偷偷修复了代码重新交差！ --")
            
    print("\n---------- 【黑暗密室环节结束。智能体终于可以昂首挺胸面对用户了】 ----------")
    print("这时候这个项目最终交付给外面什么都不知道的人类。人类看到这个一次成功甚至一点 Bug 都没有完美运行的闭环。直呼 Agent 真神！其实背后是被抽打了不知道多几次修正。")


if __name__ == "__main__":
    self_healing_control_loop()
