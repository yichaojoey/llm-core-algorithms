"""
Agent 智能体核心控制论：强制沙盒工具调用 (Tool Orchestration)
======================================================
【理论揭秘】：
普通 ChatGPT 模型只会吐普通的 Markdown 文字给你看。
如何让模型具备“行动能力 (Agentic 行动)”？比如让它帮你读取硬盘文件？帮你查天气？帮你推送 Github 代码？
**绝密做法：**
在极度私密的后台系统里，用最严厉的用词指引让大模型放弃跟人聊天。强行规定它的输出必须是极其死板的特定 JSON 或 XML 格式！
比如强逼它如果想拉代码时必须输出带有特殊拦截符的串：`[TOOL_CALL: git_pull {"branch":"main"}]`
我们的系统一旦检测到这行字，立刻截断它的输出（甚至不显示给用户看），我们在沙盒（Sandbox）里执行这条真实指令，然后再把终端输出的乱码包装成 `[TOOL_RETURN]` 也就是 Observation 送回给模型继续看！
这就是把一个语言生成器变身成全能系统骇客工程师的最底层黑科技。
"""

import json

# 假设这是我们提供给模型的两把真枪实权武器（底层沙盒工具）
def tool_read_file(args):
    filename = args.get("file")
    # 真实情况这里是系统的 OS API
    return f"【沙盒执行结果反馈】：{filename} 文件的内容被成功抓取并展示为: 'def hack(): print(1)'。"

def tool_run_bash(args):
    cmd = args.get("command")
    return f"【沙盒执行结果反馈】：Bash执行 `{cmd}` 成功结束！输出 0 错误。"

def mock_llm_agent_generator(prompt):
    """
    假设我们的大模型被洗脑后，极度死板，不再乱打招呼，而是输出了带有特殊转义符的功能请求块。
    """
    print(f"\n[大模型后台开始输出并沉思...] 发现需要查看文件，产生内部调用指令。")
    # 模型输出假造
    return """
    Let me read the target script first.
    <tool_call>
    {"name": "tool_read_file", "arguments": {"file": "core.py"}}
    </tool_call>
    """

def agent_orchestrator_loop():
    print("=" * 60)
    print("全自动修 Bug 智能体基座：工具调用沙盒（Tool Orchestration）拦截解析")
    print("=" * 60)
    
    # 1. 注册映射大模型字符串指令与本地真实函数的挂钩桥梁
    toolbox_registry = {
        "tool_read_file": tool_read_file,
        "tool_run_bash": tool_run_bash
    }
    
    # 2. 我们给大模型塞下任务
    mission_prompt = "User wants me to debug the core.py."
    
    # 3. 大模型产生带特殊定界符的输出
    raw_output = mock_llm_agent_generator(mission_prompt)
    
    # 4. 最核心的 Orchestrator (系统调度层拦截器) 启动！绝对不把这一层显示给客户端看！
    if "<tool_call>" in raw_output:
        print("\n【系统警报!!】 Agent 企图调用现实世界的沙盒武器!")
        
        # 抠出极其严格的 JSON 字符串块
        start_idx = raw_output.find("<tool_call>") + len("<tool_call>")
        end_idx = raw_output.find("</tool_call>")
        json_str = raw_output[start_idx:end_idx].strip()
        
        req = json.loads(json_str)
        func_name = req["name"]
        args = req["arguments"]
        
        print(f"【系统解析成功】: 它想调用 [{func_name}]，传入参数 [{args}]")
        
        # 5. 危险动作区：我们在安全的沙盒容器里执行它的愿望！
        if func_name in toolbox_registry:
            observation = toolbox_registry[func_name](args)
            print(observation)
        else:
            print(f"【系统驳回】: 没有这个防身的武器库映射！报错喂给 Agent 让他重试。")
            
        print("\n接着把这句 真实系统反馈的东西 拼接到下一轮输入的最前面，再次无情抽打大模型让它接着分析判断！")
        print("这就实现了大模型直接与现实终端的交互！")

if __name__ == "__main__":
    agent_orchestrator_loop()
