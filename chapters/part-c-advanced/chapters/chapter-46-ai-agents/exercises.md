
**纯推理**（如Chain-of-Thought）的问题：
- 只能基于内部知识，无法获取外部信息
- 容易产生幻觉
- 无法执行实际操作

**纯行动**（如简单工具调用）的问题：
- 缺乏系统性推理
- 容易在复杂任务中迷失方向
- 无法解释决策过程

**ReAct（Reasoning + Acting）**将两者结合，形成**思考-行动-观察**的循环。
--
思考：我已经得到最终答案
最终答案：问题的答案

开始：
思考：
```

--
        
        prompt = f"""你需要完成以下任务，通过交替进行"思考"和"行动"来解决问题。

任务：{task}

可用工具：
{tools_desc}
--
        # 模拟LLM推理过程
        if "问题" in prompt and "爱因斯坦" in prompt:
            if "搜索" in prompt:
                return "思考：我需要搜索爱因斯坦的出生日期\n行动：搜索引擎[爱因斯坦 出生日期]"
            elif "1879年3月14日" in prompt:
                return "思考：我已经找到了答案\n最终答案：爱因斯坦出生于1879年3月14日"
        
        return "思考：让我继续分析问题\n行动：搜索引擎[相关信息]"


# 定义工具
def search_engine(query: str) -> str:
    """模拟搜索引擎"""
--
        # 构建上下文
        context = f"用户问题：{user_input}\n\n"
        
        if function_results:
            context += "工具调用结果：\n"
            for result in function_results:
                if result.status == FunctionCallStatus.SUCCESS:
