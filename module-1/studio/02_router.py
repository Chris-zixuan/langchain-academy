from langchain_deepseek import ChatDeepSeek
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 工具
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# LLM 并绑定工具
llm = ChatDeepSeek(model="deepseek-chat")
llm_with_tools = llm.bind_tools([multiply])

# 节点
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 构建图
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # 如果 assistant 的最新消息（结果）是工具调用 -> tools_condition 路由到 tools
    # 如果 assistant 的最新消息（结果）不是工具调用 -> tools_condition 路由到 END
    tools_condition,
)
builder.add_edge("tools", END)

# 编译图
graph = builder.compile()