from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode

def add(a: int, b: int) -> int:
    """计算 a 和 b 的和。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """计算 a 和 b 的乘积。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a * b

def divide(a: int, b: int) -> float:
    """计算 a 除以 b 的商。

    Args:
        a: 第一个整数
        b: 第二个整数
    """
    return a / b

tools = [add, multiply, divide]

# 定义绑定了工具的 LLM
llm = ChatDeepSeek(model="deepseek-chat")
llm_with_tools = llm.bind_tools(tools)

# 系统消息
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

# 节点
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# 构建图
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # 如果来自 assistant 的最新消息（结果）是工具调用 -> tools_condition 路由到 tools
    # 如果来自 assistant 的最新消息（结果）不是工具调用 -> tools_condition 路由到 END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# 编译图
graph = builder.compile()
