from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END

# 我们将使用这个模型同时用于对话和摘要
from langchain_deepseek import ChatDeepSeek
model = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 用于存储消息和摘要的状态类
class State(MessagesState):
    summary: str

# 定义调用模型的逻辑
def call_model(state: State):

    # 获取摘要（如果存在）
    summary = state.get("summary", "")

    # 如果有摘要，则将其添加到消息中
    if summary:

        # 将摘要添加到系统消息中
        system_message = f"Summary of conversation earlier: {summary}"

        # 将摘要追加到较新的消息之前
        messages = [SystemMessage(content=system_message)] + state["messages"]

    else:
        messages = state["messages"]

    response = model.invoke(messages)
    return {"messages": response}

# 判断是结束还是进行摘要
def should_continue(state: State) -> Literal["summarize_conversation", "__end__"]:

    """返回要执行的下一个节点。"""

    messages = state["messages"]

    # 如果消息超过六条，则对对话进行摘要
    if len(messages) > 6:
        return "summarize_conversation"

    # 否则可以直接结束
    return END

def summarize_conversation(state: State):

    # 首先获取摘要（如果存在）
    summary = state.get("summary", "")

    # 创建我们的摘要 prompt
    if summary:

        # 如果已存在摘要，将其添加到 prompt 中
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        # 如果不存在摘要，则创建一个新的
        summary_message = "Create a summary of the conversation above:"

    # 将 prompt 添加到对话历史中
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # 删除除最近 2 条消息之外的所有消息，并将摘要添加到状态中
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# 定义一个新图
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# 设置入口点为 conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# 编译
graph = workflow.compile()