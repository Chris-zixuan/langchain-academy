import random 
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# State（状态）
class State(TypedDict):
    graph_state: str

# 条件边
def decide_mood(state) -> Literal["node_2", "node_3"]:

    # 通常，我们会使用状态来决定下一个要访问的节点
    user_input = state['graph_state']

    # 这里，我们简单地在节点 2 和 3 之间做 50/50 的随机选择
    if random.random() < 0.5:

        # 50% 的概率返回 Node 2
        return "node_2"

    # 50% 的概率返回 Node 3
    return "node_3"

# Nodes（节点）
def node_1(state):
    print("---Node 1---")
    return {"graph_state":state['graph_state'] +" I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state":state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state":state['graph_state'] +" sad!"}

# 构建图
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# 编译图
graph = builder.compile()