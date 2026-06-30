import uuid 

from pydantic import BaseModel, Field

from trustcall import create_extractor

from langchain_core.messages import SystemMessage
from langchain_core.messages import merge_message_runs
from langchain_core.runnables.config import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
import configuration

# 初始化 LLM
model = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 记忆 schema
class Memory(BaseModel):
    content: str = Field(description="The main content of the memory. For example: User expressed interest in learning about French.")

# 创建 Trustcall 提取器
trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    # 允许提取器插入新记忆
    enable_inserts=True,
)

# 聊天机器人指令
MODEL_SYSTEM_MESSAGE = """You are a helpful chatbot. You are designed to be a companion to a user.

You have a long term memory which keeps track of information you learn about the user over time.

Current Memory (may include updated memories from this conversation):

{memory}"""

# Trustcall 指令
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

Use the provided tools to retain any necessary memories about the user.

Use parallel tool calling to handle updates and insertions simultaneously:"""
def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """从存储中加载记忆，并用其个性化聊天机器人的回答。"""

    # 获取配置
    configurable = configuration.Configuration.from_runnable_config(config)

    # 从配置中获取用户 ID
    user_id = configurable.user_id

    # 从存储中检索记忆
    namespace = ("memories", user_id)
    memories = store.search(namespace)

    # 为系统提示格式化记忆
    info = "\n".join(f"- {mem.value['content']}" for mem in memories)
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=info)

    # 使用记忆和聊天历史进行回答
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """反思聊天历史，并将记忆保存到存储中。"""

    # 获取配置
    configurable = configuration.Configuration.from_runnable_config(config)

    # 从配置中获取用户 ID
    user_id = configurable.user_id

    # 定义记忆的命名空间
    namespace = ("memories", user_id)

    # 检索最近的记忆作为上下文
    existing_items = store.search(namespace)

    # 为 Trustcall 提取器格式化现有记忆
    tool_name = "Memory"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None
                        )

    # 合并聊天历史和指令
    updated_messages=list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"]))

    # 调用提取器
    result = trustcall_extractor.invoke({"messages": updated_messages,
                                        "existing": existing_memories})

    # 将 Trustcall 产生的记忆保存到存储中
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"),
            )

# 定义图
builder = StateGraph(MessagesState,config_schema=configuration.Configuration)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)
graph = builder.compile()