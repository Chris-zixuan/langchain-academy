from pydantic import BaseModel, Field

from trustcall import create_extractor

from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
import configuration

# 初始化 LLM
model = ChatDeepSeek(model="deepseek-chat", temperature=0)

# Schema
class UserProfile(BaseModel):
    """ 用户资料 """
    user_name: str = Field(description="The user's preferred name")
    user_location: str = Field(description="The user's location")
    interests: list = Field(description="A list of the user's interests")

# 创建提取器
trustcall_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile", # 强制使用 UserProfile 工具
)

# 聊天机器人指令
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# 提取指令
TRUSTCALL_INSTRUCTION = """Create or update the memory (JSON doc) to incorporate information from the following conversation:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """从存储中加载记忆，并用其个性化聊天机器人的回答。"""

    # 获取配置
    configurable = configuration.Configuration.from_runnable_config(config)

    # 从配置中获取用户 ID
    user_id = configurable.user_id

    # 从存储中检索记忆
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # 为系统提示格式化记忆
    if existing_memory and existing_memory.value:
        memory_dict = existing_memory.value
        formatted_memory = (
            f"Name: {memory_dict.get('user_name', 'Unknown')}\n"
            f"Location: {memory_dict.get('user_location', 'Unknown')}\n"
            f"Interests: {', '.join(memory_dict.get('interests', []))}"
        )
    else:
        formatted_memory = None

    # 在系统提示中格式化记忆
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=formatted_memory)

    # 使用记忆和聊天历史进行回答
    response = model.invoke([SystemMessage(content=system_msg)]+state["messages"])

    return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """反思聊天历史，并将记忆保存到存储中。"""

    # 获取配置
    configurable = configuration.Configuration.from_runnable_config(config)

    # 从配置中获取用户 ID
    user_id = configurable.user_id

    # 从存储中检索现有记忆
    namespace = ("memory", user_id)
    existing_memory = store.get(namespace, "user_memory")

    # 从列表中获取资料值，并将其转换为 JSON 文档
    existing_profile = {"UserProfile": existing_memory.value} if existing_memory else None

    # 调用提取器
    result = trustcall_extractor.invoke({"messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)]+state["messages"], "existing": existing_profile})

    # 将更新后的资料获取为 JSON 对象
    updated_profile = result["responses"][0].model_dump()

    # 保存更新后的资料
    key = "user_memory"
    store.put(namespace, key, updated_profile)

# 定义图
builder = StateGraph(MessagesState,config_schema=configuration.Configuration)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)
graph = builder.compile()