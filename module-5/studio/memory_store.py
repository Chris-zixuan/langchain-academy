from langchain_core.messages import SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore
import configuration

# 初始化 LLM
model = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 聊天机器人指令
MODEL_SYSTEM_MESSAGE = """You are a helpful assistant with memory that provides information about the user.
If you have memory for this user, use it to personalize your responses.
Here is the memory (it may be empty): {memory}"""

# 从聊天历史和现有记忆中创建新记忆
CREATE_MEMORY_INSTRUCTION = """"You are collecting information about the user to personalize your responses.

CURRENT USER INFORMATION:
{memory}

INSTRUCTIONS:
1. Review the chat history below carefully
2. Identify new information about the user, such as:
   - Personal details (name, location)
   - Preferences (likes, dislikes)
   - Interests and hobbies
   - Past experiences
   - Goals or future plans
3. Merge any new information with existing memory
4. Format the memory as a clear, bulleted list
5. If new information conflicts with existing memory, keep the most recent version

Remember: Only include factual information directly stated by the user. Do not make assumptions or inferences.

Based on the chat history below, please update the user information:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):

    """从存储中加载记忆，并用其个性化聊天机器人的回答。"""

    # 获取配置
    configurable = configuration.Configuration.from_runnable_config(config)

    # 从配置中获取用户 ID
    user_id = configurable.user_id

    # 从存储中检索记忆
    namespace = ("memory", user_id)
    key = "user_memory"
    existing_memory = store.get(namespace, key)

    # 提取记忆
    if existing_memory:
        # 值是一个包含 memory 键的字典
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # 在系统提示中格式化记忆
    system_msg = MODEL_SYSTEM_MESSAGE.format(memory=existing_memory_content)

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

    # 提取记忆
    if existing_memory:
        # 值是一个包含 memory 键的字典
        existing_memory_content = existing_memory.value.get('memory')
    else:
        existing_memory_content = "No existing memory found."

    # 在系统提示中格式化记忆
    system_msg = CREATE_MEMORY_INSTRUCTION.format(memory=existing_memory_content)
    new_memory = model.invoke([SystemMessage(content=system_msg)]+state['messages'])

    # 覆盖存储中的现有记忆
    key = "user_memory"
    store.put(namespace, key, {"memory": new_memory.content})

# 定义图
builder = StateGraph(MessagesState,config_schema=configuration.Configuration)
builder.add_node("call_model", call_model)
builder.add_node("write_memory", write_memory)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", "write_memory")
builder.add_edge("write_memory", END)
graph = builder.compile()