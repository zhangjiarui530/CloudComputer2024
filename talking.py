import os
from langchain_community.llms import Tongyi
from langchain_core.messages import HumanMessage,AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

os.environ["DASHSCOPE_API_KEY"] = "sk-9041beedef014f4cb57ff1ef1dc6cd33"
# 提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个有用的助手。尽你所能回答所有问题。",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 历史会话存储
store = {}

# 获取会话历史
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 使用 Tongyi LLM
model = Tongyi()

# k=10 则无法记得姓名是什么，k=20 则可以记得
def filter_messages(messages, k=10):
    return messages[-k:]

chain = (
    RunnablePassthrough.assign(messages=lambda x: filter_messages(x["messages"]))
    | prompt
    | model
)

# 历史消息
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)

config = {"configurable": {"session_id": "abc2"}}

user_input = input("输入：")

response = with_message_history.invoke(
    {"messages": [HumanMessage(content=f"{user_input}")]},
    config=config
)

print(response)