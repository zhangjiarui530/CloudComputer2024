from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
api_key = "cd493791757b49fdbc0eeba1367608f0.xGgQTGgq7e7jIORn"
model=ChatOpenAI(
temperature=0.95,
model="glm-4-flash",
openai_api_key=api_key,
openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

# 粤语到普通话翻译
messages = [
SystemMessage(content="首先判断我说的是粤语还是普通话，是粤语则翻译成普通话，是普通话则翻译成粤语，只输出翻译内容"),
HumanMessage(content="我哋今晚去边度食好？"),
]
response = model.invoke(messages)

# 仅输出翻译后的文本
translated_text = response.content  # 从响应中提取文本字段
print(translated_text)  # 打印翻译后的结果
