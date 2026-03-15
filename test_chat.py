from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os

# 加载.env文件中的环境变量
load_dotenv()

# 获取OpenAI的API密钥
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_BASE_URL", "https://chataiapi.com/v1")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")

# 初始化OpenAI模型
chat = ChatOpenAI(
    model="deepseek-r1",
    temperature=0.3,
    max_tokens=32768,
    api_key=api_key,
    base_url=base_url
)

try:
    # 构建消息列表
    messages = [
        SystemMessage(content="你是一个友好的助手，能回答简单问题。"),
        HumanMessage(content="你好，能正常交流吗？")
    ]

    # 调用模型获取响应
    response = chat.invoke(messages)

    # 打印响应内容
    print("API 调用成功，响应内容如下：")
    print(response.content)
except Exception as e:
    print(f"API 调用失败，错误信息：{e}")
