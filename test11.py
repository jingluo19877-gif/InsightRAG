import os
from zhipuai import ZhipuAI
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
api_key = os.environ.get("ZHIPUAI_API_KEY")

client = ZhipuAI(api_key=api_key)

# 设置提示词
prompt = "你是一个知识渊博的助手，能准确清晰地回答各种问题。"

# 构建初始消息列表，包含提示词
messages = [{"role": "user", "content": prompt}]

while True:
    # 获取用户输入的问题
    question = input("请输入你的问题（输入 '退出' 结束对话）：")
    if question == "退出":
        break
    # 将用户的问题添加到消息列表中
    messages.append({"role": "user", "content": question})
    try:
        # 调用模型获取回复
        response = client.chat.completions.create(
            model="GLM-4-Flash",
            messages=messages
        )
        # 提取模型的回复内容
        answer = response.choices[0].message.content
        print("模型回复：", answer)
        # 将模型的回复添加到消息列表中，以便后续对话有上下文
        messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        print(f"请求发生错误: {e}")
