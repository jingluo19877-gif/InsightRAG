import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_deepseek import ChatDeepSeek
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import LLMContextPrecisionWithReference
from ragas.metrics import NonLLMContextPrecisionWithReference
from ragas.metrics import NonLLMContextRecall
import asyncio
from ragas import SingleTurnSample
from ragas.metrics import LLMContextRecall
from ragas.metrics import ContextEntityRecall


# 加载 .env 文件中的环境变量
load_dotenv()

# 获取 DeepSeek 的 API Key
deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    raise ValueError("DEEPSEEK_API_KEY is not set in the .env file.")

# 初始化 DeepSeek 聊天模型
llm = ChatDeepSeek(
    model="deepseek-reasoner",  # 替换为实际的 DeepSeek 模型名称
    temperature=0,
    api_key=deepseek_api_key
)

# 封装模型
evaluator_llm = LangchainLLMWrapper(llm)

# 读取 naive_mode_context.txt 文件内容
context_file_path = os.path.join(os.getcwd(), "实验", "naive_mode_context.txt")
try:
    with open(context_file_path, 'r', encoding='utf-8') as file:
        context_content = file.read()
except FileNotFoundError:
    raise FileNotFoundError(f"文件 {context_file_path} 未找到，请检查文件路径和文件名是否正确。")

# 读取 naive_mode_response.txt 文件内容
response_file_path = os.path.join(os.getcwd(), "实验", "naive_mode_response.txt")
try:
    with open(response_file_path, 'r', encoding='utf-8') as file:
        response_content = file.read()
except FileNotFoundError:
    raise FileNotFoundError(f"文件 {response_file_path} 未找到，请检查文件路径和文件名是否正确。")


reference_file_path = os.path.join(os.getcwd(), "实验", "reference.txt")
try:
    with open(response_file_path, 'r', encoding='utf-8') as file:
        reference_content = file.read()
except FileNotFoundError:
    raise FileNotFoundError(f"文件 {reference_file_path} 未找到，请检查文件路径和文件名是否正确。")

# 定义测试数据
test_data = {
    #"user_input": "从评论数据来看，人们如何看待调休？",
    #"response": response_content,
    "reference": reference_content,
    "retrieved_contexts": [context_content]
}

# 创建 Context Precision without reference 评估指标
metric = ContextEntityRecall(llm=evaluator_llm)

# 将测试数据转换为 SingleTurnSample 对象
test_sample = SingleTurnSample(**test_data)

# 异步运行评估
try:
    res = asyncio.run(metric.single_turn_ascore(test_sample))
    print(res)
except RuntimeError as e:
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(metric.single_turn_ascore(test_sample))
    print("无参考的上下文精度:")
    print(res)
