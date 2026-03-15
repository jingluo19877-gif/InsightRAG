import mysql.connector
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_deepseek import ChatDeepSeek
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
import asyncio
import time
import gc
from langchain_core.exceptions import OutputParserException

# 加载.env文件中的环境变量
load_dotenv()

# 获取智谱 AI 的 API Key
api_key = os.environ.get("ZHIPUAI_API_KEY")

if not api_key:
    raise ValueError("ZHIPUAI_API_KEY is not set in the .env file.")

# #初始化智谱 AI 聊天模型
llm = ChatZhipuAI(
    model="GLM-4-Plus",
    temperature=0.5,
    zhipuai_api_key=api_key
)


# 封装模型
evaluator_llm = LangchainLLMWrapper(llm)

# 连接MariaDB
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000",
    database="rag_deal",
    charset='utf8mb4',
    collation='utf8mb4_unicode_ci'
)

cursor = connection.cursor()

# 获取数据总行数
count_query = "SELECT COUNT(*) FROM naive_question_responses WHERE id NOT IN (SELECT question_id FROM Faithfulness_naive)"
cursor.execute(count_query)
total_rows = cursor.fetchone()[0]

# 读取数据并分批次处理
batch_size = 10
offset = 0
processed_rows = 0

while True:
    # 读取一批未处理的数据
    query = f"""
    SELECT id, question, response, relevant_context 
    FROM naive_question_responses 
    WHERE id NOT IN (SELECT question_id FROM Faithfulness_naive)
    LIMIT {batch_size} OFFSET {offset}
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    if not rows:
        break  # 如果没有更多数据，退出循环

    # 处理每批数据
    for row in rows:
        question_id, question, response, relevant_context = row

        # 定义测试数据
        test_data = {
            "user_input": question,
            "response": response,
            "retrieved_contexts": [relevant_context]
        }

        # 创建Context Precision without reference评估指标
        metric = Faithfulness(llm=evaluator_llm)

        # 将测试数据转换为SingleTurnSample对象
        test_sample = SingleTurnSample(**test_data)

        # 异步运行评估
        try:
            loop = asyncio.get_event_loop()
            res = loop.run_until_complete(metric.single_turn_ascore(test_sample))
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            res = loop.run_until_complete(metric.single_turn_ascore(test_sample))
        except OutputParserException as e:
            print(f"Output parsing error for question ID {question_id}: {e}")
            res = question_id  # 将问题的id值作为得分

        # 将结果写入数据库
        insert_query = """
        INSERT INTO Faithfulness_naive (question_id, score)
        VALUES (%s, %s)
        """
        cursor.execute(insert_query, (question_id, res))
        connection.commit()

        # 显示处理进度
        processed_rows += 1
        percentage = (processed_rows / total_rows) * 100
        print(f"Processed question ID {question_id} with score {res}. Processed {percentage:.2f}% of data.")

        # 释放不再使用的变量
        del test_data, test_sample
        # 手动触发垃圾回收
        gc.collect()

    # 更新offset
    offset += batch_size

    # 休息15秒
    time.sleep(15)

# 关闭数据库连接
cursor.close()
connection.close()
