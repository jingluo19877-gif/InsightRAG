import mysql.connector
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
import asyncio
import time
import gc
from langchain_core.exceptions import OutputParserException
from ragas.exceptions import RagasOutputParserException
import json
from openai import OpenAI

# 加载.env文件中的环境变量
load_dotenv()

# 获取OpenAI的API密钥
#api_key = os.environ.get("OPENAI_API_KEY")
#base_url = os.environ.get("OPENAI_BASE_URL", "https://chatapi.littlewheat.com/v1")

#if not api_key:
    #raise ValueError("OPENAI_API_KEY is not set in the .env file.")

# 获取智谱 AI 的 API Key
api_key = os.environ.get("ZHIPUAI_API_KEY")

# 初始化OpenAI模型
llm = ChatZhipuAI(
        model="GLM-4-Flash",
        temperature=0.3,
        max_tokens=16384,
        zhipuai_api_key=api_key
    )

# 使用 LangchainLLMWrapper 包装 LLM
v_llm = LangchainLLMWrapper(llm)

# 创建 Faithfulness 评估指标并指定使用的 LLM
metric = Faithfulness()
metric.llm = v_llm

# 连接 MariaDB
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000",
    database="rag_deal",
    charset='utf8mb4',
    collation='utf8mb4_unicode_ci'
)

cursor = connection.cursor()
print(cursor)

# 获取所有已处理的 question_id
cursor.execute("SELECT question_id FROM Faithfulness_deepseek")
processed_question_ids = set([row[0] for row in cursor.fetchall()])

# 获取数据总行数
if processed_question_ids:
    count_query = "SELECT COUNT(*) FROM hybrid_question_responses_deepseek WHERE id NOT IN (%s)" % (','.join(['%s'] * len(processed_question_ids)))
    cursor.execute(count_query, tuple(processed_question_ids))
else:
    count_query = "SELECT COUNT(*) FROM hybrid_question_responses_deepseek"
    cursor.execute(count_query)

total_rows = cursor.fetchone()[0]

# 读取数据并分批次处理
batch_size = 100
offset = 0
processed_rows = 0

async def process_row(row):
    question_id, question, response, relevant_context = row
    print(f"Processing question ID {question_id}")

    # 定义测试数据
    test_data = {
        "user_input": question,
        "response": response,
        "retrieved_contexts": [relevant_context]
    }

    # 将测试数据转换为 SingleTurnSample 对象
    test_sample = SingleTurnSample(**test_data)

    # 异步运行评估
    try:
        res = await metric.single_turn_ascore(test_sample)
        print(res)
    except Exception as e:
        print(f"Error processing question ID {question_id}: {e}")
        res = 10.0

    # 检查 res 是否为浮点数
    if isinstance(res, float):
        score = res
        print(score)
    else:
        score = 10.0
        print(score)

    # 将结果写入数据库
    insert_query = """
    INSERT INTO Faithfulness_deepseek (question_id, score)
    VALUES (%s, %s)
    """
    cursor.execute(insert_query, (question_id, score))
    connection.commit()

    return question_id, score

async def process_batch(rows):
    semaphore = asyncio.Semaphore(50)  # 限制并发数量为 2
    async def sem_process_row(row):
        async with semaphore:
            return await process_row(row)
    tasks = [sem_process_row(row) for row in rows]
    results = await asyncio.gather(*tasks)
    return results

while True:
    if processed_question_ids:
        # 当 processed_question_ids 不为空时，使用 NOT IN 子句
        print(processed_question_ids)
        query = f"""
        SELECT id, question, response, relevant_context 
        FROM hybrid_question_responses_deepseek 
        WHERE id NOT IN ({','.join(['%s'] * len(processed_question_ids))})
        LIMIT {batch_size} OFFSET {offset}
        """
        print(query)
        cursor.execute(query, tuple(processed_question_ids))
    else:
        # 当 processed_question_ids 为空时，不使用 NOT IN 子句
        print(processed_question_ids)
        query = f"""
        SELECT id, question, response, relevant_context 
        FROM hybrid_question_responses_deepseek 
        LIMIT {batch_size} OFFSET {offset}
        """
        cursor.execute(query)

    rows = cursor.fetchall()
    print(f"Fetched {len(rows)} rows.")

    if not rows:
        print("All data has been processed. Exiting...")
        break  # 如果没有更多数据，退出循环

    # 并发处理每批数据
    results = asyncio.run(process_batch(rows))

    # 显示处理进度
    for question_id, score in results:
        processed_rows += 1
        percentage = (processed_rows / total_rows) * 100
        print(f"Processed question ID {question_id} with score {score}. Processed {percentage:.2f}% of data.")

    # 更新 offset
    offset += batch_size

    # 休息30秒
    time.sleep(30)

# 关闭数据库连接
cursor.close()
connection.close()
