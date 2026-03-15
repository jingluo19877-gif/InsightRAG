import json
import os
import logging
import gc  # 导入垃圾回收模块
import time
from typing import List, Dict, Any
import mysql.connector
import pymongo
from dotenv import load_dotenv
from rag import LightRAG, QueryParam
from rag.llm.zhipu import zhipu_embedding, zhipu_complete
from rag.utils import EmbeddingFunc
from rag.operate import naive_query  # 导入 naive_query 方法
from rag.llm.openai import openai_complete_if_cache


# 设置工作目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG_Test")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# 配置 MongoDB
os.environ["MONGO_URI"] = "mongodb://localhost:27017"
os.environ["MONGO_DATABASE"] = "RAG_Test"

# 配置 Neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "0000"

# 配置 Milvus
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "root"
os.environ["MILVUS_DB_NAME"] = "rag_test"

DEEPSEEK_API_KEY = "sk-d67KcHv3pMbCCui7xLn0HLUCGvD6kZOwKOImtVeSA6JlyvrB"
MODEL = "deepseek-chat"

async def llm_model_func(
        prompt, system_prompt=None, history_messages=[],  **kwargs
) -> str:
    return await openai_complete_if_cache(
        MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=DEEPSEEK_API_KEY,
        base_url="https://chatapi.littlewheat.com/v1",
        **kwargs,
    )


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    #llm_model_name="GLM-4-Flash",  # Using the most cost/performance balance model, but you can change it here.
    #llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
    embedding_dim=2048,  # Zhipu embedding-3 dimension
    max_token_size=8192,
    func=lambda texts: zhipu_embedding(texts),
    ),
    kv_storage="MongoKVStorage",
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorge",
)

# 连接 MariaDB
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0000",
    database="rag_deal",
    charset='utf8mb4',
    collation='utf8mb4_unicode_ci'
)

# 从 generated_questions 表中读取问题
cursor = connection.cursor(dictionary=True)
batch_size = 10
offset = 0

while True:
    cursor.execute("""
        SELECT question FROM generated_questions
        LIMIT %s OFFSET %s
    """, (batch_size, offset))
    questions = cursor.fetchall()
    if not questions:
        break

    print(f"Processing batch {offset // batch_size + 1}...")

    record_ids = []
    for question_data in questions:
        question = question_data["question"]
        print(f"Processing question: {question}")

        try:
            # 调用 rag.query 获取回答
            response = rag.query(
                question,
                param=QueryParam(mode="hybrid")
            )

            # 将问题和答案写入数据库，并获取插入记录的 id
            insert_cursor = connection.cursor()
            insert_cursor.execute("""
                INSERT INTO hybrid_question_responses_deepseek (question, response)
                VALUES (%s, %s)
            """, (question, response))
            connection.commit()  # 确保提交事务
            record_id = insert_cursor.lastrowid  # 获取插入记录的 id
            record_ids.append(record_id)
            insert_cursor.close()

            print(f"Saved response for question: {question}, Record ID: {record_id}")

        except Exception as e:
            print(f"Error processing question: {question}, error: {e}")

    # 每批问题处理完成后，将 naive_mode_context.txt 中的参考上下文写入数据库
    experiment_folder = os.path.join(os.getcwd(), "实验")
    file_name = "hybrid_mode_context.txt"
    file_path = os.path.join(experiment_folder, file_name)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"File content: {content}")

        # 按 “参考上下文” 分割内容，提取每条参考上下文
        contexts = [ctx.strip() for ctx in content.split("参考上下文") if ctx.strip()]
        print(f"Extracted contexts: {contexts}")

        # 依次将参考上下文写入对应的记录
        for i, context in enumerate(contexts):
            if i < len(record_ids):
                try:
                    update_cursor = connection.cursor()
                    update_cursor.execute("""
                        UPDATE hybrid_question_responses_deepseek
                        SET relevant_context = %s
                        WHERE id = %s
                    """, (context, record_ids[i]))
                    connection.commit()  # 确保提交事务
                    print(f"Updated context for record ID {record_ids[i]}")
                    update_cursor.close()
                except mysql.connector.Error as err:
                    print(f"Error updating context for record ID {record_ids[i]}: {err}")

        # 清空 naive_mode_context.txt 文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('')

    # 释放不再使用的变量
    questions = None
    content = None
    contexts = None
    record_ids = None

    # 手动触发垃圾回收
    gc.collect()

    # 每批处理完成后休眠 15 秒
    logging.info("Batch processing completed. Sleeping for 15 seconds...")
    time.sleep(20)

    offset += batch_size

connection.close()
print("All questions processed.")