import os
import mysql.connector
import logging

from dotenv import load_dotenv
from rag import RAG, QueryParam
from rag.llm.ollama import ollama_model_complete, ollama_embed
from rag.llm.zhipu import zhipu_embedding, zhipu_complete
from rag.utils import EmbeddingFunc

# 加载 .env 文件中的环境变量
load_dotenv()

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("请填写密钥")


# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG_Test_3")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")



# mongo
os.environ["MONGO_URI"] = "mongodb://localhost:27017"
os.environ["MONGO_DATABASE"] = "RAG_Test_3"

# neo4j
#BATCH_SIZE_NODES = 500
#BATCH_SIZE_EDGES = 100
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "0000"

# milvusmilvus
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "root"
os.environ["MILVUS_DB_NAME"] = "rag_test_3"


rag = RAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="GLM-4-Flash",  # Using the most cost/performance balance model, but you can change it here.
    llm_model_max_async=4,
    llm_model_max_token_size=16384,
    embedding_func=EmbeddingFunc(
    embedding_dim=2048,  # Zhipu embedding-3 dimension
    max_token_size=8192,
    func=lambda texts: zhipu_embedding(texts),
    ),
    kv_storage="MongoKVStorage",
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorge",
)

#file = "./book.txt"
#with open(file, "r") as f:
    #rag.insert(f.read())

# 连接 MariaDB 数据库
try:
    print("Starting to connect to MariaDB...")
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="0000",
        database="rag_deal",
        charset='utf8mb4',
        collation='utf8mb4_unicode_ci'
    )
    cursor = connection.cursor()
    print("Connected to MariaDB successfully.")

    offset = 500
    batch_size = 30
    total_processed = 500
    while True:
        # 查询 rag_deal 表的 reviews 字段的 contents，每次取 10 条
        query = f"SELECT contents FROM rag_deal.reviews LIMIT {batch_size} OFFSET {offset}"
        cursor.execute(query)
        results = cursor.fetchall()

        if not results:
            break

        print(f"Processing batch of {len(results)} records starting at offset {offset}")
        batch_contents = [row[0] for row in results]
        try:
            # 假设 LightRAG 支持批量插入，这里可以调用相应的批量插入方法
            # rag.insert_batch(batch_contents)
            # 如果不支持批量插入，也可以循环处理
            for content in batch_contents:
                rag.insert(content)
                total_processed += 1
                print(f"处理进度: {total_processed}")
        except Exception as e:
            print(f"Error inserting batch: {e}")

        offset += batch_size

    # 关闭数据库连接
    cursor.close()
    connection.close()
    print("Closed MariaDB connection.")

    # 执行查询
    print("Starting query...")
    #result = rag.query("请你从主要围绕假期调休制度的概念、实施情况以及相关政策和事件入手，分析抖音平台的广大用户对假期调休制度的看法是怎样的?需要你给出参考文献", param=QueryParam(mode="hybrid"))
    # Perform naive search
    #print("naive")
    #print(
        #rag.query("从评论数据来看，调休如何影响假期？",
                  #param=QueryParam(mode="naive"))
    #)

    # Perform local search
    #print("local")
    #print(
        #rag.query("从评论数据来看，调休如何影响假期？",
                  #param=QueryParam(mode="local"))
    #)

    # Perform global search
    #print("global")
    #print(
        #rag.query("从评论数据来看，调休如何影响假期？",
                  #param=QueryParam(mode="global"))
    #)

    # Perform hybrid search
    print("mix")
    print(
        rag.query("从评论数据来看，用户如何看待高速堵车？",
                  param=QueryParam(mode="mix"))
    )
    print("Query completed.")
    #print(result)

except mysql.connector.Error as err:
    print(f"Error connecting to MariaDB: {err}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))
