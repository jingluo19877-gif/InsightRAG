import json
import os
import logging

import pymongo
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
WORKING_DIR = os.path.join(ROOT_DIR, "myKG_Test")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# mongo
os.environ["MONGO_URI"] = "mongodb://localhost:27017"
os.environ["MONGO_DATABASE"] = "RAG_Test"

# neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "0000"

# milvusmilvus
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "root"
os.environ["MILVUS_DB_NAME"] = "rag_test"


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


# 执行查询
try:
    print("Starting query...")
    # Perform naive search
    print("naive")
    print(
        rag.query("从评论数据来看，人们如何看待调休？",
                  param=QueryParam(mode="naive"))
    )
    print("Query completed.")
except Exception as e:
    print(f"An unexpected error occurred during query: {e}")