from datetime import datetime
import json
import os
import logging
import chainlit as cl
from typing import Optional

import pymongo
from dotenv import load_dotenv
import asyncio

from rag import LightRAG, QueryParam
from rag.llm.zhipu import zhipu_embedding, zhipu_complete
from rag.utils import EmbeddingFunc

# 加载.env文件中的环境变量
load_dotenv()

# 初始化配置
api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("请填写ZHIPUAI API密钥")

# 工作目录设置
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG_Test_3")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# mongo
os.environ["MONGO_URI"] = "mongodb://localhost:27017"
os.environ["MONGO_DATABASE"] = "RAG_Test_3"

# neo4j
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "0000"

# milvusmilvus
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "root"
os.environ["MILVUS_DB_NAME"] = "rag_test_3"

# 实验文件夹路径
EXPERIMENT_DIR = os.path.join(ROOT_DIR, "实验")
if not os.path.exists(EXPERIMENT_DIR):
    os.makedirs(EXPERIMENT_DIR)

# 初始化RAG系统
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="GLM-4-Flash",
    llm_model_max_async=4,
    llm_model_max_token_size=16384,
    embedding_func=EmbeddingFunc(
        embedding_dim=2048,
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
    kv_storage="MongoKVStorage",
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorge",
)



def clear_files(file_paths):
    """清空指定文件内容"""
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('')
                print(f"已清空文件: {file_path}")
            except Exception as e:
                print(f"清空文件 {file_path} 时出错: {e}")


async def process_response(response) -> str:
    """处理RAG响应，确保返回字符串"""
    if isinstance(response, str):
        return response
    try:
        return str(response)
    except Exception as e:
        print(f"转换响应为字符串时出错: {e}")
        return "无法生成回复内容"


@cl.on_message
async def main(message: cl.Message):
    """基础版消息处理方案（含结果写入）"""
    try:
        files_to_clear = [
            os.path.join(EXPERIMENT_DIR, "query_result.txt"),
            os.path.join(EXPERIMENT_DIR, "mix_mode_context.txt"),
            os.path.join(EXPERIMENT_DIR, "globally_reranked_context.txt"),
            os.path.join(EXPERIMENT_DIR, "processed_context.txt")
        ]
        clear_files(files_to_clear)

        progress_msg = cl.Message(content="🔄 正在解析您的请求...")
        await progress_msg.send()

        response = await asyncio.to_thread(
            rag.query,
            message.content,
            param=QueryParam(mode="mix")
        )

        await progress_msg.stream_token("✅ 已获取知识库数据\n\n💡 正在生成回答...")

        processed_response = await process_response(response)

        # ============ 新增的核心写入部分 ============
        try:
            with open(os.path.join(EXPERIMENT_DIR, "query_result.txt"), "w", encoding="utf-8") as f:
                f.write(processed_response)
        except Exception as e:
            print(f"写入文件失败: {str(e)}")
        # =========================================

        await progress_msg.stream_token("\n\n📝 完整回复：\n")
        for chunk in [processed_response[i:i+200] for i in range(0, len(processed_response), 200)]:
            await progress_msg.stream_token(chunk)
            await asyncio.sleep(0.05)

        await progress_msg.update()

    except Exception as e:
        error_msg = f"❌ 处理异常: {str(e)}"
        await cl.Message(content=error_msg).send()


if __name__ == "__main__":
    cl.run()