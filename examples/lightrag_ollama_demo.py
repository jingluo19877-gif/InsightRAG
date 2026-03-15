import asyncio
import os
import inspect
import logging
from rag import LightRAG, QueryParam
from rag.llm.ollama import ollama_model_complete, ollama_embed
from rag.utils import EmbeddingFunc

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:14b",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:6006", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:6006"
        ),
    ),
)

#with open("E:/LightRAG_Test/LightRAG/test1.txt", "r", encoding="utf-8") as f:
    #rag.insert(f.read())

#with open("E:/LightRAG_Test/LightRAG/cleaned_reviews_1.txt", "r", encoding="utf-8") as f:
    #for line in f:
        #rag.insert(line.strip())  # 避免插入空行

# 指定文件路径
files_to_process = [
    "E:/LightRAG_Test/LightRAG/test_3.txt",
]

# 遍历指定的文件并插入内容
for file_path in files_to_process:
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            rag.insert(line.strip())  # 避免插入空行

# Perform naive search
print(
    rag.query("从评论数据来看，关于五一高速免费，大家有哪些不同的观点？请你尽量客观全面的做出回答，并列出参考文献", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("从评论数据来看，关于五一高速免费，大家有哪些不同的观点？请你尽量客观全面的做出回答，并列出参考文献", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("从评论数据来看，关于五一高速免费，大家有哪些不同的观点？请你尽量客观全面的做出回答，并列出参考文献", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("从评论数据来看，关于五一高速免费，大家有哪些不同的观点？请你尽量客观全面的做出回答，并列出参考文献", param=QueryParam(mode="hybrid"))
)


# stream response
resp = rag.query(
    "从评论数据来看，关于五一高速免费，大家有哪些不同的观点？请你尽量客观全面的做出回答，并列出参考文献",
    param=QueryParam(mode="hybrid", stream=True),
)


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)
