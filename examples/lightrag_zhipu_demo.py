import os
import logging
from rag import LightRAG, QueryParam
from rag.llm.zhipu import zhipu_complete, zhipu_embedding
from rag.utils import EmbeddingFunc
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()
WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("请填写密钥")


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="GLM-4-Flash",  # Using the most cost/performance balance model, but you can change it here.
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=2048,  # Zhipu embedding-3 dimension
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
)

#with open("E:/LightRAG_Test/LightRAG/test1.txt", "r", encoding="utf-8") as f:
    #rag.insert(f.read())

with open("E:/LightRAG_Test/LightRAG/test_3.txt", "r", encoding="utf-8") as f:
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
