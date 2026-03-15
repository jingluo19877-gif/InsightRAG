import chainlit as cl
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from dotenv import load_dotenv
import os
import logging

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取智谱 AI 的 API Key
api_key = os.environ.get("ZHIPUAI_API_KEY")

if not api_key:
    raise ValueError("ZHIPUAI_API_KEY is not set in the .env file.")

# 初始化智谱 AI 聊天模型
llm = ChatZhipuAI(
    model="GLM-4-Plus",
    temperature=0.5,
    zhipuai_api_key=api_key
)

# 初始化智谱 AI 嵌入模型
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
    zhipuai_api_key=api_key
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 定义 Chainlit 应用
@cl.on_chat_start
async def start():
    await cl.Message(content="欢迎使用 RAG 问答系统！请上传一个 txt 文件并开始提问。").send()


@cl.on_message
async def handle_message(message: cl.Message):
    # 检查是否上传了文件
    if not cl.user_session.get("vector_store"):
        # 检查是否有文件上传
        if message.elements:
            for element in message.elements:
                if element.type == "file":
                    file_path = element.path
                    try:
                        # 加载上传的 txt 文件
                        loader = TextLoader(file_path, encoding="utf-8")
                        documents = loader.load()

                        # 将文档拆分为小块
                        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
                        texts = text_splitter.split_documents(documents)

                        logger.info(f"文件已上传并开始处理。分块数量：{len(texts)}")

                        # 分批处理文本块，避免单次请求数据量过大
                        batch_size = 5  # 每批处理 5 个文本块
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i + batch_size]
                            logger.info(f"处理批次：{i // batch_size + 1}，文本块数量：{len(batch_texts)}")

                            # 将文本转换为向量并存储到 FAISS 向量数据库
                            vector_store = FAISS.from_documents(batch_texts, embeddings)
                            if i == 0:
                                cl.user_session.set("vector_store", vector_store)
                            else:
                                # 合并向量数据库
                                cl.user_session.get("vector_store").merge_from(vector_store)

                        await cl.Message(content=f"文件已上传并处理完成，可以开始提问了！").send()
                        return
                    except Exception as e:
                        logger.error(f"文件处理出错: {e}")
                        await cl.Message(content=f"文件处理出错: {e}").send()
                        return
        await cl.Message(content="请先上传一个 txt 文件。").send()
        return

    # 获取向量数据库
    vector_store = cl.user_session.get("vector_store")

    # 初始化 RetrievalQA 链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,  # 使用智谱 AI 的 GLM-4-Plus 模型
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
    )

    # 运行问答链
    response = qa_chain.run(message.content)

    # 返回答案
    await cl.Message(content=response).send()


# 运行 Chainlit 应用
if __name__ == "__main__":
    cl.run()
