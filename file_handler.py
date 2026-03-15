import chainlit as cl
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import logging


async def process_uploaded_file(file_path, embeddings, logger):
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
                final_vector_store = vector_store
            else:
                # 合并向量数据库
                final_vector_store.merge_from(vector_store)

        return final_vector_store
    except Exception as e:
        logger.error(f"文件处理出错: {e}")
        return None
