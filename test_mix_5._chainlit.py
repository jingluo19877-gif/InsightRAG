import os
import chainlit as cl
from dotenv import load_dotenv
import asyncio
from rag import RAG, QueryParam
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
rag = RAG(
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
    """增强版消息处理方案（含结果写入和上下文展示）"""
    try:
        # 清理历史文件
        files_to_clear = [
            os.path.join(EXPERIMENT_DIR, "query_result.txt"),
            os.path.join(EXPERIMENT_DIR, "mix_mode_context.txt"),
            os.path.join(EXPERIMENT_DIR, "globally_reranked_context.txt"),
            os.path.join(EXPERIMENT_DIR, "processed_context.txt")
        ]
        clear_files(files_to_clear)

        # 初始化进度消息
        progress_msg = cl.Message(content="🔄 正在解析您的请求...")
        await progress_msg.send()

        # 执行RAG查询
        response = await asyncio.to_thread(
            rag.query,
            message.content,
            param=QueryParam(mode="mix")
        )

        await progress_msg.stream_token("✅ 已获取知识库数据\n\n💡 正在生成回答...")

        # 处理响应结果
        processed_response = await process_response(response)

        # 写入处理结果
        try:
            with open(os.path.join(EXPERIMENT_DIR, "query_result.txt"), "w", encoding="utf-8") as f:
                f.write(processed_response)
        except Exception as e:
            print(f"写入文件失败: {str(e)}")

        # 流式传输最终响应
        await progress_msg.stream_token("\n\n📝 完整回复：\n")
        for chunk in [processed_response[i:i + 200] for i in range(0, len(processed_response), 200)]:
            await progress_msg.stream_token(chunk)
            await asyncio.sleep(0.05)

        # 更新进度消息
        await progress_msg.update()

        # 新增：自然格式上下文展示
        context_files = [
            #("mix_mode_context.txt", "【混合模式上下文】"),
            ("globally_reranked_context.txt", "【全局重排上下文】"),
            ("processed_context.txt", "【处理后上下文】")
        ]

        # 发送上下文分割提示
        separator = "\n" + "━" * 40  # 使用纯文本分隔线
        await cl.Message(content=f"{separator}\n📚 本次查询上下文溯源：").send()

        for filename, title in context_files:
            try:
                file_path = os.path.join(EXPERIMENT_DIR, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 自然格式消息构建
                msg_content = (
                    f"{title}\n"
                    f"{'-' * 35}\n"  # 标题下划线
                    f"{content[:5000]}"  # 限制内容长度
                    f"\n{'▔' * 40}"  # 底部装饰线
                )

                # 发送独立消息块
                await cl.Message(content=msg_content).send()

                # 保持消息间隔
                await asyncio.sleep(0.3)

            except FileNotFoundError:
                await cl.Message(content=f"⚠️ {title[2:-2]}文件尚未生成").send()
            except Exception as e:
                await cl.Message(content=f"❌ 读取{title[2:-2]}内容失败: {str(e)}").send()

    except Exception as e:
        error_msg = f"❌ 处理异常: {str(e)}"
        await cl.Message(content=error_msg).send()

if __name__ == "__main__":
    cl.run()
