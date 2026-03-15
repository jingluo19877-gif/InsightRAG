import os
import mysql.connector
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatZhipuAI
from pydantic import BaseModel, Field
from typing import List
from langchain.output_parsers import PydanticOutputParser
import time


# --- 定义数据结构 ---
class QuestionList(BaseModel):
    questions: List[str] = Field(description="生成的观点问题列表", min_items=2, max_items=2)


# --- 配置输出解析器 ---
output_parser = PydanticOutputParser(pydantic_object=QuestionList)

# --- 构建提示模板 ---
template = """
你是一个专业的问题生成专家，擅长根据用户评论内容生成高质量的观点型问题。请根据以下用户评论组合生成两个观点型问题，严格遵循以下要求：

评论内容：
{comments}

生成要求：
1. 仅生成问题，不要包含答案，使用中文。
2. 问题必须从评论组合中找出共性事件进行询问，紧密围绕评论数据中的共同主题、情感或现象展开，确保覆盖评论里的主要话题和用户情绪。
3. 使用中文口语化表达，问题以“？”结尾。
4. 问题应聚焦于用户对某件事的看法、感受、态度或观点。
5. 设计成开放式问题，避免简单的是非题，也不要在问题中给出预设的选项或暗示特定的观点方向，鼓励回答者从多个角度表达自己的观点，可使用“评论者如何看待……”“评论者对…… 持有怎样的看法？”“评论者在……方面有哪些不同的观点？”“评论者在……问题上持有哪些观点？”等表述。
6. 问题的措辞要中立，不使用引导性语言或暗示某种“正确答案”，以收集多元化和真实的意见。
7. 问题应当清晰明确，避免使用过于专业或复杂的语言，确保所有目标用户都能理解问题意图。
8. 问题要能激发回答者对问题背景和相关因素的深度思考，可要求回答者提供理由或案例支持自己的观点。
9. 避免生成笼统或无意义的问题
10. 问题中不要出现“你”“我”等字眼
11. 问题必须是一个完整的问句，问句中间不能包含中文逗号，不能包含补充说明或引导性内容（例如“是……还是……”）。
12. 格式示例：
   - 评论者对调休安排有什么看法？
   - 评论者如何看待高速堵车现象？

请严格按以下格式输出：
{format_instructions}
"""

prompt = PromptTemplate(
    input_variables=["comments"],
    template=template,
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# 加载.env 文件中的环境变量
load_dotenv()

# --- 初始化智谱 AI 模型 ---
api_key = os.environ.get("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("ZHIPUAI_API_KEY is not set in the.env file.")

llm = ChatZhipuAI(
    model="GLM-4-Flash",
    temperature=0.5,
    zhipuai_api_key=api_key
)
chain = LLMChain(llm=llm, prompt=prompt, output_parser=output_parser)


# --- 连接 MariaDB 数据库 ---
def connect_to_mariadb():
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
        return connection, cursor
    except Exception as e:
        print(f"Failed to connect to MariaDB: {e}")
        raise


# --- 从数据库加载指定范围的评论数据 ---
def load_comments_from_db(cursor, offset, limit):
    try:
        query = "SELECT contents FROM reviews LIMIT %s OFFSET %s"  # 表名 reviews，字段名 contents
        cursor.execute(query, (limit, offset))
        comments = [row[0] for row in cursor.fetchall()]
        print(f"Loaded {len(comments)} comments from database.")
        return comments
    except Exception as e:
        print(f"Failed to load comments: {e}")
        return []


# --- 将评论分组，每 10 条为一组 ---
def group_comments(comments, group_size=10):
    return [comments[i:i + group_size] for i in range(0, len(comments), group_size)]


# --- 问题生成处理 ---
def generate_questions(comment_groups):
    results = []
    error_log = []

    for idx, group in enumerate(comment_groups):
        try:
            # 将 10 条评论组合为一个字符串
            combined_comments = "\n".join(group)

            # 调用模型生成问题
            response = chain.run(comments=combined_comments)

            # 解析结果
            if isinstance(response, QuestionList):
                for q in response.questions:
                    if validate_question(q):
                        results.append({
                            "source_comment": combined_comments[:500],  # 截取部分原文
                            "question": q.strip()
                        })

                # 进度显示
                print(f"已处理第 {idx + 1} 组评论，生成 {len(response.questions)} 个问题")

                # API速率限制
                time.sleep(1.2)  # 根据智谱 API 速率限制调整

        except Exception as e:
            error_log.append({
                "comment_group": group,
                "error": str(e)
            })

    return results, error_log


def validate_question(q):
    """基础问题验证"""
    return (
            len(q) > 6 and
            q.endswith("？") and
            "?" not in q and  # 排除英文标点
            q.count("？") == 1 and  # 只能有一个问号
            not any(word in q for word in ["答案", "回答", "建议", "如何解决", "怎么办", "为什么"]) and
            "," not in q and
            "，" not in q  # 排除逗号
    )


# --- 保存结果到数据库 ---
def save_results_to_db(cursor, connection, results):
    try:
        insert_query = "INSERT INTO generated_questions (source_comment, question) VALUES (%s, %s)"
        cursor.executemany(insert_query, [(q["source_comment"], q["question"]) for q in results])
        connection.commit()
        print(f"成功保存 {len(results)} 个问题到数据库。")
    except Exception as e:
        print(f"保存结果失败：{e}")


# --- 主程序 ---
def main():
    # 连接数据库
    connection, cursor = connect_to_mariadb()

    total_records = 10407
    batch_size = 60
    offset = 0
    total_batches = (total_records + batch_size - 1) // batch_size

    while offset < total_records:
        current_batch = offset // batch_size + 1
        print(f"\n正在处理第 {current_batch} 个批次，共 {total_batches} 个批次...")

        # 加载当前批次的评论数据
        comments = load_comments_from_db(cursor, offset, batch_size)

        # 将评论分组，每 10 条为一组
        comment_groups = group_comments(comments, group_size=10)

        # 生成问题
        questions, errors = generate_questions(comment_groups)

        # 保存结果到数据库
        if questions:
            save_results_to_db(cursor, connection, questions)

        # 输出错误日志
        if errors:
            print("\n错误日志：")
            for error in errors:
                print(f"评论组：{error['comment_group']}")
                print(f"错误：{error['error']}")
                print()

        offset += batch_size
        progress = min(offset, total_records) / total_records * 100
        print(f"总体进度: {progress:.2f}%")

    # 关闭数据库连接
    cursor.close()
    connection.close()


# --- 执行 ---
if __name__ == "__main__":
    main()
