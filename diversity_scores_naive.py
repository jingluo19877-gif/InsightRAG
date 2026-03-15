import os
import getpass
import time
import asyncio
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import mysql.connector
from typing import List, Dict

# 加载环境变量
load_dotenv()

# 优化后的提示词模板
DIVERSITY_PROMPT_TEMPLATE = """
请依据以下明确的评分标准，结合相应的上下文信息，对给定回答的观点多样性进行评估。评分范围设定为 0.00 - 1.00，结果需保留两位小数。最终评分将基于五个关键维度：**观点广度（Coverage）、观点深度（Depth）、观点层次（Hierarchy）、语义关联性（Semantic Connectivity）、全局理解能力（Global Understanding）**。

# **重要说明**
1. **评估依据**：如果上下文信息中未明确某些信息，则不能在评估中使用。
2. **评估依据**：所有评分和推理必须基于上下文信息中的内容，不能引入外部知识或假设。


# **评估标准**

## 1. **广度（Coverage）**
- **评估要点**：此维度主要衡量回答中涉及的主题的丰富程度。重点关注回答是否能够广泛涵盖多个相关方面，如不同主题（例如经济、文化、用户体验等）
- **评分范围**：
    - **0.00 - 0.30**：回答仅涉及单一的主题内容极为局限。
    - **0.31 - 0.50**：回答涉及 2 - 3 个主题,有一定的拓展但仍较为有限。
    - **0.51 - 0.70**：回答涉及 4 - 5 个主题，覆盖范围有明显提升。
    - **0.71 - 0.90**：回答涉及 6 - 7 个主题，具有较为丰富的信息。
    - **0.91 - 1.00**：回答涉及 8 个及以上的主题，对各个方面都有涉及。

## 2. **深度（Depth）、层次（Hierarchy）与语义关联性（Semantic Connectivity）评估**
### 评估要点
- **深度（Depth）**：着重考察观点是否通过合理推理展开，是否提供可靠数据、具体案例或严密逻辑支撑，是否能挖掘隐含因果关系。
- **层次（Hierarchy）**：主要评估观点是否具备清晰逻辑结构，能否体现因果、比较等逻辑关系，重点在于是否能形成完整的“现象 - 原因 - 影响 ”逻辑链条。
- **语义关联性（Semantic Connectivity）**：此维度着重衡量回答能否有效挖掘并关联上下文中相关信息里的隐含语义，开展深层语义理解与信息整合工作。

### 评分范围
|维度|0.00 - 0.30|0.31 - 0.50|0.51 - 0.70|0.71 - 0.90|0.91 - 1.00|
| ---- | ---- | ---- | ---- | ---- | ---- |
|**深度（Depth）**|仅简单提及观点，无推理过程或支持性内容|有初步推理尝试，但缺乏必要细节或证据，说服力不足|有部分推理过程，提供少量示例或数据支持，仍有提升空间|推理较充分，提供具体数据或案例支撑观点，逻辑性较强|逻辑严密，推理深入，能充分利用上下文进行推理，展现较高专业水平|
|**层次（Hierarchy）**|观点无明显层次，缺乏基本逻辑组织，难以理解|观点有基本逻辑，但层次不清晰，各部分关系不明确|观点结构较清晰，能体现部分因果或比较关系，但逻辑链条可能不完整|观点层次清晰，有明确逻辑链条，各部分关系紧密|观点层次分明，能串联多个推理层次，形成完整严密逻辑体系|
|**语义关联性（Semantic Connectivity）**|仅依据上下文的表层信息阐述，完全未挖掘隐含语义，缺乏深度和对信息的有效关联|能察觉到少量隐含语义，但未能有效关联上下文进行整合，关联程度低且对隐含语义的利用不充分|可部分识别上下文中的隐含语义，尝试进行关联推理，但推理不够深入，信息整合存在明显不足|能够有效关联上下文，充分利用上下文挖掘丰富的隐含语义，体现复杂语义关系，关联性较高|深度挖掘上下文中的隐含语义，全面且精准地进行复杂语义推理，实现高度的信息整合，回答具有卓越的逻辑性和深度|

## 3. **全局理解能力（Global Understanding）**
- **评估要点**：此维度主要考察回答是否能够整合多源信息，从而提高 RAG（检索增强生成）处理复杂查询的能力。关注回答是否能够从多个角度进行综合分析，并提供多维度的见解。
- **评分范围**：
    - **0.00 - 0.30**：回答仅基于局部信息进行分析，缺乏整体的视角，对问题的理解较为片面。
    - **0.31 - 0.50**：回答涉及部分全局信息，但分析角度单一，没有全面考虑问题的各个方面。
    - **0.51 - 0.70**：回答结合了多个维度的信息，但信息整合度有限，各个部分之间的关联性不够强。
    - **0.71 - 0.90**：回答全面整合了多个数据源的信息，分析较为完整，能够从多个角度看待问题。
    - **0.91 - 1.00**：回答具备高度的全局视角，能够综合多维信息进行深入分析，提供全面、准确的见解。

# **评估方式**
1. **拆解核心观点**：仔细识别回答中的主要观点，并深入分析这些观点之间的语义关系，明确它们是如何相互关联和支撑的，同时考量与上下文信息的关联。
2. **分析五个维度**：严格按照上述详细的标准，对每个维度分别进行评分，确保评分的客观性和准确性。

# **上下文信息**
{context}

# **待评估回答**
{answer}

# **输出格式**
请严格按照以下 JSON 格式输出评估结果：
{{
    "coverage": "观点广度评分（0.00 - 1.00）",
    "depth": "观点深度评分（0.00 - 1.00）",
    "hierarchy": "观点层次评分（0.00 - 1.00）",
    "semantic_connectivity": "语义关联性评分（0.00 - 1.00）",
    "global_understanding": "全局理解能力评分（0.00 - 1.00）",
    "key_evidence": [
        {{
            "viewpoint": "观点 1 概述",
            "context": "观点涉及的具体情境或推理过程，包含与上下文信息的关联说明"
        }},
        {{
            "viewpoint": "观点 2 概述",
            "context": "..."
        }}
    ]
}}
"""

class DiversityEvaluator:
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser()

    async def evaluate(self, response: str, context: str) -> Dict:
        prompt = ChatPromptTemplate.from_template(DIVERSITY_PROMPT_TEMPLATE)
        chain = prompt | self.llm | self.parser
        try:
            # 打印输入数据
            print(f"Input - response: {response}")
            print(f"Input - context: {context}")

            # 获取模型输出
            result = await chain.ainvoke({"answer": response, "context": context})

            # 打印模型输出
            print(f"Model output: {result}")

            # 检查 result 是否为 None
            if result is None:
                print(f"模型未生成有效输出: response={response}, context={context}")
                return {
                    "score": 0.0,
                    "coverage": 0.0,
                    "depth": 0.0,
                    "hierarchy": 0.0,
                    "semantic_connectivity": 0.0,
                    "global_understanding": 0.0,
                    "key_evidence": []
                }

            # 检查 result 是否包含必要的字段
            required_fields = ["coverage", "depth", "hierarchy", "semantic_connectivity", "global_understanding"]
            if not all(field in result for field in required_fields):
                print(f"模型输出格式不正确: {result}")
                return {
                    "score": 0.0,
                    "coverage": 0.0,
                    "depth": 0.0,
                    "hierarchy": 0.0,
                    "semantic_connectivity": 0.0,
                    "global_understanding": 0.0,
                    "key_evidence": []
                }

            # 手动计算综合评分
            coverage = float(result["coverage"])
            depth = float(result["depth"])
            hierarchy = float(result["hierarchy"])
            semantic_connectivity = float(result["semantic_connectivity"])
            global_understanding = float(result["global_understanding"])

            # 权重分配
            score = (
                coverage * 0.20 +
                depth * 0.20 +
                hierarchy * 0.20 +
                semantic_connectivity * 0.20 +
                global_understanding * 0.20
            )

            return {
                "score": round(score, 4),  # 综合评分，保留四位小数
                "coverage": round(coverage, 4),
                "depth": round(depth, 4),
                "hierarchy": round(hierarchy, 4),
                "semantic_connectivity": round(semantic_connectivity, 4),
                "global_understanding": round(global_understanding, 4),
                "key_evidence": result.get("key_evidence", [])
            }
        except Exception as e:
            print(f"评估失败: {str(e)}")
            return {
                "score": 0.0,
                "coverage": 0.0,
                "depth": 0.0,
                "hierarchy": 0.0,
                "semantic_connectivity": 0.0,
                "global_understanding": 0.0,
                "key_evidence": []
            }

# 数据库配置
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "0000",
    "database": "rag_deal",
    "charset": 'utf8mb4',
    "collation": 'utf8mb4_unicode_ci'
}

async def process_batch(batch: List[tuple], evaluator: DiversityEvaluator):
    semaphore = asyncio.Semaphore(10)  # 控制并发量

    async def evaluate_row(row):
        async with semaphore:
            question_id, response, context = row
            try:
                result = await evaluator.evaluate(response, context)
                return (question_id, result)
            except Exception as e:
                print(f"处理失败 ID {question_id}: {str(e)}")
                return (question_id, None)

    tasks = [evaluate_row(row) for row in batch]
    return await asyncio.gather(*tasks)

async def main():
    # 初始化模型
    llm = ChatZhipuAI(
        model="GLM-4-Flash",
        temperature=0.3,
        zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY")
    )
    evaluator = DiversityEvaluator(llm)

    # 数据库连接
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # 获取已处理ID
    cursor.execute("SELECT DISTINCT question_id FROM diversity_scores_naive")
    processed_ids = {row[0] for row in cursor.fetchall()}

    # 分批处理
    batch_size = 50
    offset = 0

    while True:
        if processed_ids:
            # 当 processed_ids 不为空时，查询语句有 3 个占位符
            query = """
            SELECT id, response, relevant_context 
            FROM naive_question_responses
            WHERE id NOT IN (%s)
            LIMIT %s OFFSET %s
            """
            # 将 processed_ids 转换为逗号分隔的字符串
            processed_ids_str = ",".join(map(str, processed_ids))
            params = (processed_ids_str, batch_size, offset)
        else:
            # 当 processed_ids 为空时，查询语句有 2 个占位符
            query = """
            SELECT id, response, relevant_context 
            FROM naive_question_responses
            LIMIT %s OFFSET %s
            """
            params = (batch_size, offset)

        cursor.execute(query, params)
        batch = cursor.fetchall()

        if not batch:
            break

        results = await process_batch(batch, evaluator)

        # 写入数据库
        insert_query = """
        INSERT INTO diversity_scores_naive
        (question_id, score, coverage, depth, hierarchy, semantic_connectivity, global_understanding, key_evidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        score = VALUES(score),
        coverage = VALUES(coverage),
        depth = VALUES(depth),
        hierarchy = VALUES(hierarchy),
        semantic_connectivity = VALUES(semantic_connectivity),
        global_understanding = VALUES(global_understanding),
        key_evidence = VALUES(key_evidence)
        """

        for question_id, result in results:
            if result:
                cursor.execute(insert_query, (
                    question_id,
                    result["score"],
                    result["coverage"],
                    result["depth"],
                    result["hierarchy"],
                    result["semantic_connectivity"],
                    result["global_understanding"],
                    str(result["key_evidence"])  # 将列表转换为字符串存储
                ))

        conn.commit()
        offset += batch_size
        print(f"已处理 {offset} 条记录")
        time.sleep(5)  # 控制请求频率

    cursor.close()
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())