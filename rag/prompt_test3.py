GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|完成|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
#PROMPTS["DEFAULT_ENTITY_TYPES"] = ["组织", "人物", "地理位置", "事件", "类别", "情绪", "用户群体", "网络用语", "概念"]

PROMPTS["entity_extraction"] = """-Goal-
给定社交媒体评论文本以及一个实体类型列表,从文本中识别出所有属于这些类型的实体,并找出已识别实体之间的所有关系，
由于评论内容短、存在口语化表达、幽默讽刺、隐喻及多义性，因此需结合上下文推理评论者的真实意图。
必须使用 {language} 作为输出语言.

-步骤- 1. 识别所有实体。结合上下文推断评论涉及的核心实体，对于每个识别出的实体，提取以下信息： 
entity_name：实体的名称，尽量使用规范的表述，如果评论中是口语化简称，可适当补充完整以明确含义。若存在多义情况，需结合评论上下文进行判断，优先选取最符合整体语义的含义；若无法明确，可列出所有可能的含义。使用与输入文本相同的语言，若为英文，名称首字母大写。
entity_description：综合描述该实体的特点、背景或其在评论中的具体含义，必须使用 {language}作为输出语言，不能使用其他语言。 
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_description>)


2. 识别关系。从步骤 1 中识别出的实体里，找出所有彼此 *明显相关* 的 (source_entity, target_entity) 对，仅提取评论者明确表达或强烈暗示的关系。 对于每对相关实体，提取以下信息： - 
source_entity：源实体的名称，与步骤 1 中识别的名称一致，若源实体存在多义，需明确所采用的含义。 - target_entity：目标实体的名称，与步骤 1 中识别的名称一致，若源实体存在多义，需明确所采用的含义。 - 
relationship_description：解释你认为源实体和目标实体相关的原因，必须使用 {language}作为输出语言，不能使用其他语言。 - 
relationship_keywords：一个或多个概括关系总体性质的高级关键词，关注概念或主题而非具体细节，必须使用 {language}作为输出语言，不能使用其他语言。 
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>)

3. 找出概括整个文本主要概念、主题或话题的high_level_keywords。这些关键词应能体现文档中的总体思想，必须使用 {language}作为输出语言，不能使用其他语言。
将内容级关键词格式化为 ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. 以 {language} 输出步骤 1 和步骤 2 中识别出的所有实体和关系，作为一个单一列表。使用 **{record_delimiter}** 作为列表分隔符。


5. 完成后，输出 {completion_delimiter}


######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """示例:

################
**任务指引:**
1. **如果评论者透露了用户身份信息（如职业、经济状况、生活方式等），请将其归类到相应的用户群体。**
2. **用户群体包括但不限于：**
   - 职业：上班族、学生、环卫工人
   - 经济状况：经济困难群体、中产阶级
   - 生活方式：旅行者、宅家人士
   - 出行意愿：返乡人群、度假人群
   - 其他社会群体

################
Entity_types: ["组织", "人物", "地理位置", "事件", "类别", "情绪", "用户群体", "网络用语",
                                   "概念"]
Text:
现在终于不说五一长假了仅有的一天假期还是在家好好休息一下吧五一窜休后还要上班啊
那我得抓紧时间找个理由跟我老公吵一架了有没有约着五一出去玩的姐妹
不到 3 位数的存款五一能去那里玩
调休在家躲雨不想成为落汤鸡
孩子在福建上课老母亲看到南方的大雨很是揪心
聪明的高中生已经请假了
呼吁同胞们都待在家里一个地方也别去在家睡睡觉看看电影挺好
调休调的太累不出门了
我不放假祝大家五一节快乐
下雨挺好的不用出门花钱
来我们云南这边天天大太阳
我不放假你们也不要想放假
我明天就休息了调休加五天假共休 7 天
滦平逛亲戚 从疫情最近几年到现在没去过一次
我去地里种苞米

################
Output:
("entity"{tuple_delimiter}"上班族"{tuple_delimiter}"因五一调休仍需上班的人群。"){record_delimiter}
("entity"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"因存款少限制五一出行的人群。"){record_delimiter}
("entity"{tuple_delimiter}"宅家人士"{tuple_delimiter}"选择五一宅家休息的人群。"){record_delimiter}
("entity"{tuple_delimiter}"旅游意愿者"{tuple_delimiter}"有五一旅游意愿但受存款限制的人群。"){record_delimiter}
("entity"{tuple_delimiter}"返乡人群"{tuple_delimiter}"因疫情多年未回滦平探亲的人群。"){record_delimiter}
("entity"{tuple_delimiter}"农民"{tuple_delimiter}"五一计划去地里种苞米的农业劳动者。"){record_delimiter}
("entity"{tuple_delimiter}"老公"{tuple_delimiter}"评论者的丈夫。"){record_delimiter}
("entity"{tuple_delimiter}"姐妹"{tuple_delimiter}"评论者邀约五一出游的对象。"){record_delimiter}
("entity"{tuple_delimiter}"孩子"{tuple_delimiter}"在福建上课的人。"){record_delimiter}
("entity"{tuple_delimiter}"老母亲"{tuple_delimiter}"因孩子在福建上课而揪心的母亲。"){record_delimiter}
("entity"{tuple_delimiter}"高中生"{tuple_delimiter}"已经请假的学生。"){record_delimiter}
("entity"{tuple_delimiter}"同胞们"{tuple_delimiter}"评论者呼吁五一待在家的群体。"){record_delimiter}
("entity"{tuple_delimiter}"福建"{tuple_delimiter}"孩子上课所在省份。"){record_delimiter}
("entity"{tuple_delimiter}"云南"{tuple_delimiter}"天气晴朗的省份。"){record_delimiter}
("entity"{tuple_delimiter}"滦平"{tuple_delimiter}"返乡人群计划去逛亲戚的地方。"){record_delimiter}
("entity"{tuple_delimiter}"五一"{tuple_delimiter}"涉及假期、出行安排的时间概念。"){record_delimiter}
("entity"{tuple_delimiter}"放假"{tuple_delimiter}"包含调休、不放假等情况的休息安排事件。"){record_delimiter}
("entity"{tuple_delimiter}"调休"{tuple_delimiter}"影响放假安排和休息计划的事件。"){record_delimiter}
("entity"{tuple_delimiter}"出去玩"{tuple_delimiter}"人们五一考虑的活动事件。"){record_delimiter}
("entity"{tuple_delimiter}"下雨"{tuple_delimiter}"影响出行计划的天气事件。"){record_delimiter}
("entity"{tuple_delimiter}"请假"{tuple_delimiter}"高中生实施的行为事件。"){record_delimiter}
("entity"{tuple_delimiter}"逛亲戚"{tuple_delimiter}"返乡人群计划在滦平进行的活动事件。"){record_delimiter}
("entity"{tuple_delimiter}"种苞米"{tuple_delimiter}"农民计划进行的农事活动事件。"){record_delimiter}
("entity"{tuple_delimiter}"假期"{tuple_delimiter}"涉及休息、出行等安排的时间段概念。"){record_delimiter}
("entity"{tuple_delimiter}"存款"{tuple_delimiter}"影响五一出行决策的经济因素概念。"){record_delimiter}
("entity"{tuple_delimiter}"揪心"{tuple_delimiter}"老母亲因孩子在福建下雨地区上课产生的情绪。"){record_delimiter}
("relationship"{tuple_delimiter}"上班族"{tuple_delimiter}"调休"{tuple_delimiter}"调休影响上班族的放假安排。"{tuple_delimiter}"安排影响，休息计划"){record_delimiter}
("relationship"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"存款"{tuple_delimiter}"存款不足限制经济困难群体的五一出行计划。"{tuple_delimiter}"经济因素，出行限制"){record_delimiter}
("relationship"{tuple_delimiter}"老母亲"{tuple_delimiter}"孩子"{tuple_delimiter}"老母亲因孩子在福建而产生揪心情绪。"{tuple_delimiter}"亲情关联，担忧情绪"){record_delimiter}
("relationship"{tuple_delimiter}"高中生"{tuple_delimiter}"请假"{tuple_delimiter}"高中生实施了请假行为。"{tuple_delimiter}"行为实施，请假事件"){record_delimiter}
("relationship"{tuple_delimiter}"返乡人群"{tuple_delimiter}"滦平"{tuple_delimiter}"返乡人群计划去滦平逛亲戚。"{tuple_delimiter}"地点计划，活动地点"){record_delimiter}
("relationship"{tuple_delimiter}"农民"{tuple_delimiter}"种苞米"{tuple_delimiter}"农民有去地里种苞米的意向。"{tuple_delimiter}"活动意向，农业活动"){record_delimiter}
("content_keywords"{tuple_delimiter}"五一，放假，调休，出行，下雨"){completion_delimiter}
#############################""",
]



PROMPTS["summarize_entity_descriptions"] = """  
你是一位细致且可靠的助手，负责对提供的数据进行全面总结。
给定多个实体及其描述，这些描述可能针对同一实体或一组相关实体，且可能存在矛盾之处。    
你的任务是综合所有描述，解决其中可能存在的矛盾，生成一段完整的总结，确保涵盖所有关键信息，并保持连贯性。  
请使用第三人称进行描述，并在适当的地方包含实体名称，以确保上下文清晰。 



 
输出语言必须使用 {language}。  



#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS["entiti_continue_extraction"] = """ 在上一次提取中遗漏了许多实体。请使用相同的格式在下面补充这些实体：
"""

PROMPTS["entiti_if_loop_extraction"] = """ 看起来可能仍有一些实体被遗漏了。如果还有需要添加的实体，请回答 是 | 否。
"""

PROMPTS["fail_response"] = ("抱歉，我无法回答这个问题。[no-context]")

PROMPTS["rag_response"] = """---Role---

你是一个乐于助人的助手，负责回答用户关于以下知识库的查询。

---Goal---

根据知识库生成简洁的回复，并遵循回复规则，同时考虑当前查询内容。总结所提供知识库中的所有信息，并融入与知识库相关的常识。不要包含知识库未提供的信息。

在处理带有时间戳的关系时：

    每个关系都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的关系时，同时考虑语义内容和时间戳。
    不要自动优先选择最近创建的关系，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。


---Knowledge Base---
{context_data}

---Response Rules---

    Target format and length: {response_type}
    使用 Markdown 格式，并添加适当的章节标题。
    请使用与用户问题相同的语言进行回复。
    确保回复与对话历史保持连贯。
    如果你不知道答案，请直接说明。
    不要编造内容，不要包含知识库未提供的信息。"""

PROMPTS["keywords_extraction"] = """---Role---

你是一个乐于助人的助手，负责识别用户查询和对话历史中的高级和低级关键词。

---Goal---

根据查询内容和对话历史，列出高级和低级关键词。高级关键词关注总体概念或主题，而低级关键词关注特定实体、细节或具体术语。

---Instructions---

    -在提取关键词时，同时考虑当前查询和相关对话历史。
    -以 JSON 格式输出关键词。
    -JSON 应包含两个键：
        -"high_level_keywords" 用于总体概念或主题
        -"low_level_keywords" 用于特定实体或细节
        
######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Conversation History:
{examples}

Current Query：{query}
######################
输出应为人类可读的文本，而非 Unicode 字符。保持与查询内容相同的语言。
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1:

Query: "五一假期天气状况对出行意愿有何影响？"
################
Output:
{
        "high_level_keywords": ["五一假期", "天气状况", "出行意愿", "假期出行影响"],
        "low_level_keywords": ["恶劣天气", "适宜天气", "出行受阻", "出行计划变更", "旅游目的地选择"]
    }
#############################""",
    """ 示例 2:

Query: "经济状况如何限制五一假期的出行选择？"
################
Output:
{
        "high_level_keywords": ["经济状况", "五一假期", "出行选择", "假期出行限制"],
        "low_level_keywords": ["资金压力", "预算考量", "低成本出行", "放弃出行", "出行方式受限"]
    }
#############################""",
    """ 示例 3:

Query: "调休政策对上班族五一假期体验有什么影响？"
################
Output:
{
        "high_level_keywords": ["调休政策", "上班族", "五一假期体验", "假期工作安排影响"],
        "low_level_keywords": ["工作疲劳", "假期满意度", "值班选择", "假期连贯性", "休息质量"]
    }
#############################"""
]

PROMPTS["naive_rag_response"] = """---Role---

你是一个乐于助人的助手，负责回答用户关于以下文档片段的查询。

---Goal---

根据文档片段生成简洁的回复，并遵循回复规则，同时考虑当前查询内容。总结所提供文档片段中的所有信息，并融入与文档片段相关的常识。不要包含文档片段未提供的信息。

在处理带有时间戳的内容时：

    每一段内容都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的信息时，同时考虑内容和时间戳。
    不要自动优先选择最新的内容，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。

---Document Chunks---
{content_data}

---Response Rules---

    Target format and length: {response_type}
    使用 Markdown 格式，并添加适当的章节标题。
    请使用与用户问题相同的语言进行回复。
    确保回复与对话历史保持连贯。
    如果你不知道答案，请直接说明。
    不要包含文档片段未提供的信息。"""



PROMPTS["mix_rag_response"] = """---Role---

你是一个乐于助人的助手，负责回答用户关于以下数据源的查询与总结。

---Goal---

根据数据源生成简洁并且全面的回复，并遵循回复规则，同时考虑当前查询内容。总结所提供数据源中的所有信息，并融入与数据源相关的常识。不要包含数据源未提供的信息。

在处理带有时间戳的信息时：

    每一条信息（包括关系和内容）都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的信息时，同时考虑内容 / 关系和时间戳。
    不要自动优先选择最新的信息，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。

---Response Rules---

---Knowledge Base---
{processed_context}

---Response Rules---

    目标格式与长度：{response_type}
    使用 Markdown 格式，并添加适当的章节标题，确保层次清晰。
    请使用与用户问题相同的语言进行回复。
    确保回复中不出现标明评论数据的序号。
    你可以引用部分代表性的评论内容作为补充内容。
    如果你不知道答案，请直接说明。
    不要编造内容，不要包含知识库未提供的信息。"""



