GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|完成|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["组织", "人物", "地理位置", "事件", "类别", "情绪", "用户群体", "网络用语", "概念"]

PROMPTS["entity_extraction"] = """-Goal-
给定社交媒体评论文本以及一个实体类型列表,从文本中识别出所有属于这些类型的实体,并找出已识别实体之间的所有关系，
由于评论内容短、存在口语化表达、幽默讽刺、隐喻及多义性，因此需结合上下文推理评论者的真实意图。
必须使用 {language} 作为输出语言.

-步骤- 1. 识别所有实体。结合上下文推断评论涉及的核心实体，对于每个识别出的实体，提取以下信息： 
entity_name：实体的名称，尽量使用规范的表述，如果评论中是口语化简称，可适当补充完整以明确含义。若存在多义情况，需结合评论上下文进行判断，优先选取最符合整体语义的含义；若无法明确，可列出所有可能的含义。使用与输入文本相同的语言，若为英文，名称首字母大写。
entity_type：以下类型之一：[{entity_types}]，必须使用 {language}作为输出语言，不能使用其他语言。 
entity_description：综合描述该实体的特点、背景或其在评论中的具体含义，必须使用 {language}作为输出语言，不能使用其他语言。 
将每个实体格式化为 ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)


2. 识别关系。从步骤 1 中识别出的实体里，找出所有彼此 *明显相关* 的 (source_entity, target_entity) 对，仅提取评论者明确表达或强烈暗示的关系。
对于每对相关实体，提取以下信息：
- source_entity：源实体的名称，与步骤 1 中识别的名称一致，若源实体存在多义，需明确所采用的含义。
- target_entity：目标实体的名称，与步骤 1 中识别的名称一致，若源实体存在多义，需明确所采用的含义。
- relationship_description：解释你认为源实体和目标实体相关的原因，必须使用 {language}作为输出语言，不能使用其他语言。
- relationship_strength：一个数值分数，表明源实体和目标实体之间关系的强度，范围为 1-10，数值越大代表关系越强。
- relationship_keywords：一个或多个概括关系总体性质的高级关键词，关注概念或主题而非具体细节，必须使用 {language}作为输出语言，不能使用其他语言。
将每个关系格式化为 ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

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
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """示例 1:

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
("entity"{tuple_delimiter}"上班族"{tuple_delimiter}"用户群体"{tuple_delimiter}"因五一调休仍需上班的人群。"){record_delimiter}
("entity"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"用户群体"{tuple_delimiter}"因存款少限制五一出行的人群。"){record_delimiter}
("entity"{tuple_delimiter}"宅家人士"{tuple_delimiter}"用户群体"{tuple_delimiter}"选择五一宅家休息的人群。"){record_delimiter}
("entity"{tuple_delimiter}"旅游意愿者"{tuple_delimiter}"用户群体"{tuple_delimiter}"有五一旅游意愿但受存款限制的人群。"){record_delimiter}
("entity"{tuple_delimiter}"返乡人群"{tuple_delimiter}"用户群体"{tuple_delimiter}"因疫情多年未回滦平探亲的人群。"){record_delimiter}
("entity"{tuple_delimiter}"农民"{tuple_delimiter}"用户群体"{tuple_delimiter}"五一计划去地里种苞米的农业劳动者。"){record_delimiter}
("entity"{tuple_delimiter}"老公"{tuple_delimiter}"人物"{tuple_delimiter}"评论者的丈夫。"){record_delimiter}
("entity"{tuple_delimiter}"姐妹"{tuple_delimiter}"人物"{tuple_delimiter}"评论者邀约五一出游的对象。"){record_delimiter}
("entity"{tuple_delimiter}"孩子"{tuple_delimiter}"人物"{tuple_delimiter}"在福建上课的人。"){record_delimiter}
("entity"{tuple_delimiter}"老母亲"{tuple_delimiter}"人物"{tuple_delimiter}"因孩子在福建上课而揪心的母亲。"){record_delimiter}
("entity"{tuple_delimiter}"高中生"{tuple_delimiter}"用户群体"{tuple_delimiter}"已经请假的学生。"){record_delimiter}
("entity"{tuple_delimiter}"同胞们"{tuple_delimiter}"人物"{tuple_delimiter}"评论者呼吁五一待在家的群体。"){record_delimiter}
("entity"{tuple_delimiter}"福建"{tuple_delimiter}"地理位置"{tuple_delimiter}"孩子上课所在省份。"){record_delimiter}
("entity"{tuple_delimiter}"云南"{tuple_delimiter}"地理位置"{tuple_delimiter}"天气晴朗的省份。"){record_delimiter}
("entity"{tuple_delimiter}"滦平"{tuple_delimiter}"地理位置"{tuple_delimiter}"返乡人群计划去逛亲戚的地方。"){record_delimiter}
("entity"{tuple_delimiter}"五一"{tuple_delimiter}"时间"{tuple_delimiter}"涉及假期、出行安排的时间概念。"){record_delimiter}
("entity"{tuple_delimiter}"放假"{tuple_delimiter}"事件"{tuple_delimiter}"包含调休、不放假等情况的休息安排事件。"){record_delimiter}
("entity"{tuple_delimiter}"调休"{tuple_delimiter}"事件"{tuple_delimiter}"影响放假安排和休息计划的事件。"){record_delimiter}
("entity"{tuple_delimiter}"出去玩"{tuple_delimiter}"事件"{tuple_delimiter}"人们五一考虑的活动事件。"){record_delimiter}
("entity"{tuple_delimiter}"下雨"{tuple_delimiter}"事件"{tuple_delimiter}"影响出行计划的天气事件。"){record_delimiter}
("entity"{tuple_delimiter}"请假"{tuple_delimiter}"事件"{tuple_delimiter}"高中生实施的行为事件。"){record_delimiter}
("entity"{tuple_delimiter}"逛亲戚"{tuple_delimiter}"事件"{tuple_delimiter}"返乡人群计划在滦平进行的活动事件。"){record_delimiter}
("entity"{tuple_delimiter}"种苞米"{tuple_delimiter}"事件"{tuple_delimiter}"农民计划进行的农事活动事件。"){record_delimiter}
("entity"{tuple_delimiter}"假期"{tuple_delimiter}"概念"{tuple_delimiter}"涉及休息、出行等安排的时间段概念。"){record_delimiter}
("entity"{tuple_delimiter}"存款"{tuple_delimiter}"概念"{tuple_delimiter}"影响五一出行决策的经济因素概念。"){record_delimiter}
("entity"{tuple_delimiter}"揪心"{tuple_delimiter}"情绪"{tuple_delimiter}"老母亲因孩子在福建下雨地区上课产生的情绪。"){record_delimiter}
("relationship"{tuple_delimiter}"上班族"{tuple_delimiter}"调休"{tuple_delimiter}"调休影响上班族的放假安排。"{tuple_delimiter}"安排影响，休息计划"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"存款"{tuple_delimiter}"存款不足限制经济困难群体的五一出行计划。"{tuple_delimiter}"经济因素，出行限制"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"老母亲"{tuple_delimiter}"孩子"{tuple_delimiter}"老母亲因孩子在福建而产生揪心情绪。"{tuple_delimiter}"亲情关联，担忧情绪"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"高中生"{tuple_delimiter}"请假"{tuple_delimiter}"高中生实施了请假行为。"{tuple_delimiter}"行为实施，请假事件"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"返乡人群"{tuple_delimiter}"滦平"{tuple_delimiter}"返乡人群计划去滦平逛亲戚。"{tuple_delimiter}"地点计划，活动地点"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"农民"{tuple_delimiter}"种苞米"{tuple_delimiter}"农民有去地里种苞米的意向。"{tuple_delimiter}"活动意向，农业活动"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"五一，放假，调休，出行，下雨"){completion_delimiter}
#############################""",

    """示例 2:
Entity_types: ["组织", "人物", "地理位置", "事件", "类别", "情绪", "用户群体", "网络用语",
                                   "概念"]
Text:
特种兵去上海苏州南京加路费大约一千多点
南方下冰雹河北承德下雪这个五一你还敢去哪儿玩
想问问有下冰雹的地方
何时假期能休退休吧
怎么回事调休办没通知大雨让他五一期间调休吗
南方下冰雹河北承德下雪这个五一你还敢去哪儿玩
别找我我在家追剧
呼吁同胞们都待在家里一个地方也别去在家睡睡觉看看电影挺好
我想回湖南现在还没抢到票
我一个环卫工操啥心
还有两天放假了烦死了好穷哪里都去不了
羡慕你们我除了大姨妈经常放假一放就是几个月平时从来没有假期
洛阳晴天欢迎大家来
是不是全国都在下雨
没事咱挑天晴的地方去再不济抖音里看
这几天每天都想着出去玩哪有心思上班可是看看自己囊中羞怯还是算了吧客厅房间五日游
来雄安吧没有雨
家里路边思考一下能去哪里蹭饭
买个地球仪在家转转
周边电动车游

################
**任务指引:**
1. **如果评论者透露了身份信息（如职业、经济状况、生活方式等），请将其归类到相应的用户群体。**
2. **用户群体包括但不限于：**
   - 职业：上班族、学生、环卫工人
   - 经济状况：经济困难群体、中产阶级
   - 生活方式：旅行者、宅家人士
   - 出行意愿：返乡人群、度假人群
   - 其他社会群体

################
Output:
("entity"{tuple_delimiter}"特种兵"{tuple_delimiter}"用户群体"{tuple_delimiter}"计划去上海、苏州、南京游玩的群体。"){record_delimiter}
("entity"{tuple_delimiter}"返乡人群"{tuple_delimiter}"用户群体"{tuple_delimiter}"表达想回湖南但没抢到票的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"环卫工人"{tuple_delimiter}"用户群体"{tuple_delimiter}"表明自己职业为环卫工的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"用户群体"{tuple_delimiter}"因没钱无法在假期出行的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"无固定工作者"{tuple_delimiter}"用户群体"{tuple_delimiter}"羡慕他人有假期，自己平时没假期的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"蹭饭人群"{tuple_delimiter}"用户群体"{tuple_delimiter}"思考能去哪里蹭饭的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"同胞们"{tuple_delimiter}"人物"{tuple_delimiter}"被呼吁待在家里的群体。"){record_delimiter}
("entity"{tuple_delimiter}"你们"{tuple_delimiter}"人物"{tuple_delimiter}"被‘无固定工作者’羡慕有假期的群体。"){record_delimiter}
("entity"{tuple_delimiter}"上海"{tuple_delimiter}"地理位置"{tuple_delimiter}"特种兵计划去游玩的城市。"){record_delimiter}
("entity"{tuple_delimiter}"苏州"{tuple_delimiter}"地理位置"{tuple_delimiter}"特种兵计划去游玩的城市。"){record_delimiter}
("entity"{tuple_delimiter}"南京"{tuple_delimiter}"地理位置"{tuple_delimiter}"特种兵计划去游玩的城市。"){record_delimiter}
("entity"{tuple_delimiter}"河北承德"{tuple_delimiter}"地理位置"{tuple_delimiter}"五一期间下雪的地方。"){record_delimiter}
("entity"{tuple_delimiter}"湖南"{tuple_delimiter}"地理位置"{tuple_delimiter}"‘返乡人群’想回去的省份。"){record_delimiter}
("entity"{tuple_delimiter}"洛阳"{tuple_delimiter}"地理位置"{tuple_delimiter}"晴天，被邀请去游玩的城市。"){record_delimiter}
("entity"{tuple_delimiter}"雄安"{tuple_delimiter}"地理位置"{tuple_delimiter}"没有雨，被邀请去的地方。"){record_delimiter}
("entity"{tuple_delimiter}"五一"{tuple_delimiter}"时间"{tuple_delimiter}"涉及假期、出行安排的时间概念。"){record_delimiter}
("entity"{tuple_delimiter}"放假"{tuple_delimiter}"事件"{tuple_delimiter}"人们关注的休息安排事件，有调休等情况。"){record_delimiter}
("entity"{tuple_delimiter}"调休"{tuple_delimiter}"事件"{tuple_delimiter}"影响放假安排的事件。"){record_delimiter}
("entity"{tuple_delimiter}"假期"{tuple_delimiter}"概念"{tuple_delimiter}"与休息、出行相关的时间段概念。"){record_delimiter}
("entity"{tuple_delimiter}"路费"{tuple_delimiter}"概念"{tuple_delimiter}"出行游玩涉及的费用概念。"){record_delimiter}
("entity"{tuple_delimiter}"钱"{tuple_delimiter}"概念"{tuple_delimiter}"影响出行游玩决策的因素概念。"){record_delimiter}
("entity"{tuple_delimiter}"下冰雹"{tuple_delimiter}"事件"{tuple_delimiter}"南方出现的天气事件。"){record_delimiter}
("entity"{tuple_delimiter}"下雪"{tuple_delimiter}"事件"{tuple_delimiter}"河北承德出现的天气事件。"){record_delimiter}
("entity"{tuple_delimiter}"追剧"{tuple_delimiter}"行为"{tuple_delimiter}"宅家人士在家进行的行为。"){record_delimiter}
("entity"{tuple_delimiter}"出去玩"{tuple_delimiter}"行为"{tuple_delimiter}"人们在假期考虑的行为。"){record_delimiter}
("entity"{tuple_delimiter}"蹭饭"{tuple_delimiter}"行为"{tuple_delimiter}"蹭饭人群思考的行为。"){record_delimiter}
("entity"{tuple_delimiter}"买地球仪"{tuple_delimiter}"行为"{tuple_delimiter}"评论者的行为。"){record_delimiter}
("entity"{tuple_delimiter}"周边电动车游"{tuple_delimiter}"行为"{tuple_delimiter}"一种出行游玩的行为。"){record_delimiter}
("entity"{tuple_delimiter}"烦死了"{tuple_delimiter}"情绪"{tuple_delimiter}"经济困难群体因没钱出行产生的情绪。"){record_delimiter}
("entity"{tuple_delimiter}"羡慕"{tuple_delimiter}"情绪"{tuple_delimiter}"无固定工作者对他人有假期产生的情绪。"){record_delimiter}
("relationship"{tuple_delimiter}"特种兵"{tuple_delimiter}"上海"{tuple_delimiter}"特种兵计划前往上海游玩。"{tuple_delimiter}"出行目的地，旅行计划"{tuple_delimiter}9){record_delimiter}  
("relationship"{tuple_delimiter}"特种兵"{tuple_delimiter}"苏州"{tuple_delimiter}"特种兵计划前往苏州游玩。"{tuple_delimiter}"出行目的地，旅行计划"{tuple_delimiter}9){record_delimiter}  
("relationship"{tuple_delimiter}"特种兵"{tuple_delimiter}"南京"{tuple_delimiter}"特种兵计划前往南京游玩。"{tuple_delimiter}"出行目的地，旅行计划"{tuple_delimiter}9){record_delimiter}  
("relationship"{tuple_delimiter}"返乡人群"{tuple_delimiter}"湖南"{tuple_delimiter}"返乡人群希望回到湖南但未买到票。"{tuple_delimiter}"返乡目的地，出行障碍"{tuple_delimiter}8){record_delimiter}  
("relationship"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"放假"{tuple_delimiter}"经济困难群体因缺钱无法在放假期间出行。"{tuple_delimiter}"经济状况，出行限制"{tuple_delimiter}7){record_delimiter}  
("relationship"{tuple_delimiter}"调休"{tuple_delimiter}"放假"{tuple_delimiter}"调休影响了放假安排。"{tuple_delimiter}"假期调整，出行计划"{tuple_delimiter}9){record_delimiter}  
("content_keywords"{tuple_delimiter}"五一，放假，调休，出行，下雨，经济状况"){completion_delimiter}  
#############################""",

    """示例 3:

Entity_types: ["组织", "人物", "地理位置", "事件", "类别", "情绪", "用户群体", "网络用语", 
                                   "概念"]
Text:
来长春吧天气好没有雨
下不下雨和我没关系反正我又没钱出去玩儿
欢迎你来洛阳玩感受一下千年帝都文化灵机一动订民宿找我吧
塞罕坝都下雪了
五一去哪个城市避暑比较适合推荐一下
该死的五一害得我连续上14天班
五一可以去北方避雨么
估计都在家 出去的都是勇士
想出去但是没去的方向有没有和我一样的也没有同伴
云南不下雨欢迎来云南
就说能不能值班三薪不想人挤人
转眼五一了快过年了
调休太累还没放假呢就累的够呛五一家里蹲五天
看新闻知天下事
家门口游
回村里水库抓不花钱的鱼吃家里新收的土豆子
客厅卧室五日游
今天就放假了
请叫我靓仔 精武鸭脖华埠店 何似在人间 庸人自扰    小确幸 洪不才  我是谁
现在是再上两天班了
在家呆着欣赏暴风雨的美景
不要下了还要去旅游呢一边旅游一边睁眼看世界
想去丽水 有本地的朋友嘛 天气好不好
大连不下雨来大连吧
北京不下雨来北京吧看
连上了9天放3天又接着连上9天思考了很久放这个假来干嘛
五一没有休息的举手

################
**任务指引:**
1. **如果评论者评论者透露了身份信息（如职业、经济状况、生活方式等），请将其归类到相应的用户群体。**
2. **用户群体包括但不限于：**
   - 职业：上班族、学生、环卫工人
   - 经济状况：经济困难群体、中产阶级
   - 生活方式：旅行者、宅家人士
   - 出行意愿：返乡人群、度假人群
   - 其他社会群体

################
Output:
("entity"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"用户群体"{tuple_delimiter}"因资金短缺无法外出游玩的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"旅行者"{tuple_delimiter}"用户群体"{tuple_delimiter}"有旅游意愿，邀请他人去洛阳、云南等地游玩或自己想去旅游的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"上班族"{tuple_delimiter}"用户群体"{tuple_delimiter}"受五一调休影响，连续上班或关注值班薪资的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"度假人群"{tuple_delimiter}"用户群体"{tuple_delimiter}"考虑五一期间去城市避暑、避雨度假的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"宅家人士"{tuple_delimiter}"用户群体"{tuple_delimiter}"选择在五一假期进行家里蹲、家门口游等活动的评论者群体。"){record_delimiter}
("entity"{tuple_delimiter}"长春"{tuple_delimiter}"地理位置"{tuple_delimiter}"天气良好、无雨，被推荐前往的城市。"){record_delimiter}
("entity"{tuple_delimiter}"洛阳"{tuple_delimiter}"地理位置"{tuple_delimiter}"具有千年帝都文化，被邀请前往游玩的城市。"){record_delimiter}
("entity"{tuple_delimiter}"塞罕坝"{tuple_delimiter}"地理位置"{tuple_delimiter}"出现下雪天气的地方。"){record_delimiter}
("entity"{tuple_delimiter}"云南"{tuple_delimiter}"地理位置"{tuple_delimiter}"不下雨，被邀请前往的省份。"){record_delimiter}
("entity"{tuple_delimiter}"北方"{tuple_delimiter}"地理位置"{tuple_delimiter}"五一期间被考虑去避雨的区域。"){record_delimiter}
("entity"{tuple_delimiter}"丽水"{tuple_delimiter}"地理位置"{tuple_delimiter}"有人想去旅游并询问天气的城市。"){record_delimiter}
("entity"{tuple_delimiter}"大连"{tuple_delimiter}"地理位置"{tuple_delimiter}"不下雨，被邀请前往的城市。"){record_delimiter}
("entity"{tuple_delimiter}"北京"{tuple_delimiter}"地理位置"{tuple_delimiter}"不下雨，被邀请前往的城市。"){record_delimiter}
("entity"{tuple_delimiter}"村里水库"{tuple_delimiter}"地理位置"{tuple_delimiter}"有人计划回去抓鱼的地方。"){record_delimiter}
("entity"{tuple_delimiter}"五一"{tuple_delimiter}"概念"{tuple_delimiter}"涉及放假、出行、调休等安排的重要时间节点概念。"){record_delimiter}
("entity"{tuple_delimiter}"今天"{tuple_delimiter}"概念"{tuple_delimiter}"评论者开始放假的时间概念。"){record_delimiter}
("entity"{tuple_delimiter}"出去玩"{tuple_delimiter}"事件"{tuple_delimiter}"部分评论者因金钱、方向等因素受限而难以实施的活动行为。"){record_delimiter}
("entity"{tuple_delimiter}"旅游"{tuple_delimiter}"事件"{tuple_delimiter}"部分评论者计划或期望进行的活动行为。"){record_delimiter}
("entity"{tuple_delimiter}"值班三薪"{tuple_delimiter}"事件"{tuple_delimiter}"评论者关注并询问的工作报酬相关事件。"){record_delimiter}
("entity"{tuple_delimiter}"调休"{tuple_delimiter}"事件"{tuple_delimiter}"影响放假安排，导致部分评论者感到疲惫的事件。"){record_delimiter}
("entity"{tuple_delimiter}"放假"{tuple_delimiter}"事件"{tuple_delimiter}"与工作休息相关的事件，不同评论者情况各异。"){record_delimiter}
("entity"{tuple_delimiter}"下雪"{tuple_delimiter}"事件"{tuple_delimiter}"发生在塞罕坝的天气事件。"){record_delimiter}
("entity"{tuple_delimiter}"下雨"{tuple_delimiter}"事件"{tuple_delimiter}"影响出行旅游决策的天气事件。"){record_delimiter}
("entity"{tuple_delimiter}"抓鱼"{tuple_delimiter}"事件"{tuple_delimiter}"评论者计划在村里水库进行的活动行为。"){record_delimiter}
("entity"{tuple_delimiter}"钱"{tuple_delimiter}"概念"{tuple_delimiter}"限制部分评论者外出游玩的关键经济因素概念。"){record_delimiter}
("entity"{tuple_delimiter}"同伴"{tuple_delimiter}"概念"{tuple_delimiter}"部分评论者外出游玩所缺乏的要素概念。"){record_delimiter}
("entity"{tuple_delimiter}"方向"{tuple_delimiter}"概念"{tuple_delimiter}"部分评论者外出游玩所缺乏的要素概念。"){record_delimiter}
("entity"{tuple_delimiter}"假期"{tuple_delimiter}"概念"{tuple_delimiter}"与休息、出行相关的时间段概念。"){record_delimiter}
("entity"{tuple_delimiter}"值班薪资"{tuple_delimiter}"概念"{tuple_delimiter}"评论者关注的工作报酬概念。"){record_delimiter}
("entity"{tuple_delimiter}"疲惫"{tuple_delimiter}"情绪"{tuple_delimiter}"上班族因调休产生的感受。"){record_delimiter}
("relationship"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"钱"{tuple_delimiter}"经济困难群体因资金不足，无法进行出去玩的行为。"{tuple_delimiter}"经济制约，出行意向"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"旅行者"{tuple_delimiter}"洛阳"{tuple_delimiter}"旅行者向他人发出去洛阳旅游的邀请。"{tuple_delimiter}"旅游邀请，地点关联"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"上班族"{tuple_delimiter}"五一"{tuple_delimiter}"五一调休安排使上班族需连续上班。"{tuple_delimiter}"时间关联，工作安排"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"度假人群"{tuple_delimiter}"北方"{tuple_delimiter}"度假人群在五一期间有去北方避雨的意向。"{tuple_delimiter}"时间关联，出行意向"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"旅行者"{tuple_delimiter}"同伴"{tuple_delimiter}"旅行者在出行旅游时缺乏同伴。"{tuple_delimiter}"出行要素，缺乏关联"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"旅行者"{tuple_delimiter}"方向"{tuple_delimiter}"旅行者在出行旅游时缺乏明确方向。"{tuple_delimiter}"出行要素，缺乏关联"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"上班族"{tuple_delimiter}"值班三薪"{tuple_delimiter}"上班族对值班三薪事件表示关注并进行询问。"{tuple_delimiter}"工作询问，事件关联"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"宅家人士"{tuple_delimiter}"五一"{tuple_delimiter}"宅家人士在五一假期选择家里蹲的休息方式。"{tuple_delimiter}"时间关联，休息安排"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"上班族"{tuple_delimiter}"今天"{tuple_delimiter}"上班族在今天这个时间开始放假。"{tuple_delimiter}"时间关联，放假安排"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"旅行者"{tuple_delimiter}"丽水"{tuple_delimiter}"旅行者有去丽水旅游的意向，并关注当地天气。"{tuple_delimiter}"出行意向，地点关联"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"长春"{tuple_delimiter}"下雨"{tuple_delimiter}"长春当前天气特征为无雨。"{tuple_delimiter}"地点特征，天气关联"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"云南"{tuple_delimiter}"下雨"{tuple_delimiter}"云南当前天气特征为无雨。"{tuple_delimiter}"地点特征，天气关联"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"大连"{tuple_delimiter}"下雨"{tuple_delimiter}"大连当前天气特征为无雨。"{tuple_delimiter}"地点特征，天气关联"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"北京"{tuple_delimiter}"下雨"{tuple_delimiter}"北京当前天气特征为无雨。"{tuple_delimiter}"地点特征，天气关联"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"塞罕坝"{tuple_delimiter}"下雪"{tuple_delimiter}"塞罕坝发生了下雪这一天气事件。"{tuple_delimiter}"地点特征，天气事件"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"上班族"{tuple_delimiter}"调休"{tuple_delimiter}"调休导致上班族产生疲惫情绪。"{tuple_delimiter}"事件影响，情绪反应"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"五一，放假，出行，天气，钱"){completion_delimiter}
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

根据知识库生成简洁的回复，并遵循回复规则，同时考虑对话历史和当前查询内容。总结所提供知识库中的所有信息，并融入与知识库相关的常识。不要包含知识库未提供的信息。

在处理带有时间戳的关系时：

    每个关系都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的关系时，同时考虑语义内容和时间戳。
    不要自动优先选择最近创建的关系，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。


---Conversation History---
{history}

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

根据文档片段生成简洁的回复，并遵循回复规则，同时考虑对话历史和当前查询内容。总结所提供文档片段中的所有信息，并融入与文档片段相关的常识。不要包含文档片段未提供的信息。

在处理带有时间戳的内容时：

    每一段内容都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的信息时，同时考虑内容和时间戳。
    不要自动优先选择最新的内容，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。


---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

    Target format and length: {response_type}
    使用 Markdown 格式，并添加适当的章节标题。
    请使用与用户问题相同的语言进行回复。
    确保回复与对话历史保持连贯。
    如果你不知道答案，请直接说明。
    不要包含文档片段未提供的信息。"""

PROMPTS["similarity_check"] = """ 请分析以下两个问题的相似度：

Question 1: {original_prompt}
Question 2: {cached_prompt}

请评估以下两点，并直接给出一个 0 到 1 之间的相似度分数：

    这两个问题在语义上是否相似
    问题 2 的答案是否可用于回答问题 1
    相似度分数标准：
    0: 完全不相关或答案无法复用，包括但不限于以下情况：
        问题主题不同
        问题中提及的地点不同
        问题中提及的时间不同
        问题中提及的具体人物不同
        问题中提及的具体事件不同
        问题中的背景信息不同
        问题中的关键条件不同
        1: 完全相同且答案可直接复用
        0.5: 部分相关，答案需修改后才可使用
        仅返回一个 0 - 1 之间的数字，不添加任何额外内容。
        """

PROMPTS["mix_rag_response"] = """---Role---

你是一个乐于助人的助手，负责回答用户关于以下数据源的查询。

---Goal---

根据数据源生成简洁的回复，并遵循回复规则，同时考虑对话历史和当前查询内容。数据源包含两部分：知识图谱（KG）和文档片段（DC）。总结所提供数据源中的所有信息，并融入与数据源相关的常识。不要包含数据源未提供的信息。

在处理带有时间戳的信息时：

    每一条信息（包括关系和内容）都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的信息时，同时考虑内容 / 关系和时间戳。
    不要自动优先选择最新的信息，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。


---Conversation History---
{history}

---Data Sources---

1. **来自知识图谱 (KG)：**  
{kg_context}  

2. **来自文档片段 (DC)：**  
{vector_context}  

---Response Rules---

- **目标格式与长度：** {response_type}  
- **语言要求：**  
  - 回复必须使用与用户问题相同的语言。  
  - 语言风格需清晰、直接，并适合社交媒体评论分析场景。  

- **结构要求：**  
  - 使用 Markdown 格式，并确保内容分章节组织。  
  - 每个章节需有清晰的标题，反映该部分的核心内容。  
  - 采用逻辑分层方式，例如按观点、主题、争议点等分类。  
  - 如果涉及多个观点，需分别呈现，并用客观方式归纳总结。  

- **数据一致性与参考来源：**  
  - 回复内容必须基于提供的数据，不得引入外部信息。  
  - **所有参考来源必须在 "参考文献" 部分列出**，并明确来源类型：  
    - `[KG] 知识图谱来源`  
    - `[DC] 向量数据来源`  
  - 确保所有引用的内容均来自已知数据源，**不得编造信息**。  

- **连贯性与完整性：**  
  - 需确保回复与对话历史保持连贯，不重复、不矛盾。  
  - 采用用户易理解的方式总结关键信息，并尽量提供实用性结论。  

- **当数据不足时的处理方式：**  
  - **如果数据源未提供足够信息，直接说明 "暂无足够数据支持该问题"。**  
  - **不得编造内容填充答案。**  """

