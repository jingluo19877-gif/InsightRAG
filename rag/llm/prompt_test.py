GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "中文"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|完成|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROMPTS["DEFAULT_ENTITY_TYPES"] = ["组织", "人物", "地理位置", "事件", "类别"]

PROMPTS["entity_extraction"] = """-目标-
给定一份可能与本任务相关的文本文档以及一个实体类型列表,从文本中识别出所有属于这些类型的实体,并找出已识别实体之间的所有关系.
必须使用 {language} 作为输出语言.

-步骤-
1. 识别所有实体。对于每个识别出的实体，提取以下信息：
- 实体名称：实体的名称，必须使用与输入文本相同的语言，不能用其他语言。
- 实体类型：以下类型之一：[{entity_types}]
- 实体描述：对实体的属性和活动进行全面描述，必须使用{language}语言，不能用其他语言。
将每个实体格式化为 ("实体"{tuple_delimiter}<实体名称>{tuple_delimiter}<实体类型>{tuple_delimiter}<实体描述>)

2. 从步骤 1 中识别出的实体里，找出所有彼此 *明显相关* 的 (源实体, 目标实体) 对。
对于每对相关实体，提取以下信息：
- 源实体：源实体的名称，与步骤 1 中识别的名称一致
- 目标实体：目标实体的名称，与步骤 1 中识别的名称一致
- 关系描述：解释你认为源实体和目标实体相关的原因，必须使用{language}语言，不能用其他语言。
- 关系强度：一个数值分数，表明源实体和目标实体之间关系的强度
- 关系关键词：一个或多个概括关系总体性质的高级关键词，关注概念或主题而非具体细节，必须使用{language}语言，不能用其他语言。
将每个关系格式化为 ("关系"{tuple_delimiter}<源实体>{tuple_delimiter}<目标实体>{tuple_delimiter}<关系描述>{tuple_delimiter}<关系关键词>{tuple_delimiter}<关系强度>)

3. 找出概括整个文本主要概念、主题或话题的高级关键词。这些关键词应能体现文档中的总体思想，必须使用{language}语言，不能用其他语言。
将内容级关键词格式化为 ("内容关键词"{tuple_delimiter}<高级关键词>)

4. 以 {language} 输出步骤 1 和步骤 2 中识别出的所有实体和关系，作为一个单一列表。使用 **{record_delimiter}** 作为列表分隔符。

5. 完成后，输出 {completion_delimiter}


######################
- 示例 -
######################
{examples}

#############################
- 实际数据 -
######################
实体类型: {entity_types}
文本: {input_text}
######################
输出:

"""


PROMPTS["entity_extraction_examples"] = [
    """示例 1:

################
**任务指引:**
1. **提取评论中的实体，并归类为人物、地点、时间、事件、概念等类别。**
2. **如果评论者表达了特定的身份信息（如职业、经济状况、生活方式等），请将其归类到相应的用户群体。**
3. **用户群体包括但不限于：**
   - 职业：上班族、学生、环卫工人
   - 经济状况：经济困难群体、中产阶级
   - 生活方式：旅行者、宅家人士
   - 出行意愿：返乡人群、度假人群
   - 其他社会群体

################
实体类型: [人物, 地点, 时间, 事件, 概念]
文本:
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
输出:
("实体"{tuple_delimiter}"上班族"{tuple_delimiter}"人物"{tuple_delimiter}"因五一调休仍需上班的上班族。"){record_delimiter}
("实体"{tuple_delimiter}"经济困难群体 1"{tuple_delimiter}"人物"{tuple_delimiter}"因存款不到三位数而难以出行的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"宅家人士"{tuple_delimiter}"人物"{tuple_delimiter}"选择在五一假期宅家休息的人。"){record_delimiter}
("实体"{tuple_delimiter}"旅游意愿者"{tuple_delimiter}"人物"{tuple_delimiter}"考虑五一外出旅游但受存款限制的群体。"){record_delimiter}
("实体"{tuple_delimiter}"返乡人群"{tuple_delimiter}"人物"{tuple_delimiter}"因疫情多年未能回乡探亲的群体。"){record_delimiter}
("实体"{tuple_delimiter}"农民"{tuple_delimiter}"人物"{tuple_delimiter}"五一期间计划去地里种苞米的农业劳动者。"){record_delimiter}
("实体"{tuple_delimiter}"老公"{tuple_delimiter}"人物"{tuple_delimiter}"评论者的丈夫，评论者提到可能会和其吵架。"){record_delimiter}
("实体"{tuple_delimiter}"姐妹"{tuple_delimiter}"人物"{tuple_delimiter}"评论者邀约五一出去玩的群体。"){record_delimiter}
("实体"{tuple_delimiter}"孩子"{tuple_delimiter}"人物"{tuple_delimiter}"在福建上课，让老母亲揪心的人。"){record_delimiter}
("实体"{tuple_delimiter}"老母亲"{tuple_delimiter}"人物"{tuple_delimiter}"看到南方大雨为孩子揪心的人。"){record_delimiter}
("实体"{tuple_delimiter}"高中生"{tuple_delimiter}"人物"{tuple_delimiter}"已经请假的群体。"){record_delimiter}
("实体"{tuple_delimiter}"同胞们"{tuple_delimiter}"人物"{tuple_delimiter}"评论者呼吁待在家里的群体。"){record_delimiter}
("实体"{tuple_delimiter}"福建"{tuple_delimiter}"地点"{tuple_delimiter}"孩子上课所在的省份。"){record_delimiter}
("实体"{tuple_delimiter}"云南"{tuple_delimiter}"地点"{tuple_delimiter}"天气天天大太阳的省份。"){record_delimiter}
("实体"{tuple_delimiter}"滦平"{tuple_delimiter}"地点"{tuple_delimiter}"评论者打算去逛亲戚的地方。"){record_delimiter}
("实体"{tuple_delimiter}"五一"{tuple_delimiter}"时间"{tuple_delimiter}"涉及假期、出行等安排的时间节点。"){record_delimiter}
("实体"{tuple_delimiter}"明天"{tuple_delimiter}"时间"{tuple_delimiter}"评论者开始休息的时间。"){record_delimiter}
("实体"{tuple_delimiter}"放假"{tuple_delimiter}"事件"{tuple_delimiter}"涉及休息安排的事件，有调休、不放假等情况。"){record_delimiter}
("实体"{tuple_delimiter}"调休"{tuple_delimiter}"事件"{tuple_delimiter}"影响放假安排和人们休息计划的事件。"){record_delimiter}
("实体"{tuple_delimiter}"出去玩"{tuple_delimiter}"事件"{tuple_delimiter}"人们在五一等假期考虑的活动。"){record_delimiter}
("实体"{tuple_delimiter}"下雨"{tuple_delimiter}"事件"{tuple_delimiter}"影响人们出行计划的天气相关事件。"){record_delimiter}
("实体"{tuple_delimiter}"请假"{tuple_delimiter}"事件"{tuple_delimiter}"高中生采取的行为。"){record_delimiter}
("实体"{tuple_delimiter}"逛亲戚"{tuple_delimiter}"事件"{tuple_delimiter}"评论者计划在滦平进行的活动。"){record_delimiter}
("实体"{tuple_delimiter}"种苞米"{tuple_delimiter}"事件"{tuple_delimiter}"评论者打算去地里做的事情。"){record_delimiter}
("实体"{tuple_delimiter}"假期"{tuple_delimiter}"概念"{tuple_delimiter}"涉及休息、出行、调休等相关安排的时间段概念。"){record_delimiter}
("实体"{tuple_delimiter}"存款"{tuple_delimiter}"概念"{tuple_delimiter}"影响人们出行游玩决策的因素。"){record_delimiter}
("关系"{tuple_delimiter}"上班族"{tuple_delimiter}"调休"{tuple_delimiter}"调休影响上班族的放假安排。"{tuple_delimiter}"安排影响，休息计划"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"经济困难群体 "{tuple_delimiter}"存款"{tuple_delimiter}"存款不足影响经济困难群体的出行计划。"{tuple_delimiter}"经济因素，出行限制"{tuple_delimiter}7){record_delimiter}
("关系"{tuple_delimiter}"老母亲"{tuple_delimiter}"下雨"{tuple_delimiter}"老母亲因孩子在南方下雨地区上课而揪心。"{tuple_delimiter}"亲情关联，担忧情绪"{tuple_delimiter}9){record_delimiter}
("关系"{tuple_delimiter}"高中生"{tuple_delimiter}"请假"{tuple_delimiter}"高中生实施了请假的行为。"{tuple_delimiter}"行为实施，请假事件"{tuple_delimiter}9){record_delimiter}
("关系"{tuple_delimiter}"返乡人群"{tuple_delimiter}"滦平"{tuple_delimiter}"返乡人群计划去滦平逛亲戚。"{tuple_delimiter}"地点计划，活动地点"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"农民"{tuple_delimiter}"种苞米"{tuple_delimiter}"农民打算去地里种苞米。"{tuple_delimiter}"活动意向，农业活动"{tuple_delimiter}8){record_delimiter}
("内容关键词"{tuple_delimiter}"五一，放假，调休，出行，下雨"){completion_delimiter}
#############################""",

    """示例 2:
实体类型: [人物, 地点, 时间, 事件, 概念]
文本:
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
1. **提取评论中的实体，并归类为人物、地点、时间、事件、概念等类别。**
2. **如果评论者表达了特定的身份信息（如职业、经济状况、生活方式等），请将其归类到相应的用户群体。**
3. **用户群体包括但不限于：**
   - 职业：上班族、学生、环卫工人
   - 经济状况：经济困难群体、中产阶级
   - 生活方式：旅行者、宅家人士
   - 出行意愿：返乡人群、度假人群
   - 其他社会群体

**示例输出:**
("实体"{tuple_delimiter}"特种兵"{tuple_delimiter}"人物"{tuple_delimiter}"计划去上海、苏州、南京游玩的人。"){record_delimiter}
("实体"{tuple_delimiter}"返乡人群"{tuple_delimiter}"人物"{tuple_delimiter}"表达想回湖南但没抢到票的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"环卫工人"{tuple_delimiter}"人物"{tuple_delimiter}"表明自己是环卫工的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"人物"{tuple_delimiter}"还有两天放假，因没钱哪儿都去不了的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"无固定工作者"{tuple_delimiter}"人物"{tuple_delimiter}"羡慕他人有假期，自己平时没假期的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"人物"{tuple_delimiter}"想出去玩但因没钱只能‘客厅房间五日游’的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"蹭饭人群"{tuple_delimiter}"人物"{tuple_delimiter}"思考能去哪里蹭饭的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"同胞们"{tuple_delimiter}"人物"{tuple_delimiter}"被呼吁待在家里的群体。"){record_delimiter}
("实体"{tuple_delimiter}"你们"{tuple_delimiter}"人物"{tuple_delimiter}"被‘无固定工作者’羡慕有假期的群体。"){record_delimiter}
("实体"{tuple_delimiter}"上海"{tuple_delimiter}"地点"{tuple_delimiter}"特种兵计划去游玩的城市。"){record_delimiter}
("实体"{tuple_delimiter}"苏州"{tuple_delimiter}"地点"{tuple_delimiter}"特种兵计划去游玩的城市。"){record_delimiter}
("实体"{tuple_delimiter}"南京"{tuple_delimiter}"地点"{tuple_delimiter}"特种兵计划去游玩的城市。"){record_delimiter}
("实体"{tuple_delimiter}"河北承德"{tuple_delimiter}"地点"{tuple_delimiter}"五一期间下雪的地方。"){record_delimiter}
("实体"{tuple_delimiter}"湖南"{tuple_delimiter}"地点"{tuple_delimiter}"‘返乡人群’想回去的省份。"){record_delimiter}
("实体"{tuple_delimiter}"洛阳"{tuple_delimiter}"地点"{tuple_delimiter}"晴天，被邀请去游玩的城市。"){record_delimiter}
("实体"{tuple_delimiter}"雄安"{tuple_delimiter}"地点"{tuple_delimiter}"没有雨，被邀请去的地方。"){record_delimiter}
("实体"{tuple_delimiter}"五一"{tuple_delimiter}"时间"{tuple_delimiter}"涉及假期出行安排的时间节点。"){record_delimiter}
("实体"{tuple_delimiter}"还有两天"{tuple_delimiter}"时间"{tuple_delimiter}"‘经济困难群体’即将放假的时间。"){record_delimiter}
("实体"{tuple_delimiter}"放假"{tuple_delimiter}"事件"{tuple_delimiter}"人们关注的休息安排事件，有调休等情况。"){record_delimiter}
("实体"{tuple_delimiter}"调休"{tuple_delimiter}"事件"{tuple_delimiter}"影响放假安排的事件。"){record_delimiter}
("实体"{tuple_delimiter}"假期"{tuple_delimiter}"概念"{tuple_delimiter}"与休息、出行相关的时间段概念。"){record_delimiter}
("实体"{tuple_delimiter}"路费"{tuple_delimiter}"概念"{tuple_delimiter}"出行游玩涉及的费用概念。"){record_delimiter}
("实体"{tuple_delimiter}"钱"{tuple_delimiter}"概念"{tuple_delimiter}"影响出行游玩决策的因素概念。"){record_delimiter}
#############################""",

       """示例 3:

实体类型: [人物, 地点, 时间, 事件, 概念]
文本:
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
1. **提取评论中的实体，并归类为人物、地点、时间、事件、概念等类别。**
2. **如果评论者表达了特定的身份信息（如职业、经济状况、生活方式等），请将其归类到相应的用户群体。**
3. **用户群体包括但不限于：**
   - 职业：上班族、学生、环卫工人
   - 经济状况：经济困难群体、中产阶级
   - 生活方式：旅行者、宅家人士
   - 出行意愿：返乡人群、度假人群
   - 其他社会群体

**示例输出:**
("实体"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"人物"{tuple_delimiter}"因没钱不能出去玩的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"旅行者"{tuple_delimiter}"人物"{tuple_delimiter}"邀请去洛阳玩并可帮忙订民宿的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"上班族"{tuple_delimiter}"人物"{tuple_delimiter}"因五一调休连续上班，抱怨的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"度假人群"{tuple_delimiter}"人物"{tuple_delimiter}"考虑五一去北方避雨、去城市避暑的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"宅家人士"{tuple_delimiter}"人物"{tuple_delimiter}"五一家里蹲、家门口游、客厅卧室五日游的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"旅行者"{tuple_delimiter}"人物"{tuple_delimiter}"想出去旅游但没方向且没同伴的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"上班族"{tuple_delimiter}"人物"{tuple_delimiter}"询问值班三薪情况的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"宅家人士"{tuple_delimiter}"人物"{tuple_delimiter}"在家追剧、欣赏暴风雨美景的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"旅行者"{tuple_delimiter}"人物"{tuple_delimiter}"想去丽水旅游并询问天气的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"上班族"{tuple_delimiter}"人物"{tuple_delimiter}"对放假安排有疑问的评论者。"){record_delimiter}
("实体"{tuple_delimiter}"长春"{tuple_delimiter}"地点"{tuple_delimiter}"被推荐前往，天气好无雨的城市。"){record_delimiter}
("实体"{tuple_delimiter}"洛阳"{tuple_delimiter}"地点"{tuple_delimiter}"被邀请前往，可感受千年帝都文化的城市。"){record_delimiter}
("实体"{tuple_delimiter}"塞罕坝"{tuple_delimiter}"地点"{tuple_delimiter}"下雪的地方。"){record_delimiter}
("实体"{tuple_delimiter}"云南"{tuple_delimiter}"地点"{tuple_delimiter}"不下雨，被邀请前往的省份。"){record_delimiter}
("实体"{tuple_delimiter}"北方"{tuple_delimiter}"地点"{tuple_delimiter}"五一被考虑去避雨的区域。"){record_delimiter}
("实体"{tuple_delimiter}"丽水"{tuple_delimiter}"地点"{tuple_delimiter}"有人想去并询问天气的城市。"){record_delimiter}
("实体"{tuple_delimiter}"大连"{tuple_delimiter}"地点"{tuple_delimiter}"不下雨，被邀请前往的城市。"){record_delimiter}
("实体"{tuple_delimiter}"北京"{tuple_delimiter}"地点"{tuple_delimiter}"不下雨，被邀请前往的城市。"){record_delimiter}
("实体"{tuple_delimiter}"村里水库"{tuple_delimiter}"地点"{tuple_delimiter}"有人想回去抓鱼的地方。"){record_delimiter}
("实体"{tuple_delimiter}"五一"{tuple_delimiter}"时间"{tuple_delimiter}"涉及放假、出行、调休等安排的时间节点。"){record_delimiter}
("实体"{tuple_delimiter}"今天"{tuple_delimiter}"时间"{tuple_delimiter}"评论者放假的时间。"){record_delimiter}
("实体"{tuple_delimiter}"出去玩"{tuple_delimiter}"事件"{tuple_delimiter}"部分评论者有意愿但受金钱、方向等因素限制的活动。"){record_delimiter}
("实体"{tuple_delimiter}"旅游"{tuple_delimiter}"事件"{tuple_delimiter}"部分评论者计划或期望进行的活动。"){record_delimiter}
("实体"{tuple_delimiter}"值班三薪"{tuple_delimiter}"事件"{tuple_delimiter}"评论者询问的工作相关事件。"){record_delimiter}
("实体"{tuple_delimiter}"调休"{tuple_delimiter}"事件"{tuple_delimiter}"影响放假安排，让部分评论者疲惫的事件。"){record_delimiter}
("实体"{tuple_delimiter}"放假"{tuple_delimiter}"事件"{tuple_delimiter}"与工作休息相关的事件，不同评论者情况不同。"){record_delimiter}
("实体"{tuple_delimiter}"下雪"{tuple_delimiter}"事件"{tuple_delimiter}"发生在塞罕坝的天气事件。"){record_delimiter}
("实体"{tuple_delimiter}"下雨"{tuple_delimiter}"事件"{tuple_delimiter}"影响出行旅游决策的天气事件。"){record_delimiter}
("实体"{tuple_delimiter}"抓鱼"{tuple_delimiter}"事件"{tuple_delimiter}"评论者想在村里水库进行的活动。"){record_delimiter}
("实体"{tuple_delimiter}"钱"{tuple_delimiter}"概念"{tuple_delimiter}"限制部分评论者出去玩的因素。"){record_delimiter}
("实体"{tuple_delimiter}"同伴"{tuple_delimiter}"概念"{tuple_delimiter}"部分评论者出去游玩缺乏的要素。"){record_delimiter}
("实体"{tuple_delimiter}"方向"{tuple_delimiter}"概念"{tuple_delimiter}"部分评论者出去游玩缺乏的要素。"){record_delimiter}
("实体"{tuple_delimiter}"假期"{tuple_delimiter}"概念"{tuple_delimiter}"与休息、出行相关的时间段概念。"){record_delimiter}
("实体"{tuple_delimiter}"值班薪资"{tuple_delimiter}"概念"{tuple_delimiter}"评论者关注的工作报酬概念。"){record_delimiter}
("关系"{tuple_delimiter}"经济困难群体"{tuple_delimiter}"钱"{tuple_delimiter}"经济困难群体因没钱不能出去玩。"{tuple_delimiter}"经济制约，出行意向"{tuple_delimiter}7){record_delimiter}
("关系"{tuple_delimiter}"旅行者"{tuple_delimiter}"洛阳"{tuple_delimiter}"旅行者邀请他人去洛阳玩并可帮忙订民宿。"{tuple_delimiter}"旅游邀请，地点关联"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"上班族"{tuple_delimiter}"五一"{tuple_delimiter}"上班族因五一调休连续上班。"{tuple_delimiter}"时间关联，工作安排"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"度假人群"{tuple_delimiter}"北方"{tuple_delimiter}"度假人群考虑五一去北方避雨。"{tuple_delimiter}"时间关联，出行意向"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"旅行者"{tuple_delimiter}"同伴"{tuple_delimiter}"旅行者出去游玩缺乏同伴。"{tuple_delimiter}"出行要素，缺乏关联"{tuple_delimiter}7){record_delimiter}
("关系"{tuple_delimiter}"旅行者"{tuple_delimiter}"方向"{tuple_delimiter}"旅行者出去游玩缺乏方向。"{tuple_delimiter}"出行要素，缺乏关联"{tuple_delimiter}7){record_delimiter}
("关系"{tuple_delimiter}"上班族"{tuple_delimiter}"值班三薪"{tuple_delimiter}"上班族询问值班三薪情况。"{tuple_delimiter}"工作询问，事件关联"{tuple_delimiter}7){record_delimiter}
("关系"{tuple_delimiter}"宅家人士"{tuple_delimiter}"五一"{tuple_delimiter}"宅家人士五一家里蹲。"{tuple_delimiter}"时间关联，休息安排"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"上班族"{tuple_delimiter}"今天"{tuple_delimiter}"上班族今天放假。"{tuple_delimiter}"时间关联，放假安排"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"旅行者"{tuple_delimiter}"丽水"{tuple_delimiter}"旅行者想去丽水并询问天气。"{tuple_delimiter}"出行意向，地点关联"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"长春"{tuple_delimiter}"下雨"{tuple_delimiter}"长春天气好没有雨。"{tuple_delimiter}"地点特征，天气关联"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"云南"{tuple_delimiter}"下雨"{tuple_delimiter}"云南不下雨。"{tuple_delimiter}"地点特征，天气关联"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"大连"{tuple_delimiter}"下雨"{tuple_delimiter}"大连不下雨。"{tuple_delimiter}"地点特征，天气关联"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"北京"{tuple_delimiter}"下雨"{tuple_delimiter}"北京不下雨。"{tuple_delimiter}"地点特征，天气关联"{tuple_delimiter}8){record_delimiter}
("关系"{tuple_delimiter}"塞罕坝"{tuple_delimiter}"下雪"{tuple_delimiter}"塞罕坝下雪。"{tuple_delimiter}"地点特征，天气事件"{tuple_delimiter}9){record_delimiter}
("内容关键词"{tuple_delimiter}"五一，放假，出行，天气，钱"){completion_delimiter}
#############################""",
]

PROMPTS["summarize_entity_descriptions"] = """ 你是一个乐于助人的助手，负责对以下提供的数据进行全面总结。
给定一到两个实体，以及一系列相关描述，所有这些都与同一个实体或一组实体相关。
请将所有这些描述合并成一个全面的描述。务必包含从所有描述中收集到的信息。
确保描述采用第三人称，并包含实体名称，以便我们了解完整的上下文。
输出语言使用{language}。


#######
- 数据 -
实体: {entity_name}
描述列表: {description_list}
#######
输出:
"""

PROMPTS["entiti_continue_extraction"] = """ 在上一次提取中遗漏了许多实体。请使用相同的格式在下面补充这些实体：
"""

PROMPTS["entiti_if_loop_extraction"] = """ 看起来可能仍有一些实体被遗漏了。如果还有需要添加的实体，请回答 是 | 否。
"""

PROMPTS["fail_response"] = ("抱歉，我无法回答这个问题。[no-context]")

PROMPTS["rag_response"] = """--- 角色 ---

你是一个乐于助人的助手，负责回答用户关于以下知识库的查询。

--- 目标 ---

根据知识库生成简洁的回复，并遵循回复规则，同时考虑对话历史和当前查询内容。总结所提供知识库中的所有信息，并融入与知识库相关的常识。不要包含知识库未提供的信息。

在处理带有时间戳的关系时：

    每个关系都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的关系时，同时考虑语义内容和时间戳。
    不要自动优先选择最近创建的关系，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。


--- 对话历史 ---
{history}

--- 知识库 ---
{context_data}

--- 回复规则 ---

    目标格式和长度：{response_type}
    使用 Markdown 格式，并添加适当的章节标题。
    请使用与用户问题相同的语言进行回复。
    确保回复与对话历史保持连贯。
    如果你不知道答案，请直接说明。
    不要编造内容，不要包含知识库未提供的信息。"""


PROMPTS["keywords_extraction"] = """--- 角色 ---

你是一个乐于助人的助手，负责识别用户查询和对话历史中的高级和低级关键词。

--- 目标 ---

根据查询内容和对话历史，列出高级和低级关键词。高级关键词关注总体概念或主题，而低级关键词关注特定实体、细节或具体术语。

--- 说明 ---

    -在提取关键词时，同时考虑当前查询和相关对话历史。
    -以 JSON 格式输出关键词。
    -JSON 应包含两个键：
        -"高级关键词" 用于总体概念或主题
        -"低级关键词" 用于特定实体或细节
        
######################
- 示例 -
######################
{examples}

#############################
- 实际数据 -
######################
对话历史：
{examples}

当前查询：{query}
######################
输出应为人类可读的文本，而非 Unicode 字符。保持与查询内容相同的语言。
输出：

"""

PROMPTS["keywords_extraction_examples"] = [
    """示例 1:

查询内容: "国际贸易如何影响全球经济稳定？"
################
输出:
{
"高级关键词": ["国际贸易", "全球经济稳定", "经济影响"],
"低级关键词": ["贸易协定", "关税", "货币兑换", "进口", "出口"]
}
#############################""",
    """ 示例 2:

查询内容: "森林砍伐对生物多样性有哪些环境后果？"
################
输出:
{
"高级关键词": ["环境后果", "森林砍伐", "生物多样性丧失"],
"低级关键词": ["物种灭绝", "栖息地破坏", "碳排放", "热带雨林", "生态系统"]
}
#############################""",
    """ 示例 3:

查询内容: "教育在减少贫困方面起到什么作用？"
################
输出:
{
"高级关键词": ["教育", "减贫", "社会经济发展"],
"低级关键词": ["入学机会", "识字率", "职业培训", "收入不平等"]
}
#############################"""
]

PROMPTS["naive_rag_response"] = """--- 角色 ---

你是一个乐于助人的助手，负责回答用户关于以下文档片段的查询。

--- 目标 ---

根据文档片段生成简洁的回复，并遵循回复规则，同时考虑对话历史和当前查询内容。总结所提供文档片段中的所有信息，并融入与文档片段相关的常识。不要包含文档片段未提供的信息。

在处理带有时间戳的内容时：

    每一段内容都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的信息时，同时考虑内容和时间戳。
    不要自动优先选择最新的内容，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。


--- 对话历史 ---
{history}

--- 文档片段 ---
{content_data}

--- 回复规则 ---

    目标格式和长度：{response_type}
    使用 Markdown 格式，并添加适当的章节标题。
    请使用与用户问题相同的语言进行回复。
    确保回复与对话历史保持连贯。
    如果你不知道答案，请直接说明。
    不要包含文档片段未提供的信息。"""


PROMPTS["similarity_check"] = """ 请分析以下两个问题的相似度：

问题 1: {original_prompt}
问题 2: {cached_prompt}

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

PROMPTS["mix_rag_response"] = """--- 角色 ---

你是一个乐于助人的助手，负责回答用户关于以下数据源的查询。

--- 目标 ---

根据数据源生成简洁的回复，并遵循回复规则，同时考虑对话历史和当前查询内容。数据源包含两部分：知识图谱（KG）和文档片段（DC）。总结所提供数据源中的所有信息，并融入与数据源相关的常识。不要包含数据源未提供的信息。

在处理带有时间戳的信息时：

    每一条信息（包括关系和内容）都有一个 “创建时间” 时间戳，表明我们获取该知识的时间。
    当遇到相互冲突的信息时，同时考虑内容 / 关系和时间戳。
    不要自动优先选择最新的信息，要根据上下文进行判断。
    对于特定时间的查询，在考虑创建时间戳之前，优先考虑内容中的时间信息。


--- 对话历史 ---
{history}

--- 数据源 ---

    1.来自知识图谱（KG）：
    {kg_context}
    2.来自文档片段（DC）：
    {vector_context}


--- 回复规则 ---

    目标格式和长度：{response_type}
    使用 Markdown 格式，并添加适当的章节标题。
    请使用与用户问题相同的语言进行回复。
    确保回复与对话历史保持连贯。
    将答案按主要观点或方面分节组织。
    使用清晰且能反映内容的描述性章节标题。
    必须在结尾的 “参考文献” 部分列出全部参考来源。必须明确指出每个来源是来自知识图谱（KG）还是向量数据（DC），格式如下：[KG/DC] 来源内容
    如果你不知道答案，请直接说明。不要编造任何内容。
    不要包含数据源未提供的信息。"""







