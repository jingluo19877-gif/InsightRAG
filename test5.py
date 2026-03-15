import requests
import json


class OllamaChat():
    def __init__(self, system_message="你的名字叫做minglog，是一个由骆明开发的大语言模型。",
                 url="http://localhost:6006/api/chat", model_name="qwen2.5:14b"):
        """
            url: ChatModel API. Default is Local Ollama
                # url = "http://localhost:6006/api/chat"  # AutoDL
                # url = "http://localhost:11434/api/chat"  # localhost
            model_name: ModelName.
                Default is Qwen:7b
        """
        self.url = url
        self.model_name = model_name
        self.system_message = {
            "role": "system",
            "content": f"""{system_message}"""
        }
        self.message = [self.system_message]

    def __ouput_response(self, response, stream=False, is_chat=True):
        if stream:
            return_text = ''
            # 创建JSON解析器
            decoder = json.JSONDecoder()
            buffer = ''

            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    buffer += decoded_chunk

                    # 尝试解析多个JSON对象
                    while buffer:
                        try:
                            # 尝试从缓存中解码一个JSON对象
                            obj, index = decoder.raw_decode(buffer)
                            # 根据is_chat选择提取相应字段
                            if is_chat:
                                text = obj['message']['content'] if 'message' in obj else ''
                            else:
                                text = obj['response'] if 'response' in obj else ''
                            return_text += text
                            buffer = buffer[index:].lstrip()  # 去掉已经解析的部分
                        except ValueError:
                            # 如果没有完整的JSON对象，继续读取更多数据
                            break

            return return_text
        else:
            return_text = ""
            try:
                # 处理多个JSON对象
                decoder = json.JSONDecoder()
                buffer = response.text
                while buffer:
                    try:
                        obj, index = decoder.raw_decode(buffer)
                        if is_chat:
                            text = obj['message']['content'] if 'message' in obj else ''
                        else:
                            text = obj['response'] if 'response' in obj else ''
                        return_text += text
                        buffer = buffer[index:].lstrip()
                    except ValueError:
                        break

                # 提取第一个单词作为最终结果
                words = return_text.strip().split()
                if words:
                    return_text = words[0]
                else:
                    return_text = '无法解析返回数据'
            except Exception as e:
                print(f"解析JSON时出错: {e}")
                print(f"响应内容: {response.text}")
                return_text = '无法解析返回数据'

            return return_text

    def chat(self, prompt, message=None, stream=False, system_message=None, **options):
        """
            prompt: Type Str, User input prompt words.
            messages: Type List, Dialogue History. role in [system, user, assistant]
            stream: Type Boolean, Is it streaming output. if `True` streaming output, otherwise not streaming output.
            system_message: Type Str, System Prompt. Default self.system_message.
            **options: option items.
        """
        if message is not None:
            self.message = message
        if message == []:
            self.message.append(self.system_message)
        if system_message:
            self.message[0]['content'] = system_message
        self.message.append({"role": "user", "content": prompt})
        if 'max_tokens' in options:
            options['num_ctx'] = options['max_tokens']
        data = {
            "model": self.model_name,
            "messages": self.message,
            "options": options
        }
        headers = {
            "Content-Type": "application/json"
        }
        responses = requests.post(self.url, headers=headers, json=data, stream=stream)
        return_text = self.__ouput_response(responses, stream)
        self.message.append({"role": "assistant", "content": return_text})
        return return_text, self.message

    def generate(self, prompt, stream=False, **options):
        generate_url = self.url.replace('chat', 'generate')
        if 'max_tokens' in options:
            options['num_ctx'] = options['max_tokens']
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "options": options
        }
        headers = {
            "Content-Type": "application/json"
        }
        responses = requests.post(generate_url, headers=headers, json=data, stream=stream)
        return_text = self.__ouput_response(responses, stream, is_chat=False)
        return return_text


# 假设评论数据在这个文件中
file_path = r"E:\毕业设计\cleaned_reviews\cleaned_reviews_2.txt"
output_file_path = r"E:\毕业设计\cleaned_reviews\filtered_reviews_2.txt"  # 新的文件路径


def filter_reviews_with_entities_and_relations(file_path, output_file_path):
    """
    筛选包含实体和关系的评论
    """
    Chat = OllamaChat()

    with open(file_path, "r", encoding="utf-8") as f:
        reviews = f.readlines()

    valid_reviews = []

    for i, review in enumerate(reviews):  # 只处理前50条评论
        review = review.strip()
        if not review:
            continue

        # 使用LLM判断评论是否包含可用于知识图谱的实体和关系
        prompt = (
            f"请仔细判断以下抖音用户评论中是否包含可用于构建知识图谱的‘实体’和‘关系’。"
            f"以下为常见实体类型及其定义与丰富示例："
            f"1. 人物：指具有明确身份、可被识别的个体，包括但不限于历史人物（如孔子、拿破仑）、公众人物（如明星、运动员）、专家学者（如霍金、屠呦呦）、企业高管（如马斯克、董明珠）等；也涵盖评论中以泛称、昵称、描述性称呼出现的人物，像‘那位热心肠的大哥’‘隔壁可爱的小姐姐’，甚至是通过行为或特征暗示存在的人物，例如‘帮忙修水管的师傅’‘在路边卖花的人’。\n"
            f"2. 组织机构：涵盖正式注册的各类团体，如企业（苹果公司、阿里巴巴集团）、政府部门（国家税务总局、教育局）、学校（清华大学、哈佛大学）、科研机构（中国科学院、美国国家航空航天局）、非营利组织（红十字会、绿色和平组织）；还包括临时组建、未正式登记的团体，如‘本次自驾游的车队’‘小区自发组织的抗疫志愿队’。\n"
            f"3. 地点：包含精确的地理名称，如国家（中国、美国）、城市（北京、纽约）、地区（长三角地区、珠三角地区）、景点（故宫、埃菲尔铁塔）、建筑（鸟巢、白宫）；也包括相对模糊但有实际指向的地点表述，像‘经常去的那家商场附近’‘村子东边的那片树林’，以及虚拟空间中的地点，如‘某网络游戏里的新手村’‘社交平台上的热门群组空间’。\n"
            f"4. 产品：包括有形的实物商品，如电子产品（iPhone 手机、华为电脑）、生活用品（牙刷、毛巾）、交通工具（汽车、自行车）；无形的软件、服务等也在此列，如操作系统（Windows、macOS）、在线教育服务、外卖配送服务；对于评论中未明确提及名称，但可根据语境推断出的产品，如‘刚入手的那个智能设备’‘在网上买的好吃的零食’，也应视为产品实体。\n"
            f"5. 事件：涉及大型的、具有广泛影响力的事件，如历史事件（第二次世界大战、美国独立战争）、体育赛事（奥运会、世界杯）、文化活动（春节联欢晚会、戛纳电影节）；也包括日常生活中的各类小事件，如‘小区举办的亲子活动’‘朋友间的家庭聚会’‘在公园进行的晨练活动’。\n"
            f"6. 概念：指抽象的、具有一定内涵和外延的知识或观念，包括学术领域的概念（人工智能、量子力学）、生活中的理念（低碳生活、可持续发展）、社会现象（内卷、躺平）等。\n"
            f"7. 学科：自然科学领域（物理学、化学、生物学）和社会科学领域（社会学、经济学、心理学）的各类学科。\n"
            f"8. 时间：包含精确的时间点（2024 年 10 月 1 日、上午 9 点）、时间段（20 世纪、暑假）；也有模糊的时间表述，如‘很久以前’‘最近这段日子’‘周末的时候’。\n"
            f"9. 疾病：医学范畴内的各种病症，如感冒、癌症、糖尿病等。\n"
            f"10. 书籍：包括文学作品（《红楼梦》《哈姆雷特》）、学术著作（《资本论》《时间简史》）、教材（高中数学教材、大学物理教材）等。\n"
            f"11. 电影：各类影视作品，如《泰坦尼克号》《流浪地球 2》等。\n"
            f"12. 音乐：涉及歌曲（《青花瓷》《月亮代表我的心》）、音乐专辑（周杰伦的《范特西》、林俊杰的《新地球》）、音乐家（贝多芬、周杰伦）等相关实体。\n"
            f"以下是常见关系类型及其详细阐释与多样示例："
            f"1. 属于：表示实体归属于某个特定的类别、集合或范畴，除了明确的分类归属，如 “苹果公司属于企业”“猫属于哺乳动物”，还包括宽泛意义上的从属关系，如‘这个创意属于团队集体智慧的成果’‘这种行为属于不文明现象’。\n"
            f"2. 包含：不仅表示整体与部分之间的物理包含关系，如 “中国包含多个省份”“一本书包含多个章节”，还涵盖内容上的涵盖关系，如‘这条新闻包含政治、经济、文化等多方面信息’‘这个软件包含多种实用功能’。\n"
            f"3. 创作：体现创作者与所创作作品之间的关系，如 “曹雪芹创作了《红楼梦》”“达芬奇创作了《蒙娜丽莎》”，也可拓展到创作相关的行为活动，如‘程序员编写了这个新的应用程序’‘摄影师拍摄了一组精彩的照片’。\n"
            f"4. 参演：用于描述演员与影视作品之间的参与关系，如 “莱昂纳多・迪卡普里奥参演了《泰坦尼克号》”“吴京参演了《战狼》系列电影”。\n"
            f"5. 成立：用于说明组织机构的创建时间、地点等关键信息，如 “苹果公司成立于 1976 年”“某志愿者协会成立于社区活动中心”。\n"
            f"6. 发生：表示事件与时间、地点之间的关联，如 “奥运会发生在特定的城市和时间”“交通事故发生在十字路口”，对于日常小事件同样适用，如‘家庭聚会发生在周末晚上’。\n"
            f"7. 治疗：在医学领域，描述疾病与治疗方法、药物之间的关系，如 “药物 X 可以治疗感冒”“针灸疗法可用于治疗颈椎病”。\n"
            f"8. 学习：体现人物与学科、知识概念、技能等之间的学习行为关系，如 “学生学习数学”“上班族利用业余时间学习编程技能”。\n"
            f"9. 研究：表示科研人员、机构与学科、概念之间的探索、钻研关系，如 “科学家研究人工智能”“研究团队对气候变化进行研究”，也可宽泛到普通人的探索性行为，如‘天文爱好者研究星座奥秘’。\n"
            f"10. 生产：表示企业、工厂、个人等与所生产产品之间的关系，如 “特斯拉生产电动汽车”“手工艺人制作手工艺品”，包括各种形式的制造、生产活动。\n"
            f"11. 关联：这是一种较为宽泛的关系，表明两个实体之间存在某种联系，如 “人工智能与机器学习密切关联”“环境污染与工业发展存在关联”，还包括潜在的因果、伴随、对比等关系，如‘暴雨天气和航班延误有关联’‘消费升级与经济发展水平提升相关联’。\n"
            f"12. 时间先后：用于描述事件或时间之间的先后顺序关系，如 “第一次世界大战发生在第二次世界大战之前”“早餐时间在午餐时间之前”。\n"
            f"13. 喜爱：表达人物对实体（如产品、电影、音乐、人物等）的积极情感态度，例如 “小李喜爱《青花瓷》这首歌”“小张喜爱苹果手机”“小王喜爱篮球明星詹姆斯”。\n"
            f"14. 厌恶：体现人物对实体（如产品、事件、行为等）的消极情感态度，例如 “小王厌恶这部无聊的电影”“小赵厌恶这家服务差劲的餐厅”“小李厌恶插队这种不文明行为”。\n"
            f"15. 中立评价：表示人物对实体的客观、无明显情感倾向的态度描述，例如 “小孙对这个学术概念持中立评价，只是简单提及”“小周对这次活动进行了客观中立的描述”。\n"
            f"请你基于以上全面且详细的实体和关系类型说明，对以下抖音用户评论进行严格但不失灵活的分析。若评论中包含上述类型的实体或关系，或相关的潜在信息，请只返回‘是’；若确实不包含，则只返回‘否’\n{review}")

        response, _ = Chat.chat(prompt)
        print(response)  # 打印返回结果

        # 判断返回结果
        if response == "是":
            valid_reviews.append(review)

    # 将有效评论写入新的文件
    with open(output_file_path, "w", encoding="utf-8") as f:
        for review in valid_reviews:
            f.write(review + '\n')

    # 删除不再使用的变量
    del reviews, valid_reviews, Chat


# 执行筛选
filter_reviews_with_entities_and_relations(file_path, output_file_path)
