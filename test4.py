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
file_path = r"E:\毕业设计\cleaned_reviews\cleaned_reviews_1.txt"
output_file_path = r"E:\毕业设计\cleaned_reviews\filtered_reviews_1.txt"  # 新的文件路径


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
        prompt = (f"请判断以下用户评论中是否包含可用于构建高质量知识图谱的‘实体’和‘关系’，"
                  f"首先明确常见的实体类型："
                  f"人物：包括历史人物、公众人物、专家学者、企业高管等各种角色，如牛顿、马云、钟南山等。\n"
                  f"组织机构：涵盖公司企业、政府部门、学校、科研机构、非营利组织等，例如苹果公司、中国科学院、北京大学等。\n"
                  f"地点：包含国家、城市、地区、景点、建筑等，像中国、北京市、故宫博物院等。\n"
                  f"产品：涉及各类商品、软件、硬件设备等，如 iPhone 手机、Windows 操作系统、特斯拉汽车等。\n"
                  f"事件：包括历史事件、体育赛事、文化活动、政治事件等，例如奥运会、第二次世界大战、春节联欢晚会等。\n"
                  f"概念：抽象的知识概念，如人工智能、机器学习、经济学原理等。\n"
                  f"学科：自然科学、社会科学等领域的学科，如物理学、化学、社会学等。\n"
                  f"时间：具体的时间点、时间段，如 2024 年、上午 9 点、夏季等。\n"
                  f"疾病：医学领域中的各种疾病，如感冒、肺炎、糖尿病等。\n"
                  f"书籍：文学作品、学术著作、教材等，如《红楼梦》《物种起源》等。\n"
                  f"电影：各类影视作品，如《泰坦尼克号》《流浪地球》等。\n"
                  f"音乐：歌曲、音乐专辑、音乐家等相关实体，如《青花瓷》、周杰伦的《范特西》专辑等。\n"
                  f"其次明确常见的关系类型：\n"
                  f"属于：表示实体属于某个类别或集合，如 “苹果公司属于企业”“狗属于哺乳动物”。\n"
                  f"包含：用于描述整体与部分的关系，如 “中国包含北京市”“一本书包含多个章节”。\n"
                  f"创作：体现创作者与作品之间的关系，如 “曹雪芹创作了《红楼梦》”“贝多芬创作了《命运交响曲》”。\n"
                  f"参演：用于描述演员与影视作品之间的关系，如 “莱昂纳多・迪卡普里奥参演了《泰坦尼克号》”。\n"
                  f"成立：表示组织机构的成立时间或地点等信息，如 “苹果公司成立于 1976 年”。\n"
                  f"发生：用于描述事件发生的地点或时间，如 “奥运会发生在特定的城市和时间”。\n"
                  f"治疗：在医学领域，描述疾病与治疗方法之间的关系，如 “药物 X 可以治疗感冒”。\n"
                  f"学习：体现人物与学科或知识概念之间的关系，如 “学生学习数学”。\n"
                  f"研究：用于描述科研人员、机构与学科、概念之间的关系，如 “科学家研究人工智能”。\n"
                  f"生产：表示企业或工厂与产品之间的关系，如 “特斯拉生产电动汽车”。\n"
                  f"关联：较为宽泛的关系，表示两个实体之间存在某种联系，如 “人工智能与机器学习相关联”。\n"
                  f"时间先后：用于描述事件或时间之间的先后顺序，如 “第一次世界大战发生在第二次世界大战之前”。\n"
                  f"喜爱：表达人物对实体（如产品、电影、音乐等）的积极情感关系，例如 “小李喜爱《青花瓷》这首歌”“小张喜爱苹果手机”。”。\n"
                  f"厌恶：体现人物对实体（如产品、事件等）的消极情感关系，例如 “小王厌恶这部无聊的电影”“小赵厌恶这家服务差劲的餐厅”。\n"
                  f"中立评价：表示人物对实体的客观态度，不带有明显的积极或消极情感倾向，例如 “小孙对这个学术概念持中立评价，只是简单提及”“小周对这次活动进行了中立的描述”。\n"
                  f"请你基于以上实体和关系类型的说明，对以下抖音用户评论进行分析。若评论中包含上述类型的实体或关系，或相关的信息，请返回‘是’；若不包含，则返回‘否’”。\n{review}")

        response, _ = Chat.chat(prompt)
        print(response)  # 打印返回结果

        # 判断返回结果
        if response == "是":
            valid_reviews.append(review)

    # 将有效评论写入新的文件
    with open(output_file_path, "w", encoding="utf-8") as f:
        for review in valid_reviews:
            f.write(review + '\n')


# 执行筛选
filter_reviews_with_entities_and_relations(file_path, output_file_path)

