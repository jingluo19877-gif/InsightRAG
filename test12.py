import os
import time
import mysql.connector
from zhipuai import ZhipuAI
from dotenv import load_dotenv

# 实体和关系类型说明常量
ENTITY_TYPES = """
1. 人物：指具有明确身份、可被识别的个体，包括但不限于历史人物（如孔子、拿破仑）、公众人物（如明星、运动员）、专家学者（如霍金、屠呦呦）、企业高管（如马斯克、董明珠）等；也涵盖评论中以泛称、昵称、描述性称呼出现的人物，像‘那位热心肠的大哥’‘隔壁可爱的小姐姐’，甚至是通过行为或特征暗示存在的人物，例如‘帮忙修水管的师傅’‘在路边卖花的人’。
2. 组织机构：涵盖正式注册的各类团体，如企业（苹果公司、阿里巴巴集团）、政府部门（国家税务总局、教育局）、学校（清华大学、哈佛大学）、科研机构（中国科学院、美国国家航空航天局）、非营利组织（红十字会、绿色和平组织）；还包括临时组建、未正式登记的团体，如‘本次自驾游的车队’‘小区自发组织的抗疫志愿队’。
3. 地点：包含精确的地理名称，如国家（中国、美国）、城市（北京、纽约）、地区（长三角地区、珠三角地区）、景点（故宫、埃菲尔铁塔）、建筑（鸟巢、白宫）；也包括相对模糊但有实际指向的地点表述，像‘经常去的那家商场附近’‘村子东边的那片树林’，以及虚拟空间中的地点，如‘某网络游戏里的新手村’‘社交平台上的热门群组空间’。
4. 产品：包括有形的实物商品，如电子产品（iPhone 手机、华为电脑）、生活用品（牙刷、毛巾）、交通工具（汽车、自行车）；无形的软件、服务等也在此列，如操作系统（Windows、macOS）、在线教育服务、外卖配送服务；对于评论中未明确提及名称，但可根据语境推断出的产品，如‘刚入手的那个智能设备’‘在网上买的好吃的零食’，也应视为产品实体。
5. 事件：涉及大型的、具有广泛影响力的事件，如历史事件（第二次世界大战、美国独立战争）、体育赛事（奥运会、世界杯）、文化活动（春节联欢晚会、戛纳电影节）；也包括日常生活中的各类小事件，如‘小区举办的亲子活动’‘朋友间的家庭聚会’‘在公园进行的晨练活动’。
6. 概念：指抽象的、具有一定内涵和外延的知识或观念，包括学术领域的概念（人工智能、量子力学）、生活中的理念（低碳生活、可持续发展）、社会现象（内卷、躺平）等。
7. 学科：自然科学领域（物理学、化学、生物学）和社会科学领域（社会学、经济学、心理学）的各类学科。
8. 时间：包含精确的时间点（2024 年 10 月 1 日、上午 9 点）、时间段（20 世纪、暑假）；也有模糊的时间表述，如‘很久以前’‘最近这段日子’‘周末的时候’。
9. 疾病：医学范畴内的各种病症，如感冒、癌症、糖尿病等。
10. 书籍：包括文学作品（《红楼梦》《哈姆雷特》）、学术著作（《资本论》《时间简史》）、教材（高中数学教材、大学物理教材）等。
11. 电影：各类影视作品，如《泰坦尼克号》《流浪地球 2》等。
12. 音乐：涉及歌曲（《青花瓷》《月亮代表我的心》）、音乐专辑（周杰伦的《范特西》、林俊杰的《新地球》）、音乐家（贝多芬、周杰伦）等相关实体。
"""

RELATION_TYPES = """
1. 属于：表示实体归属于某个特定的类别、集合或范畴，除了明确的分类归属，如 “苹果公司属于企业”“猫属于哺乳动物”，还包括宽泛意义上的从属关系，如‘这个创意属于团队集体智慧的成果’‘这种行为属于不文明现象’。
2. 包含：不仅表示整体与部分之间的物理包含关系，如 “中国包含多个省份”“一本书包含多个章节”，还涵盖内容上的涵盖关系，如‘这条新闻包含政治、经济、文化等多方面信息’‘这个软件包含多种实用功能’。
3. 创作：体现创作者与所创作作品之间的关系，如 “曹雪芹创作了《红楼梦》”“达芬奇创作了《蒙娜丽莎》”，也可拓展到创作相关的行为活动，如‘程序员编写了这个新的应用程序’‘摄影师拍摄了一组精彩的照片’。
4. 参演：用于描述演员与影视作品之间的参与关系，如 “莱昂纳多・迪卡普里奥参演了《泰坦尼克号》”“吴京参演了《战狼》系列电影”。
5. 成立：用于说明组织机构的创建时间、地点等关键信息，如 “苹果公司成立于 1976 年”“某志愿者协会成立于社区活动中心”。
6. 发生：表示事件与时间、地点之间的关联，如 “奥运会发生在特定的城市和时间”“交通事故发生在十字路口”，对于日常小事件同样适用，如‘家庭聚会发生在周末晚上’。
7. 治疗：在医学领域，描述疾病与治疗方法、药物之间的关系，如 “药物 X 可以治疗感冒”“针灸疗法可用于治疗颈椎病”。
8. 学习：体现人物与学科、知识概念、技能等之间的学习行为关系，如 “学生学习数学”“上班族利用业余时间学习编程技能”。
9. 研究：表示科研人员、机构与学科、概念之间的探索、钻研关系，如 “科学家研究人工智能”“研究团队对气候变化进行研究”，也可宽泛到普通人的探索性行为，如‘天文爱好者研究星座奥秘’。
10. 生产：表示企业、工厂、个人等与所生产产品之间的关系，如 “特斯拉生产电动汽车”“手工艺人制作手工艺品”，包括各种形式的制造、生产活动。
11. 关联：这是一种较为宽泛的关系，表明两个实体之间存在某种联系，如 “人工智能与机器学习密切关联”“环境污染与工业发展存在关联”，还包括潜在的因果、伴随、对比等关系，如‘暴雨天气和航班延误有关联’‘消费升级与经济发展水平提升相关联’。
12. 时间先后：用于描述事件或时间之间的先后顺序关系，如 “第一次世界大战发生在第二次世界大战之前”“早餐时间在午餐时间之前”。
13. 喜爱：表达人物对实体（如产品、电影、音乐、人物等）的积极情感态度，例如 “小李喜爱《青花瓷》这首歌”“小张喜爱苹果手机”“小王喜爱篮球明星詹姆斯”。
14. 厌恶：体现人物对实体（如产品、事件、行为等）的消极情感态度，例如 “小王厌恶这部无聊的电影”“小赵厌恶这家服务差劲的餐厅”“小李厌恶插队这种不文明行为”。
15. 中立评价：表示人物对实体的客观、无明显情感倾向的态度描述，例如 “小孙对这个学术概念持中立评价，只是简单提及”“小周对这次活动进行了客观中立的描述”。
"""

# 加载 .env 文件中的环境变量
load_dotenv()
api_key = os.environ.get("ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)


def filter_reviews_with_entities_and_relations():
    try:
        # 连接到 rag_test 数据库
        test_conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="0000",
            database="rag_test",
            charset='utf8mb4',
            collation='utf8mb4_unicode_ci'
        )
        test_cursor = test_conn.cursor()

        # 连接到 rag_deal 数据库
        deal_conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="0000",
            database="rag_deal",
            charset='utf8mb4',
            collation='utf8mb4_unicode_ci'
        )
        deal_cursor = deal_conn.cursor()

        # 获取总记录数
        test_cursor.execute("SELECT COUNT(*) FROM reviews_id")
        total_records = test_cursor.fetchone()[0]

        # 从第 4546 条记录开始处理，偏移量为 4545（因为记录编号从 0 开始）
        offset = 4545
        batch_size = 100

        while True:
            # 查询 rag_test 数据库中 reviews_id 表的指定范围记录
            test_cursor.execute(f"SELECT id, contents, cid FROM reviews_id LIMIT {batch_size} OFFSET {offset}")
            records = test_cursor.fetchall()

            if not records:
                break

            for record in records:
                record_id, contents, cid = record
                # 使用LLM判断评论是否包含可用于知识图谱的实体和关系
                prompt = (
                    f"请仔细判断以下抖音用户评论中是否包含可用于构建知识图谱的‘实体’和‘关系’。"
                    f"以下为常见实体类型及其定义与丰富示例："
                    f"{ENTITY_TYPES}"
                    f"以下是常见关系类型及其详细阐释与多样示例："
                    f"{RELATION_TYPES}"
                    f"请你基于以上全面且详细的实体和关系类型说明，对以下抖音用户评论进行严格但不失灵活的分析。若评论中包含上述类型的实体或关系，或相关的潜在信息，请只返回‘是’；若确实不包含，则只返回‘否’\n{contents}")

                messages = [{"role": "user", "content": prompt}]
                try:
                    response = client.chat.completions.create(
                        model="GLM-4-Flash",
                        messages=messages
                    )
                    response_text = response.choices[0].message.content.strip()
                    print(response_text)  # 打印返回结果

                    # 判断返回结果
                    if response_text == "是":
                        insert_query = "INSERT INTO reviews (id, contents, cid) VALUES (%s, %s, %s)"
                        try:
                            deal_cursor.execute(insert_query, (record_id, contents, cid))
                            deal_conn.commit()
                            print(f"记录 {record_id} 插入成功。")
                        except mysql.connector.Error as insert_error:
                            print(f"插入记录 {record_id} 时出错: {insert_error}")
                            deal_conn.rollback()
                except Exception as e:
                    print(f"请求发生错误: {e}")

            offset += batch_size
            # 计算处理进度
            progress = ((offset + 1) / total_records) * 100 if total_records > 0 else 100
            print(f"已处理 {offset + 1} 条记录，处理进度: {progress:.2f}%，休息 30 秒...")
            time.sleep(30)

    except mysql.connector.Error as db_error:
        print(f"数据库操作出错: {db_error}")
    except Exception as general_error:
        print(f"发生未知错误: {general_error}")
    finally:
        # 关闭数据库连接
        if 'test_cursor' in locals():
            test_cursor.close()
        if 'test_conn' in locals():
            test_conn.close()
        if 'deal_cursor' in locals():
            deal_cursor.close()
        if 'deal_conn' in locals():
            deal_conn.close()


# 执行筛选
filter_reviews_with_entities_and_relations()
