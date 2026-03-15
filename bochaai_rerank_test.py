import requests
import re
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


def gradient_based_chunk_selection(chunks, scores, min_k, gradient_threshold):
    sorted_data = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    sorted_chunks = [x[0] for x in sorted_data]
    sorted_scores = [x[1] for x in sorted_data]

    selected_chunks = []
    selected_scores = []

    for i in range(len(sorted_scores)):
        if i == 0:
            selected_chunks.append(sorted_chunks[i])
            selected_scores.append(sorted_scores[i])
        else:
            score_diff = sorted_scores[i] - sorted_scores[i - 1]
            if score_diff > gradient_threshold:
                selected_chunks.append(sorted_chunks[i])
                selected_scores.append(sorted_scores[i])
            else:
                break

    if len(selected_chunks) < min_k:
        selected_chunks = sorted_chunks[:min_k]
        selected_scores = sorted_scores[:min_k]

    return selected_chunks, selected_scores


api_key = "sk-7f18abe89eb445648a5a20b9077c926d"
url = "https://api.bochaai.com/v1/rerank"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

query = """
评论者如何看待假期出行难的问题？
"""

original_document = """
-----Entities-----id,	entity,	type,	description,	rank
1,	"停车","概念","在路边或地下室进行的行为。"<SEP>"将车辆停放。"<SEP>"车辆停放在某个位置的行为。"<SEP>停车是一种涉及车辆停放的行为，这一概念不仅涵盖了车辆停放在指定地点的动作，还包括寻找停车位、支付停车费用、遵守停车规定等多个环节。停车行为在城市生活中扮演着重要角色，与个人出行、社会生活、经济活动和城市管理的多个层面紧密相关。

停车服务在不同地区实施情况各异，如中山陵、洛阳等地实施了停车服务，江苏淮安、银泰中心、酒店和旅游景点等地也体现了停车服务的多样性和复杂性。停车费用受市场供求关系影响，如旅游旺季时费用上涨，同时停车难问题也成为社会广泛关注的问题。在江浙沪地区，停车服务呈现出较高的费用，而皖地则相对较低。

停车行为不仅包括车辆停放在指定地点的动作，还涵盖了电动车停放需要支付费用的行为，例如杭州商场和小区停车事件，以及摩托车和电动车在关键车位上的停放。淄博对外地车辆不收费的停车行为，以及评论者住宿酒店停车费用的概念，进一步丰富了停车行为的内涵。

评论者在上海、济南、西安等地经历的停车事件，以及提到的在村子里需要停车的情况，都反映了停车行为在不同场合下的表现形式。在许昌等地，停车行为可能指车辆停放的位置，也可能涉及停车费用。停车这个行为，指在商场等场所的停车行为，或是在评论中提到的行为活动，涉及车辆停放的行为或活动。

综上所述，停车已成为城市生活中不可或缺的一部分，其相关活动对于城市管理和居民生活具有重要意义。<SEP>"评论者在西湖附近发生的停车事件。"<SEP>"车辆在服务区停留的行为。"<SEP>"车辆停止行驶并停放的概念。",1
2,	"自驾","活动","指通过自驾车进行旅行或游玩的活动。"<SEP>"评论者提到的出行方式，即自驾游。"<SEP>自驾作为一种出行方式，主要是指个人或团体自行驾驶车辆进行旅行或出行活动的行为，包括自驾游、租车自驾等多种形式，通常涉及驾驶私家车或汽车进行旅行。评论者对自驾出行有着深刻的体验和认识，其提及的自驾活动涵盖了多个方面，如十年前的活动、前往灵隐寺、宁波旅游计划等，体现了其对自由出行和探索的态度。

评论者对自驾的便利性表示认可，但也指出存在一些问题，如停车不便、停车费用高等，尤其是在一线城市和二线城市。尽管如此，自驾仍是一种常见的出行方式，尤其在清明节期间与朋友出游时，如前往千岛湖的游玩方式。

评论者及其同伴在选择自驾出行时，会考虑到停车问题，并在实际出行中注意驾驶行为的安全。自驾游也被视为一种私人汽车驾驶旅游，与浙江旅游形成对比，显示出自驾游的独特性。

综合以上描述，自驾作为一种出行方式，既受到人们的青睐，也面临一些挑战，但它在人们的生活中扮演着重要的角色。<SEP>"评论者计划采用的出行方式概念。"<SEP>"评论者计划采用的出行方式，即自己开车。",1
-----Relationships-----id,	source,	target,	description,	keywords,	weight,	rank,	created_at
3,	"停车","自驾","停车问题是自驾出行时担忧的问题。"<SEP>"自驾出行涉及停车问题，评论者认为停车不方便。","出行方式，相关问题"<SEP>"出行问题，担忧关联",15.0,2,
4,	"假期","路桥费","路桥费影响了假期的出行和体验。","费用影响，出行体验",14.0,2,
5,	"放假","票","放假期间票务紧张，导致难以购买。"<SEP>"放假需要购买来回的票，增加了出行的成本。","假期安排，出行成本"<SEP>"时间关联，出行限制",13.0,2,
6,	"假期","直接放或者不放","评论者认为理想的假期安排应该直接放或者不放。","理想假期，政策建议",9.0,2,
7,	"开车出门","评论者","评论者去年五一不敢开车出门。","出行限制，活动关联",8.0,2,

-----Text_units-----id,	content
8,	出去自驾停车真的超不方便
9,	出去玩我觉得坐动车挺好自驾是方便就怕停车不方便哈哈哈
10,	免得是路桥费赌车浪费的是有钱也买不到的黄金般假期时间
11,	你就说一放假啥都贵人贼多买票都买不上玩个屁
12,	我觉得还不如不放周六日正常就行放个假还得整来回的票路上两天
13,	国家快点改革吧其实就是放一天假而已还不如不调休直接放或者不放
14,	去年五一开车去北京车进入宾馆停车场的时刻它的功能就结束了后续游玩儿全程打车地铁完全不敢开车出门    """

paragraphs = re.findall(r'^\d+\,.*?(?=^\d+\,|\Z)', original_document, re.MULTILINE | re.DOTALL)
paragraph_scores = []

for i, paragraph in enumerate(paragraphs):
    data = {
        "model": "gte-rerank",
        "query": query,
        "top_n": 1,
        "return_documents": True,
        "documents": [paragraph]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        score = result['data']['results'][0]['relevance_score']
        paragraph_scores.append((paragraph, score))
        print(f"段落 {i + 1} 的分数: {score}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP 错误发生: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"请求错误发生: {req_err}")

chunks = [p[0] for p in paragraph_scores]
scores = [p[1] for p in paragraph_scores]

min_k = 12
gradient_threshold = -0.01
selected_chunks, selected_scores = gradient_based_chunk_selection(chunks, scores, min_k, gradient_threshold)

print("\n选择的段落及其分数：")
for i, score in enumerate(selected_scores, start=1):
    print(f"段落 {i} 的分数: {score}")

sorted_data = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
sorted_chunks = [x[0] for x in sorted_data]
sorted_scores = [x[1] for x in sorted_data]

print("\n重排后的段落及其分数：")
for i, score in enumerate(sorted_scores, start=1):
    print(f"段落 {i} 的分数: {score}")

plt.plot(range(len(sorted_scores)), sorted_scores, marker='o', label="段落分数")
plt.axvline(x=len(selected_scores) - 1, color='r', linestyle='--', label="梯度选择点")
plt.xlabel("段落序号")
plt.ylabel("分数")
plt.title("段落分数梯度变化（降序）")
plt.legend()
plt.show()
