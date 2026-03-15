import requests
import re
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
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
评论者对调休制度持有哪些观点？
"""

original_document = """
1, "放假","一天假期的时间概念。"<SEP>"五一放假一天的事件。"<SEP>"评论中提到的休息时间安排。"<SEP>"评论者提到的休息安排，包括调休在内的放假时间。"<SEP>放假是一个与休息、出行相关的时间段概念，尤其在五一劳动节期间，它代表了国家规定的法定节假日、公司安排的休息时间、学校组织的假期以及与节日相关的休息时间。

2,"调休","影响放假安排的事件。"<SEP>调休是一种旨在平衡生产和消费需求的工作制度，与放假安排紧密相关。这种制度通过调整工作日来增加假期时间，特别在大型节假日和传统节日如五一、端午、中秋等期间，调休政策被广泛实施。调休不仅影响放假安排，还涉及调整休息时间的工作安排，允许学校、公司等机构根据实际需求或节假日安排灵活调整原有假期，例如放假后补班或占用周末值班。

3, "假期",
"与休息、出行相关的时间段概念。"<SEP>"五一期间休息的时间段概念。"

4, "调休安排，假期安排","放假","调休","放假期间可能涉及调休的情况。"<SEP>"调休与放假相对，调休导致放假天数减少。"<SEP>"调休可能影响放假安排，进而影响人们的旅游和消费行为。"<SEP>"调休是放假的一种特殊形式，通常是为了平衡工作日和休息日的分配。"<SEP>调休的实施可能会改变原本的放假安排，甚至导致实际放假时间变短，从而影响放假效果。评论者普遍反映，调休措施影响了他们的休息计划，导致疲惫和不满。因此，调休的安排需要充分考虑放假情况，避免与放假计划产生矛盾，确保员工能够得到应有的休息。
5, "政策影响，假期安排","假期","调休","假期如果调休，则会打乱原本的休息计划。<SEP>假期调整可能使员工连续工作，影响企业运营。<SEP>评论者提出调休不是理想的假期安排方式。<SEP>调休事件影响假期安排。<SEP>调休影响了原本的放假安排，导致假期时间减少。

6, 我五一之前连上9天五一放四天然后又连上7天
7, 无所谓你们双休调休的事关我单休人啥事搞得我两周才放一天假
8, 调休每次放假都在说没有一次取消过调休
9, 单休调休节假日值班里外里放个假我还多上了几天
"""

paragraphs = re.findall(r'^\d+\,.*?(?=^\d+\,|\Z)', original_document, re.MULTILINE | re.DOTALL)

# 打印匹配到的段落，检查是否有遗漏
print("匹配到的段落：")
for i, paragraph in enumerate(paragraphs, start=1):
    print(f"段落 {i}: {paragraph}")

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

min_k = 5
gradient_threshold = 0.2
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

# 数据
text_blocks = [f'r_{i+1}' for i in range(len(sorted_scores))]
scores = sorted_scores
selected_blocks = [f'r_{chunks.index(chunk)+1}' for chunk in selected_chunks]
k = 3  # 初始选择前 k 个文本块
g = 2  # 梯度阈值

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(text_blocks, scores, marker='o', linestyle='-', color='b', label='相关性得分')

# 标注初始选择的文本块
for i in range(k):
    plt.scatter(text_blocks[i], scores[i], color='green', zorder=5, label='初始选择' if i == 0 else "")
    plt.text(text_blocks[i], scores[i] + 0.02, f'{scores[i]:.2f}', fontsize=9, ha='center', color='green')

# 标注梯度检查过程
for i in range(k, len(text_blocks)):
    prev_score = scores[i - 1]
    threshold = prev_score / g  # 梯度阈值
    current_score = scores[i]

    # 绘制梯度阈值线
    plt.hlines(threshold, text_blocks[i - 1], text_blocks[i], colors='r', linestyles='dashed',
               label='梯度阈值' if i == k else "")

    # 标注当前文本块
    if current_score > threshold:
        plt.scatter(text_blocks[i], scores[i], color='orange', zorder=5, label='加入选择' if i == k else "")
        plt.text(text_blocks[i], scores[i] + 0.02, f'{scores[i]:.2f}', fontsize=9, ha='center', color='orange')
    else:
        plt.scatter(text_blocks[i], scores[i], color='gray', zorder=5, label='停止选择' if i == k else "")
        plt.text(text_blocks[i], scores[i] + 0.02, f'{scores[i]:.2f}', fontsize=9, ha='center', color='gray')
        break  # 停止选择

# 添加标题和标签
plt.title('文本块选择过程（梯度阈值 g=2）', fontsize=16)
plt.xlabel('文本块', fontsize=14)
plt.ylabel('相关性得分', fontsize=14)
plt.ylim(0, 1.0)  # 设置 y 轴范围
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 显示图形
plt.show()

