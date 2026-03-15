import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
text_blocks = ['r_3', 'r_2', 'r_6', 'r_4', 'r_5', 'r_1']
scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
selected_blocks = ['r_3', 'r_2', 'r_6', 'r_4', 'r_5']  # 最终选择的文本块
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
