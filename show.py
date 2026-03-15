import networkx as nx
import matplotlib.pyplot as plt

# 解决中文显示问题，设置合适的字体
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 使用黑体字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 读取 GraphML 文件
graph_path = "E:/LightRAG_Test/LightRAG/examples/dickens/graph_chunk_entity_relation.graphml"
G = nx.read_graphml(graph_path)

print("节点数:", len(G.nodes))
print("边数:", len(G.edges))

# 使用 spring_layout 计算位置
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # 固定布局，使得每次显示一致

# 绘制节点
nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")

# 绘制边 (增加边宽度，确保边可见)
nx.draw_networkx_edges(G, pos, edge_color="gray", width=2, alpha=0.6, arrows=True, arrowstyle="->")

# 绘制标签，确保使用 SimHei 字体支持中文
nx.draw_networkx_labels(G, pos, font_size=10, font_family="SimHei", font_color="black")

# 添加标题
plt.title("图形可视化")
plt.show()
