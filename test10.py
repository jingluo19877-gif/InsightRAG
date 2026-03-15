from py2neo import Graph
import csv

# 连接到 Neo4j 数据库
graph = Graph("bolt://localhost:7687", auth=("neo4j", "0000"))

# 查询数据
query = "MATCH (n) RETURN n"
result = graph.run(query)

# 保存为 CSV
with open('output.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "data"])  # 根据返回数据修改列名
    for record in result:
        writer.writerow([record['n'].identity, record['n']])  # 修改为正确的属性
