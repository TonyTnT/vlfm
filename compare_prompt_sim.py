import pandas as pd

# 读取 Excel 文件
with_prompt_df = pd.read_excel("similarities_with_prompt.xlsx", index_col=0)
wo_prompt_df = pd.read_excel("similarities_wo_prompt.xlsx", index_col=0)

# 提取 'coffee table' 行的数据
with_prompt_coffee_table = with_prompt_df.loc["coffee table"]
wo_prompt_coffee_table = wo_prompt_df.loc["coffee table"]

# 对每一列的元素从高到低排序，取前20
top20_with_prompt = with_prompt_coffee_table.sort_values(ascending=False).head(20)
top20_wo_prompt = wo_prompt_coffee_table.sort_values(ascending=False).head(20)

# 输出结果
print("Top 20 similarities with prompt:")
print(top20_with_prompt)
print("\nTop 20 similarities without prompt:")
print(top20_wo_prompt)
import matplotlib.pyplot as plt

# 将索引转换为中文标签
chinese_labels_with_prompt = ["标签" + str(i) for i in range(1, 21)]
chinese_labels_wo_prompt = ["标签" + str(i) for i in range(1, 21)]

# 绘制带提示和不带提示的相似度条形图在同一图中
plt.figure(figsize=(15, 7))

# 带提示的相似度条形图
plt.bar(chinese_labels_with_prompt, top20_with_prompt.values, color="b", alpha=0.7, label="With Prompt")

# 不带提示的相似度条形图
plt.bar(chinese_labels_wo_prompt, top20_wo_prompt.values, color="r", alpha=0.7, label="Without Prompt")

plt.xticks(rotation=0)  # 将 x 轴标签横着放
plt.title("Top 20 similarities with and without prompt")
plt.xlabel("Items")
plt.ylabel("Similarity")
plt.legend()
plt.tight_layout()
plt.show()
