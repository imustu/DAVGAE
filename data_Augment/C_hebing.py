import pandas as pd
threshold=0.91
# 读取out_fu.csv和out_zheng.csv文件
df_fu = pd.read_csv('../data/datasetA/append_'+ str(threshold*100)+'.csv')
df_zheng = pd.read_csv('../data/datasetA/old_edges.csv')

# 将两个DataFrame合并为一个DataFrame
df_all = pd.concat([df_fu, df_zheng])

# 保存合并后的数据到out_all.csv文件
df_all.to_csv('../data/datasetA/out_all_'+ str(threshold*100)+'.csv', index=False)