import csv
# 将基因，疾病向量求出来的，d-g关联概率，转化为0-1

import csv

# 将基因，疾病向量求出来的，d-g关联概率，转化为0-1

count = 0  # 记录值为 1.0 的位置数
threshold=0.91
with open('../data/datasetA/gd_probability.csv','r') as f:
    reader = csv.reader(f)

    with open('../data/datasetA/append_'+ str(threshold*100)+'.csv','w') as f_out_coords:  # Add this line to open the output file for coordinates

        for row_idx, row in enumerate(reader):
            new_row = []
            for col_idx, val in enumerate(row):
                if float(val) >= threshold:
                    print(val)
                    new_row.append(1.0)
                    count += 1
                    f_out_coords.write(f'{row_idx},{col_idx},1\n')  # Add this line to write the row and column index to the file
                else:
                    new_row.append(0.0)


print(f'Total number of positions with value 1.0: {count}')