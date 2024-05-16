
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

# Read the CSV file
df = pd.read_csv('../data/datasetA/gd_01.csv',header=None)

# Convert to COO sparse matrix
sparse_matrix = coo_matrix(df.values)

# Create an empty list to collect non-zero elements
non_zero_elements = []

# Iterate through non-zero elements in the COO sparse matrix
for i, j, v in zip(sparse_matrix.row, sparse_matrix.col, sparse_matrix.data):
    if v != 0:  # We only care about non-zero elements
        non_zero_elements.append((i, j, 0))  # +1 because we need 1-based indices

# Create a DataFrame to store non-zero elements, then write to CSV
df_out = pd.DataFrame(non_zero_elements, columns=['Row', 'Column', 'Value'])

df_out.to_csv('../data/datasetA/old_edges.csv', index=False)
print(len(non_zero_elements))