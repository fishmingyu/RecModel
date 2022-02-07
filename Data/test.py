import pandas as pd
from scipy.sparse import csr_matrix

df = pd.read_csv("output.csv")
row = df['user'].values
col = df['item'].values
rate = df['rating'].values

m = max(row) + 1
n = max(col) + 1

sp = csr_matrix((rate, (row, col)), shape=(m, n))