import pandas as pd
from AutoClean import AutoClean
from scipy import stats
import numpy as np

df = pd.read_csv("optiver-trading-at-the-close/train.csv")
df = df.drop(columns = ['far_price', 'near_price'])
print(df.head())
print(df.isna().any().any())
rows_nan = df[df.isna().any(axis=1)]
print(rows_nan.index)
print(df.columns)
print(df['date_id'].mean())
print(df['date_id'].unique())
print(df['date_id'].value_counts())
print(len(df))
df = df.dropna()
print(len(df))
#df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
for col in df.columns:
    if "id" not in col:
        df = df[np.abs(df[col]-df[col].mean()) <= (3*df[col].std())]
#     q_low = df[col].quantile(0.01)
#     q_hi  = df[col].quantile(0.99)
#     print("xxxxx")

#     print(len(df[col] < q_hi))
#     print(len(df[col] > q_low))
#     df = df[(df[col] < q_hi) & (df[col] > q_low)]
    print("length check:", len(df))
    print("col check:", df[col])
print("length:", len(df))
#pipeline = AutoClean(df, mode = 'manual', outliers = 'auto', missing_num='auto')
print(df.isna().any().any())
#df.to_csv('train_clean.csv', index=False)
print("done")