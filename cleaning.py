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
df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
print(len(df))
#pipeline = AutoClean(df, mode = 'manual', outliers = 'auto', missing_num='auto')
print(df.isna().any().any())
print("done")