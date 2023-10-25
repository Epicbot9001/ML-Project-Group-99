import pandas as pd
from AutoClean import AutoClean

df = pd.read_csv("optiver-trading-at-the-close/train.csv")
print(df.head())
print(df.isna().any().any())
#pipeline = AutoClean(df)
print(df.isna().any().any())
print("done")