import pandas as pd
from attr.validators import matches_re

df=pd.read_csv("train.csv", low_memory=False)
print(df.head())
print(df.info())

df_small=df.sample(100000, random_state=42)
df_small.to_csv("expedia_small.csv", index=False)

