import pandas as pd

df = pd.read_parquet("./hf_data/").sample(n=200_000, random_state=42)
df.to_parquet("sample_200k.parquet", index=False)