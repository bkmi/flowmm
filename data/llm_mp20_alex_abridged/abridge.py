import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)

df = pd.read_csv("/home/bkmi/flowmm/data/llm_mp20_alex/train.csv")
initial_len = len(df)
n_samples = initial_len // 6

abridged = df.sample(n_samples + n_samples // 100, random_state=rng, axis="index")
train = abridged.iloc[:n_samples]
val = abridged.iloc[n_samples:]

train.to_csv("/home/bkmi/flowmm/data/llm_mp20_alex_abridged/train.csv")
val.to_csv("/home/bkmi/flowmm/data/llm_mp20_alex_abridged/val.csv")

print("train len", len(train))
print("val len", len(val))

print()

print(np.linspace(0, len(train), 50).round())
