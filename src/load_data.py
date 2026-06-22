from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

ds = load_dataset("Tobi-Bueck/customer-support-tickets")

df = pd.DataFrame(ds["train"])

print("\nColumns:")
print(df.columns)

print("\nShape:")
print(df.shape)

print("\nQueue Distribution:")
print(df["queue"].value_counts())

df.to_csv("data/raw/customer_tickets.csv", index=False)

print("Dataset saved successfully!")