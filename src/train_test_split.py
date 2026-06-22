import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "data/processed/encoded_tickets.csv"
)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["queue_encoded"]
)

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)

print("Train Shape:", train_df.shape)
print("Test Shape:", test_df.shape)

print("Train data saved.")
print("Test data saved.")