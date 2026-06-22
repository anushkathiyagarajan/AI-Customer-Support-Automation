import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/cleaned_tickets.csv")

encoder = LabelEncoder()

df["queue_encoded"] = encoder.fit_transform(df["queue"])

print("Number of Classes:", len(encoder.classes_))

print("\nSample Classes:")
for i, label in enumerate(encoder.classes_[:10]):
    print(i, "->", label)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

df.to_csv(
    "data/processed/encoded_tickets.csv",
    index=False
)

print("\nLabel encoding completed successfully!")