import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

df = pd.read_csv("data/raw/customer_tickets.csv")

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df["clean_body"] = df["body"].apply(clean_text)

print(df[["body", "clean_body"]].head())

df.to_csv(
    "data/processed/cleaned_tickets.csv",
    index=False
)

print("Preprocessing completed successfully!")
