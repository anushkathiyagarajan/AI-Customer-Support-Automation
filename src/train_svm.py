import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load data
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Handle missing values
train_df["clean_body"] = train_df["clean_body"].fillna("")
test_df["clean_body"] = test_df["clean_body"].fillna("")

# Features and labels
X_train = train_df["clean_body"]
X_test = test_df["clean_body"]

y_train = train_df["queue_encoded"]
y_test = test_df["queue_encoded"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# SVM Model
model = LinearSVC()

# Train
model.fit(X_train_tfidf, y_train)

# Predict
predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# Save model
joblib.dump(
    model,
    "models/svm_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nSVM Model Saved Successfully!")