import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load train and test datasets
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Handle missing values
train_df["clean_body"] = train_df["clean_body"].fillna("")
test_df["clean_body"] = test_df["clean_body"].fillna("")

# Check missing values
print("Missing values in train:",
      train_df["clean_body"].isnull().sum())

print("Missing values in test:",
      test_df["clean_body"].isnull().sum())

# Features and labels
X_train = train_df["clean_body"]
X_test = test_df["clean_body"]

y_train = train_df["queue_encoded"]
y_test = test_df["queue_encoded"]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train_tfidf, y_train)

# Predictions
predictions = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

# Detailed report
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Save model and vectorizer
joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel Saved Successfully!")