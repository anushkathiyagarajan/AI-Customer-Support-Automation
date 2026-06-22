import joblib

# Load model, vectorizer, and encoder
model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
encoder = joblib.load("models/label_encoder.pkl")

while True:
    ticket = input("\nEnter Customer Ticket: ")

    if ticket.lower() == "exit":
        break

    ticket_tfidf = vectorizer.transform([ticket])

    prediction = model.predict(ticket_tfidf)

    queue_name = encoder.inverse_transform(prediction)

    print("Predicted Queue:", queue_name[0])