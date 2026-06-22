import os
import joblib
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Gemini Client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load ML models
model = joblib.load("models/svm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
encoder = joblib.load("models/label_encoder.pkl")

while True:

    ticket = input("\nEnter Customer Ticket: ")

    if ticket.lower() == "exit":
        break

    # Predict Queue
    ticket_tfidf = vectorizer.transform([ticket])

    prediction = model.predict(ticket_tfidf)

    queue_name = encoder.inverse_transform(prediction)[0]

    print("\nPredicted Queue:", queue_name)

    # Gemini Prompt
    prompt = f"""
    Customer Ticket:
    {ticket}

    Predicted Department:
    {queue_name}

    Generate a professional customer support acknowledgement email.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAI Response:\n")
    print(response.text)