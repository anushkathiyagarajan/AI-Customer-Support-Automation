# 🤖 AI-Powered Customer Support Automation using Generative AI

## 📌 Overview

This project automates customer support ticket routing using Machine Learning, Deep Learning, and Generative AI. The system classifies customer support tickets into the appropriate departments and generates professional acknowledgment responses using Google Gemini.

The application is deployed using Streamlit to provide an interactive chat-based support experience.

---

## 🎯 Objectives

- Automate customer support ticket classification.
- Reduce manual effort in ticket routing.
- Improve response efficiency.
- Generate professional acknowledgment messages using Generative AI.
- Provide an interactive user interface through Streamlit.

---

## 🚀 Features

✅ Customer support ticket classification

✅ NLP-based text preprocessing

✅ Multiple classification models implemented:
- Logistic Regression
- Naïve Bayes
- Support Vector Machine (SVM)
- LSTM

✅ Automatic department prediction

✅ Google Gemini integration for AI-generated responses

✅ Interactive Streamlit application

✅ Chat-based support interface

---

## 📂 Project Structure

```text
FINAL/
│
├── data/
│
├── models/
│   ├── label_encoder.pkl
│   ├── logistic_regression.pkl
│   ├── lstm_model.h5
│   ├── naive_bayes.pkl
│   ├── svm_model.pkl
│   ├── svm_model_improved.pkl
│   ├── tfidf_vectorizer.pkl
│   └── tokenizer.pkl
│
├── src/
│   ├── ai_support_system.py
│   ├── label_encoding.py
│   ├── load_data.py
│   ├── predict_ticket.py
│   ├── preprocessing.py
│   ├── test_env.py
│   ├── test_gemini.py
│   ├── train_logistic_regression.py
│   ├── train_lstm.py
│   ├── train_naive_bayes.py
│   ├── train_svm.py
│   └── train_test_split.py
│
├── streamlit_app/
│   └── app.py
│
├── .env
├── README.md
├── requirements.txt
└── venv/
```

---

## 🛠 Technologies Used

### Programming Language
- Python

### Machine Learning
- Logistic Regression
- Naïve Bayes
- Support Vector Machine (SVM)

### Deep Learning
- LSTM (Long Short-Term Memory)

### Natural Language Processing
- Text Cleaning
- Tokenization
- TF-IDF Vectorization
- Label Encoding

### Generative AI
- Google Gemini API

### Frameworks & Libraries
- Streamlit
- Scikit-learn
- TensorFlow/Keras
- Pandas
- NumPy
- NLTK
- Joblib
- Python-dotenv
- Hugging Face Datasets

---

## 📊 Dataset

This project utilizes the **Customer Support Tickets Dataset** obtained from Hugging Face.

The dataset contains customer complaints and their corresponding support queues, enabling supervised learning for automated ticket routing.

---

## ⚙️ Workflow

### 1. Dataset Preparation
- Load customer support ticket dataset.
- Explore dataset characteristics.
- Identify support categories.

### 2. Data Preprocessing
- Convert text to lowercase.
- Remove special characters.
- Remove stopwords.
- Clean customer ticket descriptions.

### 3. Feature Extraction
- Apply TF-IDF Vectorization.
- Convert text into numerical representations.

### 4. Label Encoding
- Encode support departments into numerical labels.

### 5. Dataset Splitting
- Perform train-test split.

### 6. Model Training
Train multiple models:
- Logistic Regression
- Naïve Bayes
- SVM
- LSTM

### 7. Model Evaluation
Evaluate models using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 8. Ticket Prediction
Predict the appropriate support department for new customer tickets.

### 9. Gemini Integration
Generate professional acknowledgment responses using Google Gemini.

### 10. Streamlit Deployment
Provide a user-friendly interface for real-time interaction.

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/your-repository-name.git

cd FINAL
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root directory.

```env
GEMINI_API_KEY=your_api_key_here
```

---

## ▶ Running the Application

Launch the Streamlit application:

```bash
streamlit run streamlit_app/app.py
```

Open the browser at:

```text
http://localhost:8501
```

---

## 📷 Application Workflow

User enters a support query

⬇

Text preprocessing

⬇

TF-IDF transformation

⬇

SVM model predicts support department

⬇

Gemini generates acknowledgment response

⬇

Results displayed in Streamlit interface

---

## 📈 Future Enhancements

- Multi-language support
- Sentiment analysis
- Email integration
- Cloud deployment
- Voice-based support
- Advanced conversational memory

---

## 📄 License

This project is developed for academic and educational purposes.