import os
import joblib
import streamlit as st
from dotenv import load_dotenv
from google import genai

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Support Assistant",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

.stApp{
    background-color:#ffffff;
}

.main .block-container{
    max-width:900px;
    padding-top:2rem;
}

[data-testid="stSidebar"]{
    background-color:#f7f7f8;
}

h1{
    font-size:52px !important;
    font-weight:700 !important;
    color:#202123 !important;
}

h2,h3{
    color:#202123 !important;
}

p,div,span{
    color:#444654 !important;
}

.stButton button{
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)
# =====================================
# LOAD ENV
# =====================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =====================================
# LOAD MODELS
# =====================================

model = joblib.load(
    "models/svm_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)

encoder = joblib.load(
    "models/label_encoder.pkl"
)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🤖 AI Support")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

# =====================================
# SESSION STATE
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================
# WELCOME SCREEN
# =====================================
if len(st.session_state.messages) == 0:
    st.title("🤖 AI Support Assistant")

    st.markdown("### How can I help you today?")
    

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "💳 Payment Issue",
            use_container_width=True
        ):
            st.session_state.sample_prompt = (
                "I made payment but order is not showing"
            )

        if st.button(
            "🔐 Account Problem",
            use_container_width=True
        ):
            st.session_state.sample_prompt = (
                "Someone accessed my account"
            )

    with col2:

        if st.button(
            "🛠 Technical Support",
            use_container_width=True
        ):
            st.session_state.sample_prompt = (
                "Application crashes during login"
            )

        if st.button(
            "📦 Wrong Product",
            use_container_width=True
        ):
            st.session_state.sample_prompt = (
                "Received wrong product"
            )
# =====================================
# DISPLAY CHAT HISTORY
# =====================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

# =====================================
# CHAT INPUT
# =====================================

user_prompt = st.chat_input(
    "Message AI Support Assistant..."
)

# =====================================
# PROCESS USER INPUT
# =====================================

if user_prompt:

    # USER MESSAGE

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    with st.chat_message("user"):

        st.markdown(user_prompt)

    # =================================
    # PREDICT DEPARTMENT
    # =================================

    ticket_tfidf = vectorizer.transform(
        [user_prompt]
    )

    prediction = model.predict(
        ticket_tfidf
    )

    department = encoder.inverse_transform(
        prediction
    )[0]

    # =================================
    # GEMINI PROMPT
    # =================================

    ai_prompt = f"""
You are a professional customer support assistant.

Customer Issue:
{user_prompt}

Assigned Department:
{department}

Generate a professional support response.

Requirements:

- Mention the department
- Be professional
- Be concise
- Be polite
- No placeholders
"""

    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing your issue..."
        ):

            try:

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=ai_prompt
                )

                reply = f"""
**Department:** {department}

---

{response.text}
"""

            except Exception:

                reply = f"""
**Department:** {department}

---

Your ticket has been successfully routed to the appropriate department.

The AI response service is temporarily unavailable.
Please try again later.
"""

            st.markdown(reply)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply
        }
    )