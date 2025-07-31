import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="Language Detector | Ankur AI",
    page_icon="🌐",
    layout="centered"
)

# ------------------------
# Custom Styling
# ------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;500;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            background: linear-gradient(to right, #f8fdff, #e0f7fa);
            color: #263238;
        }

        .main-card {
            background: white;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            margin-top: 40px;
            margin-bottom: 30px;
        }

        .title {
            font-size: 2.6em;
            font-weight: 700;
            text-align: center;
            color: #00796b;
            margin-bottom: 0.4em;
        }

        .subtitle {
            font-size: 1.1em;
            text-align: center;
            color: #555;
            margin-bottom: 2em;
        }

        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #777;
            margin-top: 3rem;
        }

        .prediction-box {
            background-color: #e0f2f1;
            border-left: 5px solid #00796b;
            padding: 1rem;
            margin-top: 1.5rem;
            border-radius: 8px;
        }

        .top-pred {
            font-weight: bold;
            color: #00796b;
        }

        textarea {
            font-family: 'Roboto', sans-serif !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# App UI
# ------------------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)
st.markdown('<div class="title">🌍 Language Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered app to identify the language of any text</div>', unsafe_allow_html=True)

with st.expander("💡 How does this work?"):
    st.markdown("""
    - This app uses a **Naive Bayes** model trained on a labeled multilingual text dataset.
    - Input text is transformed into numerical vectors using **CountVectorizer**.
    - The model then classifies the language with high accuracy.
    """)

user_input = st.text_area("✍️ Paste or type your sentence here:", placeholder="e.g., welcome to Ankur AI translator", height=150)

if st.button("🔍 Detect Language"):
    if user_input.strip():
        vect_text = vectorizer.transform([user_input])
        prediction = model.predict(vect_text)[0]

        st.markdown(f"""
            <div class="prediction-box">
                ✅ <strong>Detected Language:</strong> <span class="top-pred">{prediction}</span>
            </div>
        """, unsafe_allow_html=True)

        # Show top 3 probabilities if model supports it
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vect_text)
            top_idx = np.argsort(proba[0])[::-1][:3]
            top_labels = model.classes_[top_idx]
            top_probs = proba[0][top_idx]

            st.markdown("#### 🔢 Top 3 Predictions:")
            for i in range(len(top_labels)):
                st.write(f"- **{top_labels[i]}**: {top_probs[i]*100:.2f}%")

    else:
        st.warning("⚠️ Please enter some text.")

st.markdown('<div class="footer">Crafted with ❤️ by Ankur | Powered by Scikit-learn & Streamlit</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

