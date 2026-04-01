import streamlit as st
import pickle
import re

# -------------------- LOAD MODEL --------------------
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Sentiment Analyzer", layout="centered")

# -------------------- CUSTOM STYLE --------------------
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #AAAAAA;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">💬 AI Aspect-Based Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze sentiment for food, service, price, ambience</div>', unsafe_allow_html=True)

# -------------------- INPUT --------------------
user_input = st.text_area("✍️ Enter your sentence:", height=120)

# -------------------- ASPECT KEYWORDS --------------------
aspect_keywords = {
    'food': ['food', 'pizza', 'burger', 'meal', 'dish', 'taste', 'menu'],
    'service': ['service', 'staff', 'waiter', 'manager', 'support'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money'],
    'ambience': ['ambience', 'atmosphere', 'environment', 'place', 'vibe']
}

# -------------------- PREDICTION FUNCTION --------------------
def predict_aspect_sentiment(text):
    text = clean_text(text)
    results = {}

    negative_words = ['slow', 'bad', 'poor', 'terrible', 'worst', 'delay', 'hate']
    positive_words = ['good', 'great', 'amazing', 'excellent', 'nice', 'love']

    for aspect, keywords in aspect_keywords.items():
        if any(word in text for word in keywords):

            words = text.split()
            context = text

            for word in keywords:
                if word in words:
                    idx = words.index(word)
                    start = max(0, idx - 3)
                    end = min(len(words), idx + 4)
                    context = " ".join(words[start:end])
                    break

            # ---------------- RULE-BASED CORRECTION ----------------
            if any(w in context for w in negative_words):
                sentiment = 'negative'
                confidence = 85.0
            elif any(w in context for w in positive_words):
                sentiment = 'positive'
                confidence = 85.0
            else:
                input_text = context + " " + aspect
                vec = vectorizer.transform([input_text])

                sentiment = model.predict(vec)[0]
                confidence = max(model.predict_proba(vec)[0]) * 100

            results[aspect] = (sentiment, round(confidence, 2))

    # ---------------- FALLBACK (IMPORTANT) ----------------
    if not results:
        vec = vectorizer.transform([text])
        sentiment = model.predict(vec)[0]
        confidence = max(model.predict_proba(vec)[0]) * 100
        results['overall'] = (sentiment, round(confidence, 2))

    return results

# -------------------- BUTTON --------------------
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() != "":
        results = predict_aspect_sentiment(user_input)

        st.markdown("### 📊 Analysis Results")

        for aspect, (sentiment, confidence) in results.items():

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**🔹 {aspect.upper()}**")

            with col2:
                st.markdown(f"**{confidence}%**")

            if sentiment == "positive":
                st.success("😊 Positive")
            elif sentiment == "negative":
                st.error("😠 Negative")
            else:
                st.info("😐 Neutral")

            st.progress(int(confidence))
            st.markdown("---")

    else:
        st.warning("⚠️ Please enter some text.")