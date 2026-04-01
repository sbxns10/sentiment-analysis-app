import streamlit as st
import pickle

# -------------------- LOAD MODEL --------------------
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

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

# -------------------- PREDICTION FUNCTION --------------------
def predict_aspect_sentiment(text):
    aspects = ['food', 'service', 'price', 'ambience']
    results = {}

    negative_words = ['slow', 'bad', 'poor', 'terrible', 'worst', 'delay']
    positive_words = ['good', 'great', 'amazing', 'excellent', 'nice']

    for aspect in aspects:
        if aspect in text.lower():
            
            words = text.lower().split()
            if aspect in words:
                idx = words.index(aspect)
                start = max(0, idx - 3)
                end = min(len(words), idx + 4)
                context = " ".join(words[start:end])
            else:
                context = text

            # Rule-based correction
            if any(word in context for word in negative_words):
                sentiment = 'negative'
                confidence = 85.0
            elif any(word in context for word in positive_words):
                sentiment = 'positive'
                confidence = 85.0
            else:
                input_text = context + " " + aspect
                vec = vectorizer.transform([input_text])

                sentiment = model.predict(vec)[0]
                confidence = max(model.predict_proba(vec)[0]) * 100

            results[aspect] = (sentiment, round(confidence, 2))

    return results

# -------------------- BUTTON --------------------
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() != "":
        results = predict_aspect_sentiment(user_input)

        st.markdown("### 📊 Analysis Results")

        if results:
            for aspect, (sentiment, confidence) in results.items():
                
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"**🔹 {aspect.upper()}**")

                with col2:
                    st.markdown(f"**{confidence}%**")

                if sentiment == "positive":
                    st.success(f"😊 Positive")
                elif sentiment == "negative":
                    st.error(f"😠 Negative")
                else:
                    st.info(f"😐 Neutral")

                st.progress(int(confidence))

                st.markdown("---")

        else:
            st.warning("⚠️ No known aspects found in the sentence.")
    else:
        st.warning("⚠️ Please enter some text.")