import streamlit as st
import pickle
import re
import pandas as pd
import json

# -------------------- LOAD MODEL --------------------
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="AI Sentiment Analyzer", layout="wide")

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# -------------------- SIDEBAR --------------------
st.sidebar.title("⚙️ Settings")
show_chart = st.sidebar.checkbox("Show Confidence Chart", True)
show_history = st.sidebar.checkbox("Show History", True)

# -------------------- HEADER --------------------
st.title("💬 AI Aspect-Based Sentiment Analyzer")
st.markdown("### 🚀 Analyze sentiment for food, service, price, ambience")

# -------------------- SESSION HISTORY --------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# -------------------- INPUT --------------------
user_input = st.text_area("✍️ Enter your sentence:", height=120)

# -------------------- ASPECT KEYWORDS --------------------
aspect_keywords = {
    'food': ['food', 'pizza', 'burger', 'meal', 'dish', 'taste', 'menu'],
    'service': ['service', 'staff', 'waiter', 'manager', 'support'],
    'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money'],
    'ambience': ['ambience', 'atmosphere', 'environment', 'place', 'vibe']
}

# -------------------- PREDICTION --------------------
def predict_aspect_sentiment(text):
    text = clean_text(text)
    results = {}

    negative_words = ['slow', 'bad', 'poor', 'terrible', 'worst', 'delay', 'hate']
    positive_words = ['good', 'great', 'amazing', 'excellent', 'nice', 'love']

    for aspect, keywords in aspect_keywords.items():
        if any(word in text for word in keywords):

            words = text.split()
            context = text

            # -------- Extract context window --------
            for word in keywords:
                if word in words:
                    idx = words.index(word)
                    start = max(0, idx - 3)
                    end = min(len(words), idx + 4)
                    context = " ".join(words[start:end])
                    break

            # -------- ALWAYS use model --------
            input_text = context + " " + aspect
            vec = vectorizer.transform([input_text])

            model_sentiment = model.predict(vec)[0]
            model_conf = max(model.predict_proba(vec)[0]) * 100

            # -------- Hybrid correction --------
            if any(w in context for w in negative_words):
                sentiment = 'negative'
                confidence = max(model_conf, 80)  # ensure strong confidence
            elif any(w in context for w in positive_words):
                sentiment = 'positive'
                confidence = max(model_conf, 80)
            else:
                sentiment = model_sentiment
                confidence = model_conf

            results[aspect] = (sentiment, round(confidence, 2))

    # -------- FALLBACK --------
    if not results:
        vec = vectorizer.transform([text])
        sentiment = model.predict(vec)[0]
        confidence = max(model.predict_proba(vec)[0]) * 100
        results['overall'] = (sentiment, round(confidence, 2))

    return results

    # Fallback
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

        # Save history
        st.session_state.history.append((user_input, results))

        col1, col2 = st.columns([2, 1])

        # -------------------- LEFT PANEL --------------------
        with col1:
            st.markdown("## 📊 Results")

            for aspect, (sentiment, confidence) in results.items():

                st.markdown(f"### 🔹 {aspect.upper()}")

                if sentiment == "positive":
                    st.success(f"😊 Positive ({confidence}%)")
                elif sentiment == "negative":
                    st.error(f"😠 Negative ({confidence}%)")
                else:
                    st.info(f"😐 Neutral ({confidence}%)")

                st.progress(int(confidence))
                st.markdown("---")

        # -------------------- RIGHT PANEL --------------------
        with col2:
            st.markdown("## 🧠 Summary")

            summary = []
            for aspect, (sentiment, _) in results.items():
                emoji = "😊" if sentiment == "positive" else "😠" if sentiment == "negative" else "😐"
                summary.append(f"{aspect}: {emoji}")

            st.write(", ".join(summary))

            # -------------------- CHART --------------------
            if show_chart:
                data = []
                for aspect, (_, confidence) in results.items():
                    data.append([aspect, confidence])

                df = pd.DataFrame(data, columns=['Aspect', 'Confidence'])
                st.bar_chart(df.set_index('Aspect'))

            # -------------------- DOWNLOAD --------------------
            st.download_button(
                label="📥 Download Results",
                data=json.dumps(results),
                file_name="results.json"
            )

        # -------------------- HISTORY --------------------
        if show_history:
            st.markdown("## 🕘 Recent Inputs")

            for text, _ in reversed(st.session_state.history[-3:]):
                st.write(f"👉 {text}")

    else:
        st.warning("⚠️ Please enter some text.")