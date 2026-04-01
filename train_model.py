import xml.etree.ElementTree as ET
import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------- TEXT CLEANING --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# -------------------- LOAD DATA --------------------
tree = ET.parse('Restaurants_Train.xml')
root = tree.getroot()

data = []

for sentence in root.iter('sentence'):
    text = sentence.find('text').text
    aspects = sentence.find('aspectTerms')
    
    if aspects is not None:
        for aspect in aspects.findall('aspectTerm'):
            term = aspect.get('term')
            polarity = aspect.get('polarity')
            
            if polarity != 'conflict':
                cleaned_text = clean_text(text)
                data.append([cleaned_text, term, polarity])

df = pd.DataFrame(data, columns=['text', 'aspect', 'sentiment'])

print("Dataset loaded. Total rows:", len(df))


# -------------------- PREPARE DATA --------------------
# Combine cleaned text + aspect (important for aspect-based learning)
df['input'] = df['text'] + " " + df['aspect']


# -------------------- VECTORIZATION --------------------
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['input'])
y = df['sentiment']


# -------------------- TRAIN MODEL --------------------
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

print("Model trained successfully!")


# -------------------- SAVE MODEL --------------------
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Model and vectorizer saved successfully!")