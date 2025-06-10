import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load data dari CSV
data = pd.read_csv("model/training_data.csv")

# Stopwords bahasa Indonesia sederhana
indonesian_stop_words = [
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan",
    "adalah", "ini", "itu", "atau", "juga", "karena", "sebagai", "oleh", "agar"
]

# Vectorizer dan Model
vectorizer = TfidfVectorizer(stop_words=indonesian_stop_words)
X = vectorizer.fit_transform(data["text"])
model = LogisticRegression()
model.fit(X, data["label"])

# Simpan vectorizer dan model
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(model, "model/classifier.pkl")

print("Model dan vectorizer berhasil disimpan.")
print(f"Total data training: {len(data)}")
