import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Path ke file CSV
csv_path = "data/dataset.csv"

# Baca data dari CSV
data = pd.read_csv(csv_path)

# Stopword Bahasa Indonesia sederhana (bisa ditambah lagi)
indonesian_stop_words = [
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah", "ini", "itu", "atau", "juga", "karena",
    "dalam", "agar", "dapat", "sebagai", "oleh", "bagi", "lebih", "tanpa", "bisa", "saja", "akan", "telah", "kalau"
]

# Vectorizer dan Model
vectorizer = TfidfVectorizer(stop_words=indonesian_stop_words)
X = vectorizer.fit_transform(data["text"])
model = LogisticRegression(max_iter=1000)
model.fit(X, data["label"])

# Buat folder model jika belum ada
os.makedirs("model", exist_ok=True)

# Simpan vectorizer dan model ke folder model/
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(model, "model/classifier.pkl")

print(f"✅ Model dan vectorizer berhasil disimpan. Total data training: {len(data)} sample.")
