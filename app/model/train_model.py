import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Data sederhana untuk contoh awal
data = pd.DataFrame({
    "text": [
        "Selamat kamu menang hadiah dari shopee",
        "Klik link ini untuk situs gacor terbaru",
        "Vaksin membuat mandul, jangan percaya pemerintah",
        "Mari kita belajar bersama untuk ujian sekolah",
        "Ada diskon produk elektronik minggu ini"
    ],
    "label": ["penipuan", "judi", "hoax", "aman", "aman"]
})

indonesian_stop_words = [
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan", "adalah", "ini", "itu", "atau", "juga", "karena"
    # tambahkan sesuai kebutuhan
]
# Vectorizer dan Model
vectorizer = TfidfVectorizer(stop_words=indonesian_stop_words)
X = vectorizer.fit_transform(data["text"])
model = LogisticRegression()
model.fit(X, data["label"])

# Simpan vectorizer dan model
joblib.dump(vectorizer, "app/model/vectorizer.pkl")
joblib.dump(model, "app/model/classifier.pkl")
print("Model dan vectorizer berhasil disimpan.")
