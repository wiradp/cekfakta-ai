from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # agar frontend bisa akses dari file lokal

# Load model dan vectorizer
vectorizer = joblib.load("model/vectorizer.pkl")
model = joblib.load("model/classifier.pkl")

# Logic penjelasan sederhana
def get_explanation(label):
    explanations = {
        "penipuan": "Teks ini mengandung pola umum penipuan seperti undian atau hadiah palsu.",
        "judi": "Teks ini mirip promosi judi online, seperti 'situs gacor' atau taruhan.",
        "hoax": "Teks mengandung konten hoax seperti konspirasi atau ketidakpercayaan publik.",
        "aman": "Tidak terdeteksi sebagai pesan berbahaya. Tetap waspada."
    }
    return explanations.get(label, "Tidak ada penjelasan.")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Teks kosong"}), 400

    X = vectorizer.transform([text])
    label = model.predict(X)[0]
    explanation = get_explanation(label)
    
    return jsonify({"label": label, "explanation": explanation})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
