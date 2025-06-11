from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import openai
from dotenv import load_dotenv

# Muat file .env jika ada (untuk lokal)
load_dotenv()

# Inisialisasi Flask
app = Flask(__name__, static_folder="static")
CORS(app)

# Load model lokal
vectorizer = joblib.load("model/vectorizer.pkl")
model = joblib.load("model/classifier.pkl")

# Penjelasan hasil model lokal
def get_explanation(label):
    explanations = {
        "penipuan": "Teks ini mengandung pola umum penipuan seperti undian atau hadiah palsu.",
        "judi": "Teks ini mirip promosi judi online, seperti 'situs gacor' atau taruhan.",
        "hoax": "Teks mengandung konten hoax seperti konspirasi atau ketidakpercayaan publik.",
        "aman": "Tidak terdeteksi sebagai pesan berbahaya. Tetap waspada."
    }
    return explanations.get(label, "Tidak ada penjelasan.")

# Konfigurasi Azure OpenAI
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_type = "azure"
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# 👇 ROOT "/" → Serve index.html
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# 👇 PREDIKSI LOKAL (Model scikit-learn)
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

# 👇 PREDIKSI GPT-4 dari Azure OpenAI
@app.route("/cek-gpt", methods=["POST"])
def cek_pesan_gpt():
    data = request.get_json()
    pesan = data.get("text", "")  # pakai "text" agar konsisten
    if not pesan:
        return jsonify({"error": "Teks kosong"}), 400

    prompt = f"""
Teks: "{pesan}"

Apakah ini termasuk salah satu kategori berikut: 'penipuan', 'judi', 'hoax', atau 'aman'? 
Jawab dalam format JSON:
{{
    "prediksi": "kategori",
    "alasan": "penjelasan singkat mengapa"
}}
"""

    try:
        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )

        hasil = response.choices[0].message["content"]
        return jsonify({"hasil": hasil})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 👇 RUN
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
