from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging

# ✅ Muat variabel lingkungan dari .env
load_dotenv()

# ✅ Inisialisasi klien OpenAI
# (Dihapus karena duplikat dan salah referensi 'openai')

# ✅ Inisialisasi Flask
app = Flask(__name__, static_folder="static")
CORS(app)

# ✅ Load model lokal (scikit-learn)
vectorizer = joblib.load("model/vectorizer.pkl")
model = joblib.load("model/classifier.pkl")

# ✅ Penjelasan berdasarkan label
def get_explanation(label):
    explanations = {
        "penipuan": "Teks ini mengandung pola umum penipuan seperti undian atau hadiah palsu.",
        "judi": "Teks ini mirip promosi judi online, seperti 'situs gacor' atau taruhan.",
        "hoax": "Teks mengandung konten hoax seperti konspirasi atau ketidakpercayaan publik.",
        "aman": "Tidak terdeteksi sebagai pesan berbahaya. Tetap waspada."
    }
    return explanations.get(label, "Tidak ada penjelasan.")

# ✅ Inisialisasi klien Azure OpenAI
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# ✅ Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# ✅ Route: /
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

# ✅ Route: Prediksi lokal
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

# ✅ Route: Prediksi GPT (Azure OpenAI)
@app.route("/predict_gpt", methods=["POST"])
def predict_gpt():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Teks kosong"}), 400

    prompt = (
        "Klasifikasikan teks berikut ke dalam salah satu kategori: penipuan, judi, hoax, atau aman. "
        "Jawab dengan format:\nLabel: <label>\nPenjelasan: <penjelasan alami, ramah, mudah dimengerti dan singkat>\n\n"
        f"Teks: {text}"
    )

    try:
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        hasil = response.choices[0].message.content
        return jsonify({"hasil": hasil})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
