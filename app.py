from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import logging

# ✅ Muat variabel lingkungan dari .env
load_dotenv()

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
    logging.info(f"Request /predict_gpt: {text}")
    if not text:
        logging.warning("Teks kosong pada /predict_gpt")
        return jsonify({"error": "Teks kosong"}), 400

    prompt = (
        "Klasifikasikan teks berikut ke dalam salah satu kategori: penipuan, judi, hoax, atau aman. "
        "Jawab dengan format:\nLabel: <label>\nPenjelasan: <penjelasan alami, ramah, mudah dimengerti dan singkat>\n\n"
        f"Teks: {text}"
    )

    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "Kamu adalah detektor pesan mencurigakan."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        content = response.choices[0].message.content.strip()
        logging.info(f"Response GPT: {content}")
        # Parsing label dan penjelasan dari response
        label, explanation = None, None
        for line in content.splitlines():
            if line.lower().startswith("label:"):
                label = line.split(":", 1)[1].strip().lower()
                label = label.replace(".", "").replace(":", "").strip()
            elif line.lower().startswith("penjelasan:"):
                explanation = line.split(":", 1)[1].strip()
        if label not in ["penipuan", "judi", "hoax", "aman"]:
            logging.error(f"Label tidak dikenali: {label}. Jawaban model: {content}")
            return jsonify({"error": f"Label tidak dikenali: {label}. Jawaban model: {content}"}), 400
        if not explanation:
            explanation = "Tidak ada penjelasan dari model."
        logging.info(f"Prediksi: {label}, Penjelasan: {explanation}")
        return jsonify({"label": label, "explanation": explanation})

    except Exception as e:
        logging.exception("Error pada /predict_gpt")
        return jsonify({"error": str(e)}), 500

# ✅ Run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
