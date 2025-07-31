from flask import Flask, request, jsonify
import joblib

# Cargar el vectorizador previamente entrenado
vectorizer = joblib.load("assets/tfidf_vectorizer.joblib")  # Ajusta la ruta si es necesario

app = Flask(__name__)

@app.route("/vectorize", methods=["POST"])
def vectorize():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    texto = data["text"]
    vector = vectorizer.transform([texto])
    row = vector[0]
    result = {
        "indices": row.indices.tolist(),
        "values": row.data.tolist()
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
