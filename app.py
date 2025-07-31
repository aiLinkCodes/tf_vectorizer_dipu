from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Cargar el vectorizador entrenado
vectorizer = joblib.load("tfidf_vectorizer.joblib")

class TextInput(BaseModel):
    text: str

@app.post("/vectorize")
def vectorize(input: TextInput):
    vector = vectorizer.transform([input.text])
    indices = vector.indices.tolist()
    values = vector.data.tolist()
    return {"indices": indices, "values": values}
