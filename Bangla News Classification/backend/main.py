from fastapi import FastAPI
from pydantic import BaseModel
from .model.model_loader import load_model_and_tokenizer, predict_category

app = FastAPI()

# Load model and tokenizer
model, tokenizer, label_map = load_model_and_tokenizer()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    category = predict_category(request.text, model, tokenizer, label_map)
    return {"category": category}
