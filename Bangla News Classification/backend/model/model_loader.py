import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_and_tokenizer():
    
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "smbert_category_model.pt")
    tokenizer_path = os.path.join(base_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    

    label_map = {0: 'sports', 1: 'international', 2: 'entertainment', 3: 'national'}

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert_small", num_labels=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")), strict=True)
    model.eval()

    return model, tokenizer, label_map

def predict_category(text, model, tokenizer, label_map):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_map[prediction]
