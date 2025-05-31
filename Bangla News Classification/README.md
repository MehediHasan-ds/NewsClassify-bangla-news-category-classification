# 📰 Bangla News Category Classifier

A full-stack machine learning project that uses a fine-tuned [BanglaBERT-small](https://huggingface.co/csebuetnlp/banglabert_small) model to classify Bangla news texts into four categories:

* **Sports** ⚽
* **International** 🌐
* **Entertainment** 🎬
* **National** 🏛️

### 🖥️ User Interface of the Product

Below is a screenshot of the product's user interface where users can input Bangla news text and instantly see the predicted category:

![User Interface](user_interface.png)



This project features:
- ✅ A `FastAPI` backend for serving model inference
- ✅ A `Streamlit` frontend for user interaction
- ✅ A fine-tuned BanglaBERT model for Bangla news classification

---

## 📂 Project Structure
```
Bangla-News-Classifier/
├── backend/
│   ├── main.py
│   └── model/
│       ├── __pycache__
│       ├── model_loader.py
│       ├── smbert_category_model.pt
│       └── tokenizer/
├── frontend/
│   └── app.py
├── README.md
├── requirements.txt

```


---

## 🧠 How the main Model we are using for deployment

- **Base Model**: `csebuetnlp/banglabert_small`
- **Fine-tuned** using a labeled Bangla news dataset with 4 categories.
- **Tokenizer**: BanglaBERT-small tokenizer
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Training Details**:
  - Used early stopping based on validation loss
  - Saved the best model weights to `smbert_category_model.pt`

---


## Here's how we have made the deployment

### Step 1: Model Loading Logic (backend/model/model\_loader.py)

We created a utility that:

* Loads the fine-tuned `BanglaBERT-Small` model and tokenizer
* Removes the need to re-train or re-download during deployment
* Maps prediction outputs to category labels

```python
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert_small", num_labels=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, tokenizer, label_map
```

---

### Step 2: FastAPI Backend (backend/main.py)

We built a simple **FastAPI** app that:

* Loads model and tokenizer at startup
* Defines a POST endpoint `/predict`
* Accepts Bangla news text as input
* Returns the predicted category

```python
from fastapi import FastAPI
from pydantic import BaseModel
from .model.model_loader import load_model_and_tokenizer, predict_category

app = FastAPI()

model, tokenizer, label_map = load_model_and_tokenizer()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    category = predict_category(request.text, model, tokenizer, label_map)
    return {"category": category}
```

---

### Step 3: Streamlit Frontend (frontend/app.py)

We created a user-friendly **Streamlit UI** that:

* Lets users paste news text or upload a `.txt` file
* Sends the input to the FastAPI backend using `requests`
* Displays the predicted category

```python
response = requests.post("http://localhost:8000/predict", json={"text": input_text})
st.success(f"Predicted Category: {result['category'].capitalize()}")
```

---

### Step 4: Install Dependencies

Use a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```txt
fastapi
uvicorn
torch
transformers
streamlit
requests
```

---

### Step 5: Run the Backend (FastAPI)

From the root directory:

```bash
uvicorn backend.main:app --reload
```

This will start the API at `http://127.0.0.1:8000`

---

### Step 6: Run the Frontend (Streamlit)

From the `frontend/` folder:

```bash
streamlit run app.py
```

Visit: `http://localhost:8501`

---

### Step 7: Test the Application

* Paste or upload Bangla news text
* Click "Predict"
* See the predicted category displayed on screen

### Some example Texts to Try
You can input any Bangla news headline or paragraph. Here are a few examples:
```
আজ বিশ্বজুড়ে পালিত হচ্ছে আন্তর্জাতিক নারী দিবস।
ঢালিউডে নতুন একটি সিনেমার শুটিং শুরু হয়েছে।
রাজধানীতে আজ মন্ত্রিপরিষদের এক জরুরি বৈঠক অনুষ্ঠিত হয়েছে।

```


## 🚀 Test the application 

### 1. Clone the Repository

```
git clone https://github.com/MehediHasan-ds/NewsClassify-bangla-news-category-classification.git

cd banglabert-news-classifier
```

### 2. Create and Activate Virtual Environment
```
python -m venv venv
# For Windows
venv\Scripts\activate
# For macOS/Linux
source venv/bin/activate

```

###  3. Setup the Backend (FastAPI)

```
cd backend
pip install -r requirements.txt
uvicorn backcend.main:app --reload

```

### 4. Setup the Frontend (Streamlit)
Open a new terminal tab:

```
cd frontend
pip install -r requirements.txt
streamlit run app.py

```