Sure! Here's how you can present the user interface in your project documentation:

---

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
│       └── banglabert_tokenizer/
├── frontend/
│   └── app.py
├── README.md
├── requirements.txt

```


---

## 🧠 How the Model Was Trained

- **Base Model**: `csebuetnlp/banglabert_small`
- **Fine-tuned** using a labeled Bangla news dataset with 4 categories.
- **Tokenizer**: BanglaBERT tokenizer
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Training Details**:
  - Used early stopping based on validation loss
  - Saved the best model weights to `smbert_category_model.pt`

---

## 🚀 Getting Started

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
## 📮 API Endpoint
POST /predict

Request (JSON):

```
{
  "text": "বাংলাদেশ আজ ভারতের বিরুদ্ধে দুর্দান্ত এক জয় পেয়েছে।"
}

```
Response (JSON):

```
{
  "category": "sports"
}

```

## 📌 Example Texts to Try
You can input any Bangla news headline or paragraph. Here are a few examples:
```
আজ বিশ্বজুড়ে পালিত হচ্ছে আন্তর্জাতিক নারী দিবস।
ঢালিউডে নতুন একটি সিনেমার শুটিং শুরু হয়েছে।
রাজধানীতে আজ মন্ত্রিপরিষদের এক জরুরি বৈঠক অনুষ্ঠিত হয়েছে।

```

## 📦 Dependencies


The backend requires the following Python libraries:

- `fastapi`
- `uvicorn`
- `torch`
- `transformers`
- `streamlit`
- `requests`

You can install them with:

pip install -r /requirements.txt

```
Then run:
uvicorn backend.main:app --reload
streamlit run app.py

and visit the link.
