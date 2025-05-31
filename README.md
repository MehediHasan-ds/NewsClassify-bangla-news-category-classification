# Bangla News Categorization using BERT

This project focuses on classifying Bangla news articles into predefined categories using a fine-tuned BERT-based language model (`sagorsarker/bangla-bert-base`), Bangla-Electra and mBert.

The reason for choosing these model is that they performed state of the art in text/nlp classification task
which you can check on the huggingface sagorsarker/bangla-bert-base: https://huggingface.co/sagorsarker/bangla-bert-base here you will find that Bangla Bert Base has. outperformed the other two model and Bangla Benchmarks in  https://colab.research.google.com/drive/1vltPI81atzRvlALv4eCvEB0KdFoEaCOb#scrollTo=pqsH5gU-QxVf
we will also try csebuetnlp/banglabert_small because of its very low number of parameters(13M) just to check how it performs in comparison the other three.

The workflow includes preprocessing raw Bangla text, removing irrelevant tokens, training a transformer model with early stopping, and visualizing word distributions across categories.

The aim is to develop an accurate and robust model that understands Bangla language nuances and effectively categorizes news content for downstream NLP applications.

Here’s a revised version of the **"Data Load and Exploration", "Data Cleaning", "Feature Engineering"**, and **"Model Setup and Training"** sections. The rewrite focuses on **what has been done and what has been achieved**, emphasizing your **techniques and decisions**, not the detailed step-by-step process.


#### my notebook link: https://colab.research.google.com/drive/1V7T8yeZq4snNm3FLRE2pwKDgfDhMWoDM?usp=sharing


## 📂 Data Load and Exploration

The dataset was inspected and cleaned by discarding irrelevant columns such as `reporter`, which contained a large number of null values and had no influence on the classification task. Only the `title`, `content`, and `category` fields were retained for modeling.

An initial analysis confirmed that the dataset is balanced across four distinct news categories. Additionally, content strings were analyzed to detect and eliminate repetitive, irrelevant phrases like “ছবি: সংগৃহীত” and “ফাইল ছবি”, which often appear at the beginning of articles but offer no contextual value for classification.


## 🧹 Data Cleaning

To enhance the quality of input data, punctuation and Bangla stopwords were removed from the content. Stopwords were merged from multiple sources to ensure comprehensive coverage. This significantly reduced noise and preserved only meaningful tokens, especially important when dealing with a token limit of 512.

Additional low-value terms (e.g., সাল, মিনিট) that were domain-irrelevant were identified and excluded to further refine the dataset. This targeted cleaning approach ensured higher-quality inputs for the model and reduced the risk of truncating informative tokens during tokenization.


## 🛠️ Feature Engineering

The data was restructured to create a new processed text field that combines the cleaned and filtered content. Once finalized, the original unprocessed content field was removed to optimize memory usage and keep the dataset concise.


## 🧠 Model Setup and Dataset Preparation

The cleaned dataset was prepared for training by tokenizing the text using the `sagorsarker/bangla-bert-base` model. The label space was mapped, and the model architecture was adapted for multi-class classification with four output categories.


## 🔁 Model Training with Early Stopping

The model was fine-tuned using a robust training loop that incorporates early stopping to prevent overfitting. A linear learning rate scheduler was employed to stabilize learning and optimize convergence. This approach resulted in a well-generalized model capable of accurately classifying Bangla news content across all categories.


## Training Loop with Validation and Early Stopping

The model is trained for up to `num_epochs`, but it can stop earlier if the validation loss does not improve significantly for a number of `patience` consecutive epochs. This helps prevent overfitting and saves training time.

### Training Phase

- The model is set to **train mode**.
- Each batch of data is **forwarded** through the model.
- **Loss is computed**, then **backpropagated** to update the model's parameters.
- The **learning rate** is updated using the defined **scheduler**.

### Validation Phase

- The model is set to **evaluation mode** (no gradient updates).
- **Validation loss** is calculated using the validation dataset.
- No weights are updated during this phase.

### Early Stopping Check

- If the **validation loss improves** by more than the `threshold`, the model's state is saved and the `patience_counter` is **reset**.
- If there is **no significant improvement**, the `patience_counter` is **incremented**.
- If the `patience_counter` **exceeds** the allowed `patience`, training is **stopped early**.

This strategy ensures efficient training and prevents the model from overfitting to the training data.

# Load Trained Model for Evaluation and Inference

In this step, we load the trained Bangla BERT model and tokenizer to evaluate it on the test data. This setup is crucial for making predictions and analyzing the model's performance using various metrics.

### Configuration

- **Model Name:** We use `"sagorsarker/bangla-bert-base"`, a pretrained Bangla BERT model.
- **Number of Labels:** Set to `4` corresponding to the categories: `sports`, `international`, `entertainment`, `national`.
- **Device Setup:** Uses `GPU` if available, otherwise defaults to `CPU`.

### Label Mapping

- `label2id`: A dictionary mapping category names to numerical IDs.
- `id2label`: The inverse mapping to convert prediction IDs back to readable category names.

This mapping is necessary for both training and interpreting model predictions.

### Load Tokenizer and Model

- **Tokenizer**: Loaded from the saved `banglabert_tokenizer` directory.
- **Model**: Initialized using the pretrained base and modified for sequence classification with 4 output labels.
- **Weights**: Loaded from the saved model checkpoint (`banglabert_category_model.pt`).

```python
model.load_state_dict(torch.load("banglabert_category_model.pt", map_location=device))
````

# Model Evaluation on Test Set

After training and saving the best model, we now evaluate its performance on the **test dataset**. This step helps assess how well the model generalizes to unseen data.

### Inference and Predictions

* We loop through the test data in batches and pass them through the model in **evaluation mode**.
* **Predictions** are obtained by selecting the class with the highest logit value using `argmax`.
* Predictions and true labels are collected for metric calculation.

```python
with torch.no_grad():
    for batch in test_loader:
        ...
        preds = torch.argmax(outputs.logits, dim=1)
        ...
```

## Evaluation Summary

The BanglaBERT-based text classification model achieved an impressive **accuracy of 96.47%** on the test dataset.

### Key Takeaways from the Classification Report:

* All four categories (`sports`, `international`, `entertainment`, `national`) have **high precision, recall, and F1-scores**, indicating that the model performs well across all classes.
* **Macro and Weighted Averages** are both above **0.96**, confirming balanced performance regardless of class size.
* The model shows the highest performance on the `sports` category (F1-score: 0.99), and consistently strong results on the other categories as well.

This suggests that the model generalizes well and is highly effective for multilingual news classification in Bangla.

### Confusion Matrix Analysis

The confusion matrix provides a detailed breakdown of the model’s classification performance across the four news categories:

* **Diagonal cells** represent correct predictions.
* **Off-diagonal cells** indicate misclassifications.

#### Observations:

* **Sports**: Predicted almost perfectly with only 3 misclassifications (1 as international, 2 as entertainment).
* **International**: 283 out of 298 correctly predicted. Most confusion was with `entertainment` (7) and `national` (7).
* **Entertainment**: 290 correct predictions. A few were misclassified as `international` (5), `national` (2), or `sports` (1).
* **National**: 282 correct. Slight confusion with all other classes, especially `entertainment` (7).

Overall, the model performs exceptionally well with minimal confusion, confirming its ability to distinguish between Bangla news categories with high precision.

All the four predictions were correct.

---

## Evaluation Results for BanglaBERT-Small News Classifier

The **`csebuetnlp/banglabert_small`** model was fine-tuned on a Bangla news classification task with four categories: `sports`, `international`, `entertainment`, and `national`. After training and applying early stopping, the model's performance was evaluated on a held-out test set.

### Accuracy

**Overall Accuracy:** `0.9723`

### Classification Report

| Category         | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Sports           | 0.99      | 0.99   | 0.99     | 297     |
| International    | 0.99      | 0.94   | 0.96     | 298     |
| Entertainment    | 0.96      | 0.98   | 0.97     | 298     |
| National         | 0.95      | 0.98   | 0.97     | 298     |
| **Macro Avg**    | 0.97      | 0.97   | 0.97     | 1191    |
| **Weighted Avg** | 0.97      | 0.97   | 0.97     | 1191    |

### Confusion Matrix

* Shows strong diagonal dominance with minimal misclassification.
* Most confusion appears between `international` and `entertainment`.

### ROC Curve (Multi-Class)

* All four categories achieve an **AUC of \~1.00**, indicating excellent separability between classes.
* The ROC curve confirms the model’s robustness across all classes.

---

Hence Banglabert-small performs better than others in metrics evaluation and with its small size of parameters.

```


