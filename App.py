from flask import Flask, render_template, request
import torch
import torch.nn.functional as F
import re
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification
warnings.filterwarnings("ignore")
MODEL_PATH = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
def strong_clean(text):
    text = text.lower()
    text = re.sub(r'\b(facebook|share|click|subscribe|viral|subscribe)\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
def predict_news(text):
    if len(text.split()) < 20:
        return "Please enter at least 20 words.", 0.0
    cleaned = strong_clean(text)
    encoded = tokenizer(
        cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=192
    )

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    label_map = {0: "Fake News", 1: "Real News"}
    return label_map.get(pred, "Unknown"), round(confidence, 2)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__)
@app.route("/")
def home():
    return render_template("Index.html")  # Your landing page
@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result_text = None
    confidence = 0.0
    if request.method == "POST":
        news_text = request.form.get("news")
        result_text, confidence = predict_news(news_text)
    return render_template("prediction.html", prediction_text=result_text, confidence=confidence)
if __name__ == "__main__":
    app.run(debug=True)
