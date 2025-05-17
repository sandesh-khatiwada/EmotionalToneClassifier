import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

# Define emotion columns (must match training script)
emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Load model and tokenizer
model_path = "./mood_extractor_model"  # Use "./results/checkpoint-20782" for a checkpoint
try:
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Input text
text = "I’m so excited about my new job!"

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
inputs = {key: val.to(device) for key, val in inputs.items()}

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).cpu().numpy()[0]

# Get predicted emotions (threshold > 0.5)
threshold = 0.5
predicted_emotions = [
    {"label": emotion_columns[i], "score": float(probabilities[i])}
    for i in range(len(emotion_columns))
    if probabilities[i] > threshold
]

# If no emotions exceed threshold, get the top emotion
if not predicted_emotions:
    max_prob_idx = np.argmax(probabilities)
    predicted_emotions = [
        {"label": emotion_columns[max_prob_idx], "score": float(probabilities[max_prob_idx])}
    ]

# Print results
print("Predicted emotions:")
for emotion in predicted_emotions:
    print(f"{emotion['label']}: {emotion['score']:.4f}")

# Optional: Print all probabilities
print("\nAll emotion probabilities:")
for emotion, prob in zip(emotion_columns, probabilities):
    print(f"{emotion}: {prob:.4f}")