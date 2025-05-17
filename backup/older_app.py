import sys
import os
import json

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Suppress HuggingFace symlink warning

try:
    import numpy as np
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

    import transformers
    import torch
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please install dependencies by running: pip install datasets transformers torch numpy pandas scikit-learn")
    sys.exit(1)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
else:
    print("No GPU detected. Training will be slower on CPU.")

# Print environment diagnostics
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Transformers version: {transformers.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Define emotion mapping 
emotion_mapping = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance", 4: "approval",
    5: "caring", 6: "confusion", 7: "curiosity", 8: "desire", 9: "disappointment",
    10: "disapproval", 11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy", 18: "love",
    19: "nervousness", 20: "optimism", 21: "pride", 22: "realization",
    23: "relief", 24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
}

target_emotions = list(emotion_mapping.values())
label_to_id = {emo: idx for idx, emo in enumerate(target_emotions)}

# Load CSV dataset
csv_file = "labeled_texts.csv"
if not os.path.exists(csv_file):
    print(f"Error: CSV file {csv_file} not found in the current directory.")
    sys.exit(1)

try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

required_columns = ["text", "label"]
if not all(col in df.columns for col in required_columns):
    print(f"Error: CSV must contain columns: {required_columns}")
    sys.exit(1)

if not all(df["label"].apply(lambda x: isinstance(x, (int, np.integer)) and x in emotion_mapping)):
    print("Error: CSV 'label' column must contain integers from 0 to 27 corresponding to emotion_mapping.")
    sys.exit(1)

dataset = Dataset.from_pandas(df)

# Preprocess: Map labels to emotion indices
def preprocess(example):
    emotion = emotion_mapping.get(example["label"])
    if emotion in target_emotions:
        return {"text": example["text"], "label": label_to_id[emotion]}
    return None

filtered_dataset = dataset.filter(lambda x: preprocess(x) is not None)
filtered_dataset = filtered_dataset.map(preprocess)

if len(filtered_dataset) == 0:
    print("Error: No valid samples remain after preprocessing. Check CSV data and labels.")
    sys.exit(1)

filtered_dataset = filtered_dataset.remove_columns([col for col in filtered_dataset.column_names if col not in ["text", "label"]])

# Train-validation split
train_test_split = filtered_dataset.train_test_split(test_size=0.2, seed=42)
dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# Tokenize
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)
tokenized_dataset = dataset_dict.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(
    [col for col in tokenized_dataset["train"].column_names if col not in ["text", "label", "input_ids", "attention_mask"]]
)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(target_emotions)
)
model.to(device)

batch_size = 8
approx_steps_per_epoch = len(tokenized_dataset["train"]) // batch_size + 1

# === Updated metrics for single-label ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=2e-5,
    save_strategy="steps",
    eval_strategy="steps",
    logging_steps=100,
    save_steps=approx_steps_per_epoch,
    eval_steps=approx_steps_per_epoch,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# Save model & tokenizer
model.save_pretrained("./mood_extractor_model")
tokenizer.save_pretrained("./mood_extractor_model")
print("Model and tokenizer saved to ./mood_extractor_model")

# Evaluate and save metrics
print("Final Evaluation Metrics on Validation Set:")
final_metrics = trainer.evaluate()
for k, v in final_metrics.items():
    print(f"{k}: {v:.4f}")

with open("final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=4)

print("Final evaluation metrics saved to final_metrics.json")

# GPU memory usage
if device.type == "cuda":
    print("GPU Memory Usage:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")
    print(f"  Cached:    {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f} MB")
