import sys
import os
import json
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
import transformers
from torch.nn import BCEWithLogitsLoss

# Suppress HuggingFace symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

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

# Define emotion columns
emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Step 1: Load combined CSV
combined_csv = "combined_goemotions.csv"
if not os.path.exists(combined_csv):
    print(f"Error: CSV file {combined_csv} not found.")
    sys.exit(1)

try:
    combined_df = pd.read_csv(combined_csv)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    sys.exit(1)

print(f"Loaded CSV: {combined_csv}")

# Verify required columns
required_columns = ["text"] + emotion_columns
if not all(col in combined_df.columns for col in required_columns):
    print(f"Error: CSV must contain columns: {required_columns}")
    sys.exit(1)

# Filter out unclear examples
combined_df = combined_df[combined_df["example_very_unclear"] == False]

# Step 2: Compute class weights based on label frequencies
label_counts = combined_df[emotion_columns].sum()
total_samples = len(combined_df)
# Compute inverse frequency weights, capped at 100 to avoid extreme values
class_weights = torch.tensor([min(1.0 / (count / total_samples), 100.0) for count in label_counts], dtype=torch.float32, device=device)
print("Class weights for emotions:", dict(zip(emotion_columns, class_weights.tolist())))

# Step 3: Prepare multi-label data
def get_labels(row):
    return [emotion for emotion in emotion_columns if row[emotion] == 1]

combined_df["labels"] = combined_df.apply(get_labels, axis=1)
mlb = MultiLabelBinarizer(classes=emotion_columns)
binary_labels = mlb.fit_transform(combined_df["labels"])
df = pd.DataFrame({
    "text": combined_df["text"],
    "labels": [list(row.astype(np.float32)) for row in binary_labels]
})

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
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
    [col for col in tokenized_dataset["train"].column_names if col not in ["text", "labels", "input_ids", "attention_mask"]]
)

# Ensure labels are float32
def format_labels(examples):
    examples["labels"] = torch.tensor(examples["labels"], dtype=torch.float32)
    return examples

tokenized_dataset = tokenized_dataset.map(format_labels, batched=False)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(emotion_columns),
    problem_type="multi_label_classification"
)
model.to(device)

# Custom Trainer with class-weighted loss, updated to handle num_items_in_batch
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = BCEWithLogitsLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(int)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }

# Training arguments
batch_size = 8
approx_steps_per_epoch = len(tokenized_dataset["train"]) // batch_size + 1
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
    fp16=(device.type == "cuda"),
    report_to="none",
)

# Initialize trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics
)

# Train
print("Starting training...")
trainer.train()

# Save model and tokenizer
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