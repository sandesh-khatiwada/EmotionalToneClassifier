import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Define emotion columns (same as in your training script)
emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Step 1: Load the combined CSV
combined_csv = "combined_goemotions.csv"
if not os.path.exists(combined_csv):
    print(f"Error: CSV file {combined_csv} not found.")
    exit(1)

try:
    df = pd.read_csv(combined_csv)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

# Step 2: Verify emotion columns
if not all(col in df.columns for col in emotion_columns):
    print(f"Error: CSV must contain all emotion columns: {emotion_columns}")
    exit(1)

# Step 3: Filter out unclear examples (as in your training script)
df = df[df["example_very_unclear"] == False]

# Step 4: Calculate label frequencies
label_counts = df[emotion_columns].sum()  # Sum of 1s for each emotion
total_samples = len(df)
label_percentages = (label_counts / total_samples) * 100  # Percentage of samples with each emotion

# Step 5: Print statistics
print(f"Total samples after filtering: {total_samples}")
print("\nLabel Frequencies (count and percentage):")
for emotion, count in label_counts.items():
    percentage = label_percentages[emotion]
    print(f"{emotion}: {count} samples ({percentage:.2f}%)")

print("\nSummary Statistics:")
print(f"Mean percentage: {label_percentages.mean():.2f}%")
print(f"Standard deviation: {label_percentages.std():.2f}%")
print(f"Min percentage: {label_percentages.min():.2f}% ({label_percentages.idxmin()})")
print(f"Max percentage: {label_percentages.max():.2f}% ({label_percentages.idxmax()})")
print(f"Imbalance ratio (max/min): {label_percentages.max() / label_percentages.min():.2f}")

# Step 6: Visualize the distribution
plt.figure(figsize=(12, 6))
label_percentages.sort_values(ascending=False).plot(kind='bar')
plt.title('Emotion Label Distribution in GoEmotions Dataset')
plt.xlabel('Emotion')
plt.ylabel('Percentage of Samples (%)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('emotion_distribution.png')
plt.show()
print("Bar plot saved as 'emotion_distribution.png'")