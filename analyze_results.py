# analyze_results.py
import pandas as pd
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    print("ERROR: scikit-learn is not installed. Install it with: pip install scikit-learn")
    raise
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------------------
# CSV path
csv_path = "results/batch_results.csv"

# -------------------------------
# Check if CSV exists
if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at {csv_path}")
    exit()

# -------------------------------
# Load CSV
df = pd.read_csv(csv_path)
print("CSV loaded successfully!")
print("First 5 rows:")
print(df.head())

# -------------------------------
# Ensure required columns exist
required_cols = ["actual", "predicted"]
for col in required_cols:
    if col not in df.columns:
        print(f"ERROR: Column '{col}' not found in CSV")
        exit()

# -------------------------------
# Overall accuracy
total = len(df)
correct = (df["actual"].str.lower() == df["predicted"].str.lower()).sum()  # ignore capitalization
accuracy = correct / total * 100
print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")

# -------------------------------
# Emotion-wise accuracy
emotions = df["actual"].str.lower().unique()
print("\nAccuracy by emotion:")
for emo in emotions:
    subset = df[df["actual"].str.lower() == emo]
    acc = (subset["actual"].str.lower() == subset["predicted"].str.lower()).sum() / len(subset) * 100
    print(f"{emo}: {acc:.2f}% ({len(subset)} samples)")

# -------------------------------
# Confusion matrix
labels = sorted([e.capitalize() for e in emotions])  # keep order consistent
cm = confusion_matrix(df["actual"].str.capitalize(), df["predicted"].str.capitalize(), labels=labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Emotion Detection Confusion Matrix")
plt.tight_layout()

# Save the confusion matrix
os.makedirs("results", exist_ok=True)
plt.savefig("results/confusion_matrix.png")
print("\nConfusion matrix saved as results/confusion_matrix.png")

# -------------------------------
# Normalized confusion matrix (by actual/row)
cm_normalized = cm.astype('float')
row_sums = cm_normalized.sum(axis=1)
# avoid division by zero
row_sums[row_sums == 0] = 1
cm_normalized = (cm_normalized.T / row_sums).T
plt.figure(figsize=(6,5))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap="Blues", vmin=0, vmax=1)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Normalized Confusion Matrix (by Actual)")
plt.tight_layout()
plt.savefig("results/confusion_matrix_normalized.png")
print("Normalized confusion matrix saved as results/confusion_matrix_normalized.png")

# -------------------------------
# Per-emotion accuracy bar chart
acc_list = []
for emo in labels:
    # labels were capitalized; compare capitalized
    subset = df[df["actual"].str.capitalize() == emo]
    if len(subset) == 0:
        acc = 0.0
    else:
        acc = (subset["actual"].str.lower() == subset["predicted"].str.lower()).sum() / len(subset) * 100
    acc_list.append({"emotion": emo, "accuracy": acc, "samples": len(subset)})

acc_df = pd.DataFrame(acc_list)
plt.figure(figsize=(6,4))
sns.barplot(data=acc_df, x='emotion', y='accuracy', palette='viridis')
plt.ylim(0,100)
plt.xlabel('Emotion')
plt.ylabel('Accuracy (%)')
plt.title('Per-Emotion Accuracy')
for idx, row in acc_df.iterrows():
    plt.text(idx, row['accuracy'] + 1, f"{row['accuracy']:.1f}%\n(n={row['samples']})", ha='center')
plt.tight_layout()
plt.savefig("results/accuracy_by_emotion.png")
print("Per-emotion accuracy chart saved as results/accuracy_by_emotion.png")

# -------------------------------
# Structural plateau graph (running/cumulative accuracy with plateau detection)
correct_flags = (df["actual"].str.lower() == df["predicted"].str.lower()).astype(int)
n = len(correct_flags)
running_acc = correct_flags.cumsum() / np.arange(1, n+1) * 100

# rolling/smoothed accuracy for plateau detection
window = max(3, min(20, int(n * 0.1)))
rolling_acc = pd.Series(running_acc).rolling(window=window, min_periods=1).mean()

# detect plateau: first index where the maximum change over the next `window` samples is < epsilon
epsilon = 0.5  # percent points
plateau_index = None
for i in range(0, len(rolling_acc) - window):
    window_range = rolling_acc[i:i+window]
    if (window_range.max() - window_range.min()) < epsilon:
        plateau_index = i + window  # use end of window as plateau point
        break

plt.figure(figsize=(8,4))
plt.plot(range(1, n+1), running_acc, label='Cumulative accuracy', marker='o', markersize=4)
plt.plot(range(1, n+1), rolling_acc, label=f'Rolling mean (window={window})', linewidth=2)
if plateau_index is not None:
    plt.axvline(plateau_index, color='red', linestyle='--', label=f'Plateau at sample {plateau_index}')
    plt.text(plateau_index, running_acc[plateau_index-1] + 2, f"Plateau\n{running_acc[plateau_index-1]:.1f}%", color='red', ha='center')
plt.xlabel('Sample index')
plt.ylabel('Accuracy (%)')
plt.title('Structural Plateau: Cumulative Accuracy over Samples')
plt.ylim(0,100)
plt.legend()
plt.tight_layout()
plt.savefig("results/plateau_plot.png")
print("Structural plateau plot saved as results/plateau_plot.png")

plt.show()