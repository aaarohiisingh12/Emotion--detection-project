import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# -------------------------------
# loading saved results from emotion detection
# csv has actual vs predicted labels
# refer to pg 88 
df = pd.read_csv("your_file.csv")

# checking if prediction is correct or not
df["correct"] = df["actual"] == df["predicted"]

# setting graph style (just for better visuals)
sns.set(style="whitegrid")

# -------------------------------
# confusion matrix
# shows how many times each emotion was predicted correctly or wrongly
cm = confusion_matrix(df["actual"], df["predicted"])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# accuracy for each emotion separately
# helps to see which emotion works well and which fails
accuracy_by_emotion = df.groupby("actual")["correct"].mean()

plt.figure(figsize=(7,5))
sns.barplot(x=accuracy_by_emotion.index, y=accuracy_by_emotion.values)

plt.ylabel("Accuracy")
plt.xlabel("Emotion")
plt.title("Accuracy by Emotion")
plt.ylim(0,1)

plt.savefig("accuracy_by_emotion.png", dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# overall accuracy of the system
overall_accuracy = df["correct"].mean()

plt.figure(figsize=(4,4))
plt.bar(["Accuracy"], [overall_accuracy])

plt.ylim(0,1)
plt.title("Overall Accuracy")

plt.savefig("overall_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()

# printing final result
print("Overall Accuracy:", overall_accuracy)
print("Graphs saved as PNG files.")