import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_crfsuite import metrics
from sklearn.metrics import confusion_matrix

# Load processed data
with open("processed_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X_test, y_test = data["X_test"], data["y_test"]

# Load trained model
crf = joblib.load("models/crf_model.pkl")

# Convert test data into CRF-compatible format (sequence-based)
def extract_sequences(X, y):
    """Convert dataset into CRF-compatible sequences"""
    X_seq, y_seq = [], []
    temp_x, temp_y = [], []
    
    for i in range(len(X)):
        temp_x.append(X[i])  # Add feature dict to sequence
        temp_y.append(y[i])  # Add label to sequence
        
        # Assume each sentence ends when a word has a period (.)
        if X[i]['word'].endswith('.') or i == len(X) - 1:
            X_seq.append(temp_x)
            y_seq.append(temp_y)
            temp_x, temp_y = [], []  # Reset for next sequence

    return X_seq, y_seq

X_test_sequences, y_test_sequences = extract_sequences(X_test, y_test)

# Predict labels using CRF model
y_pred = crf.predict(X_test_sequences)

# Ensure y_test and y_pred have the same length
assert len(y_test_sequences) == len(y_pred), "Mismatch in test data and predictions!"

# Print classification report
print("🔎 Classification Report:\n")
print(metrics.flat_classification_report(y_test_sequences, y_pred, digits=3))

# Compute accuracy
accuracy = metrics.flat_f1_score(y_test_sequences, y_pred, average='weighted')

# ----------------------------
# 🔍 ERROR ANALYSIS: Incorrect Predictions
# ----------------------------
incorrect_preds = []

for sentence_idx in range(len(y_test_sequences)):  # Iterate through sentences
    for word_idx in range(len(y_test_sequences[sentence_idx])):  # Iterate through words
        true_label = y_test_sequences[sentence_idx][word_idx]
        pred_label = y_pred[sentence_idx][word_idx]
        word = X_test_sequences[sentence_idx][word_idx]['word']
        
        if true_label != pred_label:
            incorrect_preds.append((word, true_label, pred_label))

# Show First 5 Incorrect Predictions
print("\n❌ Incorrect Predictions:")
for word, true_label, pred_label in incorrect_preds[:5]:
    print(f"Word: {word} | True: {true_label} | Predicted: {pred_label}")

print("\n🔍 Why Does CRF Fail in Some Cases?")
print("1️⃣ Some words may be ambiguous in singular/plural form.")
print("2️⃣ CRF depends on **context features**, which may not always capture exceptions.")
print("3️⃣ Some rare words may not have enough examples in training data.")

# ----------------------------
# 📊 CONFUSION MATRIX
# ----------------------------
labels = ["SINGULAR", "PLURAL"]
y_test_flat = [label for sublist in y_test_sequences for label in sublist]
y_pred_flat = [label for sublist in y_pred for label in sublist]

conf_matrix = confusion_matrix(y_test_flat, y_pred_flat, labels=labels)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for CRF Model")
plt.show()

# ----------------------------
# 📊 MODEL ACCURACY PLOT
# ----------------------------
plt.figure(figsize=(5, 3))
plt.bar(["Accuracy"], [accuracy], color='blue')
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Accuracy")
plt.show()


# import json
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn_crfsuite import metrics
# from sklearn.metrics import confusion_matrix
# from collections import Counter

# # ----------------------------
# # 📥 LOAD DATA & MODEL
# # ----------------------------

# # Load processed data
# with open("processed_data.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# X_test, y_test = data["X_test"], data["y_test"]

# # Load trained **best-tuned** CRF model
# crf = joblib.load("models/crf_model_tuned.pkl")

# # Convert test data into CRF-compatible format (sequence-based)
# def extract_sequences(X, y):
#     """Convert dataset into CRF-compatible sequences"""
#     X_seq, y_seq = [], []
#     temp_x, temp_y = [], []
    
#     for i in range(len(X)):
#         temp_x.append(X[i])  # Add feature dict to sequence
#         temp_y.append(y[i])  # Add label to sequence
        
#         # Assume each sentence ends when a word has a period (.)
#         if X[i]['word'].endswith('.') or i == len(X) - 1:
#             X_seq.append(temp_x)
#             y_seq.append(temp_y)
#             temp_x, temp_y = [], []  # Reset for next sequence

#     return X_seq, y_seq

# X_test_sequences, y_test_sequences = extract_sequences(X_test, y_test)

# # Predict labels using CRF model
# y_pred = crf.predict(X_test_sequences)

# # Ensure y_test and y_pred have the same length
# assert len(y_test_sequences) == len(y_pred), "Mismatch in test data and predictions!"

# # ----------------------------
# # 🔍 CLASSIFICATION REPORT
# # ----------------------------

# print("🔎 Classification Report:\n")
# print(metrics.flat_classification_report(y_test_sequences, y_pred, digits=3))

# # Compute accuracy
# accuracy = metrics.flat_f1_score(y_test_sequences, y_pred, average='weighted')

# # ----------------------------
# # 🔍 ERROR ANALYSIS: Incorrect Predictions
# # ----------------------------

# incorrect_preds = []
# for sentence_idx in range(len(y_test_sequences)):  # Iterate through sentences
#     for word_idx in range(len(y_test_sequences[sentence_idx])):  # Iterate through words
#         true_label = y_test_sequences[sentence_idx][word_idx]
#         pred_label = y_pred[sentence_idx][word_idx]
#         word = X_test_sequences[sentence_idx][word_idx]['word']
        
#         if true_label != pred_label:
#             sentence = " ".join([w['word'] for w in X_test_sequences[sentence_idx]])  # Get full sentence
#             incorrect_preds.append((word, true_label, pred_label, sentence))

# # Show First 5 Incorrect Predictions with Context
# print("\n❌ Incorrect Predictions:")
# for word, true_label, pred_label, sentence in incorrect_preds[:5]:
#     print(f"Word: {word} | True: {true_label} | Predicted: {pred_label} | Context: \"{sentence}\"")

# # 🔎 **Why Does CRF Fail in Some Cases?**
# print("\n🔍 Possible Causes for Incorrect Predictions:")
# print("1️⃣ Some words may be ambiguous in singular/plural form.")
# print("2️⃣ CRF depends on **context features**, which may not always capture exceptions.")
# print("3️⃣ Some rare words may not have enough examples in training data.")

# # ----------------------------
# # 📊 CLASS-SPECIFIC ACCURACY
# # ----------------------------
# y_test_flat = [label for sublist in y_test_sequences for label in sublist]
# y_pred_flat = [label for sublist in y_pred for label in sublist]

# class_accuracy = {
#     label: np.mean([y_pred_flat[i] == label for i in range(len(y_pred_flat)) if y_test_flat[i] == label])
#     for label in set(y_test_flat)
# }

# print("\n📊 Class-Specific Accuracy:")
# for label, acc in class_accuracy.items():
#     print(f"{label}: {acc:.3f}")

# # ----------------------------
# # 📊 CONFUSION MATRIX
# # ----------------------------
# labels = sorted(set(y_test_flat))  # Ensure labels match actual test set

# conf_matrix = confusion_matrix(y_test_flat, y_pred_flat, labels=labels)

# plt.figure(figsize=(6, 5))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix for CRF Model")
# plt.show()

# # ----------------------------
# # 📊 MODEL ACCURACY PLOT
# # ----------------------------
# plt.figure(figsize=(5, 3))
# plt.bar(["Accuracy"], [accuracy], color='blue')
# plt.ylim(0, 1)
# plt.ylabel("Score")
# plt.title("Model Accuracy")
# plt.show()
