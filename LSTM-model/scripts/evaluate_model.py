# import json
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, accuracy_score

# # Load preprocessed data
# with open("../data/preprocessed.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# X_test = np.array(data["X_test"])
# y_test = np.array(data["y_test"])

# # Load trained model
# model = tf.keras.models.load_model("models/lstm_model.h5")

# # Predict
# y_pred_probs = model.predict(X_test)
# y_pred = (y_pred_probs > 0.5).astype(int)  # Convert probabilities to binary labels

# # Classification report
# print("🔎 Classification Report:\n")
# print(classification_report(y_test, y_pred, digits=3))

# # Compute accuracy
# overall_accuracy = accuracy_score(y_test, y_pred)
# plural_accuracy = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])  # Accuracy for plural cases

# print(f"📌 Overall Model Accuracy: {overall_accuracy:.3f}")
# print(f"📌 Pluralization Accuracy: {plural_accuracy:.3f}")

# # Plot Accuracy Graphs
# labels = ["Overall Accuracy", "Pluralization Accuracy"]
# values = [overall_accuracy, plural_accuracy]

# plt.figure(figsize=(6, 4))
# plt.bar(labels, values, color=["blue", "green"])
# plt.ylim(0, 1)
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy Comparison")
# plt.show()


# import json
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# # Load preprocessed data
# with open("../data/preprocessed.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# X_test = np.array(data["X_test"])
# y_test = np.array(data["y_test"])

# # Load trained model
# model = tf.keras.models.load_model("models/lstm_model.h5")

# # Predict
# y_pred_probs = model.predict(X_test)
# y_pred = (y_pred_probs > 0.5).astype(int)

# # Classification report
# print("🔎 Classification Report:\n")
# print(classification_report(y_test, y_pred, digits=3))

# # Compute accuracy
# overall_accuracy = accuracy_score(y_test, y_pred)
# plural_accuracy = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])  # Accuracy for plural cases
# singular_accuracy = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])  # Accuracy for singular cases

# print(f"📌 Overall Model Accuracy: {overall_accuracy:.3f}")
# print(f"📌 Singular Accuracy: {singular_accuracy:.3f}")
# print(f"📌 Plural Accuracy: {plural_accuracy:.3f}")

# # Error Analysis - Show incorrect predictions
# incorrect_indices = np.where(y_pred != y_test)[0]
# if len(incorrect_indices) > 0:
#     print("\n❌ Incorrect Predictions (First 5 shown):")
#     for idx in incorrect_indices[:5]:
#         print(f"Word: {data['X_test'][idx]} | True: {y_test[idx]} | Predicted: {y_pred[idx]}")

# # Plot Accuracy Graphs
# labels = ["Overall Accuracy", "Singular Accuracy", "Plural Accuracy"]
# values = [overall_accuracy, singular_accuracy, plural_accuracy]

# plt.figure(figsize=(6, 4))
# plt.bar(labels, values, color=["blue", "green", "red"])
# plt.ylim(0, 1)
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy Comparison")
# plt.show()

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed data
with open("../data/preprocessed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X_test = np.array(data["X_test"])
y_test = np.array(data["y_test"])
word_index = data["word_index"]
max_length = data["max_length"]

# Reverse word index to map indices to characters
index_to_char = {index: char for char, index in word_index.items()}

# Load trained model
model = tf.keras.models.load_model("models/lstm_model.h5")

# Predict
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()  # Ensure it's a 1D array

# Classification report
print("🔎 Classification Report:\n")
print(classification_report(y_test, y_pred, digits=3))

# Compute accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
plural_accuracy = accuracy_score(y_test[y_test == 1], y_pred[y_test == 1])  # Accuracy for plural cases
singular_accuracy = accuracy_score(y_test[y_test == 0], y_pred[y_test == 0])  # Accuracy for singular cases

print(f"📌 Overall Model Accuracy: {overall_accuracy:.3f}")
print(f"📌 Singular Accuracy: {singular_accuracy:.3f}")
print(f"📌 Plural Accuracy: {plural_accuracy:.3f}")

# Error Analysis - Show incorrect predictions with actual words
incorrect_indices = np.where(y_pred != y_test)[0]  # Only where predictions are wrong
unique_incorrect_words = set()  # To avoid duplicates

if len(incorrect_indices) > 0:
    print("\n❌ Incorrect Predictions (First 5 unique errors shown):")
    for idx in incorrect_indices:
        # Convert sequence of character indices back to word
        char_indices = X_test[idx]  # Character indices of the word
        word = "".join(index_to_char.get(i, "?") for i in char_indices if i != 0)  # Convert indices to characters
        
        true_label = "Plural" if y_test[idx] == 1 else "Singular"
        predicted_label = "Plural" if y_pred[idx] == 1 else "Singular"

        if (word, true_label, predicted_label) not in unique_incorrect_words:
            print(f"Word: {word} | True: {true_label} | Predicted: {predicted_label}")
            unique_incorrect_words.add((word, true_label, predicted_label))

        if len(unique_incorrect_words) >= 5:  # Stop after 5 unique errors
            break

# Plot Accuracy Graphs
labels = ["Overall Accuracy", "Singular Accuracy", "Plural Accuracy"]
values = [overall_accuracy, singular_accuracy, plural_accuracy]

plt.figure(figsize=(6, 4))
plt.bar(labels, values, color=["blue", "green", "red"])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()

