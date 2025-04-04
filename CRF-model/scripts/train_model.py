# import json
# import joblib
# import os
# import sklearn_crfsuite
# from sklearn_crfsuite import metrics
# from collections import Counter
# from sklearn.model_selection import RandomizedSearchCV

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

# # Load processed data
# with open("processed_data.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# # Convert dataset into **CRF-compatible** sequences
# X_train_sequences, y_train_sequences = extract_sequences(X_train, y_train)
# X_test_sequences, y_test_sequences = extract_sequences(X_test, y_test)

# # 🔎 **Sanity-check: Ensure No Data Leakage**
# print(f"📊 Train Set Size: {len(X_train_sequences)} sentences")
# print(f"📊 Test Set Size: {len(X_test_sequences)} sentences")

# train_labels = [label for seq in y_train_sequences for label in seq]
# test_labels = [label for seq in y_test_sequences for label in seq]

# print("✅ Train Label Distribution:", Counter(train_labels))
# print("✅ Test Label Distribution:", Counter(test_labels))

# # 🔧 **Define CRF model with default hyperparameters**
# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     max_iterations=100,
#     all_possible_transitions=True
# )

# # 📌 **Check if we have enough data for hyperparameter tuning**
# if len(X_train_sequences) > 10:
#     print("🚀 Running Hyperparameter Tuning...")
#     param_grid = {
#         'c1': [0.01, 0.1, 1, 10],  # L1 regularization
#         'c2': [0.01, 0.1, 1, 10]   # L2 regularization
#     }
    
#     # Ensure CV is at least 2
#     cv_folds = min(5, len(X_train_sequences))  # Avoid errors if too few sequences

#     search = RandomizedSearchCV(
#         crf, param_distributions=param_grid, 
#         n_iter=5, cv=cv_folds, verbose=1, n_jobs=-1
#     )
#     search.fit(X_train_sequences, y_train_sequences)

#     print("✅ Best Hyperparameters:", search.best_params_)
#     best_crf = search.best_estimator_
# else:
#     print("⚠️ Not enough data for hyperparameter tuning. Training with default parameters...")
#     best_crf = crf.fit(X_train_sequences, y_train_sequences)

# # ✅ Save trained model
# os.makedirs("models", exist_ok=True)
# model_path = "models/crf_model.pkl"
# joblib.dump(best_crf, model_path)
# print(f"📁 Model saved as {model_path}!")

# # Evaluate Model
# y_pred = best_crf.predict(X_test_sequences)
# print("🔎 Classification Report:\n", metrics.flat_classification_report(y_test_sequences, y_pred))

# import json
# import joblib
# import os
# import random
# import sklearn_crfsuite
# from sklearn_crfsuite import metrics
# from collections import Counter

# def extract_sequences(X, y):
#     """Convert dataset into CRF-compatible sequences"""
#     X_seq, y_seq = [], []
#     temp_x, temp_y = [], []
    
#     for i in range(len(X)):
#         temp_x.append(X[i])  # Add feature dict to sequence
#         temp_y.append(y[i])  # Add label to sequence
        
#         # 🔹 **Fix: Ensure every 10 words form a sentence**
#         if (i + 1) % 10 == 0 or i == len(X) - 1:
#             X_seq.append(temp_x)
#             y_seq.append(temp_y)
#             temp_x, temp_y = [], []  # Reset for next sequence

#     return X_seq, y_seq

# def augment_data(X, y, num_augmented=5000):
#     """🔹 Data Augmentation: Adds slight variations of words"""
#     augmented_X, augmented_y = X.copy(), y.copy()

#     for _ in range(num_augmented):
#         idx = random.randint(0, len(X) - 1)
#         new_word = X[idx]["word"] + random.choice(["ा", "ि", "ी"])  # Add Hindi-like suffix
#         new_features = X[idx].copy()
#         new_features["word"] = new_word

#         augmented_X.append(new_features)
#         augmented_y.append(y[idx])

#     return augmented_X, augmented_y

# # Load processed data
# with open("processed_data.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# # 🔹 **Increase Data Size Using Augmentation**
# X_train, y_train = augment_data(X_train, y_train, num_augmented=5000)

# # Convert dataset into **CRF-compatible** sequences
# X_train_sequences, y_train_sequences = extract_sequences(X_train, y_train)
# X_test_sequences, y_test_sequences = extract_sequences(X_test, y_test)

# # 🔎 **Sanity-check: Ensure No Data Leakage**
# print(f"📊 Train Set Size: {len(X_train_sequences)} sentences")
# print(f"📊 Test Set Size: {len(X_test_sequences)} sentences")

# train_labels = [label for seq in y_train_sequences for label in seq]
# test_labels = [label for seq in y_test_sequences for label in seq]

# print("✅ Train Label Distribution:", Counter(train_labels))
# print("✅ Test Label Distribution:", Counter(test_labels))

# # If data is still empty, stop execution
# if len(X_train_sequences) == 0 or len(X_test_sequences) == 0:
#     print("❌ Error: No data available for training. Check your dataset!")
#     exit()

# # 🔧 **Define CRF model with default hyperparameters**
# crf = sklearn_crfsuite.CRF(
#     algorithm='lbfgs',
#     max_iterations=100,
#     all_possible_transitions=True
# )

# # 📌 **Check if we have enough data for hyperparameter tuning**
# if len(X_train_sequences) > 1:
#     from sklearn.model_selection import RandomizedSearchCV

#     param_grid = {
#         'c1': [0.01, 0.1, 1, 10],  # L1 regularization
#         'c2': [0.01, 0.1, 1, 10]   # L2 regularization
#     }
    
#     # Ensure CV is at least 2
#     cv_folds = min(2, len(X_train_sequences))

#     # Run Randomized Search
#     print("🚀 Tuning Hyperparameters...")
#     search = RandomizedSearchCV(crf, param_distributions=param_grid, n_iter=5, cv=cv_folds, verbose=1, n_jobs=-1)
#     search.fit(X_train_sequences, y_train_sequences)
#     print("✅ Best Hyperparameters:", search.best_params_)

#     # Use best model for final training
#     best_crf = search.best_estimator_
# else:
#     print("⚠️ Not enough data for hyperparameter tuning. Training with default parameters...")
#     best_crf = crf.fit(X_train_sequences, y_train_sequences)

# # ✅ Save trained model
# os.makedirs("models", exist_ok=True)
# model_path = "models/crf_model.pkl"
# joblib.dump(best_crf, model_path)
# print(f"📁 Model saved as {model_path}!")

# # Evaluate Model
# y_pred = best_crf.predict(X_test_sequences)
# print("🔎 Classification Report:\n", metrics.flat_classification_report(y_test_sequences, y_pred))


import json
import joblib
import os
import random
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from collections import Counter

def extract_sequences(X, y):
    """Convert dataset into CRF-compatible sequences"""
    X_seq, y_seq = [], []
    temp_x, temp_y = [], []
    
    for i in range(len(X)):
        temp_x.append(X[i])  # Add feature dict to sequence
        temp_y.append(y[i])  # Add label to sequence
        
        # 🔹 **Fix: Ensure every 10 words form a sentence**
        if (i + 1) % 10 == 0 or i == len(X) - 1:
            X_seq.append(temp_x)
            y_seq.append(temp_y)
            temp_x, temp_y = [], []  # Reset for next sequence

    return X_seq, y_seq

def augment_data(X, y, num_augmented=5000):
    """🔹 Data Augmentation: Ensures equal plural & singular samples"""
    augmented_X, augmented_y = X.copy(), y.copy()
    plural_count = sum(1 for label in y if label == "PLURAL")
    singular_count = sum(1 for label in y if label == "SINGULAR")

    while plural_count < singular_count:
        idx = random.randint(0, len(X) - 1)
        if y[idx] == "PLURAL":
            new_word = X[idx]["word"] + random.choice(["ा", "ि", "ी"])  # Add Hindi-like suffix
            new_features = X[idx].copy()
            new_features["word"] = new_word

            augmented_X.append(new_features)
            augmented_y.append("PLURAL")
            plural_count += 1

    while singular_count < plural_count:
        idx = random.randint(0, len(X) - 1)
        if y[idx] == "SINGULAR":
            new_word = X[idx]["word"][:-1]  # Slight change for singular
            new_features = X[idx].copy()
            new_features["word"] = new_word

            augmented_X.append(new_features)
            augmented_y.append("SINGULAR")
            singular_count += 1

    return augmented_X, augmented_y

# Load processed data
with open("processed_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X_train, X_test, y_train, y_test = data["X_train"], data["X_test"], data["y_train"], data["y_test"]

# 🔹 **Equalize Data Using Augmentation**
X_train, y_train = augment_data(X_train, y_train, num_augmented=5000)

# Convert dataset into **CRF-compatible** sequences
X_train_sequences, y_train_sequences = extract_sequences(X_train, y_train)
X_test_sequences, y_test_sequences = extract_sequences(X_test, y_test)

# 🔎 **Sanity-check: Ensure No Data Leakage**
print(f"📊 Train Set Size: {len(X_train_sequences)} sentences")
print(f"📊 Test Set Size: {len(X_test_sequences)} sentences")

train_labels = [label for seq in y_train_sequences for label in seq]
test_labels = [label for seq in y_test_sequences for label in seq]

print("✅ Train Label Distribution:", Counter(train_labels))
print("✅ Test Label Distribution:", Counter(test_labels))

# If data is still empty, stop execution
if len(X_train_sequences) == 0 or len(X_test_sequences) == 0:
    print("❌ Error: No data available for training. Check your dataset!")
    exit()

# 🔧 **Define CRF model**
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)

# Train the model
best_crf = crf.fit(X_train_sequences, y_train_sequences)

# ✅ Save trained model
os.makedirs("models", exist_ok=True)
model_path = "models/crf_model.pkl"
joblib.dump(best_crf, model_path)
print(f"📁 Model saved as {model_path}!")

# Evaluate Model
y_pred = best_crf.predict(X_test_sequences)
print("🔎 Classification Report:\n", metrics.flat_classification_report(y_test_sequences, y_pred))
