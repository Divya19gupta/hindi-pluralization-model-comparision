# import json
# import re
# from collections import Counter
# from sklearn.model_selection import train_test_split

# def load_data(file_path):
#     """Load and process data from hin.txt"""
#     data = []
    
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split("\t")
#             if len(parts) != 3:
#                 continue  # Skip malformed lines
            
#             root, inflected, features = parts
#             features_set = set(features.split(";"))
            
#             # Determine number (Singular/Plural)
#             number = "PLURAL" if "PL" in features_set else "SINGULAR"
            
#             data.append({
#                 "word": inflected,
#                 "root": root,
#                 "pos": features.split(";")[0],  # First feature = POS
#                 "number": number
#             })
    
#     return data

# def extract_features_labels(data):
#     """Extract features and labels"""
#     X, y = [], []
    
#     for entry in data:
#         features = {
#             "word": entry["word"].lower(),
#             "pos": entry["pos"],
#             "length": len(entry["word"]),
#             "suffix": entry["word"][-3:] if len(entry["word"]) >= 3 else entry["word"],  # Handle short words
#         }
#         labels = entry["number"]
        
#         X.append(features)
#         y.append(labels)
    
#     return X, y

# # Load dataset
# raw_data = load_data("../data/hin.txt")

# # Sanity check: Data distribution
# label_counts = Counter(entry["number"] for entry in raw_data)
# print("🔍 Label Distribution Before Split:", label_counts)

# # Extract features and labels
# X, y = extract_features_labels(raw_data)

# # Split into train/test sets (Stratified to maintain balance)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42, stratify=y)

# # Sanity check after split
# print("✅ Train Label Distribution:", Counter(y_train))
# print("✅ Test Label Distribution:", Counter(y_test))

# # Save processed data
# with open("processed_data.json", "w", encoding="utf-8") as f:
#     json.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, f, ensure_ascii=False, indent=2)

# print("✅ Data preprocessing completed successfully!")


import json
import re
from collections import Counter
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load and process data from hin.txt"""
    data = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue  # Skip malformed lines
            
            root, inflected, features = parts
            features_set = set(features.split(";"))
            
            # Determine number (Singular/Plural)
            number = "PLURAL" if "PL" in features_set else "SINGULAR"
            
            data.append({
                "word": inflected,
                "root": root,
                "pos": features.split(";")[0],  # First feature = POS
                "number": number
            })
    
    return data

def extract_features_labels(data):
    """Extract features and labels with contextual features"""
    X, y = [], []
    
    for i, entry in enumerate(data):
        word = entry["word"].lower()
        
        # Extract suffixes (Hindi plurals often change suffixes)
        suffix_3 = word[-3:] if len(word) >= 3 else word
        suffix_2 = word[-2:] if len(word) >= 2 else word
        
        # Extract features
        features = {
            "word": word,
            "pos": entry["pos"],
            "length": len(word),
            "suffix_3": suffix_3,
            "suffix_2": suffix_2
        }

        # 🔹 Contextual Features (Previous & Next Words)
        if i > 0:
            prev_word = data[i - 1]["word"].lower()
            features["prev_word"] = prev_word
            features["prev_suffix_3"] = prev_word[-3:] if len(prev_word) >= 3 else prev_word
        else:
            features["prev_word"] = "START"

        if i < len(data) - 1:
            next_word = data[i + 1]["word"].lower()
            features["next_word"] = next_word
            features["next_suffix_3"] = next_word[-3:] if len(next_word) >= 3 else next_word
        else:
            features["next_word"] = "END"

        X.append(features)
        y.append(entry["number"])
    
    return X, y

# Load dataset
raw_data = load_data("../data/hin.txt")

# Sanity check: Data distribution
label_counts = Counter(entry["number"] for entry in raw_data)
print("🔍 Label Distribution Before Split:", label_counts)

# Extract features and labels
X, y = extract_features_labels(raw_data)

# Ensure **equal distribution** of singular & plural forms
min_count = min(label_counts["PLURAL"], label_counts["SINGULAR"])
X_balanced, y_balanced = [], []
plural_count, singular_count = 0, 0

for features, label in zip(X, y):
    if label == "PLURAL" and plural_count < min_count:
        X_balanced.append(features)
        y_balanced.append(label)
        plural_count += 1
    elif label == "SINGULAR" and singular_count < min_count:
        X_balanced.append(features)
        y_balanced.append(label)
        singular_count += 1

# Split into train/test sets (Stratified to maintain balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, train_size=0.7, random_state=42, stratify=y_balanced
)

# Sanity check after split
print("✅ Train Label Distribution:", Counter(y_train))
print("✅ Test Label Distribution:", Counter(y_test))

# Save processed data
with open("processed_data.json", "w", encoding="utf-8") as f:
    json.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, f, ensure_ascii=False, indent=2)

print("✅ Data preprocessing completed successfully!")
