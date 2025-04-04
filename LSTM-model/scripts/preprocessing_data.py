# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split

# # Load hin.txt
# file_path = "../data/hin.txt"
# data = []
# with open(file_path, "r", encoding="utf-8") as file:
#     for line in file:
#         parts = line.strip().split("\t")
#         if len(parts) == 3:
#             base_word, inflected_word, tags = parts
#             if "PL" in tags:
#                 label = 1  # 1 for PLURAL
#             elif "SG" in tags:
#                 label = 0  # 0 for SINGULAR
#             else:
#                 continue
#             data.append((inflected_word, label))

# # Split into train & test
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# # Tokenize words
# words = [word for word, _ in data]
# tokenizer = Tokenizer(char_level=True)  # Character-level tokenization
# tokenizer.fit_on_texts(words)

# # Convert words to sequences
# def convert_data(data):
#     words, labels = zip(*data)
#     sequences = tokenizer.texts_to_sequences(words)
#     sequences = pad_sequences(sequences, padding="post")  # Pad to same length
#     return sequences, np.array(labels)

# X_train, y_train = convert_data(train_data)
# X_test, y_test = convert_data(test_data)

# # Save processed data
# with open("../data/preprocessed.json", "w", encoding="utf-8") as f:
#     json.dump({
#         "X_train": X_train.tolist(),  
#         "y_train": y_train.tolist(),  
#         "X_test": X_test.tolist(),    
#         "y_test": y_test.tolist(),    
#         "word_index": tokenizer.word_index  # ✅ Ensure this is added
#     }, f, indent=4, ensure_ascii=False)  

# print("✅ Data preprocessed and saved!")


import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load hin.txt
file_path = "../data/hin.txt"
data = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            base_word, inflected_word, tags = parts
            if "PL" in tags:
                label = 1  # 1 for PLURAL
            elif "SG" in tags:
                label = 0  # 0 for SINGULAR
            else:
                continue
            data.append((inflected_word, label))

# Shuffle to ensure random distribution
np.random.shuffle(data)

# Ensure balanced dataset before splitting
singular_data = [d for d in data if d[1] == 0]
plural_data = [d for d in data if d[1] == 1]
min_samples = min(len(singular_data), len(plural_data))

balanced_data = singular_data[:min_samples] + plural_data[:min_samples]
np.random.shuffle(balanced_data)

# Split into train & test
train_data, test_data = train_test_split(balanced_data, test_size=0.2, random_state=42)

# Tokenize words
words = [word for word, _ in balanced_data]
tokenizer = Tokenizer(char_level=True, filters="")  # Character-level tokenization
tokenizer.fit_on_texts(words)

# Convert words to sequences
def convert_data(data):
    words, labels = zip(*data)
    sequences = tokenizer.texts_to_sequences(words)
    max_length = max(len(seq) for seq in sequences)  # Dynamic padding length
    sequences = pad_sequences(sequences, maxlen=max_length, padding="post")
    return sequences, np.array(labels)

X_train, y_train = convert_data(train_data)
X_test, y_test = convert_data(test_data)

# Save processed data
with open("../data/preprocessed.json", "w", encoding="utf-8") as f:
    json.dump({
        "X_train": X_train.tolist(),
        "y_train": y_train.tolist(),
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "word_index": tokenizer.word_index,
        "max_length": X_train.shape[1]  # Save max length for consistency
    }, f, indent=4, ensure_ascii=False)

print(f"✅ Data preprocessed and saved! Train size: {len(y_train)}, Test size: {len(y_test)}")

