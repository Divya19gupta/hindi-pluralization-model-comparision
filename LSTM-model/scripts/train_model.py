# import json
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional

# # Load preprocessed data
# with open("../data/preprocessed.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# X_train = np.array(data["X_train"])
# y_train = np.array(data["y_train"])
# word_index = data["word_index"]

# # Define LSTM Model
# model = Sequential([
#     Embedding(input_dim=len(word_index) + 1, output_dim=32, input_length=X_train.shape[1]),
#     Bidirectional(LSTM(32)),  # Bidirectional for better learning
#     Dense(1, activation="sigmoid")  # Binary classification
# ])

# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # Train model
# print("🚀 Training LSTM model...")
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# # Save model
# model.save("models/lstm_model.h5")
# print("✅ LSTM Model Trained & Saved!")


import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load preprocessed data
with open("../data/preprocessed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X_train = np.array(data["X_train"])
y_train = np.array(data["y_train"])
word_index = data["word_index"]
max_length = data["max_length"]

# Define LSTM Model
model = Sequential([
    Embedding(input_dim=len(word_index) + 1, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),  # More LSTM units for better feature extraction
    Dropout(0.3),  # Helps prevent overfitting
    Bidirectional(LSTM(32)),  # Added another LSTM layer
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # Binary classification
])

# Compile model with learning rate scheduler
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss="binary_crossentropy", metrics=["accuracy"])

# Learning rate adjustment
lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)

# Train model
print("🚀 Training LSTM model with hyperparameter tuning...")
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, 
                    callbacks=[lr_scheduler], verbose=2)

# Save model
model.save("models/lstm_model.h5")
print("✅ LSTM Model Trained & Saved!")

# Save training history for analysis
with open("../data/training_history.json", "w") as f:
    json.dump(history.history, f)
