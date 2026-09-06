import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback  # progress bar callback

# Load your labeled data from the JSON file (final_data.json)

def train(data):
    # Extract features and labels.
    # Each item is a dictionary with keys "filename" and "data".
    # "data" is a list: [background_rgb, flake_rgb, label]
    features = []
    labels = []
    for item in data:
        background, flake, label = item["data"]
        # Combine background and flake RGB values into a single feature vector (length 6)
        features.append(background + flake)
        labels.append(label)

    # Convert lists to numpy arrays.
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)  # Use int32 for class labels (0, 1, 2)

    # Normalize the RGB values (0-255 to 0-1)
    features = features / 255.0

    # Split the dataset into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Build a deep neural network model using Keras
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(6,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')  # 3 classes: for example, 0 = flake type 1, 1 = flake type 2, 2 = background
    ])

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Print a summary of the model architecture
    model.summary()

    # Set up an early stopping callback to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model (with 20% of training data used for validation)
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, TqdmCallback(verbose=1)]
    )

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.2f}")

# Save the trained model locally
    model.save('TIT_10x.h5')

if __name__ == "__main__":
    data_path = 'final_data.json'
    with open(data_path, 'r') as f:
        data = json.load(f)
    train(data)