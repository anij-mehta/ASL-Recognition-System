import os
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---- Load Dataset ----
def load_dataset(dataset_path):
    print(f"Loading dataset from {dataset_path}...")
    labels = os.listdir(dataset_path)
    data = []
    target = []

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    
    for label in labels:
        label_path = os.path.join(dataset_path, label)
        for file in os.listdir(label_path):
            img_path = os.path.join(label_path, file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmarks.append(landmark.x)
                        landmarks.append(landmark.y)
                data.append(landmarks)
                target.append(label)
    
    hands.close()

    print(f"Total samples: {len(data)}")
    
    return np.array(data), np.array(target)

# ---- Data Augmentation ----
def augment_images(dataset_path):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True
    )
    return datagen

# ---- Preprocess Data ----
def preprocess_data(X, y):
    label_map = {label: idx for idx, label in enumerate(np.unique(y))}
    y_encoded = np.array([label_map[label] for label in y])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_map

# ---- Model Training ----
def create_cnn(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---- Train and Evaluate Models ----
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    dump(model, f"{model_name.lower().replace(' ', '_')}_asl.joblib")
    print(f"{model_name} model saved!\n")
    
    return model

# ---- Main Function ----
def main():
    start_time = time.time()
    dataset_path = 'C:\\Users\\anijm\\Downloads\\archive\\asl_alphabet_train\\asl_alphabet_train\\' 
    X, y = load_dataset(dataset_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, label_map = preprocess_data(X, y)

    # Create and train models
    cnn_model = create_cnn(X_train.shape[1], len(label_map))

    cnn_model = train_and_evaluate(cnn_model, X_train, y_train, X_test, y_test, "CNN")
    
    # Evaluate Performance
    y_pred = np.argmax(cnn_model.predict(X_test), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_map.keys()))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    print(f"\nExecution completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
