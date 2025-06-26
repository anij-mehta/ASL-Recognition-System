import streamlit as st
import numpy as np
import joblib
import cv2
import mediapipe as mp

# Load the trained model
model = joblib.load("cnn_asl.joblib")
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_hand_landmarks(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
        return np.array(landmarks).reshape(1, -1)  # Reshape to (1, 42)
    else:
        return None

# Function to preprocess and predict
def predict_image(image_path):
    landmarks = extract_hand_landmarks(image_path)
    if landmarks is None:
        return "No hand detected"
    prediction = model.predict(landmarks)
    predicted_label = class_labels[np.argmax(prediction)]
    return predicted_label

# Streamlit UI
st.title("ASL Sign Language Recognition")
st.write("Upload an image of a hand sign, and the model will predict the corresponding letter.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Predicting...")
    
    # Save uploaded file temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    predicted_sign = predict_image("temp_image.jpg")
    st.write(f"Predicted Sign: **{predicted_sign}**")

st.markdown("""
    <style>
    .card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 30px;
    }
    </style>
    <div class="card">
    <h3>About the Project</h3>
    <p>This project leverages computer vision and machine learning to recognize American Sign Language (ASL) hand signs from images. 
    By uploading an image of a hand performing an ASL gesture, the model predicts the corresponding letter or symbol.</p>

    <p>The core components of this system include:</p>
    <ul>
        <li><strong>MediaPipe Hands:</strong> This library is used to detect and extract hand landmarks from the uploaded image.</li>
        <li><strong>Trained CNN Model:</strong> A Convolutional Neural Network (CNN) model is used to classify the hand signs based on the extracted landmarks.</li>
        <li><strong>Class Labels:</strong> The model is trained to recognize ASL letters from 'A' to 'Z', along with special signs like 'del' (delete), 'nothing' (no hand detected), and 'space' (space between letters).</li>
    </ul>
    </div>
""", unsafe_allow_html=True)
