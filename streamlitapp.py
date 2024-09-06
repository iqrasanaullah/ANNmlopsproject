import streamlit as st
import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# App Title
st.title("Digit Classification with Neural Networks")
st.write("""
This app predicts the label of an uploaded digit image (0-9) using a neural network model trained on the digits dataset.
""")

# Load dataset
digits = load_digits()
X = digits.data
y = digits.target

# One-hot encode the labels
y = to_categorical(y)

# Neural network model architecture
def create_model():
    model = Sequential()
    model.add(Dense(12, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # 10 output classes for digits 0-9
    return model

# Compile and train the model
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the entire dataset
with st.spinner('Training the model...'):
    model.fit(X, y, epochs=200, batch_size=10, verbose=0)

st.success("Model trained successfully!")

# Accepting image upload from user
st.write("### Upload an Image of a Digit (0-9) for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the uploaded image
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = img.resize((8, 8))  # Resize to match the dataset (8x8 pixels)
    img_array = img_to_array(img) / 16.0  # Scale pixel values to match training data (0-16 range)
    img_array = img_array.flatten().reshape(1, -1)  # Flatten and reshape for prediction
    
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    pred = model.predict(img_array)
    predicted_digit = np.argmax(pred)
    
    st.write(f"### Predicted Digit: {predicted_digit}")