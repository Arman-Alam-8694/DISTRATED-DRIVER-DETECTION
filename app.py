    
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.image as mpimg
import os

# Load pre-trained model
model = load_model(r"C:\Users\arman\Downloads\model\model.h5")

# Function to predict image class
def predict_image_class(image_path):
    img = keras_image.load_img(image_path, target_size=(100, 100, 3))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Function to get the meaning of the predicted class
def get_class_meaning(class_index):
    class_meanings = {
        0: 'Safe driving',
        1: 'Texting - right',
        2: 'Talking on the phone - right',
        3: 'Texting - left',
        4: 'Talking on the phone - left',
        5: 'Operating the radio',
        6: 'Drinking',
        7: 'Reaching behind',
        8: 'Hair and makeup',
        9: 'Talking to passenger'
    }
    return class_meanings.get(class_index, 'Unknown class')

# Streamlit UI for image upload and prediction
st.title("Distracted Driver Detection")
st.write("Upload an image to classify distracted driving behavior.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    
    # Predict the class of the uploaded image
    predicted_class = predict_image_class("temp_image.jpg")
    class_meaning = get_class_meaning(predicted_class)
    
    # Display the prediction
    st.write(f"Predicted Class: {predicted_class} - {class_meaning}")

    # Display the image again for clarity
    img = mpimg.imread("temp_image.jpg")
    st.image(img, use_column_width=True)

    # Clean up temporary image
    os.remove("temp_image.jpg")