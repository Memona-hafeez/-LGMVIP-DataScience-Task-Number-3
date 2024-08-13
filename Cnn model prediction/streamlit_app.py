import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your model
model = load_model('cnn_model.h5')

# Function to make a prediction
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)

# Streamlit app interface
st.title('CNN Model Prediction')

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict_image(uploaded_file)
    st.write(f"Predicted label: {label}")
