# 📦 Import Libraries
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import pickle

# 🔍 Load the trained model
model = tf.keras.models.load_model("hindi_character_model.h5")

# 🏷️ Load the LabelEncoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# 🎨 Streamlit UI Setup
st.title("📝 Hindi Handwritten Character Recognition")
st.write("📤 Upload a 32x32 grayscale image of a Hindi handwritten character.")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 🖼️ Load and preprocess the image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    display_image = image.resize((32, 32))
    st.image(display_image, caption="🖼️ Resized Image (32×32)", width=150)

    image = display_image
    img_array = np.array(image) / 255.0  # 🔄 Normalize pixel values
    img_array = img_array.reshape(1, 32, 32, 1)  # ➡️ Reshape for model

    # 🤖 Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    # 🏷️ Decode label
    predicted_label = le.inverse_transform([predicted_index])[0]

    # 📊 Display result
    st.subheader("🔎 Prediction")
    st.write(f"**🅰️ Character:** {predicted_label}")
    st.write(f"**📈 Confidence:** {confidence:.2%}")