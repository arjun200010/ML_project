import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np

# Load the pre-trained model
model_path = r'C:\Users\rahul\PycharmProjects\Arjun_DSML\model_cnn.h5'
model = load_model(model_path)

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Create a file uploader in the app
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the uploaded image
    processed_img = preprocess_image(uploaded_file)

    # Make predictions using the model
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)

    # Display the predicted class
    st.write(f"Predicted Class: {predicted_class}")
