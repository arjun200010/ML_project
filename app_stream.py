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

    # Define a dictionary mapping class labels to plant species names
    species_names = {0: "Aloevera", 1: "Banana", 2: "Bilimbi" ,3 : "Cantaloupe" ,4 : "Cassava",5 : "Coconut" ,6 : "Coconut" , 7 : "Corn" , 8 : "Cucumber" , 9 : "Curcuma" , 10 : "Eggplant" ,11 : "Galangal" , 12 : "Ginger" , 13 : "Guava",14 : "Kale" , 15 : "Longbean" ,16 :  "Mango" ,17: "Melon",18 : "Orange",19 : "Paddy",20 : "Papaya"   }  # Update with your actual mappings

    # Define a dictionary mapping plant species names to NPK ratios
    npk_ratios = {"Aloevera": "1:1:2", "Banana": "1:2:6", "Bilimbi": "1:1:2","Cantaloupe": "2:3:4" ,"Cassava" : "2:3:4","Coconut" : "3:1:6","Corn" : "2:1:3","Cucumber" : "1:2:3","Curcuma" : "1:1:2","Eggplant" : "1:2:3","Galangal" : "2:3:3","Ginger" : "2:3:3","Guava" : "2:2:4","Kale" : "2:3:3","Longbean" : "2:3:3","Mango": "2:3:3","Melon" : "2:3:3","Orange" : "2:3:3","Paddy" : "2:1:1","Papaya" : "1:1:3"}  # Update with actual ratios

    # Check if the predicted class exists in the dictionary
    if predicted_class in species_names:
        predicted_species = species_names[predicted_class]
        st.write(f"Predicted Species: {predicted_species}")

        # Check if the predicted species has an NPK ratio available
        if predicted_species in npk_ratios:
            st.write(f"NPK Ratio for {predicted_species} is : {npk_ratios[predicted_species]}")
        else:
            st.write(f"NPK Ratio not available for {predicted_species}")
    else:
        st.write("Species not found")
