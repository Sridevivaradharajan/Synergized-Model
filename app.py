import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from PIL import Image

# Load Models
with open("xgboost_sepsis_model.pkl", "rb") as f:
    sepsis_model = pickle.load(f)

jaundice_model = tf.keras.models.load_model("keras_model.h5", compile=False)

# Define Feature Names for Sepsis
selected_features = ["Hour", "HR", "O2Sat", "Temp", "MAP", "Resp", "BUN", "Chloride", 
                     "Creatinine", "Glucose", "Hct", "Hgb", "WBC", "Platelets"]

# Function to load and resize images
def load_and_resize_image(image_path, target_size=(400, 400)):  # Increased size
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)  # Resize to larger size
        return img
    except:
        return None  # Return None if there's an error

# Streamlit Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Sepsis Prediction", "Jaundice Prediction"])

# Home Page
if page == "Home":
    st.title("🩺 Healthcare Prediction System")
    st.markdown("### AI-powered predictions for **Sepsis** & **Jaundice** using medical data & images.")

    col1, col2 = st.columns(2)
    
    img1 = load_and_resize_image("1736724598574.jpg")
    img2 = load_and_resize_image("1691577720897.jpg")
    
    if img1:
        with col1:
            st.image(img1, caption="Sepsis Symptoms", use_container_width=True)
    
    if img2:
        with col2:
            st.image(img2, caption="Jaundice Symptoms", use_container_width=True)

    st.subheader("Why Early Detection Matters?")
    st.write("- **Sepsis**: A life-threatening condition requiring immediate intervention.")
    st.write("- **Jaundice**: Early diagnosis can prevent complications in newborns and adults.")
    st.write("Use the sidebar to navigate to prediction tools.")

# Sepsis Prediction Page
elif page == "Sepsis Prediction":
    st.title("Sepsis Prediction")
    st.write("Enter patient data to predict the likelihood of sepsis.")

    # Input fields for user data
    input_data = [st.number_input(f"{feature}", value=0.0) for feature in selected_features]
    input_array = np.array(input_data).reshape(1, -1)

    if st.button("Predict Sepsis"):
        try:
            sepsis_prob = sepsis_model.predict_proba(input_array)[0]  # Get probabilities
            prob_no_sepsis, prob_sepsis = round(sepsis_prob[0], 2), round(sepsis_prob[1], 2)
            predicted_class = "Sepsis" if prob_sepsis > prob_no_sepsis else "No Sepsis"
            st.success(f"XGBoost Prediction: **{predicted_class}**")
            st.write(f"**Probability of Sepsis:** {prob_sepsis}")
            st.write(f"**Probability of No Sepsis:** {prob_no_sepsis}")
        except:
            st.error("An error occurred. Please check input values.")

# Jaundice Prediction Page
elif page == "Jaundice Prediction":
    st.title("Jaundice Prediction")
    st.write("Upload an image to predict jaundice.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_resized = image.resize((400, 400))  # Increased size
        st.image(image_resized, caption="Uploaded Image", use_container_width=True)

        # Preprocess image for the model
        img_resized = image.resize((224, 224))  # Resize to model's input shape
        img_array = np.array(img_resized) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        if st.button("Predict Jaundice"):
            try:
                confidence_score = jaundice_model.predict(img_array)[0][0]  # Get model output
                confidence_percentage = round(confidence_score * 100, 2)
                predicted_class = "Jaundice" if confidence_score > 0.5 else "No Jaundice"
                st.success(f"Prediction: **{predicted_class}**")
                st.write(f"**Confidence Score:** {confidence_percentage}%")
            except:
                st.error("Error processing the image. Please upload a valid image.")
