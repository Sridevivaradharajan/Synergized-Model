## Synergized Model for Predicting Sepsis and Jaundice

This project integrates two machine learning models for predicting **Sepsis** (from tabular clinical data) and detecting **Jaundice** (from eye images), with a unified web interface built using **Streamlit**. The project aims to showcase early disease prediction using AI techniques in a user-friendly application.

---

## Project Structure

- `xgboost_sepsis_model/` — Code and trained model for sepsis prediction using tabular data.
- `keras_model/` — Code and trained CNN model for jaundice detection from images.
- `app/` — Web interface allowing users to interact with both models.
- `Dataset/` — Sample data files for testing the models.
- `requirements.txt` — All necessary Python packages.

---

## How to Run
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Sridevivaradharajan/Synergized-Model.git
   cd Synergized-Model
2. Install Required Libraries pip install -r requirements.txt
3. Run the Streamlit App streamlit run app.py

---

## Models Used
🔹Sepsis Prediction

Model: XGBoost Classifier (XGBClassifier)

Input: Clinical tabular data (e.g., HR, O2Sat, MAP, Resp, etc.)

Output: Binary classification — Sepsis or No Sepsis

Code Adapted From: Lakshya Soni’s Kaggle Notebook

Dataset: Prediction of Sepsis Dataset

🔹Jaundice Detection

Model: Convolutional Neural Network (CNN) based on Inception architecture

Input: Eye images (categorized as Normal vs Jaundiced)

Output: Classification — Normal or Jaundiced

Code: Fully implemented by me. The CNN model architecture, training logic, and class balancing methods were built from scratch using Keras and TensorFlow.

Dataset: A sample image per class is included for demonstration. The full dataset remains undisclosed.

Note: The full training code is not publicly shared here to maintain code originality and prevent unauthorized use.

---

## Web Interface
The unified Streamlit app allows:

-Uploading a user input for Sepsis prediction

-Uploading an image for Jaundice detection

-Visualizing prediction outputs with confidence levels

-Educational demonstration of healthcare AI

---

## Acknowledgements
-Sepsis prediction logic and code were adapted from:Lakshya Soni’s Kaggle Notebook

-Sepsis dataset used:Prediction of Sepsis Dataset


**Note:** To maintain the originality and integrity of the work, the complete jaundice training code and dataset have not been disclosed in full. Only a minimal setup is shared for demonstration purposes.The Artificial Intelligence models for both sepsis and jaundice were trained independently and are original to this project.The Streamlit web interface was designed and implemented as a unified platform to demonstrate both models interactively.
No direct code from the above sources was reused without modification. Instead, concepts and logic were adapted and extended.

---

## License

This project is proprietary and all rights are reserved by the author.

You may not copy, modify, reuse, distribute, or incorporate any part of this project into other works without explicit written permission from the author.

For inquiries, visit: https://github.com/Sridevivaradharajan


