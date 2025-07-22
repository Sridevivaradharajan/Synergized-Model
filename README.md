## Synergized Model for Predicting Sepsis and Jaundice

This project integrates two machine learning models for predicting **Sepsis** (from tabular clinical data) and detecting **Jaundice** (from eye images), with a unified web interface built using **Streamlit**. The project aims to showcase early disease prediction using AI techniques in a user-friendly application.

---

## Project Structure

- `sepsis_model/` — Code and trained model for sepsis prediction using tabular data.
- `jaundice_model/` — Code and trained CNN model for jaundice detection from images.
- `streamlit_app/` — Web interface allowing users to interact with both models.
- `data/` — Sample data files for testing the models.
- `requirements.txt` — All necessary Python packages.

---

## How to Run
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. Install Required Libraries pip install -r requirements.txt
3. Run the Streamlit App streamlit run autism_app.py

## Models Used
Sepsis Prediction

Model: XGBoost Classifier (XGBClassifier)

Input: Clinical tabular data (e.g., HR, O2Sat, MAP, Resp, etc.)

Output: Binary classification — Sepsis or No Sepsis

Code Adapted From: Lakshya Soni’s Kaggle Notebook

Dataset: Prediction of Sepsis Dataset

Jaundice Detection

Model: Convolutional Neural Network (CNN)

Input: Eye images (categorized as Normal vs Jaundiced)

Output: Classification — Normal or Jaundiced

Approach Adapted From: Nirmal Gaud’s Notebook

Dataset: Eye images were preprocessed and adapted based on the method in the above source

## Web Interface
The unified Streamlit app allows:

-Uploading a user input for Sepsis prediction

-Uploading an image for Jaundice detection

-Visualizing prediction outputs with confidence levels

-Educational demonstration of healthcare AI

## Acknowledgements
-Sepsis prediction logic and code were adapted from:Lakshya Soni’s Kaggle Notebook

-Sepsis dataset used:Prediction of Sepsis Dataset

-Jaundice classification method adapted from:Nirmal Gaud’s Notebook

**Note:** The machine learning models for both sepsis and jaundice were trained independently and are original to this project.
The Streamlit web interface was designed and implemented as a unified platform to demonstrate both models interactively.
No direct code from the above sources was reused without modification. Instead, concepts and logic were adapted and extended.

