from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import os

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

try:
    # Load the model
    model_path = r"C:/Detection/keras_model.h5"
    model = load_model(model_path, compile=True)
    print(f"Model loaded successfully from {model_path}")

    # Load the labels
    labels_path = r"C:/Detection/labels.txt"
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    class_names = open(labels_path, "r").readlines()
    print("Labels loaded successfully.")

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image_path = r"C:\Detection\PreProcess Dataset\Positive\im2.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    # Resizing the image to 224x224
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predict using the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print(f"Class: {class_name}")
    print(f"Confidence Score: {confidence_score:.2f}")

except FileNotFoundError as e:
    print(e)
except Exception as e:
    print(f"An error occurred: {e}")
