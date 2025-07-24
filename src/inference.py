import sys
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess  # or paste the function above here

# Load the trained model
model = load_model('models/image_forgery_detector.h5')

def predict_image(image_path):
    x = preprocess(image_path)
    pred = model.predict(x[None, ...])[0][0]
    label = "FORGED" if pred > 0.5 else "AUTHENTIC"
    return label, pred

if __name__ == "__main__":
    # Example usage: python src/inference.py test_images/anyfile.jpg
    test_img_path = sys.argv[1]
    label, conf = predict_image(test_img_path)
    print(f"Prediction: {label}, Confidence: {conf:.2f}")

