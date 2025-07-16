import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

heart_model = pickle.load(open("heart_model.pkl", "rb"))
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
try:
    pneumonia_model = load_model("pneumonia_model.h5")
except:
    pneumonia_model = None

def predict_heart(data):
    return heart_model.predict([data])[0]

def predict_diabetes(data):
    return diabetes_model.predict([data])[0]

def predict_pneumonia(img_file):
    if pneumonia_model is None:
        return "Model Not Available"
    img = image.load_img(img_file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = pneumonia_model.predict(img_array)
    return "Pneumonia" if prediction[0][0] > 0.5 else "Normal"
