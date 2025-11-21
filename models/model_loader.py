import pickle
import os

MODEL_PATH = "models/iris_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Train model first.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
