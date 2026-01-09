import os
import joblib

def check_models():
    print(f"CWD: {os.getcwd()}")
    
    model_path = "models/type_beat_model.pkl"
    scaler_path = "models/scaler.pkl"
    encoder_path = "models/encoder.pkl"
    
    print(f"Checking {model_path}: {os.path.exists(model_path)}")
    print(f"Checking {scaler_path}: {os.path.exists(scaler_path)}")
    print(f"Checking {encoder_path}: {os.path.exists(encoder_path)}")
    
    if os.path.exists(model_path):
        print(f"Size {model_path}: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    check_models()