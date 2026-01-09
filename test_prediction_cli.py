import os
import joblib
import librosa
import numpy as np
import pandas as pd
import glob
import random
from sklearn.preprocessing import StandardScaler

# --- COPIED FROM APP.PY (To modify app.py without breaking it, we test logic here) ---

def load_system():
    # Paths
    model_path = "models/type_beat_model.pkl"
    scaler_path = "models/scaler.pkl"
    encoder_path = "models/encoder.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
        print("⚠️ Modèles introuvables !")
        return None, None, None, None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        expected_features = getattr(scaler, 'feature_names_in_', None)
        return model, scaler, encoder, expected_features
    except Exception as e:
        print(f"Erreur de chargement du modèle: {e}")
        return None, None, None, None

def extract_features_silent(file_path):
    SAMPLE_RATE = 22050
    DURATION_SLICE = 30
    try:
        y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        total_samples = len(y_full)
        samples_per_slice = sr * DURATION_SLICE
        
        if total_samples < samples_per_slice:
            y_slices = [np.pad(y_full, (0, samples_per_slice - total_samples), 'constant')]
        else:
            y_slices = []
            for i in range(0, total_samples, samples_per_slice):
                segment = y_full[i : i + samples_per_slice]
                if len(segment) >= (sr * 10): 
                    if len(segment) < samples_per_slice:
                        segment = np.pad(segment, (0, samples_per_slice - len(segment)), 'constant')
                    y_slices.append(segment)
        
        all_features_list = []
        for y in y_slices:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            zcr = librosa.feature.zero_crossing_rate(y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
            features = {"tempo": tempo_val}
            
            for name, data in [("rms", rms), ("zcr", zcr), ("spec_cent", spec_cent), ("spec_roll", spec_roll)]:
                features[f"{name}_mean"] = np.mean(data)
                features[f"{name}_var"]  = np.var(data)

            for i in range(13):
                features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
                features[f"mfcc_{i}_var"]  = np.var(mfccs[i])
            for i in range(12):
                features[f"chroma_{i}_mean"] = np.mean(chroma[i])
                features[f"chroma_{i}_var"]  = np.var(chroma[i])
                
            all_features_list.append(features)
            
        return pd.DataFrame(all_features_list)
    except Exception as e:
        print(f"Error extraction: {e}")
        return None

def main():
    print("Initializing logic test...")
    model, scaler, encoder, feature_cols = load_system()
    if model is None: return

    # Search for a file
    # We look in data/audio_test recursively
    files = glob.glob("data/audio_test/**/*.mp3", recursive=True) or \
            glob.glob("data/audio_test/**/*.m4a", recursive=True)
    
    if not files:
        print("No files found to test in data/audio_test.")
        return

    test_file = random.choice(files)
    print(f"Testing on file: {test_file}")
    
    df = extract_features_silent(test_file)
    if df is None or df.empty:
        print("Extraction failed.")
        return
        
    # Logic
    if feature_cols is not None:
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        input_df = df[feature_cols]
    else:
        input_df = df.drop(columns=['filename'], errors='ignore')
        
    X_scaled = scaler.transform(input_df)
    probas = model.predict_proba(X_scaled) 
    avg_probas = probas.mean(axis=0)
    
    results = list(zip(encoder.classes_, avg_probas))
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\nCorrectly Predicted:")
    print(f"🥇 Winner: {results[0][0]} ({results[0][1]*100:.1f}%)")
    print("Top 5:")
    for art, p in results[:5]:
        print(f"  {art}: {p*100:.1f}%")

if __name__ == "__main__":
    main()
