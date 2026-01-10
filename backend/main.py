from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import numpy as np
import pandas as pd
import librosa
from model_loader import load_ai_models

# --- CONFIGURATION ---
app = FastAPI(title="Type Beat AI API")

# Configuration CORS (CRITIQUE pour que le frontend Vercel puisse parler au backend)
origins = [
    "http://localhost:3000",          # Pour le dev local
    "https://votre-app.vercel.app",   # Remplacez par votre URL Vercel réelle
    "*"                               # À restreindre en production si possible
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CHARGEMENT DU MODÈLE (Au démarrage) ---
# On charge une seule fois pour éviter de le refaire à chaque requête
MODEL, SCALER, ENCODER, EXPECTED_FEATURES = load_ai_models("models")

# --- FONCTIONS UTILITAIRES ---
def extract_visual_features(file_path):
    """
    Réplique exacte de la logique d'extraction utilisée dans l'app Streamlit
    pour garantir la cohérence des prédictions.
    """
    SAMPLE_RATE = 22050
    DURATION_SLICE = 30
    
    try:
        y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        total_samples = len(y_full)
        samples_per_slice = sr * DURATION_SLICE
        
        # Découpage (Slicing)
        if total_samples < samples_per_slice:
            y_slices = [np.pad(y_full, (0, samples_per_slice - total_samples), 'constant')]
        else:
            y_slices = []
            for i in range(0, total_samples, samples_per_slice):
                segment = y_full[i : i + samples_per_slice]
                if len(segment) >= (sr * 10): # Ignorer segments trop courts (<10s)
                    if len(segment) < samples_per_slice:
                        segment = np.pad(segment, (0, samples_per_slice - len(segment)), 'constant')
                    y_slices.append(segment)
        
        all_features_list = []
        
        for y in y_slices:
            # Extraction
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            zcr = librosa.feature.zero_crossing_rate(y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
            
            features = {"tempo": tempo_val}
            
            # Helper pour cleaner le code
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
        print(f"Erreur extraction: {e}")
        return None

# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "Type Beat AI Backend is ready"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Format de fichier non supporté.")

    temp_filename = f"temp_{file.filename}"
    
    try:
        # 1. Sauvegarder le fichier temporairement
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Extraire les features
        features_df = extract_visual_features(temp_filename)
        
        if features_df is None or features_df.empty:
            raise HTTPException(status_code=422, detail="Impossible d'extraire les caractéristiques audio.")

        # 3. Préparer les données (Alignement colonnes)
        if EXPECTED_FEATURES is not None:
            for col in EXPECTED_FEATURES:
                if col not in features_df.columns:
                    features_df[col] = 0
            input_df = features_df[EXPECTED_FEATURES]
        else:
            input_df = features_df

        # 4. Standardisation
        X_scaled = SCALER.transform(input_df)

        # 5. Prédiction (Moyenne des probabilités des slices)
        probas = MODEL.predict_proba(X_scaled)
        avg_probas = probas.mean(axis=0)
        
        # 6. Formatting Result
        classes = ENCODER.classes_
        results = list(zip(classes, avg_probas))
        results.sort(key=lambda x: x[1], reverse=True)
        
        top_result = results[0]
        winner_artist = top_result[0]
        confidence = float(top_result[1])

        # Top 5 details
        top_5 = []
        for artist, score in results[:5]:
            top_5.append({"artist": artist, "score": float(score)})

        return {
            "prediction": winner_artist,
            "confidence": confidence,
            "details": top_5
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Nettoyage
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
