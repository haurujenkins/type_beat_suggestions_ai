from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import numpy as np
import librosa
import csv
import gc
import ctypes
from model_loader import load_ai_models

# --- CONFIGURATION OPTIMISÉE ---
app = FastAPI(title="Type Beat AI API (Light)")

# Configuration CORS
origins = [
    "http://localhost:3000",
    "https://votre-app.vercel.app",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CHARGEMENT DU MODÈLE (GLOBAL & OPTIMISÉ) ---
MODEL, SCALER, ENCODER, EXPECTED_FEATURES = load_ai_models("models")
# Force un clean immédiat après le chargement lourd
gc.collect()

# --- CACHE LÉGER POUR LES MÉTHADONNÉES ---
METADATA_CACHE = {}

def load_artist_metadata_light():
    """Charge artist_popularity.csv sans Pandas (csv natif)."""
    global METADATA_CACHE
    if METADATA_CACHE:
        return METADATA_CACHE
        
    csv_path = "data/artist_popularity.csv"
    # Fallback si le dossier data n'est pas à la racine du backend
    if not os.path.exists(csv_path):
        # Essayer de chercher 'dataset_audio.csv' si c'est ce que l'utilisateur voulait
        # Mais on privilégie artist_popularity pour la légèreté
        return {}

    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # On ne garde que l'essentiel pour économiser la RAM
                try:
                    search_name = row.get('search_name', 'Unknown')
                    views = float(row.get('youtube_avg_views', 0))
                    METADATA_CACHE[search_name] = {'views': views}
                except ValueError:
                    continue
        print(f"✅ Metadata chargées : {len(METADATA_CACHE)} artistes.")
        return METADATA_CACHE
    except Exception as e:
        print(f"⚠️ Erreur chargement metadata : {e}")
        return {}

# Chargeons les metadata au démarrage
load_artist_metadata_light()


# --- FONCTIONS UTILITAIRES ---
def extract_features_light(file_path):
    """
    Extraction ULTRA-LÉGÈRE (Memory Optimized).
    Ne charge que les 30 premières secondes une seule fois.
    """
    SAMPLE_RATE = 22050
    DURATION_SLICE = 30
    
    try:
        # --- CHARGEMENT UNIQUE ET LIMITÉ ---
        # On charge SEULEMENT 30 secondes, et on force le sample rate
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION_SLICE)
        
        # Ignorer si trop court (<5s)
        if len(y) < (sr * 5):
            return []
            
        # Padding si < 30s
        target_length = sr * DURATION_SLICE
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        
        # --- EXTRACTION ---
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)

        tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
        
        features = {"tempo": tempo_val}
        
        features["rms_mean"] = float(np.mean(rms))
        features["rms_var"] = float(np.var(rms))
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_var"] = float(np.var(zcr))
        features["spec_cent_mean"] = float(np.mean(spec_cent))
        features["spec_cent_var"] = float(np.var(spec_cent))
        features["spec_roll_mean"] = float(np.mean(spec_roll))
        features["spec_roll_var"] = float(np.var(spec_roll))

        for i in range(13):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i}_var"]  = float(np.var(mfccs[i]))
        for i in range(12):
            features[f"chroma_{i}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{i}_var"]  = float(np.var(chroma[i]))
            
        # Libération explicite
        del y, rms, zcr, spec_cent, spec_roll, mfccs, chroma
        
        return [features] # Single dict in list
        
    except Exception as e:
        print(f"Erreur extraction: {e}")
        return []

# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "online", "mode": "light_optimized"}

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Format non supporté")

    temp_filename = f"temp_{file.filename}"
    
    try:
        # 1. Sauvegarde streamée pour limiter la RAM
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Extraction
        features_list_dicts = extract_features_light(temp_filename)
        
        if not features_list_dicts:
            raise HTTPException(status_code=422, detail="Échec extraction audio")

        # 3. Alignement des colonnes (Critique : Scaler attend un ordre précis)
        X_input = []
        if EXPECTED_FEATURES is not None:
            # Conversion optimisée
            feature_names = list(EXPECTED_FEATURES)
            for f_dict in features_list_dicts:
                # On utilise .get(k, 0.0) pour la robustesse
                row = [f_dict.get(k, 0.0) for k in feature_names]
                X_input.append(row)
        else:
            # Fallback (Dangereux si l'ordre change, mais mieux que crash)
            for f_dict in features_list_dicts:
                X_input.append(list(f_dict.values()))

        # 4. Conversion Numpy (Type float32 pour économiser 50% RAM vs float64)
        X_np = np.array(X_input, dtype=np.float32)

        # 5. Prédiction
        # Transform
        X_scaled = SCALER.transform(X_np)
        
        # Predict Proba
        probas = MODEL.predict_proba(X_scaled)
        
        # Moyenne
        avg_probas = probas.mean(axis=0)
        
        # 6. Mapping Classes
        classes = ENCODER.classes_
        results = []
        for i, class_name in enumerate(classes):
            results.append((class_name, float(avg_probas[i])))
            
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 7. Enrichissement avec Metadata (Sans Pandas)
        top_5 = []
        metadata = load_artist_metadata_light()
        
        for artist, score in results[:5]:
            meta = metadata.get(artist, {})
            top_5.append({
                "artist": artist,
                "score": score,
                "views": meta.get('views', 0)
            })

        top_result = top_5[0]

        # Clean Memory
        del X_input
        del X_np
        del X_scaled
        del probas
        gc.collect()

        return {
            "prediction": top_result["artist"],
            "confidence": top_result["score"],
            "details": top_5
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        gc.collect()
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except:
            pass
