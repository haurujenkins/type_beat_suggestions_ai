from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import librosa
import os
import shutil
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import gc

# --- VARIABLES GLOBALES ---
DATASET_PATH = "data/dataset_audio.csv"
DF_AUDIO = None
SCALER = None
MODEL_KNN = None
FEATURE_COLUMNS = []

# --- LIFESPAN (Gestion au démarrage) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global DF_AUDIO, SCALER, MODEL_KNN, FEATURE_COLUMNS
    print("🔄 Démarrage de l'API & Entraînement du modèle à la volée...")
    
    try:
        # 1. Charger le CSV
        # On suppose que le CWD est la racine du projet, donc data/ est accessible
        # Sinon ajuster le path
        if not os.path.exists(DATASET_PATH):
            # Fallback si on est dans le dossier backend
            if os.path.exists("../data/dataset_audio.csv"):
                 DATASET_PATH_FIX = "../data/dataset_audio.csv"
            else:
                 print(f"⚠️ Warning: Dataset introuvable à {DATASET_PATH}")
                 DATASET_PATH_FIX = DATASET_PATH
        else:
            DATASET_PATH_FIX = DATASET_PATH

        if os.path.exists(DATASET_PATH_FIX):
            df = pd.read_csv(DATASET_PATH_FIX)
            print(f"📊 Dataset brut chargé : {df.shape}")

            # 2. Nettoyage et Sélection des Features
            # On ne garde que les colonnes numériques pertinentes
            # La consigne est d'utiliser les MOYENNES ('_mean') et le 'tempo'.
            # On exclut explicitement les variances ('_var') pour correspondre à la logique "light".
            cols_to_keep = [c for c in df.columns if c == 'tempo' or c.endswith('_mean')]
            
            # Filtrer le DF (Exclut filename, label, et les variances)
            df_features = df[cols_to_keep].copy()
            
            # Nettoyage des NaN éventuels
            df_features = df_features.dropna()
            
            # Sauvegarde la liste finale des colonnes pour l'alignement lors de l'inférence
            FEATURE_COLUMNS = df_features.columns.tolist()
            print(f"🎯 Features sélectionnées ({len(FEATURE_COLUMNS)}) : {FEATURE_COLUMNS[:5]}...")

            # 3. Entraînement du Scaler
            SCALER = StandardScaler()
            X_scaled = SCALER.fit_transform(df_features)
            
            # 4. Entraînement du modèle KNN
            # Metric 'cosine' est souvent meilleure pour l'audio que 'euclidean'
            MODEL_KNN = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
            MODEL_KNN.fit(X_scaled)
            
            # 5. Stockage du 'référentiel' (pour récupérer les infos artistes après prédiction)
            # On garde le DF original aligné avec X_scaled (attention aux index si dropna)
            DF_AUDIO = df.loc[df_features.index].reset_index(drop=True)
            
            # Nettoyage RAM
            del df
            del df_features
            del X_scaled
            gc.collect()
            
            print("✅ Modèle entraîné et prêt !")
        else:
            print("❌ IMPOSSIBLE de charger le dataset. L'API ne pourra pas prédire.")
        
    except Exception as e:
        print(f"❌ Erreur critique au démarrage : {e}")
        # On ne quitte pas forcément, mais l'API sera dégradée
    
    yield
    
    # Clean up à l'extinction
    print("🛑 Arrêt de l'API. Nettoyage mémoire.")
    del DF_AUDIO
    del SCALER
    del MODEL_KNN
    gc.collect()

# --- APP SETUP ---
app = FastAPI(title="Type Beat AI API (On-the-fly)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- FONCTION D'EXTRACTION OPTIMISÉE ---
def extract_features(audio_path):
    """
    Extrait exactement les features attendues par le modèle (FEATURE_COLUMNS).
    Charge max 30s de l'audio.
    """
    try:
        # 1. Chargement léger (30s)
        y, sr = librosa.load(audio_path, duration=30)
        
        # Features dict
        features = {}
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # Features spectrales (Moyennes)
        features['rms_mean'] = np.mean(librosa.feature.rms(y=y))
        features['spec_cent_mean'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['spec_bw_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        features['spec_roll_mean'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # Chroma (12 notes) - Moyennes
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            
        # MFCC (20 coefficients) - Moyennes
        # Note : Le CSV peut en contenir moins (ex: 13), le filtrage final gérera ça.
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            
        # 2. Alignement et Construction du Vecteur
        # On ne renvoie QUE les colonnes connues du modèle, dans le BON ORDRE.
        # Si une colonne manque (ex: 'spec_bw_mean' non présent dans le CSV), elle est ignorée.
        # Si une colonne du CSV manque ici (ex: variances), on a un problème -> Mais on a filtré sur '_mean' au load.
        
        final_vector = []
        if FEATURE_COLUMNS is None:
             print("⚠️ Modèle non initialisé, pas de colonnes de features.")
             return None

        for col in FEATURE_COLUMNS:
            if col in features:
                final_vector.append(features[col])
            else:
                # Fallback critique : si une feature attendue n'est pas calculée
                # Pour éviter le crash, on met 0.0, mais ça ne devrait pas arriver si le CSV et ce code sont synchrones.
                # print(f"⚠️ Feature manquante lors de l'extraction : {col}") 
                # (Commenté pour éviter le spam de logs si spec_bw manque)
                final_vector.append(0.0)
                
        return np.array(final_vector)

    except Exception as e:
        print(f"Erreur extraction audio: {e}")
        return None

# --- ROUTES ---

@app.get("/")
def read_root():
    return {"status": "API is running", "model_loaded": MODEL_KNN is not None}

@app.post("/predict")
async def predict_type_beat(file: UploadFile = File(...)):
    """
    Reçoit un fichier audio, l'analyse, et retourne les 5 beats les plus proches.
    """
    if MODEL_KNN is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt ou le dataset a échoué.")
    
    # Création dossier temp si besoin
    os.makedirs("temp_audio", exist_ok=True)
    
    # Chemin temporaire
    file_location = f"temp_audio/{file.filename}"
    
    try:
        # 1. Sauvegarde disque
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Extraction
        features_vector = extract_features(file_location)
        
        if features_vector is None or len(features_vector) == 0:
            raise HTTPException(status_code=400, detail="Impossible d'extraire les features audio.")
            
        # Reshape pour predict (1, n_features)
        features_vector = features_vector.reshape(1, -1)
        
        # 3. Scaling
        scaled_features = SCALER.transform(features_vector)
        
        # 4. Recherche KNN
        distances, indices = MODEL_KNN.kneighbors(scaled_features)
        
        # 5. Récupération des Résultats
        recommendations = []
        
        for i, idx in enumerate(indices[0]):
            neighbor_row = DF_AUDIO.iloc[idx]
            dist = float(distances[0][i])
            
            # Gestion des champs qui peuvent manquer dans le CSV
            filename = str(neighbor_row.get('filename', 'Unknown'))
            label = str(neighbor_row.get('label', 'Unknown'))
            
            recommendations.append({
                "rank": i + 1,
                "filename": filename,
                "label": label, 
                "distance": round(dist, 4),
                "preview_path": f"/audio/{filename}" # Exemple
            })
            
        return {
            "input_filename": file.filename,
            "recommendations": recommendations
        }

    except Exception as e:
        print(f"Erreur route predict: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Nettoyage fichier temp
        if os.path.exists(file_location):
            try:
                os.remove(file_location)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    # Pour tester en local
    uvicorn.run(app, host="0.0.0.0", port=8000)
