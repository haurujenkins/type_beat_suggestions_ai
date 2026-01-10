import os
import shutil
import numpy as np
import pandas as pd
import librosa
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import warnings
import tempfile

# Ignorer les warnings de metadata audio (courant avec les mp3 web)
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset_audio.csv")
POPULARITY_PATH = os.path.join(BASE_DIR, "data", "artist_popularity.csv")

# Configuration du cache Numba pour éviter les erreurs de permission
os.environ['NUMBA_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), 'numba_cache')

DF_AUDIO = None
DF_POPULARITY = None
SCALER = None
MODEL_KNN = None
FEATURE_COLUMNS = []

# --- UTILITAIRES AUDIO OPTIMISÉS ---

def get_audio_content_optimized(audio_path, analyze_duration=30):
    """
    OPTIMISATION I/O:
    - Charge UNE SEULE FOIS un buffer de 30s situé au milieu du fichier.
    - Utilise res_type='linear' pour une vitesse maximale.
    - Fallback: Si échec (MP3 VBR, metadata), charge le début du fichier.
    """
    sr = 22050
    try:
        # 1. Tentative Optimisée : Duration -> Offset -> Load Slice
        total_duration = librosa.get_duration(path=audio_path)
        
        # 2. Calculer l'offset pour viser le milieu
        # Si le fichier est plus court que 30s, offset = 0
        if total_duration <= analyze_duration:
            offset = 0.0
            duration = None # Charger tout
        else:
            offset = (total_duration - analyze_duration) / 2
            duration = analyze_duration

        # 3. Chargement unique optimisé
        y, _ = librosa.load(
            audio_path,
            sr=sr,
            mono=True,
            offset=offset,
            duration=duration,
            res_type='linear' # Beaucoup plus rapide que kaiser_best/fast
        )
        
        # Padding si trop court
        target_samples = int(analyze_duration * sr)
        if len(y) < target_samples:
             y = np.pad(y, (0, max(0, target_samples - len(y))), 'constant')
             
        return y, sr
    
    except Exception as e:
        print(f"⚠️ Erreur chargement optimisé ({e}). Tentative de fallback (début du fichier)...")
        try:
            # 4. Fallback : Chargement simple des 30 premières secondes
            # Parfois get_duration/offset fail sur certains MP3 (VBR) ou headers incomplets.
            # On charge depuis le début (offset=0 par défaut), c'est plus robuste.
            y, _ = librosa.load(
                audio_path,
                sr=sr,
                mono=True,
                duration=analyze_duration,
                res_type='linear'
            )
            # Padding fallback
            target_samples = int(analyze_duration * sr)
            if len(y) < target_samples:
                y = np.pad(y, (0, max(0, target_samples - len(y))), 'constant')
            
            return y, sr
        except Exception as e2:
            print(f"❌ Echec total chargement audio: {e2}")
            return None, None

def extract_features(y, sr, precomputed_tempo=None):
    """
    Extrait les features d'un segment audio.
    Accepte un tempo pré-calculé pour éviter la redondance CPU.
    """
    features = {}
    
    try:
        # OPTIMISATION CPU: Utiliser le tempo global si fourni
        if precomputed_tempo is not None:
             features['tempo'] = float(precomputed_tempo)
        else:
             tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
             features['tempo'] = float(tempo)
        
        # Pre-calcul spectrogramme pour vitesse
        S_full = np.abs(librosa.stft(y))
        
        # Features Spectrales
        features['rms_mean'] = np.mean(librosa.feature.rms(S=S_full))
        features['rms_var'] = np.var(librosa.feature.rms(S=S_full))
        
        features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(y))
        features['zcr_var'] = np.var(librosa.feature.zero_crossing_rate(y))
        
        features['spec_cent_mean'] = np.mean(librosa.feature.spectral_centroid(S=S_full, sr=sr))
        features['spec_cent_var'] = np.var(librosa.feature.spectral_centroid(S=S_full, sr=sr))
        
        features['spec_bw_mean'] = np.mean(librosa.feature.spectral_bandwidth(S=S_full, sr=sr))
        features['spec_bw_var'] = np.var(librosa.feature.spectral_bandwidth(S=S_full, sr=sr))
        
        features['spec_roll_mean'] = np.mean(librosa.feature.spectral_rolloff(S=S_full, sr=sr))
        features['spec_roll_var'] = np.var(librosa.feature.spectral_rolloff(S=S_full, sr=sr))
        
        # Chroma
        chroma = librosa.feature.chroma_stft(S=S_full, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i}_var'] = np.var(chroma[i])
            
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_var'] = np.var(mfcc[i])

        return features

    except Exception as e:
        print(f"❌ Erreur extraction features: {e}")
        return None

# --- LIFESPAN (Gestion au démarrage) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global DF_AUDIO, DF_POPULARITY, SCALER, MODEL_KNN, FEATURE_COLUMNS
    print("🔄 Démarrage de l'API & Entraînement du modèle à la volée...")
    print(f"📂 Base directory: {BASE_DIR}")
    
    try:
        # 1. Charger la Popularité (Metadata)
        if os.path.exists(POPULARITY_PATH):
            try:
                DF_POPULARITY = pd.read_csv(POPULARITY_PATH)
                # Création clé de recherche normalisée
                DF_POPULARITY['search_key'] = DF_POPULARITY['search_name'].astype(str).str.lower().str.strip()
                print(f"🌟 Popularité chargée : {DF_POPULARITY.shape[0]} artistes depuis {POPULARITY_PATH}")
            except Exception as e:
                print(f"⚠️ Erreur chargement popularité : {e}")
        else:
            print(f"⚠️ Fichier popularité introuvable ({POPULARITY_PATH})")

        # 2. Charger le Dataset Audio & Train
        if os.path.exists(DATASET_PATH):
            df = pd.read_csv(DATASET_PATH)
            print(f"📊 Dataset brut chargé : {df.shape} depuis {DATASET_PATH}")
            
            # 3. Préparer les Features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_drop = ['Unnamed: 0', 'id']
            FEATURE_COLUMNS = [c for c in numeric_cols if c not in cols_to_drop]
            
            print(f"🔍 {len(FEATURE_COLUMNS)} features audio identifiées pour l'entraînement.")

            # 4. Entraînement (Scaling + KNN)
            X = df[FEATURE_COLUMNS].values
            X = np.nan_to_num(X)

            SCALER = StandardScaler()
            X_scaled = SCALER.fit_transform(X)
            
            MODEL_KNN = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
            MODEL_KNN.fit(X_scaled)
            
            # Stockage du 'référentiel' aligned pour le rendu des résultats
            # Optimisation Mémoire: On ne garde que les métadonnées
            DF_AUDIO = df[['filename', 'label']].copy()
            
            # Nettoyage RAM
            del df
            del X
            del X_scaled
            gc.collect()
            
            print("✅ Modèle KNN entraîné sur le dataset complet.")
            print("🧹 Nettoyage RAM effectué.")
        else:
            print(f"❌ IMPOSSIBLE de charger le dataset ({DATASET_PATH}).")
        
    except Exception as e:
        print(f"❌ Erreur critique au démarrage : {e}")
    
    yield
    
    # Clean up à l'extinction
    print("🛑 Arrêt de l'API. Nettoyage mémoire.")
    del DF_AUDIO
    del DF_POPULARITY
    del SCALER
    del MODEL_KNN
    gc.collect()

# --- APP SETUP ---
app = FastAPI(title="Type Beat AI API (High Perf)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "API is running (High Perf Mode)", 
        "model_loaded": MODEL_KNN is not None,
        "features_count": len(FEATURE_COLUMNS)
    }

@app.post("/predict")
async def predict_type_beat(file: UploadFile = File(...)):
    """
    Route optimisée pour Render Free Tier:
    1. Chargement unique de 30s.
    2. Découpage en RAM.
    3. Calcul unique du tempo.
    """
    if MODEL_KNN is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas encore prêt.")
    
    os.makedirs("temp_audio", exist_ok=True)
    file_location = f"temp_audio/{file.filename}"
    
    try:
        # 1. Sauvegarde disque (I/O obligatoire)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Chargement Audio Optimisé (UNE FOIS SEULEMENT)
        # On charge 30s au milieu du morceau
        y_full, sr = get_audio_content_optimized(file_location, analyze_duration=30)
        
        if y_full is None or len(y_full) == 0:
            raise HTTPException(status_code=400, detail="Impossible de lire l'audio.")

        # 3. Calcul du Tempo sur le buffer global (Lourd, fait une seule fois)
        global_tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
        
        # 4. Slicing en RAM (Ultra rapide) et Analyse
        # On découpe les 30s en 3 bouts de 10s
        samples_per_slice = 10 * sr
        slices = []
        
        # Slice 1: 0-10s
        if len(y_full) >= samples_per_slice:
            slices.append(y_full[0:samples_per_slice])
        
        # Slice 2: 10-20s
        if len(y_full) >= 2 * samples_per_slice:
            slices.append(y_full[samples_per_slice:2*samples_per_slice])
            
        # Slice 3: 20-30s
        if len(y_full) >= 3 * samples_per_slice:
            slices.append(y_full[2*samples_per_slice:3*samples_per_slice])
            
        # Fallback: si audio < 10s, on prend tout comme un seul slice
        if not slices:
            slices.append(y_full)

        # 5. Soft Voting
        vote_scores = defaultdict(float)
        song_to_artist_map = {} 

        for y_slice in slices:
            # Extraction avec tempo pré-calculé
            features_dict = extract_features(y_slice, sr, precomputed_tempo=global_tempo)
            if not features_dict: continue

            # Alignement vecteur
            vector = []
            for col in FEATURE_COLUMNS:
                val = features_dict.get(col, 0.0)
                vector.append(val)
            
            # KNN
            X_input = np.array([vector])
            X_input = np.nan_to_num(X_input)
            X_scaled = SCALER.transform(X_input)
            
            distances, indices = MODEL_KNN.kneighbors(X_scaled)
            
            for dist, idx in zip(distances[0], indices[0]):
                neighbor_data = DF_AUDIO.iloc[idx]
                raw_filename = str(neighbor_data['filename'])
                artist_label = str(neighbor_data['label'])
                
                song_key = raw_filename.split('__slice')[0]
                score_contribution = max(0, 1.0 - dist)
                
                vote_scores[song_key] += score_contribution
                song_to_artist_map[song_key] = artist_label
        
        # 6. Aggregation & Top 8 (Logique Métier Préservée)
        final_results = []
        
        for song_key, total_score in vote_scores.items():
            artist = song_to_artist_map.get(song_key, "Unknown")
            
            pop_score = 0
            views = 0
            if DF_POPULARITY is not None:
                search_key = artist.lower().strip()
                match = DF_POPULARITY[DF_POPULARITY['search_key'] == search_key]
                if not match.empty:
                    try:
                        pop_score = int(match.iloc[0]['popularity'])
                        views = float(match.iloc[0]['youtube_avg_views'])
                    except: pass

            final_results.append({
                "title": song_key.replace(".mp3", "").replace("_", " "), 
                "artist": artist,
                "label": artist,
                "score": round(total_score, 4),
                "popularity": pop_score,
                "popularity_score": pop_score,
                "views": views,
                "is_trending": False 
            })

        # Tri et Deduplication
        final_results.sort(key=lambda x: x['score'], reverse=True)
        
        unique_artist_results = []
        seen_artists = set()
        
        for res in final_results:
            artist_name = res['artist']
            if artist_name not in seen_artists:
                unique_artist_results.append(res)
                seen_artists.add(artist_name)
            if len(unique_artist_results) >= 8:
                break
        
        top_8 = unique_artist_results
        
        # Trending Logic (Aligned with src/app.py - uses YouTube Views)
        if top_8:
            # Identifier les Top 3 "Most Popular" (YouTube Views)
            top_by_views = sorted(top_8, key=lambda x: x.get('views', 0), reverse=True)
            top_3_view_artists = [item['artist'] for item in top_by_views[:3]]
            
            for item in top_8:
                # Si l'artiste est dans le top 3 des vues (et qu'il a des vues), il est trending
                if item['artist'] in top_3_view_artists and item.get('views', 0) > 0:
                    item['is_trending'] = True
            
            max_s = top_8[0]['score'] if top_8[0]['score'] > 0 else 1
            for item in top_8:
                # Normalisation finale pour l'affichage (comme src/app.py implicitement via proba)
                # On ajuste le score en pourcentage relatif au premier pour la barre
                pass  # Le score est déjà une accumulation de votes ("mass"), on le garde tel quel ou on normalise
                # Dans app.py: top_artists = [(artist, score / total_mass) ...]
                # Ici nous avons déjà des scores de soft voting.
                pass

        winner = top_8[0] if top_8 else {"artist": "Inconnu", "score": 0, "label": "Inconnu"}

        return {
            "prediction": winner['artist'],
            "confidence": winner['score'],
            "recommendations": top_8,
            "input_filename": file.filename
        }

    except Exception as e:
        print(f"Erreur route predict: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(file_location):
            try:
                os.remove(file_location)
            except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

