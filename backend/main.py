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

# --- UTILITAIRES AUDIO ---

def get_audio_slices(audio_path, target_duration=15, n_slices=3):
    """
    OPTIMISATION MAXIMALE RENDER:
    - Ne charge PAS tout le fichier audio.
    - Lit uniquement les metadata pour la durée.
    - Charge uniquement 3 petits bouts de 15s.
    """
    try:
        # 1. Obtenir la durée sans charger le fichier (Rapide)
        total_duration = librosa.get_duration(path=audio_path)
        sr = 22050
        
        segments = []
        
        # Cas A : Audio trop court (< duréee minimale)
        if total_duration < target_duration:
            y, _ = librosa.load(audio_path, sr=sr, mono=True)
            # Padding
            target_samples = int(target_duration * sr)
            if len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)), 'constant')
            segments.append(y)
            return segments, sr

        # Cas B : Audio long -> On saute direct aux bons endroits
        # Positions relatives : 10% (Intro), 45% (Milieu), 80% (Fin)
        positions = [0.1, 0.45, 0.8]
        
        for pos in positions:
            # Calcul de l'offset en secondes
            # On s'assure qu'on ne dépasse pas la fin
            start_sec = (total_duration - target_duration) * pos
            if start_sec < 0: start_sec = 0
            
            # Chargement partiel chirurgical
            y_slice, _ = librosa.load(
                audio_path, 
                sr=sr, 
                mono=True, 
                offset=start_sec, 
                duration=target_duration,
                res_type='kaiser_fast'
            )
            segments.append(y_slice)
            
        return segments, sr
    
    except Exception as e:
        print(f"⚠️ Erreur slicing optimisé: {e}, fallback load complet.")
        # Fallback méthode bourrin si échec lecture metadata
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        segments.append(y[:int(target_duration*22050)])
        return segments, sr

def extract_features_from_segment(y, sr):
    """
    Extrait les features d'un segment audio brut (numpy array).
    Doit produire exactement les mêmes features que dataset_audio.csv
    (Mean AND Variances).
    """
    features = {}
    
    try:
        # 1. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo)
        
        # 2. Features Spectrales (Mean & Var)
        # RMS
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        
        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_var'] = np.var(zcr)
        
        # Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spec_cent_mean'] = np.mean(spec_cent)
        features['spec_cent_var'] = np.var(spec_cent)
        
        # Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spec_bw_mean'] = np.mean(spec_bw)
        features['spec_bw_var'] = np.var(spec_bw)
        
        # Spectral Rolloff
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spec_roll_mean'] = np.mean(spec_roll)
        features['spec_roll_var'] = np.var(spec_roll)
        
        # 3. Chroma (12 notes) -> calcul mean et var pour chaque
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            features[f'chroma_{i}_mean'] = np.mean(chroma[i])
            features[f'chroma_{i}_var'] = np.var(chroma[i])
            
        # 4. MFCC (20 coefficients) -> calcul mean et var pour chaque
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
            # On exclut filenames et labels pour ne garder que les maths
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Sécurité : on retire d'éventuels ID ou colonnes parasites si présents
            cols_to_drop = ['Unnamed: 0', 'id']
            FEATURE_COLUMNS = [c for c in numeric_cols if c not in cols_to_drop]
            
            print(f"🔍 {len(FEATURE_COLUMNS)} features audio identifiées pour l'entraînement.")

            # 4. Entraînement (Scaling + KNN)
            X = df[FEATURE_COLUMNS].values
            
            # Nettoyage des NaNs éventuels
            X = np.nan_to_num(X)

            SCALER = StandardScaler()
            X_scaled = SCALER.fit_transform(X)
            
            # Entraînement du modèle KNN (10 voisins pour avoir du choix)
            MODEL_KNN = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
            MODEL_KNN.fit(X_scaled)
            
            # Stockage du 'référentiel' aligned pour le rendu des résultats
            # Optimisation Mémoire (Render Free Tier)
            # On ne garde que les métadonnées dans DF_AUDIO, on jette les features brutes
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
app = FastAPI(title="Type Beat AI API (On-the-fly)", lifespan=lifespan)

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
        "status": "API is running", 
        "model_loaded": MODEL_KNN is not None,
        "features_count": len(FEATURE_COLUMNS),
        "dataset_size_ref": len(DF_AUDIO) if DF_AUDIO is not None else 0
    }

@app.post("/predict")
async def predict_type_beat(file: UploadFile = File(...)):
    """
    1. Reçoit l'audio
    2. Découpe en 5 segments
    3. Soft Voting des voisins
    4. Top 8 + Trending logic
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
            
        # 2. Slicing (5 segments)
        segments, sr = get_audio_slices(file_location)
        
        # Dictionnaire pour le vote : { "Song_Real_Name": accumulated_score }
        vote_scores = defaultdict(float)
        # Cache pour retrouver l'artiste lié à une chanson sans refaire des loops
        song_to_artist_map = {} 

        # 3. Boucle d'analyse sur les segments
        for i, y_seg in enumerate(segments):
            # Extraction
            features_dict = extract_features_from_segment(y_seg, sr)
            if not features_dict: continue

            # Alignement avec le modèle (IMPORTANT : Ordre des colonnes)
            vector = []
            for col in FEATURE_COLUMNS:
                val = features_dict.get(col, 0.0) # 0.0 si feature manquante (fallback)
                vector.append(val)
            
            # Predict
            X_input = np.array([vector])
            X_input = np.nan_to_num(X_input) # Sécurité nan
            X_scaled = SCALER.transform(X_input)
            
            distances, indices = MODEL_KNN.kneighbors(X_scaled)
            
            # Vote (1 - distance) ajouté au score de la track parente
            for dist, idx in zip(distances[0], indices[0]):
                neighbor_data = DF_AUDIO.iloc[idx]
                raw_filename = str(neighbor_data['filename'])
                artist_label = str(neighbor_data['label'])
                
                # Nettoyage nom de fichier pour grouper les slices d'une même track
                # Ex: "Damso_Macarena.mp3__slice_0" -> "Damso_Macarena.mp3"
                song_key = raw_filename.split('__slice')[0]
                
                # Score de similarité (plus c'est proche de 0, plus le score est haut)
                # On utilise (1 - dist) pour avoir une "confiance"
                score_contribution = max(0, 1.0 - dist)
                
                vote_scores[song_key] += score_contribution
                song_to_artist_map[song_key] = artist_label
        
        # 4. Aggregation Finale
        final_results = []
        
        # On normalise les scores par le nombre de segments de vote possible (moyenne)
        # Mais pour le classement, le cumul suffit
        num_segments = len(segments)
        
        for song_key, total_score in vote_scores.items():
            # Pour l'UI on normalise un peu le score, ce n'est pas une proba pure
            # Total score max possible = (n_slices * 10 * 1.0) si tous les voisins sont distance 0
            
            artist = song_to_artist_map.get(song_key, "Unknown")
            
            # Récup popularité
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
                "label": artist, # Compatibilité frontend
                "score": round(total_score, 4), # Score cumulé brut
                "popularity": pop_score, # Compatibilité frontend
                "popularity_score": pop_score,
                "views": views,
                "is_trending": False 
            })

        # 5. Tri et Filtrage par Artiste Unique (Deduplication)
        # On trie d'abord par score décroissant pour avoir les meilleures chansons en premier
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
        
        # 6. Logique Trending (dans le Top 8 unique)
        if top_8:
            # On identifie les 3 qui ont la plus grosse popularité
            top_by_pop = sorted(top_8, key=lambda x: x['popularity_score'], reverse=True)
            top_3_pop_titles = [item['title'] for item in top_by_pop[:3]]
            
            # Application du flag
            for item in top_8:
                if item['title'] in top_3_pop_titles and item['popularity_score'] > 0:
                    item['is_trending'] = True
                    
            # Normalisation finale du score pour l'affichage (0 à 1)
            max_s = top_8[0]['score'] if top_8[0]['score'] > 0 else 1
            for item in top_8:
                # Normalisation relative au winner
                item['score'] = round(item['score'] / max_s * 0.95, 2)
                item['distance'] = 1 - item['score'] # Compatibilité frontend

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

