import os
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_PATH = "data/dataset_audio.csv"
TEST_FOLDER = "data/audio_test"
SAMPLE_RATE = 22050
DURATION_SLICE = 30

# --- 1. FONCTIONS ---

def load_recommendation_system():
    print("⏳ Chargement du système de recommandation (KNN)...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset introuvable à {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # --- EQUILIBRAGE DES CLASSES ---
    # On limite chaque artiste au nombre de segments du plus petit (ex: ~500)
    # Pour éviter que les "gros" artistes n'écrasent les petits par simple densité.
    min_count = df['label'].value_counts().min()
    print(f"⚖️  Équilibrage du dataset : {min_count} segments max par artiste.")
    df = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
    # -------------------------------

    metadata_cols = ['filename', 'label']
    X = df.drop(columns=metadata_cols, errors='ignore')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    knn = NearestNeighbors(n_neighbors=5, metric='manhattan')
    knn.fit(X_scaled)
    
    print(f"✅ Système prêt ! ({len(df)} segments en base)")
    return knn, scaler, df, X.columns

def extract_features_v3(file_path):
    """
    Logique identique à app.py : Découpage 30s + Features
    """
    try:
        y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        total_samples = len(y_full)
        samples_per_slice = sr * DURATION_SLICE
        
        # Padding si trop court
        if total_samples < samples_per_slice:
            padding = samples_per_slice - total_samples
            y_padded = np.pad(y_full, (0, padding), 'constant')
            y_slices = [y_padded]
        else:
            y_slices = []
            for i in range(0, total_samples, samples_per_slice):
                segment = y_full[i : i + samples_per_slice]
                # On garde le segment s'il fait au moins 10 secondes
                if len(segment) >= (sr * 10): 
                    if len(segment) < samples_per_slice:
                        padding = samples_per_slice - len(segment)
                        segment = np.pad(segment, (0, padding), 'constant')
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
            
            features = {
                "tempo": tempo_val,
            }

            feature_list = {
                "rms": rms,
                "zcr": zcr,
                "spec_cent": spec_cent,
                "spec_roll": spec_roll
            }

            for name, data in feature_list.items():
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
        print(f"❌ Erreur lecture {os.path.basename(file_path)} : {e}")
        return None

# --- 2. EVALUATION ---

def run_evaluation():
    try:
        knn, scaler, database_df, feature_cols = load_recommendation_system()
    except Exception as e:
        print(e)
        return

    # Collecter les fichiers de test
    test_files = []
    for root, dirs, files in os.walk(TEST_FOLDER):
        for file in files:
            if file.lower().endswith('.mp3') or file.lower().endswith('.wav') or file.lower().endswith('.m4a'):
                full_path = os.path.join(root, file)
                # Le dossier parent est le label (ex: data/audio_test/Damso/track.mp3 -> Damso)
                label = os.path.basename(root)
                test_files.append((full_path, label))

    print(f"\n📂 Fichiers trouvés pour le test : {len(test_files)}")
    
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    details = []

    for file_path, true_label in tqdm(test_files, desc="🧠 Analyse du dataset de test", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"):
        # 1. Extraction (Silencieux)
        features_slices_df = extract_features_v3(file_path)
        
        if features_slices_df is None or features_slices_df.empty:
            continue
            
        # 2. Cumulative KNN (Weighted Voting)
        artist_scores = Counter()
        input_df = features_slices_df[feature_cols] # align columns
        
        if input_df.empty:
            continue

        try:
            for i in range(len(input_df)):
                slice_vec = input_df.iloc[[i]]
                slice_scaled = scaler.transform(slice_vec)
                distances, indices = knn.kneighbors(slice_scaled)
                
                for rank, idx in enumerate(indices[0]):
                    dist = distances[0][rank]
                    neighbor_row = database_df.iloc[idx]
                    label = neighbor_row['label']
                    
                    # Weight = 1 / (distance + epsilon)
                    # Plus la distance est petite, plus le poids est grand
                    weight = 1.0 / (dist + 1e-6)
                    artist_scores[label] += weight
            
            # 3. Consensus
            if not artist_scores:
                continue

            top_consensus = artist_scores.most_common(3) # Top 3 artists based on score
            
            # Prediction
            pred_1 = top_consensus[0][0] if len(top_consensus) > 0 else None
            pred_3 = [x[0] for x in top_consensus]
            
            is_top1 = (pred_1 == true_label)
            is_top3 = (true_label in pred_3)
            
            if is_top1: correct_top1 += 1
            if is_top3: correct_top3 += 1
            total += 1
            
            details.append({
                "file": os.path.basename(file_path),
                "true": true_label,
                "pred_1": pred_1,
                "pred_3": pred_3,
                "match_1": is_top1,
                "match_3": is_top3
            })
            
        except Exception as e:
            # Erreur silencieuse pour ne pas casser le tableau
            pass

    # --- RESULTATS ---
    print("\n" + "="*60)
    print(f"{'📊 RAPPORT DE PERFORMANCE DU MODÈLE':^60}")
    print("="*60)
    
    if total > 0:
        acc_1 = (correct_top1 / total) * 100
        acc_3 = (correct_top3 / total) * 100
        
        # Jauge Visuelle ASCII
        def jauge(percent):
            filled = int(percent / 5) # 20 blocs max
            bar = "█" * filled + "▒" * (20 - filled)
            return f"[{bar}] {percent:.1f}%"

        print("\n📈 SCORES GLOBAUX")
        print(f"   • Précision Top-1 (Exact)  : {jauge(acc_1)}")
        print(f"   • Précision Top-3 (Large)  : {jauge(acc_3)}")
        print(f"   • Fichiers testés          : {total}")
        
        # Matrice de confusion simplifiée
        print("\n🔍 ANALYSE PAR ARTISTE (TOP-1)")
        print(f"   {'ARTISTE':<15} | {'PRÉCISION':<10} | {'PREDICTIONS (Vrai -> PRED)'}")
        print("-" * 65)

        # Groupe par artiste réel
        artist_stats = {}
        for d in details:
            truth = d['true']
            pred = d['pred_1']
            if truth not in artist_stats:
                artist_stats[truth] = {"total": 0, "correct": 0, "mistakes": Counter()}
            
            artist_stats[truth]["total"] += 1
            if truth == pred:
                artist_stats[truth]["correct"] += 1
            else:
                artist_stats[truth]["mistakes"][pred] += 1
        
        for art, stats in artist_stats.items():
            rate = (stats['correct'] / stats['total']) * 100
            # Affiche les 2 erreurs les plus fréquentes
            mistakes_str = ""
            if stats['mistakes']:
                top_m = stats['mistakes'].most_common(2)
                m_str = [f"{k}({v})" for k,v in top_m]
                mistakes_str = " -> " + ", ".join(m_str)
            else:
                mistakes_str = " ✅ Parfait"

            print(f"   {art:<15} | {rate:>6.1f}%    |{mistakes_str}")

        print("\n" + "="*60 + "\n")

    else:
        print("Aucun fichier valide traité.")

if __name__ == "__main__":
    run_evaluation()
