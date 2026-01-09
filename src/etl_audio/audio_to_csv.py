import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# On ignore les warnings de Librosa pour garder la console propre
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
AUDIO_DIR = "data/raw_audio"
OUTPUT_CSV = "data/dataset_audio.csv"

# IMPORTANT : Fichiers complets, qu'on va découper
DURATION_SLICE = 30  # Durée d'un segment
# SAMPLE_RATE inchangé
SAMPLE_RATE = 22050

def extract_features(file_path):
    try:
        # 1. Chargement COMPLET du fichier (plus d'offset/duration ici)
        y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 2. Découpage en tranches de 30s
        total_samples = len(y_full)
        samples_per_slice = sr * DURATION_SLICE
        
        # Si le fichier est plus court qu'un segment, on ignore
        if total_samples < samples_per_slice:
            return []

        all_segments_features = []
        
        # On boucle par pas de 30 sec
        # On utilise une boucle while pour gérer les indices
        current_sample = 0
        slice_idx = 0
        
        base_filename = os.path.basename(file_path)

        while current_sample + samples_per_slice <= total_samples:
            # Extraction du segment
            start = current_sample
            end = current_sample + samples_per_slice
            y_slice = y_full[start:end]
            
            # --- Feature Extraction sur ce segment (y_slice) ---
            # (Même logique qu'avant mais sur y_slice)
            
            tempo, _ = librosa.beat.beat_track(y=y_slice, sr=sr)
            rms = librosa.feature.rms(y=y_slice)
            zcr = librosa.feature.zero_crossing_rate(y=y_slice)
            spec_cent = librosa.feature.spectral_centroid(y=y_slice, sr=sr)
            spec_roll = librosa.feature.spectral_rolloff(y=y_slice, sr=sr)
            mfccs = librosa.feature.mfcc(y=y_slice, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y_slice, sr=sr)

            # Calcul stats
            features = {
                "tempo": float(tempo.item()) if hasattr(tempo, 'item') else float(tempo),
                # Génération d'un filename unique par segment
                "filename": f"{base_filename}__slice_{slice_idx}", 
            }

            feature_list_simple = {
                "rms": rms,
                "zcr": zcr,
                "spec_cent": spec_cent,
                "spec_roll": spec_roll
            }

            for name, data in feature_list_simple.items():
                features[f"{name}_mean"] = np.mean(data)
                features[f"{name}_var"]  = np.var(data)

            for i in range(13):
                features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
                features[f"mfcc_{i}_var"]  = np.var(mfccs[i])
            
            for i in range(12):
                features[f"chroma_{i}_mean"] = np.mean(chroma[i])
                features[f"chroma_{i}_var"]  = np.var(chroma[i])
            
            all_segments_features.append(features)
            
            # On avance la fenetre
            current_sample += samples_per_slice
            slice_idx += 1

        return all_segments_features

    except Exception as e:
        print(f"❌ Erreur sur {file_path} : {e}")
        return []

def main():
    if not os.path.exists(AUDIO_DIR):
        print(f"⚠️ ERREUR: Le dossier {AUDIO_DIR} n'existe pas.")
        return

    data = []
    
    # --- LOGIQUE D'APPEND ---
    existing_filenames = set()
    if os.path.exists(OUTPUT_CSV):
        try:
            df_existing = pd.read_csv(OUTPUT_CSV, usecols=['filename'])
            existing_filenames = set(df_existing['filename'].values)
            print(f"📄 Dataset existant détecté : {len(existing_filenames)} segments déjà en base.")
        except Exception as e:
            print(f"⚠️ Fichier existant corrompu ou illisible ({e}), on repart de zéro.")

    artists = [d for d in os.listdir(AUDIO_DIR) if os.path.isdir(os.path.join(AUDIO_DIR, d))]
    
    print(f"🎧 Analyse audio lancée sur : {artists}")
    new_segments_count = 0

    for artist in artists:
        artist_path = os.path.join(AUDIO_DIR, artist)
        # Added .m4a support
        files = [f for f in os.listdir(artist_path) if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
        
        # NOTE : Ici, on n'a plus une correspondance 1-1 entre fichier et ligne CSV
        # Un fichier file.mp3 donnera file.mp3__slice_0, file.mp3__slice_1...
        # Pour savoir si on doit traiter le fichier, on regarde si AU MOINS le slice_0 existe.
        # Si slice_0 existe, on considère que le fichier a déjà été traité.
        
        files_to_process = []
        already_processed_count = 0
        
        for f in files:
            # On check si le premier segment existe déjà
            expected_first_segment = f"{f}__slice_0"
            if expected_first_segment not in existing_filenames:
                files_to_process.append(f)
            else:
                already_processed_count += 1
        
        if not files_to_process:
            print(f"📂 {artist} : À jour ({already_processed_count} fichiers ignorés car déjà présents).")
            continue
            
        print(f"\n📂 Traitement de {artist} : {len(files_to_process)} nouveaux, {already_processed_count} ignorés.")
        
        for file in tqdm(files_to_process):
            file_path = os.path.join(artist_path, file)
            
            # Récupère une LISTE de segments
            segments_list = extract_features(file_path)
            
            for seg in segments_list:
                seg["label"] = artist
                # filename est déjà défini dans seg
                data.append(seg)
                new_segments_count += 1


    # Sauvegarde (Append mode)
    if data:
        df = pd.DataFrame(data)
        
        # Si le fichier n'existait pas, on met un header. Sinon, non (car append).
        header_mode = not os.path.exists(OUTPUT_CSV)
        
        # On s'assure de l'ordre des colonnes si possible, ou on laisse pandas gérer
        df.to_csv(OUTPUT_CSV, mode='a', header=header_mode, index=False)
        print(f"\n✅ SUCCESS ! {new_segments_count} nouveaux segments ajoutés.")
        
        try:
            # Stats rapides
            full_df = pd.read_csv(OUTPUT_CSV)
            print("-" * 30)
            print(f"📊 Totaux actuels du dataset : {full_df.shape} segments.")
            print(full_df['label'].value_counts())
        except:
            pass
            
    else:
        print("\n💤 Aucune nouvelle donnée à ajouter.")

if __name__ == "__main__":
    main()