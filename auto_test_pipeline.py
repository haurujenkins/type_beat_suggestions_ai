import os
import yt_dlp
import pandas as pd
import re
import unicodedata
import glob

# --- CONFIGURATION ---
CSV_PATH = "data/dataset_audio.csv"
TEST_ROOT = "data/audio_test"
# AJOUTE JOLAGREEN / ROUNHAA SI TU VEUX TESTER CES NOUVEAUX AUSSI
TARGET_ARTISTS = ["Damso", "Josman", "Ninho", "Jul", "Nekfeu", "Jolagreen", "Rounhaa", "SCH", "Yvnnis", "Zamdane", "J9ueve"] 
SAMPLES_PER_ARTIST = 5

def normalize_string(s):
    """Normalisation stricte pour la comparaison de noms de fichiers."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    s = re.sub(r'\.mp3$', '', s)
    s = re.sub(r'\.wav$', '', s)
    s = re.sub(r'\.m4a$', '', s)
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def get_known_files():
    """Renvoie un set de 'fingerprints' de fichiers connus (CSV + Dossier Test existant)."""
    known = set()
    
    # 1. Fichiers du CSV (Entraînement)
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, usecols=['filename'])
            for f in df['filename'].unique():
                base = f.split("__slice_")[0]
                known.add(normalize_string(base))
        except Exception as e:
            print(f"⚠️ Warning CSV: {e}")

    # 2. Fichiers déjà dans data/audio_test (pour ne pas retélécharger les mêmes clips de test)
    if os.path.exists(TEST_ROOT):
        for root, _, files in os.walk(TEST_ROOT):
            for f in files:
                if f.endswith(('.mp3', '.m4a', '.wav')):
                    known.add(normalize_string(f))

    print(f"ℹ️  {len(known)} empreintes de fichiers déjà connus (CSV + Test Folder).")
    return known

def download_test_samples():
    known_fingerprints = get_known_files()
    
    # Options yt-dlp pour ne télécharger que de l'audio
    ydl_opts_base = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
        'ignoreerrors': True,
        'restrictfilenames': True,
        'noplaylist': True,
        'extract_flat': True,  # Juste récupérer les infos d'abord
        # REMOVE GLOBAL EXTRACTOR ARGS to reset defaults
    }

    print(f"🚀 Lancement du téléchargement de {SAMPLES_PER_ARTIST} samples de TEST par artiste...")

    for artist in TARGET_ARTISTS:
        artist_dir = os.path.join(TEST_ROOT, artist)
        os.makedirs(artist_dir, exist_ok=True)
        
        existing = len(glob.glob(os.path.join(artist_dir, "*.mp3")))
        needed = max(0, SAMPLES_PER_ARTIST - existing)
        
        if needed == 0:
            print(f"✅ {artist}: Déjà {existing} fichiers de test.")
            continue
            
        print(f"🔍 {artist}: Recherche de {needed} instrumentales INÉDITES...")
        
        query = f"{artist} type beat instrumental"
        
        with yt_dlp.YoutubeDL(ydl_opts_base) as ydl:
            try:
                # 1. Recherche (sans télécharger) - Élargi à 60 résultats pour trouver des inédits
                search_res = ydl.extract_info(f"ytsearch60:{query}", download=False)
                
                downloaded_count = 0
                
                if 'entries' in search_res:
                    for entry in search_res['entries']:
                        if downloaded_count >= needed:
                            break
                            
                        title = entry.get('title', 'Unknown')
                        
                        # Vérif doublon
                        fingerprint = normalize_string(title)
                        if fingerprint in known_fingerprints:
                            continue
                            
                        # Téléchargement effectif
                        # FORCE WEB CLIENT (CLASSIC) - iOS/Android seem blocked
                        real_opts = ydl_opts_base.copy()
                        real_opts['extract_flat'] = False
                        # remove specific client override to fallback to default
                        if 'extractor_args' in real_opts:
                            del real_opts['extractor_args']
                            
                        real_opts['outtmpl'] = os.path.join(artist_dir, '%(title)s.%(ext)s')
                        
                        print(f"   ⬇️ Téléchargement: {title}")
                        try:
                            # Utiliser 'url' ou 'webpage_url' ou construire l'URL depuis l'ID
                            video_url = entry.get('webpage_url') or entry.get('url')
                            if not video_url and entry.get('id'):
                                video_url = f"https://www.youtube.com/watch?v={entry['id']}"
                                
                            if video_url:
                                with yt_dlp.YoutubeDL(real_opts) as ydl_real:
                                    ydl_real.download([video_url])
                                
                                known_fingerprints.add(fingerprint)
                                downloaded_count += 1
                            else:
                                print("      ❌ Pas d'URL trouvée pour la vidéo")
                                
                        except Exception as e:
                            print(f"   ❌ Erreur DL: {e}")

            except Exception as e:
                print(f"❌ Erreur recherche {artist}: {e}")

if __name__ == "__main__":
    download_test_samples()
