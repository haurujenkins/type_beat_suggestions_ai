import os
import yt_dlp
import pandas as pd
import re
# from youtubesearchpython import VideosSearch <--- SUPPRIMÉ, trop instable

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = "data/raw_audio"
CSV_PATH = "data/dataset_audio.csv"

# LISTE DES ARTISTES À TÉLÉCHARGER
ARTISTS_TO_DOWNLOAD = ["Hamza"]

DURATION_START = 60
DURATION_END = 105
MAX_RESULTS_SEARCH = 1000
MAX_DOWNLOADS = 250 # Nombre de téléchargements PAR artiste
# CONFIGURATION MANUELLE: Si IP bloquée, mettez les URLs dans manual_sources.txt
MANUAL_SOURCES_FILE = "manual_sources.txt"

# --- CONFIGURATION COOKIES (Méthode fichier) ---
# 1. Installez l'extension Chrome/Firefox "Get cookies.txt LOCALLY"
# 2. Allez sur youtube.com (connecté à votre compte)
# 3. Exportez les cookies et sauvegardez le fichier sous le nom "cookies.txt" à la racine du projet
COOKIES_FILE = "cookies.txt" 

def sanitize_title(title):
    """
    Nettoyage pour comparaison plus robuste (ignore la casse et les caractères spéciaux).
    Sert de fingerprint pour détecter les doublons.
    """
    s = str(title).lower()
    # On enlève extension si présente (au cas où on traiterait un nom de fichier)
    s = re.sub(r'\.mp3$', '', s)
    s = re.sub(r'\.wav$', '', s)
    s = re.sub(r'\.m4a$', '', s)
    # On ne garde que les lettres et chiffres pour ignorer ponctuation/espaces
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def get_existing_base_filenames():
    """
    Récupère un SET de fingerprints des fichiers déjà présents dans le CSV.
    """
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, usecols=['filename'])
            fingerprints = set()
            for fname in df['filename'].values:
                # ex: "song.mp3__slice_0" -> "song.mp3"
                base_name = fname.split('__slice_')[0]
                # -> fingerprint normalisé
                fingerprints.add(sanitize_title(base_name))
            return fingerprints
        except Exception as e:
            print(f"⚠️ Erreur lecture CSV : {e}")
    return set()

def download_artist_beats(artist_name):
    # On récupère les fingerprints connus (CSV et Disque) pour éviter doublons logiques
    csv_existing_fingerprints = get_existing_base_filenames()
    
    # 1. Création du dossier
    artist_dir = os.path.join(BASE_OUTPUT_DIR, artist_name.replace(" ", "_"))
    os.makedirs(artist_dir, exist_ok=True)
    
    # Détection des fichiers locaux déjà présents
    local_files_raw = [f for f in os.listdir(artist_dir) if f.endswith('.mp3')]
    local_existing_fingerprints = {sanitize_title(f) for f in local_files_raw}
    
    count_existing = len(local_files_raw)


    if count_existing >= MAX_DOWNLOADS:
        print(f"✅ {artist_name} : Déjà {count_existing} fichiers présents. Limite ({MAX_DOWNLOADS}) atteinte. ⏭️ Skipping.")
        return

    remaining_to_download = MAX_DOWNLOADS - count_existing
    print(f"\n🔍 Recherche de 'Type Beats' pour : {artist_name}...")
    print(f"   📊 État actuel : {count_existing} fichiers locaux. Objectif : +{remaining_to_download} nouveaux.")

    # 2. Fonction de Recherche
    def perform_search(search_limit):
        search_query = f"{artist_name} type beat"
        candidates = []
        
        print(f"   🔎 Recherche standard pour : '{search_query}'...")
        
        # Options optimisées pour éviter le blocage (Simule une requête Android)
        ydl_opts_search = {
            'quiet': True,
            'extract_flat': True,
            'ignoreerrors': True,
            'extractor_args': {
                'youtube': {
                    'player_client': ['IOS', 'web']
                }
            },
            'sleep_interval': 2,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts_search) as ydl:
                # On utilise la limite définie par l'appeleur
                search_results = ydl.extract_info(f"ytsearch{search_limit}:{search_query}", download=False)
                
                if 'entries' in search_results:
                    for entry in search_results['entries']:
                        if not entry: continue
                        candidates.append({
                            'title': entry.get('title', 'Unknown'),
                            'url': entry.get('url', '')
                        })
        except Exception as e:
            print(f"⚠️ Erreur recherche : {e}")
                
        return candidates

    # Première passe
    found_videos = perform_search(MAX_RESULTS_SEARCH)
    
    # Extension si insuffisant
    if len(found_videos) < (remaining_to_download): # On compare par rapport à ce qu'il reste à chopper
        print(f"   ⚠️ Trop peu de résultats ({len(found_videos)}/{remaining_to_download}). Extension recherche...")
        found_videos = perform_search(MAX_RESULTS_SEARCH * 2)

    print(f"✅ {len(found_videos)} vidéos correspondantes trouvées.")
    print("🚀 Filtrage et Démarrage du téléchargement...")

    # 3. Configuration Téléchargement
    ydl_opts_download = {
        'format': 'bestaudio/best',
        'outtmpl': f'{artist_dir}/%(title)s.%(ext)s',
        'restrictfilenames': True, 
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        
        # --- CONFIG ANTI-BOT & STABILITÉ ---
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': True,
        'no_overwrites': True,
        
        # Astuce : On se fait passer pour un client Android pour éviter le sign-in
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web']
            }
        },
        'sleep_interval': 3,
        'max_sleep_interval': 10,
    }

    count_downloaded = 0
    
    with yt_dlp.YoutubeDL(ydl_opts_download) as ydl:
        for video in found_videos:
            if count_downloaded >= remaining_to_download:
                print(f"🛑 Limite atteinte (+{count_downloaded} ajoutés).")
                break
            
            # --- CHECK RAPIDE (Doublons CSV) ---
            # On utilise le fingerprint pour vérifier si on l'a déjà
            current_fingerprint = sanitize_title(video['title'])
            
            # Check 1: Est-ce que le fichier semble être dans le CSV ?
            if current_fingerprint in csv_existing_fingerprints:
                continue
                
            # Check 2: Est-ce que le fichier est déjà sur le disque ?
            if current_fingerprint in local_existing_fingerprints:
                continue

            try:
                # Le 'no_overwrites' de ydl_opts gère la sécurité finale disque
                print(f"⬇️ ({count_downloaded + 1}/{MAX_DOWNLOADS}) {video['title']}")
                ydl.download([video['url']])
                count_downloaded += 1
            except Exception as e:
                # Erreurs discrètes
                pass

    print(f"🎉 Terminé pour {artist_name}. {count_downloaded} nouveaux fichiers.")

# --- ZONE DE LANCEMENT ---
if __name__ == "__main__":
    print(f"🎯 Artistes ciblés : {ARTISTS_TO_DOWNLOAD}")
    
    for artiste in ARTISTS_TO_DOWNLOAD:
        download_artist_beats(artiste)