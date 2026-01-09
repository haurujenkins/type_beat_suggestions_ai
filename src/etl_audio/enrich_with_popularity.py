import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import os
import subprocess
import json
import time
from datetime import datetime
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load Env
load_dotenv()

# --- CONFIG ---
DATASET_PATH = "data/dataset_audio.csv"
OUTPUT_PATH = "data/artist_popularity.csv"
YOUTUBE_SEARCH_LIMIT = 100

# Spotify credentials
client_id = os.getenv("SPOTIPY_CLIENT_ID")
client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

# --- INIT SPOTIFY ---
sp = None
if client_id and client_secret:
    try:
        auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        print(f"Warning: Error initializing Spotify client: {e}")

# Mapping manuel pour aider la recherche des artistes ambigus
ARTIST_SEARCH_HELPER = {
    "Jul": "JuL",
}

# Mapping ID manuel pour forcer l'artiste exact (évite les homonymes)
ARTIST_ID_OVERRIDES = {
    "Laylow": "0LnhY2fzptb0QEs5Q5gM7S",
    "Hamza": "5gs4Sm2WQUkcGeikMcVHbh",
    "Booba": "58wXmynHaAWI5hwlPZP3qL",
}

# --- YOUTUBE FUNCTIONS ---
def get_youtube_popularity(artist_name):
    """
    Récupère les 100 dernières vidéos '{Artist} Type Beat'.
    Calcule la moyenne de vues.
    """
    # Nettoyage du nom (ex: Ino_Casablanca -> Ino Casablanca) pour la recherche et le filtre
    clean_name = artist_name.replace("_", " ")
    
    encoded_query = quote_plus(f"{clean_name} type beat")
    # sp=CAI%3D => Tri par date de mise en ligne (Upload date)
    search_url = f"https://www.youtube.com/results?search_query={encoded_query}&sp=CAI%3D"
    
    cmd = [
        "yt-dlp",
        search_url,
        "--playlist-end", str(YOUTUBE_SEARCH_LIMIT), 
        "--flat-playlist",  # Beaucoup plus rapide, donne accès au view_count
        "--dump-json",
        "--no-warnings",
        "--ignore-errors"
    ]
    
    try:
        # Exécution silencieuse
        result = subprocess.run(cmd, capture_output=True, text=True)
        raw_lines = result.stdout.strip().split('\n')
        
        total_views = 0
        video_count = 0
        
        for line in raw_lines:
            if not line: continue
            try:
                vid = json.loads(line)
                
                # Vérif Pertinence Titre (Pour éviter le bruit)
                title = vid.get('title', '').lower()
                clean_artist_token = clean_name.lower().strip()
                
                # Assouplissement de la vérification :
                # 1. Soit la chaîne exacte est dedans
                # 2. Soit tous les mots de l'artiste sont présents (pour gérer "Ino - Casablanca" vs "Ino Casablanca")
                match = False
                if clean_artist_token in title:
                    match = True
                else:
                    words = clean_artist_token.split()
                    if len(words) > 0 and all(w in title for w in words):
                        match = True
                
                if not match:
                    continue

                # Récupération vues
                views = vid.get('view_count')
                if views is None: 
                    # Parfois null si c'est une première qui n'a pas commencé ou bug
                    views = 0
                
                # Équilibrage : Si < 1000, on compte 1000 pour lisser la moyenne
                if views < 1000:
                    views = 1000
                
                total_views += views
                video_count += 1
                    
            except json.JSONDecodeError:
                continue
        
        # Calcul Moyenne
        avg_views = 0
        if video_count > 0:
            avg_views = int(total_views / video_count)
            
        print(f"    -> Analysed {video_count} videos. Total views: {total_views}. Avg: {avg_views}")

        return {
            "youtube_avg_views": avg_views,
            "youtube_video_count": video_count, # Je garde le count pour info/debug mais l'user veut surtout la note
            "youtube_updated": datetime.now().strftime("%Y-%m-%d")
        }

    except Exception as e:
        print(f"  [X] YouTube Error for {artist_name}: {e}")
        return None

# --- SPOTIFY FUNCTIONS ---
def get_spotify_data(artist_name):
    if not sp: return None
    
    # Check ID overrides first
    if artist_name in ARTIST_ID_OVERRIDES:
        try:
            artist = sp.artist(ARTIST_ID_OVERRIDES[artist_name])
            print(f"    [!] Used manual ID override for {artist_name}")
            return {
                "spotify_name": artist['name'],
                "popularity": artist['popularity'],
                "genres": str(artist['genres']),
                "spotify_id": artist['id'],
            }
        except Exception as e:
            print(f"    [X] Error using ID override for {artist_name}: {e}")
            # Fallback to search if ID fails
            pass

    search_query = ARTIST_SEARCH_HELPER.get(artist_name, artist_name)
    try:
        # 1. Search with 'artist:' prefix
        results = sp.search(q='artist:' + search_query, type='artist', limit=5)
        items = results['artists']['items']
        
        target_artist = None
        for item in items:
            if item['name'].lower() == search_query.lower():
                target_artist = item
                break
        
        # 2. Broad search
        if not target_artist:
            results = sp.search(q=search_query, type='artist', limit=5)
            items = results['artists']['items']
            for item in items:
                if item['name'].lower() == search_query.lower():
                    target_artist = item
                    break
        
        # 3. Best guess
        if not target_artist and items:
            target_artist = items[0]
            print(f"    [Warn] Using Spotify guess: {target_artist['name']}")

        if target_artist:
            # On ne récupère pas les followers comme demandé
            return {
                "spotify_name": target_artist['name'],
                "popularity": target_artist['popularity'],
                "genres": str(target_artist['genres']),
                "spotify_id": target_artist['id'],
                # "followers": target_artist['followers']['total'] # Désactivé
            }
    except Exception as e:
        print(f"    [Error] Spotify fetch failed: {e}")
    
    return None

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset {DATASET_PATH} not found.")
        return

    # 1. Load Audio Dataset
    print("Loading audio dataset...")
    df_audio = pd.read_csv(DATASET_PATH)
    unique_artists = df_audio['label'].unique()
    print(f"Found {len(unique_artists)} artists to transform.")

    # 2. Load Existing Data
    if os.path.exists(OUTPUT_PATH):
        df_target = pd.read_csv(OUTPUT_PATH)
        if 'search_name' not in df_target.columns:
            df_target['search_name'] = ""
            
        # --- NETTOYAGE ANCIENNES COLONNES ---
        cols_to_drop = [
            'hype_score', 'youtube_videos_7d', 'youtube_views_7d',
            'spotify_popularity', 'spotify_genres', 'spotify_followers', 'spotify_url', 'img_url'
        ]
        existing_drop = [c for c in cols_to_drop if c in df_target.columns]
        if existing_drop:
            print(f"Dropping old columns: {existing_drop}")
            df_target.drop(columns=existing_drop, inplace=True)
            
    else:
        df_target = pd.DataFrame(columns=['search_name'])

    # Set index
    df_target.set_index('search_name', inplace=True, drop=False)
    
    # 3. Iterate
    processed_count = 0
    
    for artist_name in unique_artists:
        needs_save = False
        
        if artist_name not in df_target.index:
            print(f"\nProcessing NEW artist: {artist_name}")
            df_target.loc[artist_name] = pd.Series({'search_name': artist_name})
            needs_save = True
        
        row = df_target.loc[artist_name]
        
        # --- MISE À JOUR SYSTÉMATIQUE (Force Update) ---
        print(f"🔄 Updating metrics for: {artist_name}")
        
        # 1. SPOTIFY
        # On tente de récupérer les infos Spotify à chaque run
        try:
            sp_data = get_spotify_data(artist_name)
            if sp_data:
                for k, v in sp_data.items():
                    df_target.at[artist_name, k] = v
                needs_save = True
        except Exception as e:
            print(f"   ⚠️ Spotify error: {e}")

        # 2. YOUTUBE
        # On relance le scan YouTube à chaque run
        try:
            print(f"   Fetching YOUTUBE for: {artist_name}")
            yt_data = get_youtube_popularity(artist_name)
            if yt_data:
                for k, v in yt_data.items():
                    df_target.at[artist_name, k] = v
                needs_save = True
                # Anti-ban delay (très important pour le bulk update)
                time.sleep(1) 
        except Exception as e:
            print(f"   ⚠️ YouTube error: {e}")
        
        if needs_save:
            df_target.to_csv(OUTPUT_PATH, index=False)
            processed_count += 1
    
    print(f"\nDone! Updated {processed_count} artists. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
