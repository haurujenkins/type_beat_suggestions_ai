# optimized_pipeline.py
import yt_dlp
import librosa
import numpy as np
import pandas as pd
import os
import shutil
import warnings
import time

# Suppress librosa warnings
warnings.filterwarnings('ignore')

class MyLogger:
    def debug(self, msg):
        pass
    def warning(self, msg):
        pass
    def error(self, msg):
        print(msg)

class DataPipeline:
    def __init__(self, output_csv="data/dataset_audio.csv", v2_dir="data/v2_spectrograms", temp_dir="data/temp_audio"):
        self.output_csv = output_csv
        self.v2_dir = v2_dir
        self.temp_dir = temp_dir
        
        # Audio Settings (Aligned with V1 and V2 Req)
        self.sr = 22050
        self.slice_duration = 30  # seconds
        
        # Spectrogram Settings
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512

        # Ensure directories exist
        os.makedirs(self.v2_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Create CSV Header if not exists
        if not os.path.exists(self.output_csv):
            # Create a dummy dataframe to initialize the CSV with correct columns if empty
            # But the extract_features logic returns rows, we'll let pandas handle the first write
            pass

    def run_for_artist(self, artist_name, max_videos=5, min_views=10000):
        """
        Main Pipeline Entry Point for a single artist.
        1. Search
        2. Stream Process (Download -> V1 -> V2 -> Clean)
        """
        print(f"\n🚀 Starting pipeline for: {artist_name}")
        
        # 1. SEARCH
        print(f"🔍 Searching YouTube for '{artist_name} Type Beat'...")
        videos = self.search_videos(artist_name, limit=max_videos * 2) # Search more to allow filtering
        
        # Filter
        valid_videos = []
        for v in videos:
            if v.get('view_count', 0) >= min_views:
                valid_videos.append(v)
            if len(valid_videos) >= max_videos:
                break
        
        if not valid_videos:
             print(f"⚠️ No videos found with > {min_views} views. Found {len(videos)} total results.")
             print("   -> Try lowering 'min_views' in run_for_artist().")
        else:
            print(f"✅ Found {len(valid_videos)} valid videos (views > {min_views}). processing...")

        # 2. PROCESS LOOP
        processed_count = 0
        total_videos = len(valid_videos)
        
        for idx, vid in enumerate(valid_videos, 1):
            safe_title = vid.get('title', 'Unknown')
            print(f"\n🔄 [{idx}/{total_videos}] Processing: {safe_title[:50]}...")
            
            try:
                self.process_single_video(vid, artist_name)
                processed_count += 1
            except Exception as e:
                # Simplified error logging without full traceback
                print(f"❌ Error processing '{safe_title[:30]}...': {e}")
                # import traceback
                # traceback.print_exc() 

        print(f"🎉 Artist {artist_name} done. {processed_count}/{len(valid_videos)} success.")

    def search_videos(self, artist_name, limit=10):
        search_query = f"{artist_name} Type Beat"
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'max_downloads': limit,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # "ytsearchN:query" searches for N results
            try:
                result = ydl.extract_info(f"ytsearch{limit}:{search_query}", download=False)
                if 'entries' in result:
                    return result['entries']
            except Exception as e:
                print(f"Search failed: {e}")
        return []

    def process_single_video(self, video_info, artist_label):
        title = video_info.get('title', 'unknown')
        url = video_info.get('webpage_url', video_info.get('url')) # 'url' in flat, 'webpage_url' often safer
        video_id = video_info.get('id')
        
        # print(f"  ▶️ Processing: {title[:40]}...")

        # A. DOWNLOAD (Temp)
        # Clean title for filename safe
        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).strip().replace(" ", "_")
        temp_filename = f"{artist_label}_{safe_title}_{video_id}.mp3"
        temp_path = os.path.join(self.temp_dir, temp_filename)
        
        # Skip if temp file exists (maybe failed previous run) - actually, for stream we want to overwrite
        if os.path.exists(temp_path):
            os.remove(temp_path)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': temp_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'logger': MyLogger(),
        }

        # The actual file might have .mp3 appended by yt-dlp logic automatically
        # So we define the path without extension for outtmpl sometimes, but here we enforce it.
        # Let's perform the download
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            raise Exception(f"Download failed: {e}")

        # Fix extension check: yt-dlp might append .mp3 if not in template correctly or do conversion
        # Ensure we find the file
        if not os.path.exists(temp_path):
            # Try finding it if yt-dlp added extension
            if os.path.exists(temp_path + ".mp3"):
                temp_path = temp_path + ".mp3"
            else:
                raise FileNotFoundError(f"Could not find downloaded file at {temp_path}")

        try:
            # LOAD AUDIO ONCE
            y, sr = librosa.load(temp_path, sr=self.sr)

            # B. PROCESS V1 (Append to CSV)
            self._process_v1_features(y, sr, artist_label, os.path.basename(temp_path))

            # C. PROCESS V2 (Save Spectrogram)
            self._process_v2_spectrogram(y, sr, artist_label, os.path.basename(temp_path))

        finally:
            # D. CLEANUP
            if os.path.exists(temp_path):
                os.remove(temp_path)
                # print("    🗑️ Temp file deleted.")


    def _process_v1_features(self, y, sr, label, filename):
        """
        Slices audio and extracts V1 features. Appends to CSV immediately.
        """
        total_samples = len(y)
        samples_per_slice = sr * self.slice_duration
        
        if total_samples < samples_per_slice:
            print("    ⚠️ Audio too short for slicing, skipping V1.")
            return

        segments_data = []
        slice_idx = 0
        current_sample = 0
        
        while current_sample + samples_per_slice <= total_samples:
            start = current_sample
            end = current_sample + samples_per_slice
            y_slice = y[start:end]
            
            # --- FEATURE EXTRACTION (Matches audio_to_csv.py) ---
            tempo, _ = librosa.beat.beat_track(y=y_slice, sr=sr)
            rms = librosa.feature.rms(y=y_slice)
            zcr = librosa.feature.zero_crossing_rate(y=y_slice)
            spec_cent = librosa.feature.spectral_centroid(y=y_slice, sr=sr)
            spec_roll = librosa.feature.spectral_rolloff(y=y_slice, sr=sr)
            mfccs = librosa.feature.mfcc(y=y_slice, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y_slice, sr=sr)
            
            # Pack
            features = {
                "label": label, # Important for training
                "filename": f"{filename}__slice_{slice_idx}",
                "tempo": float(tempo.item()) if hasattr(tempo, 'item') else float(tempo),
            }
            
            # Means & Vars
            stats_map = {
                "rms": rms, "zcr": zcr, "spec_cent": spec_cent, "spec_roll": spec_roll
            }
            for name, data in stats_map.items():
                features[f"{name}_mean"] = np.mean(data)
                features[f"{name}_var"]  = np.var(data)

            for i in range(13):
                features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
                features[f"mfcc_{i}_var"]  = np.var(mfccs[i])
            for i in range(12):
                features[f"chroma_{i}_mean"] = np.mean(chroma[i])
                features[f"chroma_{i}_var"]  = np.var(chroma[i])

            segments_data.append(features)
            
            current_sample += samples_per_slice
            slice_idx += 1

        # Append to CSV
        if segments_data:
            df_new = pd.DataFrame(segments_data)
            
            # --- Enforce column order to match CSV header ---
            # Define the standard column order (must match training/V1 expectation)
            # This is crucial so that 'label' is at the end or consistent
            
            # Standard columns (v1) derived from audio_to_csv.py logic
            ordered_cols = ["tempo", "filename"]
            
            # Basic Features
            for feat in ["rms", "zcr", "spec_cent", "spec_roll"]:
                ordered_cols.append(f"{feat}_mean")
                ordered_cols.append(f"{feat}_var")
            
            # MFCCs
            for i in range(13):
                ordered_cols.append(f"mfcc_{i}_mean")
                ordered_cols.append(f"mfcc_{i}_var")
                
            # Chromas
            for i in range(12):
                ordered_cols.append(f"chroma_{i}_mean")
                ordered_cols.append(f"chroma_{i}_var")
                
            ordered_cols.append("label")
            
            # Reorder DataFrame
            # Ensure all columns exist (fill 0 if logic failed somewhere, though unlikely)
            for col in ordered_cols:
                if col not in df_new.columns:
                    df_new[col] = 0
            
            df_new = df_new[ordered_cols]
            
            # Check if file exists to determine header
            header = not os.path.exists(self.output_csv)
            df_new.to_csv(self.output_csv, mode='a', header=header, index=False)
            print(f"    💾 V1: Appended {len(segments_data)} slices to CSV.")


    def _process_v2_spectrogram(self, y, sr, label, filename):
        """
        Generates Mel-Spectrogram and saves as .npz (compressed, float32)
        """
        # Ensure consistent length or treat full? 
        # User said "Audio complet (ou slices, disons complet pour l'instant)"
        
        # Generates Mel Spectrogram
        S = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=self.n_mels, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        
        # Convert to Log-Mel (dB) and cast to float32 for storage optimization
        S_dB = librosa.power_to_db(S, ref=np.max).astype(np.float32)
        
        # Save
        # Filename safe cleaning
        safe_name = os.path.splitext(filename)[0]
        
        # Create artist subfolder
        artist_dir = os.path.join(self.v2_dir, label)
        os.makedirs(artist_dir, exist_ok=True)
        
        out_name = f"{label}__{safe_name}.npz" # Convention: Artist__Title.npz
        out_path = os.path.join(artist_dir, out_name)
        
        np.savez_compressed(out_path, spec=S_dB)
        print(f"    🧠 V2: Saved Log-Mel Spectrogram to {label}/{out_name} (compressed)")


if __name__ == "__main__":
    # Example Usage
    pipeline = DataPipeline()
    
    # List of artists to process
    artists_to_process = ["Disiz"]
    
    for artist in artists_to_process:
        # Lower threshold to ensure we get results for smaller artists/queries
        # User requested up to 100 videos
        pipeline.run_for_artist(artist, max_videos=100, min_views=1000)
