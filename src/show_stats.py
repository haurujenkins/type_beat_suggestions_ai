import pandas as pd
import os

print("\n📊 --- DATASET STATISTICS ---")
try:
    if not os.path.exists('data/dataset_audio.csv'):
        print("⚠️ No dataset found at data/dataset_audio.csv")
    else:
        df = pd.read_csv('data/dataset_audio.csv')
        
        # Count unique artists
        n_artists = df['label'].nunique()
        total_segments = len(df)
        
        print(f"🎤 Registered Artists: {n_artists}")
        print(f"📦 Total Segments: {total_segments}")
        print("\n--- Distribution per Artist ---")
        print(df['label'].value_counts())
        
except Exception as e:
    print(f"❌ Error analysis: {e}")
print("----------------------------\n")
