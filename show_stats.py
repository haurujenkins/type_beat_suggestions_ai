import pandas as pd
import os

CSV_PATH = "data/dataset_audio.csv"

if os.path.exists(CSV_PATH):
    try:
        df = pd.read_csv(CSV_PATH)
        print("--- Nombre de segments (slices) par artiste ---")
        counts = df['label'].value_counts()
        print(counts)
        print("\nTotal segments:", len(df))
    except Exception as e:
        print(f"Erreur lecture CSV: {e}")
else:
    print("Fichier CSV non trouvé.")
