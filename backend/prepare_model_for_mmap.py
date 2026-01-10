import joblib
import os
import sys

def decompress_model():
    """
    Charge le modèle compressé et le ré-enregistre sans compression (zlib).
    Cela permet d'utiliser mmap_mode='r' pour charger le modèle sans saturer la RAM.
    """
    # Définir les chemins relatifs au dossier backend ou root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # On cherche dans backend/models car c'est là que l'app Docker ira chercher
    models_dir = os.path.join(base_dir, "models")
    
    input_path = os.path.join(models_dir, "type_beat_model.pkl")
    output_path = os.path.join(models_dir, "model_uncompressed.pkl")
    
    print(f"📂 Répertoire cible : {models_dir}")

    if not os.path.exists(input_path):
        print(f"❌ Erreur : Le fichier {input_path} n'existe pas.")
        sys.exit(1)

    print(f"⏳ Chargement du modèle compressé : {input_path} ...")
    try:
        model = joblib.load(input_path)
    except Exception as e:
        print(f"❌ Erreur chargement joblib : {e}")
        sys.exit(1)

    print(f"💾 Sauvegarde du modèle NON COMPRESSÉ vers : {output_path} ...")
    
    # compress=0 est CRITIQUE. C'est ce qui crée un fichier compatible mmap.
    # Protocol pickle par défaut (souvent 4 ou 5) est ok.
    joblib.dump(model, output_path, compress=0) 
    
    # Vérification taille
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Conversion réussie !")
    print(f"📊 Nouvelle taille du fichier : {size_mb:.2f} MB")
    print(f"👉 Vous pouvez maintenant utiliser mmap_mode='r' sur ce fichier.")

if __name__ == "__main__":
    decompress_model()
