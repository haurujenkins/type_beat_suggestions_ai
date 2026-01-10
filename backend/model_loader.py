import joblib
import os
import sys
import gc

def load_ai_models(models_dir="models"):
    """
    Charge les 3 fichiers nécessaires au modèle (Modèle, Scaler, Encoder).
    Gère les chemins relatifs pour Docker.
    Optimisation Mémoire : Garbage Collection immédiat.
    """
    print(f"🔄 Chargement des modèles depuis : {models_dir}...")
    
    try:
        model_path = os.path.join(models_dir, "type_beat_model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        encoder_path = os.path.join(models_dir, "encoder.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier modèle introuvable: {model_path}")

        # Chargement avec joblib
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)

        # Récupération des features attendues (si disponible dans le scaler)
        expected_features = getattr(scaler, 'feature_names_in_', None)

        # Libération immédiate de la mémoire temporaire
        gc.collect()

        print("✅ Modèles chargés avec succès.")
        return model, scaler, encoder, expected_features

    except Exception as e:
        print(f"❌ Erreur critique lors du chargement du modèle : {e}")
        # En production, on veut peut-être stopper le conteneur si le modèle ne charge pas
        sys.exit(1) 
