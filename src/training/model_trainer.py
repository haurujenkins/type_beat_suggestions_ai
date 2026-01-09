import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, top_k_accuracy_score

# --- CONFIGURATION ---
INPUT_CSV = "data/dataset_audio.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "type_beat_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "encoder.pkl")

def train_model():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    print("⏳ Chargement du dataset...")
    df = pd.read_csv(INPUT_CSV)

    # --- ÉQUILIBRAGE (UNDERSAMPLING) ---
    min_count = df['label'].value_counts().min()
    print(f"⚖️  Équilibrage actif : Limite de {min_count} segments par artiste.")
    df_balanced = df.groupby('label', group_keys=False).apply(lambda x: x.sample(min_count, random_state=42))
    
    # 1. Préparation des données
    groups = df_balanced['filename'].apply(lambda x: x.split('__slice_')[0])
    X = df_balanced.drop(columns=['label', 'filename'])
    y = df_balanced['label']
    
    print(f"   - {len(X)} segments")
    print(f"   - {len(np.unique(groups))} chansons distinctes")
    
    # 2. Encodage
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    
    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. ENSEMBLE MODEL (Le "Avengers" des modèles)
    print("🚀 Entraînement de l'Ensemble (RF + MLP + KNN)...")
    
    # Model A: Random Forest (Reduced estimators to save space)
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    
    # Model B: Réseau de Neurones simple (Bon pour les patterns audio complexes MFCC)
    clf_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    
    # Model C: KNN (Capture les structures locales)
    clf_knn = KNeighborsClassifier(n_neighbors=9, weights='distance', n_jobs=-1)

    # Combinaison (Soft Voting = Moyenne des probabilités)
    eclf = VotingClassifier(
        estimators=[('rf', clf_rf), ('mlp', clf_mlp), ('knn', clf_knn)],
        voting='soft',
        n_jobs=-1
    )
    
    eclf.fit(X_train_scaled, y_train)
    
    # 6. Évaluation TOP-K
    print("🧠 Évaluation sur le Test Set...")
    
    # Prédictions brutes
    y_pred = eclf.predict(X_test_scaled)
    # Probabilités pour le Top-K
    y_proba = eclf.predict_proba(X_test_scaled)
    
    acc_1 = accuracy_score(y_test, y_pred) * 100
    acc_3 = top_k_accuracy_score(y_test, y_proba, k=3) * 100
    acc_5 = top_k_accuracy_score(y_test, y_proba, k=5) * 100
    
    print(f"\n🏆 RÉSULTATS (Segments individuels) :")
    print(f"   🥇 Top-1 Accuracy : {acc_1:.2f}% (Le bon artiste est suggéré en 1er)")
    print(f"   🥉 Top-3 Accuracy : {acc_3:.2f}% (Le bon artiste est dans les 3 premiers)")
    print(f"   🖐️  Top-5 Accuracy : {acc_5:.2f}% (Le bon artiste est dans les 5 premiers)")
    
    print("\n📊 Rapport détaillé Top-1 :")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 7. Sauvegarde (Compressée pour passer sur GitHub < 100Mo)
    print("💾 Sauvegarde du modèle Ensemble (Compression active)...")
    joblib.dump(eclf, MODEL_PATH, compress=9)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("✅ Terminé !")

if __name__ == "__main__":
    train_model()
