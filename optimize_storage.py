import os
import glob
import numpy as np
from tqdm import tqdm

def optimize_storage():
    """
    Script d'optimisation de l'espace disque pour les spectrogrammes.
    Action : .npy (float64, non compressé) -> .npz (float32, compressé)
    """
    target_dir = os.path.join("data", "v2_spectrograms")
    
    print(f"--- Démarrage de l'optimisation du stockage ---")
    print(f"Cible : {target_dir}")

    # Recherche récursive de tous les fichiers .npy
    # Note: On utilise recursive=True pour chercher dans les sous-dossiers (artistes)
    search_pattern = os.path.join(target_dir, "**", "*.npy")
    files = glob.glob(search_pattern, recursive=True)

    if not files:
        print("Aucun fichier .npy trouvé.")
        return

    print(f"Fichiers à traiter : {len(files)}")

    total_initial_bytes = 0
    total_final_bytes = 0
    files_processed = 0
    errors = 0

    # Barre de progression
    pbar = tqdm(files, desc="Conversion et Compression", unit="file")

    for file_path in pbar:
        try:
            # 1. Calcul taille initiale
            initial_size = os.path.getsize(file_path)
            total_initial_bytes += initial_size

            # 2. Chargement et Conversion (Downcasting float64 -> float32)
            data = np.load(file_path)
            
            # Vérification de sécurité: on s'assure que c'est bien des nombres
            if data.dtype == np.float64:
                data_f32 = data.astype(np.float32)
            else:
                # Si déjà float32 ou autre, on garde tel quel pour la compression
                data_f32 = data

            # 3. Sauvegarde compressée (.npz)
            new_file_path = file_path.replace('.npy', '.npz')
            
            # On utilise le mot clé 'spec' pour faciliter le chargement futur
            np.savez_compressed(new_file_path, spec=data_f32)

            # 4. Vérification et Nettoyage
            if os.path.exists(new_file_path):
                # Vérification de la taille finale
                final_size = os.path.getsize(new_file_path)
                total_final_bytes += final_size
                
                # Suppression de l'original UNIQUEMENT si le nouveau existe
                os.remove(file_path)
                files_processed += 1
            else:
                print(f"[ERREUR] Le fichier compressé n'a pas été créé : {new_file_path}")
                errors += 1

        except Exception as e:
            print(f"\n[ERREUR] Échec sur {file_path} : {str(e)}")
            errors += 1
            # On ne supprime surtout pas le fichier original en cas d'erreur de lecture/écriture

    # 5. Statistiques Finales
    saved_bytes = total_initial_bytes - total_final_bytes
    saved_gb = saved_bytes / (1024 ** 3)  # Conversion en Gigaoctets
    initial_gb = total_initial_bytes / (1024 ** 3)
    final_gb = total_final_bytes / (1024 ** 3)

    print("\n" + "="*40)
    print("RÉSULTATS DE L'OPTIMISATION")
    print("="*40)
    print(f"Fichiers traités avec succès : {files_processed}")
    print(f"Erreurs rencontrées        : {errors}")
    print(f"Espace initial utilisé     : {initial_gb:.2f} Go")
    print(f"Espace final utilisé       : {final_gb:.2f} Go")
    print(f"ESPACE LIBÉRÉ              : {saved_gb:.2f} Go")
    print("="*40)

if __name__ == "__main__":
    optimize_storage()
