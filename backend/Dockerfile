# Utiliser une image Python légère
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# 1. Installer les dépendances système (CRITIQUE pour Librosa)
# libsndfile1 est requis pour lire les fichiers audio
# ffmpeg est souvent nécessaire pour le re-scoding mp3
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 2. Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copier le code de l'application
COPY . .

# 4. Copier explicitement le dossier models (s'il n'est pas déjà copié via le point précédent)
# Assurez-vous que le dossier 'models/' est bien DANS le dossier 'backend/' avant de build
# COPY models/ models/ 

# Exposer le port (Render utilise la variable PORT, mais 8000 par défaut)
EXPOSE 8000

# Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
