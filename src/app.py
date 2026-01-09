import streamlit as st
import librosa
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler
from collections import Counter

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Type Beat Suggest",
    page_icon="🎹",
    layout="centered"
)

# Custom Styling (Light Model & Clean Ergonomics)
st.markdown("""
<style>
    /* Global Clean Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Background - Light */
    .stApp {
        background-color: #FAFAFA;
        color: #31333F;
    }
    
    /* Headers */
    .big-title {
        font-size: 2.5rem !important;
        font-weight: 900;
        color: #1E1E1E;
        text-align: center;
        margin-top: 1rem;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #666666;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /* Clean Card Design */
    .simple-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #F0F0F0;
        margin-bottom: 0.8rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .simple-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-color: #FF4B4B;
    }

    /* Winner Section */
    .winner-container {
        background-color: #FFFFFF;
        border-left: 6px solid #FF4B4B;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 3rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }
    
    .winner-label {
        color: #FF4B4B;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .winner-name {
        font-size: 3.5rem;
        font-weight: 900;
        color: #1E1E1E;
        margin: 0;
        line-height: 1.1;
    }
    
    .winner-stat {
        font-size: 1.1rem;
        color: #666;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* Progress Customization */
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    
    /* Remove default clutter */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# Chemins
DATASET_PATH = "data/dataset_audio.csv"

# --- FONCTIONS ---

@st.cache_data(ttl=600)
def load_popularity_data():
    """Charge les données de popularité (Spotify/YouTube)."""
    pop_path = "data/artist_popularity.csv"
    if os.path.exists(pop_path):
        try:
            df = pd.read_csv(pop_path)
            # Normalisation clef
            df['search_name'] = df['search_name'].astype(str)
            return df.set_index('search_name')
        except:
            return None
    return None

@st.cache_resource
def load_recommendation_system():
    # Paths from model_trainer.py
    model_path = "models/type_beat_model.pkl"
    scaler_path = "models/scaler.pkl"
    encoder_path = "models/encoder.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path)):
        st.error("⚠️ Modèles introuvables ! Veuillez lancer l'entraînement via model_trainer.py")
        return None, None, None, None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        
        # Get expected features order
        expected_features = getattr(scaler, 'feature_names_in_', None)
        
        return model, scaler, encoder, expected_features
    except Exception as e:
        st.error(f"Erreur de chargement du modèle: {e}")
        return None, None, None, None

def extract_features_silent(file_path):
    # Version simplifiée pour l'UI, sans logs
    SAMPLE_RATE = 22050
    DURATION_SLICE = 30
    try:
        y_full, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        total_samples = len(y_full)
        samples_per_slice = sr * DURATION_SLICE
        
        if total_samples < samples_per_slice:
            y_slices = [np.pad(y_full, (0, samples_per_slice - total_samples), 'constant')]
        else:
            y_slices = []
            for i in range(0, total_samples, samples_per_slice):
                segment = y_full[i : i + samples_per_slice]
                if len(segment) >= (sr * 10): 
                    if len(segment) < samples_per_slice:
                        segment = np.pad(segment, (0, samples_per_slice - len(segment)), 'constant')
                    y_slices.append(segment)
        
        all_features_list = []
        for y in y_slices:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            rms = librosa.feature.rms(y=y)
            zcr = librosa.feature.zero_crossing_rate(y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)

            tempo_val = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo)
            features = {"tempo": tempo_val}
            
            for name, data in [("rms", rms), ("zcr", zcr), ("spec_cent", spec_cent), ("spec_roll", spec_roll)]:
                features[f"{name}_mean"] = np.mean(data)
                features[f"{name}_var"]  = np.var(data)

            for i in range(13):
                features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
                features[f"mfcc_{i}_var"]  = np.var(mfccs[i])
            for i in range(12):
                features[f"chroma_{i}_mean"] = np.mean(chroma[i])
                features[f"chroma_{i}_var"]  = np.var(chroma[i])
                
            all_features_list.append(features)
            
        return pd.DataFrame(all_features_list)
    except:
        return None

# --- UI MAIN ---

st.markdown('<div class="big-title">TYPE BEAT SUGGEST</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Importez un fichier audio pour identifier son style</div>', unsafe_allow_html=True)

model, scaler, encoder, feature_cols = load_recommendation_system()

# Upload Area - Simple
uploaded_file = st.file_uploader("Fichier audio (MP3/WAV)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None and model is not None:
    # Save temp
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Lecteur simple
    st.audio(uploaded_file, format='audio/mp3')
    
    st.write("")
    
    analysis_results = None

    # Status simple et clair (pas de termes techniques trop visibles)
    with st.status("Analyse en cours...", expanded=True) as status:
        st.write("Segmentation & Extraction des caractéristiques...")
        features_slices_df = extract_features_silent("temp_audio.mp3")
        
        if features_slices_df is not None and not features_slices_df.empty:
            st.write("Comparaison avec la base de données...")
            
            # --- LOGIC ---
            # Ensure columns consistency
            if feature_cols is not None:
                # Add missing cols if any (robustness)
                for col in feature_cols:
                    if col not in features_slices_df.columns:
                        features_slices_df[col] = 0
                input_df = features_slices_df[feature_cols]
            else:
                input_df = features_slices_df.drop(columns=['filename'], errors='ignore')

            # Prediction
            X_scaled = scaler.transform(input_df)
            
            # Get probabilities from Ensemble Model
            # shape: (n_slices, n_classes)
            probas = model.predict_proba(X_scaled) 
            
            # Average probabilities across all slices (Consensus)
            avg_probas = probas.mean(axis=0)
            
            # Map to class names
            classes = encoder.classes_
            results = list(zip(classes, avg_probas))
            
            # Sort by probability descending
            results.sort(key=lambda x: x[1], reverse=True)
            
            winner_artist = results[0][0]
            winner_conf = results[0][1] * 100
            top_artists = results[:10]
            
            # Store results
            analysis_results = {
                "winner_artist": winner_artist,
                "winner_conf": winner_conf,
                "top_artists": top_artists,
            }
            
            status.update(label="Terminé !", state="complete", expanded=False)

        else:
            status.update(label="Erreur", state="error")
            st.error("L'analyse du fichier a échoué.")
            
    # --- DISPLAY RESULTS (Outside Status Container) ---
    if analysis_results:
        winner_artist = analysis_results["winner_artist"]
        winner_conf = analysis_results["winner_conf"]
        top_artists = analysis_results["top_artists"]
        # total_neighbors removal

        # Load Popularity to enrich UI
        pop_df = load_popularity_data()
        
        # Prepare Data: Match detected artists with their Market Score (YouTube Views)
        candidates_ui = []
        # Take Top 10 matches
        for artist, proba in top_artists: 
            score = 0
            if pop_df is not None and artist in pop_df.index:
                # Use YouTube Avg Views as "Hype/Market Score"
                score = pop_df.loc[artist].get('youtube_avg_views', 0)
            candidates_ui.append({'artist': artist, 'proba': proba, 'score': score})
            
        # Identify the Top 3 "Most Popular" among these candidates to emphasize them
        sorted_by_score = sorted(candidates_ui, key=lambda x: x['score'], reverse=True)
        top3_names = {x['artist'] for x in sorted_by_score[:3]} if candidates_ui else set()

        st.markdown(f"""
        <div class="winner-container">
            <div class="winner-label">RÉSULTAT PREDOMINANT</div>
            <div class="winner-name">{winner_artist}</div>
            <div class="winner-stat">Confiance : {winner_conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Styles & Tendances")
        st.caption("Top 10 Probabilités (Encadré Vert = Top 3 Demande YouTube)")
        
        # Horizontal Layout (Flexbox)
        html_cards = '<div style="display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px;">'
        
        for item in candidates_ui:
            artist = item['artist']
            prob = int(item['proba'] * 100)
            is_trending = artist in top3_names
            
            # Styles
            border = "2px solid #4CAF50" if is_trending else "1px solid #E0E0E0"
            bg = "#E8F5E9" if is_trending else "#FFFFFF"
            text_col = "#1B5E20" if is_trending else "#31333F"
            box_shadow = "0 4px 12px rgba(76, 175, 80, 0.2)" if is_trending else "0 2px 5px rgba(0,0,0,0.03)"
            
            badge_html = ""
            if is_trending:
                 badge_html = '<div style="font-size:0.65rem; font-weight:800; color:#4CAF50; margin-top:6px; letter-spacing:0.5px;">🔥 STRONG DEMAND</div>'

            html_cards += f"""
<div style="flex: 1 1 140px; min-width: 140px; max-width: 200px; padding: 1.2rem; border-radius: 12px; border: {border}; background-color: {bg}; box-shadow: {box_shadow}; text-align: center; display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <div style="font-weight:700; font-size:1.15rem; color:{text_col}; line-height:1.2;">{artist}</div>
    <div style="font-size:0.85rem; color:#888; margin-top:4px; font-weight:500;">{prob}% match</div>
    {badge_html}
</div>"""
        
        html_cards += "</div>"
        st.markdown(html_cards, unsafe_allow_html=True)


    if os.path.exists("temp_audio.mp3"):
        os.remove("temp_audio.mp3")
