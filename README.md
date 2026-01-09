# Projet IA Musicale

Ce projet vise à constituer un dataset local à partir de type beat d'artistes variés afin de prédire et suggérer des artistes pour de nouvelles instrumentales. Cet outil est un outil SEO de suggestion d'artiste par reconnaissance IA sur des caractéristiques audio. Il est aussi doté d'un enrichissement des suggestions par analyse des tendances actuelles sur youtube et spotify.

Tester l'application : https://type-beat-suggestion-ai.streamlit.app/

to dl from youtube playlist :
     yt-dlp -x --audio-format mp3 --audio-quality 0 --restrict-filenames --download-sections "*1:00-1:45" -o "data/raw_audio/Eminem/%(title)s.%(ext)s" "https://youtube.com/playlist\?list\=PLdxBy4Ifgex-1v5u6CzhYo-6WUR_LdGpf\&si\=VFZbSSHF8NdBTvkj"

convert to csv : 
    python3.10 ./src/etl_audio/audio_to_csv.py 

to train model : 
    python3.10 ./src/training/model_trainer.py 

to test model : 
    python src/predict.py "/Users/lpr/Desktop/Code/Type_beat_suggest/data/raw_audio/Drake/BEST_Drake_-_Lemon_Pepper_Freestyle_Instrumental.mp3"

launch app :
     streamlit run src/app.py


App, à ajouter en dernier:
    le model propose une dizaine d'artistes se rapprochant de l'instrumentale. ensuite un autre algorithme prend ces 10 artistes en entrée et analysé le nombre de prods postées sur youtube les 7 derniers jours pour chaque artiste et sa popularité actuelle (streams ? top 50 ?) et propose les meilleurs options a l'utilisateur pour son SEO

# type_beat_suggestions_ai
