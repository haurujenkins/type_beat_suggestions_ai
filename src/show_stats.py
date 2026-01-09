import pandas as pd
try:
    df = pd.read_csv('data/dataset_audio.csv')
    print(df['label'].value_counts())
except Exception as e:
    print(e)
