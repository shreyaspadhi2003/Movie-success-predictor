# --- utils/preprocess.py ---

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    titles = pd.read_csv('data/titles.csv')
    credits = pd.read_csv('data/credits.csv')
    merged = pd.merge(titles, credits[credits['role'].isin(['ACTOR', 'DIRECTOR'])], 
                     on='id', how='inner')
    return merged

def engineer_features(df):
    if 'name' in df.columns:
        df['actor_avg'] = df.groupby('name')['imdb_score'].transform('mean')
        df['director_avg'] = (df[df['role']=='DIRECTOR'].groupby('name')['imdb_score'].transform('mean'))
    else:
        df['actor_avg'] = 6.5
        df['director_avg'] = 6.5

    global_actor_avg = df['imdb_score'].mean()
    global_director_avg = df[df['role']=='DIRECTOR']['imdb_score'].mean()
    df['actor_avg'] = df['actor_avg'].fillna(global_actor_avg)
    df['director_avg'] = df['director_avg'].fillna(global_director_avg)

    try:
        genres = df['genres'].apply(eval).explode()
        top_genres = genres.value_counts().head(10).index
        for genre in top_genres:
            df[f'genre_{genre}'] = df['genres'].apply(lambda x: 1 if genre in x else 0)
    except:
        top_genres = ['Drama', 'Comedy', 'Action']
        for genre in top_genres:
            df[f'genre_{genre}'] = 0

    df.attrs['top_genres'] = list(top_genres)
    features = ['runtime', 'release_year', 'actor_avg', 'director_avg'] + \
               [f'genre_{g}' for g in top_genres]
    return df[features].fillna(0)

def get_targets(df):
    clean_df = df.dropna(subset=['imdb_score'])
    y_reg = clean_df['imdb_score']
    y_clf = (y_reg > 7).astype(int)
    return y_reg, y_clf, clean_df.index

def get_dataset_stats(df):
    return {
        'min_year': int(df['release_year'].dropna().min()),
        'max_year': int(df['release_year'].dropna().max()),
        'min_runtime': int(max(1, df['runtime'].dropna().min())),
        'max_runtime': int(df['runtime'].dropna().max()),
        'common_runtime': int(df['runtime'].mode()[0])
    }

def create_name_lookups(df):
    actor_avg = df[df['role']=='ACTOR'].groupby('name')['imdb_score'].mean().to_dict()
    director_avg = df[df['role']=='DIRECTOR'].groupby('name')['imdb_score'].mean().to_dict()
    return {
        'actors': sorted(actor_avg.keys()),
        'directors': sorted(director_avg.keys()),
        'actor_avg': actor_avg,
        'director_avg': director_avg
    }

def prepare_train_test(df, test_size=0.2):
    X = engineer_features(df)
    y_reg, y_clf, valid_indices = get_targets(df)
    X = X.loc[valid_indices]
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=test_size, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=test_size, random_state=42)
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test