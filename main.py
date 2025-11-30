
print('Importing is starting')
import os
import time
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import joblib

import torch
from transformers import AutoModel, AutoTokenizer
print('Importing is ending')

# =================================================================================
# CONSTANTS
# =================================================================================

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
USER_DATA_FILENAME = "users.csv"
BOOK_DATA_FILENAME = "books.csv"
BOOK_GENRES_FILENAME = "book_genres.csv"
GENRES_FILENAME = "genres.csv"
BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
SUBMISSION_FILENAME = "submission.csv"
TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
BERT_EMBEDDINGS_FILENAME = "bert_embeddings.pkl"
BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
PROCESSED_DATA_FILENAME = "processed_features.parquet"

COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "rating"
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"
COL_HAS_READ = "has_read"
COL_TIMESTAMP = "timestamp"

F_USER_MEAN_RATING = "user_mean_rating"
F_USER_RATINGS_COUNT = "user_ratings_count"
F_BOOK_MEAN_RATING = "book_mean_rating"
F_BOOK_RATINGS_COUNT = "book_ratings_count"
F_AUTHOR_MEAN_RATING = "author_mean_rating"
F_BOOK_GENRES_COUNT = "book_genres_count"

COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"
COL_DESCRIPTION = "description"

VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

MISSING_CAT_VALUE = "-1"
MISSING_NUM_VALUE = -1
PREDICTION_MIN_VALUE = 0
PREDICTION_MAX_VALUE = 10

# =================================================================================
# CONFIG
# =================================================================================

ROOT_DIR = Path('./')
DATA_DIR = ROOT_DIR / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

N_SPLITS = 5
RANDOM_STATE = 42
TARGET = COL_TARGET
TEMPORAL_SPLIT_RATIO = 0.8
EARLY_STOPPING_ROUNDS = 50
MODEL_FILENAME = "LGBGEN2.txt"

TFIDF_MAX_FEATURES = 700 # 500
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

BERT_BATCH_SIZE = 8
BERT_MAX_LENGTH = 384 # 128
BERT_EMBEDDING_DIM = 768
BERT_DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
BERT_GPU_MEMORY_FRACTION = 0.8

CAT_FEATURES = [
    COL_USER_ID,
    COL_BOOK_ID,
    COL_GENDER,
    COL_AGE,
    COL_AUTHOR_ID,
    COL_PUBLICATION_YEAR,
    COL_LANGUAGE,
    COL_PUBLISHER,
]

LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
}

LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
}

# =================================================================================
# UTILS
# =================================================================================

def temporal_split_by_date(df, split_date, timestamp_col=COL_TIMESTAMP):
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found.")
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    train_mask = df[timestamp_col] <= split_date
    val_mask = df[timestamp_col] > split_date
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("One of the splits is empty!")
    return train_mask, val_mask

def get_split_date_from_ratio(df, ratio, timestamp_col=COL_TIMESTAMP):
    if not 0 < ratio < 1:
        raise ValueError("Ratio must be between 0 and 1")
    sorted_ts = df[timestamp_col].sort_values()
    return sorted_ts.iloc[int(len(sorted_ts) * ratio)]

# =================================================================================
# FEATURES
# =================================================================================

def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    print("Adding aggregate features...")
    user_agg = train_df.groupby(COL_USER_ID)[TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [COL_USER_ID, F_USER_MEAN_RATING, F_USER_RATINGS_COUNT]
    book_agg = train_df.groupby(COL_BOOK_ID)[TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [COL_BOOK_ID, F_BOOK_MEAN_RATING, F_BOOK_RATINGS_COUNT]
    author_agg = train_df.groupby(COL_AUTHOR_ID)[TARGET].mean().reset_index()
    author_agg.columns = [COL_AUTHOR_ID, F_AUTHOR_MEAN_RATING]

    df = df.merge(user_agg, on=COL_USER_ID, how="left")
    df = df.merge(book_agg, on=COL_BOOK_ID, how="left")
    df = df.merge(author_agg, on=COL_AUTHOR_ID, how="left")
    return df

def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    print("Adding genre count feature...")
    genre_counts = book_genres_df.groupby(COL_BOOK_ID).size().reset_index(name=F_BOOK_GENRES_COUNT)
    return df.merge(genre_counts, on=COL_BOOK_ID, how="left")

def add_text_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    tfidf_filename=TFIDF_VECTORIZER_FILENAME) -> pd.DataFrame:
    print("Adding TF-IDF features...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = MODEL_DIR / tfidf_filename

    train_books = train_df[COL_BOOK_ID].unique()
    train_desc = descriptions_df[descriptions_df[COL_BOOK_ID].isin(train_books)].copy()
    train_desc[COL_DESCRIPTION] = train_desc[COL_DESCRIPTION].fillna("")

    if vectorizer_path.exists():
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            ngram_range=TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_desc[COL_DESCRIPTION])
        joblib.dump(vectorizer, vectorizer_path)

    all_desc = descriptions_df[[COL_BOOK_ID, COL_DESCRIPTION]].copy()
    all_desc[COL_DESCRIPTION] = all_desc[COL_DESCRIPTION].fillna("")
    desc_map = dict(zip(all_desc[COL_BOOK_ID], all_desc[COL_DESCRIPTION], strict=False))
    df_desc = df[COL_BOOK_ID].map(desc_map).fillna("")
    tfidf_matrix = vectorizer.transform(df_desc)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])], index=df.index)
    return pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

def add_bert_features(
    df: pd.DataFrame,
    descriptions_df: pd.DataFrame,
    bert_filename=BERT_EMBEDDINGS_FILENAME) -> pd.DataFrame:
    print("Adding BERT embeddings...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    emb_path = MODEL_DIR / bert_filename

    if emb_path.exists():
        print(f'Loading embs...')
        embeddings_dict = joblib.load(emb_path)
    else:
        print(f"Computing BERT embeddings on {BERT_DEVICE}...")
        if BERT_DEVICE == "cuda" and torch:
            torch.cuda.set_per_process_memory_fraction(BERT_GPU_MEMORY_FRACTION)

        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(BERT_DEVICE)
        model.eval()

        desc_df = descriptions_df[[COL_BOOK_ID, COL_DESCRIPTION]].copy()
        desc_df[COL_DESCRIPTION] = desc_df[COL_DESCRIPTION].fillna("")
        unique = desc_df.drop_duplicates(subset=COL_BOOK_ID)

        embeddings_dict = {}
        batch_size = BERT_BATCH_SIZE
        for i in tqdm(range(0, len(unique), batch_size), desc="BERT batches", disable=True):
            batch = unique.iloc[i:i+batch_size]
            encoded = tokenizer(
                batch[COL_DESCRIPTION].tolist(),
                padding=True, truncation=True, max_length=BERT_MAX_LENGTH, return_tensors="pt"
            ).to(BERT_DEVICE)

            with torch.no_grad():
                outputs = model(**encoded)
                embeddings = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(embeddings.size()).float()
                summed = torch.sum(embeddings * mask, dim=1)
                counts = torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled = summed / counts

                for book_id, emb in zip(batch[COL_BOOK_ID], mean_pooled.cpu().numpy(), strict=False):
                    embeddings_dict[book_id] = emb

        joblib.dump(embeddings_dict, emb_path)

    book_ids = df[COL_BOOK_ID].tolist()
    embeddings = [embeddings_dict.get(bid, np.zeros(BERT_EMBEDDING_DIM)) for bid in book_ids]
    print(embeddings[0].shape)
    bert_df = pd.DataFrame(embeddings, columns=[f"bert_{i}" for i in range(BERT_EMBEDDING_DIM)], index=df.index)
    return pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    print("Handling missing values...")
    global_mean = train_df[TARGET].mean()
    df[COL_AGE] = df[COL_AGE].astype('float32')
    df[COL_AGE] = df[COL_AGE].fillna(df[COL_AGE].median())

    for col in [F_USER_MEAN_RATING, F_BOOK_MEAN_RATING, F_AUTHOR_MEAN_RATING]:
        if col in df.columns:
            df[col] = df[col].fillna(global_mean)
    for col in [F_USER_RATINGS_COUNT, F_BOOK_RATINGS_COUNT, F_BOOK_GENRES_COUNT]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    if COL_AVG_RATING in df.columns:
        df[COL_AVG_RATING] = df[COL_AVG_RATING].fillna(global_mean)

    for col in df.columns:
        if col.startswith(("tfidf_", "bert_")):
            df[col] = df[col].fillna(0.0)

    for col in CAT_FEATURES:
        
        if col in df.columns and df[col].isna().any():
            if df[col].dtype.name in ("category", "object"):
                df[col] = df[col].astype(str).fillna(MISSING_CAT_VALUE).astype("category")
            else:
                df[col] = df[col].fillna(MISSING_NUM_VALUE) 

    dtype_spec = {
        COL_USER_ID: "int32", COL_BOOK_ID: "int32", COL_TARGET: "float32",
        COL_GENDER: "category", COL_AGE: "float32", COL_AUTHOR_ID: "int32",
        COL_PUBLICATION_YEAR: "float32", COL_LANGUAGE: "category",
        COL_PUBLISHER: "category", COL_AVG_RATING: "float32", COL_GENRE_ID: "int16",
    } 

    for d in dtype_spec :
        if d in df.columns :
            df[d] = df[d].astype(dtype_spec[d])

    return df

# =================================================================================
# DATA LOADING
# =================================================================================

def load_and_merge_data():
    print("Loading and merging raw data...")
    dtype_spec = {
        COL_USER_ID: "int32", COL_BOOK_ID: "int32", COL_TARGET: "float32",
        COL_GENDER: "category", COL_AGE: "float32", COL_AUTHOR_ID: "int32",
        COL_PUBLICATION_YEAR: "float32", COL_LANGUAGE: "category",
        COL_PUBLISHER: "category", COL_AVG_RATING: "float32", COL_GENRE_ID: "int16",
    }

    train_df = pd.read_csv(DATA_DIR / TRAIN_FILENAME, dtype=dtype_spec, parse_dates=[COL_TIMESTAMP])
    train_df = train_df[train_df[COL_HAS_READ] == 1].copy()
    test_df = pd.read_csv(DATA_DIR / TEST_FILENAME, dtype=dtype_spec)
    user_df = pd.read_csv(DATA_DIR / USER_DATA_FILENAME, dtype=dtype_spec)
    book_df = pd.read_csv(DATA_DIR / BOOK_DATA_FILENAME, dtype=dtype_spec)
    book_df = book_df.drop_duplicates(subset=[COL_BOOK_ID])
    book_genres_df = pd.read_csv(DATA_DIR / BOOK_GENRES_FILENAME)
    book_descriptions_df = pd.read_csv(DATA_DIR / BOOK_DESCRIPTIONS_FILENAME, dtype={COL_BOOK_ID: "int32"})

    train_df[COL_SOURCE] = VAL_SOURCE_TRAIN
    test_df[COL_SOURCE] = VAL_SOURCE_TEST
    df = pd.concat([train_df, test_df], ignore_index=True)
    df = df.merge(user_df, on=COL_USER_ID, how="left")
    df = df.merge(book_df, on=COL_BOOK_ID, how="left")

    return df, book_genres_df, book_descriptions_df

def prepare_data(
    data_filename=PROCESSED_DATA_FILENAME,
    tfidf_filename=TFIDF_VECTORIZER_FILENAME,
    bert_filename=BERT_EMBEDDINGS_FILENAME
    ):
    print("="*60)
    print("STARTING DATA PREPARATION")
    print("="*60)

    merged_df, book_genres_df, book_descriptions_df = load_and_merge_data()

    print("Running feature engineering (without aggregates)...")
    df = merged_df.copy()
    df = add_genre_features(df, book_genres_df)
    df = add_text_features(
        df, df[df[COL_SOURCE] == VAL_SOURCE_TRAIN],
        book_descriptions_df,
        tfidf_filename=tfidf_filename)
    df = add_bert_features(
        df,
        book_descriptions_df,
        bert_filename=bert_filename)
    df = handle_missing_values(df, df[df[COL_SOURCE] == VAL_SOURCE_TRAIN])

    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DATA_DIR / data_filename
    df.to_parquet(path, index=False, compression="snappy")
    print(f"Processed data saved to {path}")
    print("Preparation complete!")

# =================================================================================
# TRAIN
# =================================================================================

def train(
    model_name=MODEL_FILENAME,
    data_filename=PROCESSED_DATA_FILENAME,
    train_size=TEMPORAL_SPLIT_RATIO,
    eval=True,
    gpu=False):
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)

    path = PROCESSED_DATA_DIR / data_filename
    if not path.exists():
        raise FileNotFoundError("Run prepare_data() first!")
    df = pd.read_parquet(path)

    train_df = df[df[COL_SOURCE] == VAL_SOURCE_TRAIN].copy()
    split_date = get_split_date_from_ratio(train_df, train_size, COL_TIMESTAMP)
    print(f"Temporal split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_df, split_date)
    train_split = train_df[train_mask].copy()
    val_split = train_df[val_mask].copy()

    train_split = add_aggregate_features(train_split, train_split)
    val_split = add_aggregate_features(val_split, train_split)
    train_split = handle_missing_values(train_split, train_split)
    val_split = handle_missing_values(val_split, train_split)

    exclude = [COL_SOURCE, TARGET, COL_PREDICTION, COL_TIMESTAMP]
    features = [c for c in train_split.columns if c not in exclude and train_split[c].dtype != "object"]
    X_train, y_train = train_split[features], train_split[TARGET]
    X_val, y_val = val_split[features], val_split[TARGET]

    if eval :
      print(f"Training on {len(X_train)} samples with params={LGB_PARAMS}, validating on {len(X_val)}, {len(features)} features")
      if gpu :
        LGB_PARAMS.update({'device' : 'gpu'})
      MODEL_DIR.mkdir(parents=True, exist_ok=True)
      model = lgb.LGBMRegressor(**LGB_PARAMS)
      model.fit(
          X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
      )

      preds = model.predict(X_val)
      rmse = np.sqrt(mean_squared_error(y_val, preds))
      mae = mean_absolute_error(y_val, preds)
      print(f"Validation RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    else :
      X_A = pd.concat([X_train, X_val], axis=0)
      Y_A = pd.concat([train_split[[TARGET]], val_split[[TARGET]]], axis=0)[TARGET]
      print(f"Training on {len(X_A)} samples with params={LGB_PARAMS}, validating on {len(X_A)}, {len(features)} features")

      MODEL_DIR.mkdir(parents=True, exist_ok=True)
      model = lgb.LGBMRegressor(**LGB_PARAMS)
      model.fit(
          X_A, Y_A,
          eval_set=[(X_A, Y_A)],
          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)]
      )

      preds = model.predict(X_A)
      rmse = np.sqrt(mean_squared_error(Y_A, preds))
      mae = mean_absolute_error(Y_A, preds)
      print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    model.booster_.save_model(MODEL_DIR / model_name)
    print(f"Model saved to {MODEL_DIR / model_name}")

# =================================================================================
# PREDICT
# =================================================================================

def predict(model_name=MODEL_FILENAME, data_filename=PROCESSED_DATA_FILENAME, save=True):
    print("="*60)
    print("GENERATING PREDICTIONS")
    print("="*60)

    path = PROCESSED_DATA_DIR / data_filename
    if not path.exists():
        raise FileNotFoundError("Run prepare_data() first!")
    df = pd.read_parquet(path)

    train_df = df[df[COL_SOURCE] == VAL_SOURCE_TRAIN].copy()
    test_df = df[df[COL_SOURCE] == VAL_SOURCE_TEST].copy()

    test_df = add_aggregate_features(test_df, train_df)
    test_df = handle_missing_values(test_df, train_df)

    exclude = [COL_SOURCE, TARGET, COL_PREDICTION, COL_TIMESTAMP]
    features = [c for c in test_df.columns if c not in exclude and test_df[c].dtype != "object"]
    X_test = test_df[features]

    model_path = MODEL_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError("Train model first!")
    model = lgb.Booster(model_file=str(model_path))

    preds = model.predict(X_test)
    preds = np.clip(preds, PREDICTION_MIN_VALUE, PREDICTION_MAX_VALUE)

    sub = test_df[[COL_USER_ID, COL_BOOK_ID]].copy()
    sub[COL_PREDICTION] = preds

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    if save :
      sub_path = SUBMISSION_DIR / SUBMISSION_FILENAME
      sub.to_csv(sub_path, index=False)
      print(f"Submission saved to {sub_path}")
    print(f"Predictions: mean={preds.mean():.4f}, min={preds.min():.4f}, max={preds.max():.4f}")

    return sub

def ensemble_results(model_names, data_filename=PROCESSED_DATA_FILENAME, save=True) :

  print("="*60)
  print("GENERATING ENSEMBLE")
  print("="*60)

  preds = [predict(model_name, data_filename=data_filename, save=False) for model_name in model_names]
  combined = pd.concat(preds)
  sub = combined.groupby(['book_id', 'user_id'], as_index=False)['rating_predict'].mean()

  SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
  if save :
    sub_path = SUBMISSION_DIR / SUBMISSION_FILENAME
    sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

  return sub

"""LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    'max_depth': 3,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.6,
    "bagging_freq": 1,
    "lambda_l1": 3,
    "lambda_l2": 3,
    "verbose": -1,
    "n_jobs": -1,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
}
"""
ensemble_results(['LGB_1TEMPV1.txt', 'LGB_09TEMPV1.txt'])