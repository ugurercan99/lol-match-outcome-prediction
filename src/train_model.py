# MACHINE LEARNING & PATTERN RECOGNITION - CA01
# Uğur Ercan
# Refactored for modularity and readability

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# =========================
# Configuration
# =========================

DATA_PATH = "data/raw/dataset.xlsx"
RANDOM_STATE = 42
TEST_SIZE = 0.2


# =========================
# Data Loading
# =========================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"Initial dataset shape: {df.shape}")
    return df


# =========================
# Data Cleaning & Filtering
# =========================

def filter_classic_games(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["game_mode"] == "CLASSIC"].reset_index(drop=True)
    print(f"After filtering CLASSIC games: {df.shape}")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        "game_start_utc", "game_version", "platform_id", "queue_id", "participant_id",
        "puuid", "summoner_name", "summoner_id", "champion_name", "champion_id",
        "individual_position", "lane", "role", "solo_tier", "solo_rank",
        "flex_tier", "flex_rank", "champion_mastery_lastPlayTime",
        "champion_mastery_lastPlayTime_utc"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"After dropping columns: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["team_position", "win"])
    df = df.fillna(0)
    df["win"] = df["win"].astype(int)
    print(f"After handling missing values: {df.shape}")
    return df


# =========================
# Aggregation
# =========================

def aggregate_to_team_level(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {
        "kills": "sum",
        "deaths": "sum",
        "assists": "sum",
        "baron_kills": "sum",
        "dragon_kills": "sum",
        "gold_earned": "sum",
        "gold_spent": "sum",
        "total_damage_dealt_to_champions": "sum",
        "damage_dealt_to_objectives": "sum",
        "damage_dealt_to_turrets": "sum",
        "total_damage_taken": "sum",
        "vision_score": "mean",
        "wards_placed": "sum",
        "wards_killed": "sum",
        "vision_wards_bought_in_game": "sum",
        "time_ccing_others": "mean",
        "win": "max"
    }

    agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

    team_df = (
        df.groupby(["game_id", "team_id"])
          .agg(agg_dict)
          .reset_index()
    )

    print(f"Final team-level dataset shape: {team_df.shape}")
    print("Target distribution:\n", team_df["win"].value_counts())

    return team_df


# =========================
# Feature / Target Split
# =========================

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["win", "game_id", "team_id"], errors="ignore")
    y = df["win"]
    print(f"Training features: {X.columns.tolist()}")
    return X, y


# =========================
# Pipeline Builder
# =========================

def build_pipeline(model_type: str, feature_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_columns)
        ]
    )

    if model_type == "rf":
        model = RandomForestClassifier(random_state=RANDOM_STATE)
    elif model_type == "svm":
        model = SVC(random_state=RANDOM_STATE)
    else:
        raise ValueError("Invalid model type")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    return pipeline



# =========================
# Training & Evaluation
# =========================

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test, label: str):
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print(f"\n=== {label} RESULTS ===")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))


# =========================
# Hyperparameter Tuning
# =========================

def tune_model(pipeline, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    return grid


# =========================
# Main
# =========================

def main():
    df = load_data(DATA_PATH)
    df = filter_classic_games(df)
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)

    team_df = aggregate_to_team_level(df)
    X, y = split_features_target(team_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    rf_pipeline = build_pipeline("rf", X.columns)
    svm_pipeline = build_pipeline("svm", X.columns)


    train_and_evaluate(rf_pipeline, X_train, X_test, y_train, y_test, "Random Forest (Baseline)")
    train_and_evaluate(svm_pipeline, X_train, X_test, y_train, y_test, "SVM (Baseline)")

    rf_params = {
        "classifier__n_estimators": [400, 500, 600],
        "classifier__max_depth": [None, 20],
        "classifier__min_samples_split": [2, 3],
        "classifier__min_samples_leaf": [1, 2]
    }

    svm_params = {
        "classifier__kernel": ["linear", "rbf"],
        "classifier__C": [0.1, 1, 10, 50]
    }

    rf_grid = tune_model(rf_pipeline, rf_params, X_train, y_train)
    svm_grid = tune_model(svm_pipeline, svm_params, X_train, y_train)

    print("\n=== TUNED MODEL PERFORMANCE ===")
    print(f"Random Forest Test Accuracy: {accuracy_score(y_test, rf_grid.best_estimator_.predict(X_test)):.3f}")
    print(f"SVM Test Accuracy: {accuracy_score(y_test, svm_grid.best_estimator_.predict(X_test)):.3f}")


if __name__ == "__main__":
    main()
# End of train_model.py