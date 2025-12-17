#  MACHINE LEARNING & PATTERN RECOGNITION - CA01
#  Uğur Ercan

print("LOADING LIBRARIES...")

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# load dataset
df = pd.read_excel("Dataset_Ugur_Ercan.xlsx")
print("Initial shape:", df.shape)
print("Head of the dataset:\n", df.head())
print("General info of the dataset:\n")
df.info()

# filter for CLASSIC matches
df = df[df["game_mode"] == "CLASSIC"].reset_index(drop=True)
print("After filtering CLASSIC:", df.shape)

# drop unnecessary columns
drop_cols = [
    "game_start_utc", "game_version", "platform_id", "queue_id", "participant_id",
    "puuid", "summoner_name", "summoner_id", "champion_name", "champion_id",
    "individual_position", "lane", "role", "solo_tier", "solo_rank", "flex_tier",
    "flex_rank", "champion_mastery_lastPlayTime", "champion_mastery_lastPlayTime_utc"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print("After dropping columns:", df.shape)

# handle missing values
df = df.dropna(subset=["team_position", "win"])
df = df.fillna(0)
df["win"] = df["win"].astype(int)
print("After handling missing values:", df.shape)

print("how many rows each team has: ", df.groupby(["game_id", "team_id"]).size().value_counts().sort_index())


# aggregate to team-level
agg_dict = {
    "kills": "sum", "deaths": "sum", "assists": "sum",
    "baron_kills": "sum", "dragon_kills": "sum",
    "gold_earned": "sum", "gold_spent": "sum",
    "total_damage_dealt_to_champions": "sum",
    "damage_dealt_to_objectives": "sum",
    "damage_dealt_to_turrets": "sum",
    "total_damage_taken": "sum", "vision_score": "mean",
    "wards_placed": "sum", "wards_killed": "sum",
    "vision_wards_bought_in_game": "sum",
    "time_ccing_others": "mean", "win": "max"
}
agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
team_df = df.groupby(["game_id", "team_id"]).agg(agg_dict).reset_index()
team_df = team_df.drop_duplicates(subset=["game_id", "team_id"]).reset_index(drop=True)

print("\nFinal team-level dataset shape:", team_df.shape)
print(f"\nTarget label value counts:\n{team_df["win"].value_counts()}")


# split features and target
X = team_df.drop(columns=["win", "game_id", "team_id"], errors="ignore")
y = team_df["win"]

print("\nColumns used for training:", X.columns.tolist())


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


numerical_cols = X.columns  # all features are numeric

# create transformers
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# combine into a single ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols)
    ]
)

# integrated pipeline for SVM
from sklearn.svm import SVC
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', random_state=42))
])

# integrated pipeline for Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# fit models with pipeline

rf_pipeline.fit(X_train, y_train)
svm_pipeline.fit(X_train, y_train)

y_pred_rf = rf_pipeline.predict(X_test)
y_pred_svm = svm_pipeline.predict(X_test)

print("\n=== RANDOM FOREST BASELINE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

print("\n=== SVM BASELINE ===")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\n", classification_report(y_test, y_pred_svm))

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest hyperparameter tuning (Grid Search)
param_rf = {
    "classifier__n_estimators": [400, 500, 600], # number of trees
    "classifier__max_depth": [None, 20],      # maximum depth of the tree
    "classifier__min_samples_split": [1, 2, 3], # minimum samples required to split a node
    "classifier__min_samples_leaf": [1, 2], # minimum samples required at each leaf node
}


grid_rf = GridSearchCV(
    rf_pipeline, param_rf, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_rf.fit(X_train, y_train)

# SVM hyperparameter tuning (Grid Search)
param_svm = {
    "classifier__kernel": ["linear", "rbf", "poly", "sigmoid"],     # linear kernel performed best during test runs
    "classifier__C": [0.01, 0.1, 1, 10, 50, 100],                   # margin parameter
    # "classifier__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]   # how far the influence of a single training sample reaches (not applicable for linear kernel)                 
}


grid_svm = GridSearchCV(
    svm_pipeline, param_svm, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1
)
grid_svm.fit(X_train, y_train)

print(f"Random Forest best CV Score: {grid_rf.best_score_:.3f}")
print(f"SVM best CV Score: {grid_svm.best_score_:.3f}")

print("Best RF Params:", grid_rf.best_params_)
best_rf = grid_rf.best_estimator_
y_pred_rf_tuned = best_rf.predict(X_test)
 
print("Best SVM Params:", grid_svm.best_params_)
best_svm = grid_svm.best_estimator_
y_pred_svm_tuned = best_svm.predict(X_test)


# Summary of results
print("\n=== MODEL PERFORMANCE SUMMARY ===")
print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf_tuned):.3f}")
print(f"SVM Test Accuracy: {accuracy_score(y_test, y_pred_svm_tuned):.3f}")
