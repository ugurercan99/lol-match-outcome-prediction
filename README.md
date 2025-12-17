# League of Legends Match Outcome Prediction

## Overview
This project applies supervised machine learning techniques to predict whether a team will win
a competitive League of Legends match using aggregated team-level gameplay statistics.

The workflow follows the CRISP-DM methodology and demonstrates an end-to-end ML pipeline
including data preparation, feature engineering, model training, hyperparameter tuning, and evaluation.

## Problem Statement
Given team-level performance metrics such as kills, damage, gold, objectives, and vision,
can we accurately predict match outcomes (win / loss) in competitive matches?

## Dataset
- Original data format: Player-level match statistics
- Processed format: Aggregated team-level data
- Final dataset size: 5,880 samples × 19 features
- Target variable: `win` (1 = win, 0 = loss)
- Only **CLASSIC** game mode matches were used to ensure consistency

## Data Preparation
- Removed non-predictive identifiers and irrelevant features
- Handled missing values
- Aggregated player-level statistics into team-level metrics
- Applied feature scaling using `StandardScaler`
- Used Scikit-learn Pipelines to prevent data leakage

## Models Used
- Random Forest Classifier
- Support Vector Machine (SVM)

Hyperparameter tuning was performed using `GridSearchCV` with 5-fold `StratifiedKFold`.

## Results
| Model | Test Accuracy |
|------|--------------|
| Random Forest | 0.969 |
| SVM | 0.965 |

Both models show strong and balanced performance with no signs of overfitting.

## Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib

## How to Run
pip install -r requirements.txt
python3 src/train_model.py

## Future Improvements
- Feature importance analysis (SHAP)
- Model comparison with gradient boosting
- Deployment as a REST API

