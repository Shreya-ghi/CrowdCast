"""src/train.py

Model Training Pipeline for CrowdCast.

This module:
1. Loads the prepared clean dataset (data/processed/crowdcast_clean.csv).
2. Engineers pre-arrival features (e.g. earlier occupancy ratio, prediction hour & minute).
3. Enforces strict leakage prevention: isolates 'crowding_level' as target and drops
   target-stop passenger count, target occupancy, target time, observation_id, etc.
4. Performs a trip-grouped train/test split (using GroupShuffleSplit on 'trip_id')
   so that observations from the same physical trip are never split across train and test.
5. Builds scikit-learn Pipelines:
   - Primary: Preprocessor -> DecisionTreeClassifier
   - Comparison: Preprocessor -> KNeighborsClassifier (with StandardScaler for numerical features)
6. Performs hyperparameter tuning using GridSearchCV with GroupKFold cross-validation on training data.
7. Evaluates performance, compares models, saves metrics to JSON, and serializes
   the best pipeline model and preprocessing artifact using joblib.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Target column name
TARGET_COLUMN = "crowding_level"

# Columns strictly prohibited from model feature sets (prevent target leakage & ID memorization)
COLUMNS_TO_EXCLUDE = [
    "observation_id",
    "trip_id",
    "target_passenger_count",
    "target_occupancy_percentage",
    "target_observation_time",
    "crowding_level",
    "observation_date",  # Grouping / temporal metadata, not direct causal feature
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derives pre-arrival domain features strictly available BEFORE the bus reaches the target stop."""
    X = df.copy()

    # Feature 1: Previous Occupancy Ratio (earlier passenger count / practical capacity)
    X["earlier_occupancy_ratio"] = (
        X["passengers_before_target"] / X["practical_capacity"].replace(0, np.nan)
    ).fillna(0.0)

    # Feature 2: Prediction time parsed into hour and minute components
    def parse_time_features(val):
        try:
            parts = str(val).split(":")
            h = int(parts[0])
            m = int(parts[1]) if len(parts) > 1 else 0
            return pd.Series([h, m, h + (m / 60.0)])
        except Exception:
            return pd.Series([12, 0, 12.0])

    time_feats = X["prediction_time"].apply(parse_time_features)
    time_feats.columns = ["prediction_hour", "prediction_minute", "prediction_time_decimal"]
    X = pd.concat([X, time_feats], axis=1)

    # Drop non-feature and leakage columns
    feature_cols = [c for c in X.columns if c not in COLUMNS_TO_EXCLUDE and c != "prediction_time"]
    return X[feature_cols]


def create_preprocessors(numeric_features, categorical_features):
    """Creates scikit-learn ColumnTransformer preprocessors for tree and distance models."""
    # Preprocessor for Decision Tree (StandardScaler is not strictly necessary for trees)
    num_pipe_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe_tree = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor_tree = ColumnTransformer(
        transformers=[
            ("num", num_pipe_tree, numeric_features),
            ("cat", cat_pipe_tree, categorical_features),
        ]
    )

    # Preprocessor for KNN (Scaling is CRITICAL because KNN uses Euclidean / Minkowski distances)
    num_pipe_knn = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe_knn = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor_knn = ColumnTransformer(
        transformers=[
            ("num", num_pipe_knn, numeric_features),
            ("cat", cat_pipe_knn, categorical_features),
        ]
    )

    return preprocessor_tree, preprocessor_knn


def train_and_tune_models(
    clean_csv_path: Path,
    models_dir: Path,
    reports_dir: Path,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[Pipeline, Pipeline, Dict[str, any]]:
    """Loads clean data, performs trip-grouped split, tunes Decision Tree and KNN,

    saves trained models, predictions, and metrics.
    """
    if not clean_csv_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {clean_csv_path}. Run src/prepare_data.py first!"
        )

    df = pd.read_csv(clean_csv_path)
    if len(df) < 10:
        raise ValueError(
            f"Dataset has only {len(df)} records. Please collect or prepare at least 15-30 observations."
        )

    # Verify target exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from processed data.")

    # Trip-grouped split: observations from the same trip_id stay together in train OR test
    trip_groups = df["trip_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=trip_groups))

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    print("=" * 70)
    print("           CROWDCAST: MODEL TRAINING & HYPERPARAMETER TUNING           ")
    print("=" * 70)
    print(f"Total dataset size:        {len(df)} records across {df['trip_id'].nunique()} unique trips")
    print(f"Training set:              {len(df_train)} records across {df_train['trip_id'].nunique()} trips")
    print(f"Holdout test set:          {len(df_test)} records across {df_test['trip_id'].nunique()} trips")
    print("-" * 70)

    # Features and Targets
    X_train_raw = extract_features(df_train)
    y_train_raw = df_train[TARGET_COLUMN]

    X_test_raw = extract_features(df_test)
    y_test_raw = df_test[TARGET_COLUMN]

    # Fit target LabelEncoder so classes map cleanly across both Decision Tree and KNN (especially scikit-learn 1.8+)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    target_classes = label_encoder.classes_.tolist()

    # Identify numeric and categorical columns
    numeric_features = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numerical features ({len(numeric_features)}):   {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    print("-" * 70)

    preprocessor_tree, preprocessor_knn = create_preprocessors(numeric_features, categorical_features)

    # 1. Primary Algorithm: Decision Tree Classifier
    dt_pipeline = Pipeline([
        ("preprocessor", preprocessor_tree),
        ("classifier", DecisionTreeClassifier(random_state=random_state)),
    ])

    dt_param_grid = {
        "classifier__max_depth": [3, 4, 5, 7, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__criterion": ["gini", "entropy"],
    }

    # Use GroupKFold on training trips so folds maintain trip isolation during CV
    n_splits_cv = min(5, df_train["trip_id"].nunique())
    if n_splits_cv < 2:
        cv_strategy = 2
        groups_arg = None
    else:
        cv_strategy = GroupKFold(n_splits=n_splits_cv)
        groups_arg = df_train["trip_id"]

    print(f"Tuning Decision Tree via GridSearch ({n_splits_cv}-fold GroupKFold)...")
    dt_grid = GridSearchCV(
        dt_pipeline,
        param_grid=dt_param_grid,
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=-1,
    )
    dt_grid.fit(X_train_raw, y_train, groups=groups_arg)
    best_dt = dt_grid.best_estimator_

    # 2. Comparison Algorithm: K-Nearest Neighbors Classifier
    knn_pipeline = Pipeline([
        ("preprocessor", preprocessor_knn),
        ("classifier", KNeighborsClassifier()),
    ])

    # K values must not exceed training samples
    max_k = min(25, len(df_train) - 1)
    k_candidates = [k for k in [3, 5, 7, 9, 11, 15] if k <= max_k]
    if not k_candidates:
        k_candidates = [3]

    knn_param_grid = {
        "classifier__n_neighbors": k_candidates,
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    }

    print(f"Tuning KNN via GridSearch ({n_splits_cv}-fold GroupKFold)...")
    knn_grid = GridSearchCV(
        knn_pipeline,
        param_grid=knn_param_grid,
        cv=cv_strategy,
        scoring="f1_macro",
        n_jobs=-1,
    )
    knn_grid.fit(X_train_raw, y_train, groups=groups_arg)
    best_knn = knn_grid.best_estimator_

    # Predictions on holdout test set
    dt_test_preds = best_dt.predict(X_test_raw)
    knn_test_preds = best_knn.predict(X_test_raw)

    dt_acc = accuracy_score(y_test, dt_test_preds)
    dt_f1 = f1_score(y_test, dt_test_preds, average="macro", zero_division=0)

    knn_acc = accuracy_score(y_test, knn_test_preds)
    knn_f1 = f1_score(y_test, knn_test_preds, average="macro", zero_division=0)

    print("\n--- MODEL PERFORMANCE COMPARISON ON UNSEEN HOLD-OUT TRIPS ---")
    print(f"Decision Tree (Primary):    Accuracy = {dt_acc:.4f} | Macro F1 = {dt_f1:.4f}")
    print(f"  Best DT Hyperparameters:  {dt_grid.best_params_}")
    print(f"KNN (Comparison):           Accuracy = {knn_acc:.4f} | Macro F1 = {knn_f1:.4f}")
    print(f"  Best KNN Hyperparameters: {knn_grid.best_params_}")
    print("-" * 70)

    # Save models and label encoder
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dt_model_path = models_dir / "decision_tree_model.joblib"
    knn_model_path = models_dir / "knn_model.joblib"
    le_path = models_dir / "label_encoder.joblib"

    joblib.dump(best_dt, dt_model_path)
    joblib.dump(best_knn, knn_model_path)
    joblib.dump(label_encoder, le_path)
    print(f"[SAVED] Decision Tree Pipeline -> {dt_model_path}")
    print(f"[SAVED] KNN Pipeline           -> {knn_model_path}")
    print(f"[SAVED] Target Label Encoder   -> {le_path}")

    # Save feature metadata so the Streamlit app matches columns exactly
    metadata = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_classes": target_classes,
        "training_samples": len(df_train),
        "test_samples": len(df_test),
        "training_trips": int(df_train["trip_id"].nunique()),
        "test_trips": int(df_test["trip_id"].nunique()),
        "dt_best_params": {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v) for k, v in dt_grid.best_params_.items()},
        "knn_best_params": {k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v) for k, v in knn_grid.best_params_.items()},
        "dt_metrics": {"accuracy": float(dt_acc), "macro_f1": float(dt_f1)},
        "knn_metrics": {"accuracy": float(knn_acc), "macro_f1": float(knn_f1)},
    }
    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] Model Metadata         -> {meta_path}")

    # Save test partition and decoded predictions for in-depth evaluate.py
    df_eval = df_test.copy()
    df_eval["dt_prediction"] = label_encoder.inverse_transform(dt_test_preds)
    df_eval["knn_prediction"] = label_encoder.inverse_transform(knn_test_preds)
    eval_csv_path = reports_dir / "holdout_test_predictions.csv"
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"[SAVED] Holdout Test Preds     -> {eval_csv_path}")

    # Save train set for class distribution check in evaluate.py
    df_train.to_csv(reports_dir / "training_set.csv", index=False)

    print("=" * 70)
    return best_dt, best_knn, metadata


def main():
    parser = argparse.ArgumentParser(description="Train Decision Tree and KNN models for CrowdCast.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/crowdcast_clean.csv",
        help="Path to cleaned dataset (default: data/processed/crowdcast_clean.csv)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained joblib models (default: models)",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory to save evaluation artifacts (default: reports)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Holdout test proportion grouped by trip_id (default: 0.25)",
    )
    args = parser.parse_args()

    train_and_tune_models(
        clean_csv_path=Path(args.data),
        models_dir=Path(args.models_dir),
        reports_dir=Path(args.reports_dir),
        test_size=args.test_size,
    )


if __name__ == "__main__":
    main()
