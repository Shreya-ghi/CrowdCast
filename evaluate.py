"""evaluate.py

Evaluation, Diagnostics, and Visualization Script for CrowdCast.

This script:
1. Loads holdout predictions generated during model training (reports/holdout_test_predictions.csv)
   and the training set (reports/training_set.csv).
2. Computes comprehensive evaluation metrics for both Decision Tree and KNN:
   - Overall Accuracy
   - Macro Precision, Recall, F1-Score
   - Per-class Precision, Recall, F1-Score, and Support
   - Training vs Test Class Distributions
3. Generates and saves visual diagnostic plots into reports/:
   - Confusion matrices side-by-side for Decision Tree and KNN (reports/confusion_matrices.png)
   - Class distribution comparison plot (reports/class_distribution.png)
   - Decision Tree structure visualization (reports/decision_tree_structure.png)
4. Displays a detailed, honest evaluation narrative analyzing false predictions,
   class imbalance implications, and limitations of small dataset sizes.
"""

import argparse
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for clean script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import plot_tree

ORDERED_CLASSES = ["Low", "Moderate", "High"]


def print_metrics_table(model_name: str, y_true, y_pred, labels):
    """Prints a structured academic performance breakdown."""
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    print(f"\n=======================================================")
    print(f"   EVALUATION REPORT: {model_name.upper()}")
    print(f"=======================================================")
    print(f"Overall Accuracy:       {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Macro Precision:        {macro_prec:.4f}")
    print(f"Macro Recall:           {macro_rec:.4f}")
    print(f"Macro F1-Score:         {macro_f1:.4f}")
    print("-" * 55)
    print("Per-Class Classification Breakdown:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4))


def plot_confusion_matrices(y_true, dt_preds, knn_preds, labels, output_path: Path):
    """Generates side-by-side heatmaps of confusion matrices."""
    cm_dt = confusion_matrix(y_true, dt_preds, labels=labels)
    cm_knn = confusion_matrix(y_true, knn_preds, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Decision Tree Classifier\n(Holdout Trips)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Ground Truth", fontsize=11)

    sns.heatmap(
        cm_knn,
        annot=True,
        fmt="d",
        cmap="Purples",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title("K-Nearest Neighbors (KNN)\n(Holdout Trips)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Ground Truth", fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT CREATED] Confusion Matrices -> {output_path}")


def plot_class_distributions(df_train: pd.DataFrame, df_test: pd.DataFrame, output_path: Path):
    """Plots and compares class proportions between train and test holdout splits."""
    train_counts = df_train["crowding_level"].value_counts(normalize=True).reindex(ORDERED_CLASSES, fill_value=0) * 100
    test_counts = df_test["crowding_level"].value_counts(normalize=True).reindex(ORDERED_CLASSES, fill_value=0) * 100

    plot_df = pd.DataFrame({
        "Training (%)": train_counts,
        "Testing (%)": test_counts,
    })

    ax = plot_df.plot(kind="bar", figsize=(8, 5), color=["#2b5c8f", "#d95f02"], edgecolor="black")
    plt.title("Crowding Class Distribution: Train vs Test Splits", fontsize=12, fontweight="bold")
    plt.ylabel("Proportion of Observations (%)", fontsize=11)
    plt.xlabel("Crowding Class", fontsize=11)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[PLOT CREATED] Class Distribution  -> {output_path}")


def plot_decision_tree_diagram(dt_pipeline, label_encoder, output_path: Path):
    """Visualizes the trained Decision Tree logic and feature decisions."""
    try:
        classifier = dt_pipeline.named_steps["classifier"]
        preprocessor = dt_pipeline.named_steps["preprocessor"]

        # Extract generated feature names from ColumnTransformer
        feature_names = preprocessor.get_feature_names_out()

        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            classifier,
            feature_names=[str(f).replace("num__", "").replace("cat__", "") for f in feature_names],
            class_names=[str(c) for c in label_encoder.classes_],
            filled=True,
            rounded=True,
            fontsize=8,
            max_depth=3,  # Top levels for clear readable diagram
            ax=ax,
        )
        plt.title("Decision Tree Classifier (Top 3 Levels of Rule Splits)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=220)
        plt.close()
        print(f"[PLOT CREATED] Decision Tree Visual -> {output_path}")
    except Exception as e:
        print(f"[WARNING] Could not render decision tree diagram: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CrowdCast classification models.")
    parser.add_argument(
        "--reports-dir",
        type=str,
        default="reports",
        help="Directory with test predictions (default: reports)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained joblib models (default: models)",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir)

    preds_file = reports_dir / "holdout_test_predictions.csv"
    train_file = reports_dir / "training_set.csv"

    if not preds_file.exists():
        print(f"[ERROR] Prediction file not found at {preds_file}. Run 'python src/train.py' first.")
        raise SystemExit(1)

    df_test = pd.read_csv(preds_file)
    y_true = df_test["crowding_level"]
    dt_preds = df_test["dt_prediction"]
    knn_preds = df_test["knn_prediction"]

    # Filter active classes
    present_classes = [c for c in ORDERED_CLASSES if c in y_true.unique() or c in dt_preds.unique()]

    # Terminal Metrics Reports
    print_metrics_table("Decision Tree Classifier (Primary)", y_true, dt_preds, present_classes)
    print_metrics_table("K-Nearest Neighbors (KNN Classifier)", y_true, knn_preds, present_classes)

    # Confusion matrix plots
    plot_confusion_matrices(
        y_true,
        dt_preds,
        knn_preds,
        present_classes,
        reports_dir / "confusion_matrices.png",
    )

    # Class distribution plot
    if train_file.exists():
        df_train = pd.read_csv(train_file)
        plot_class_distributions(df_train, df_test, reports_dir / "class_distribution.png")

    # Decision tree diagram
    dt_model_path = models_dir / "decision_tree_model.joblib"
    le_path = models_dir / "label_encoder.joblib"
    if dt_model_path.exists() and le_path.exists():
        dt_pipeline = joblib.load(dt_model_path)
        label_encoder = joblib.load(le_path)
        plot_decision_tree_diagram(dt_pipeline, label_encoder, reports_dir / "decision_tree_structure.png")

    print("\n" + "=" * 70)
    print("                    CRITICAL ACADEMIC ANALYSIS & INSIGHTS             ")
    print("=" * 70)
    print("1. WHY DECISION TREE AND KNN BEHAVE DIFFERENTLY:")
    print("   - Feature Geometry: Decision Tree makes orthogonal, axis-aligned splits on")
    print("     one feature at a time (e.g. earlier_occupancy_ratio >= 0.72). It is naturally")
    print("     robust to monotonic transformations and unscaled categorical one-hot vectors.")
    print("   - KNN uses geometric distance (Manhattan or Euclidean) across an expanded")
    print("     multi-dimensional space. Even with standard scaling, sparse one-hot encoded")
    print("     stop names dilute geometric distance, causing KNN to suffer in higher dimensions.")
    print("\n2. GENERALIZATION ON UNSEEN TRIPS (TRIP-GROUPED SPLITTING):")
    print("   - By grouping by trip_id, no observations from the same bus trip exist in both")
    print("     train and test sets. This mirrors real life: we predict for a completely new trip!")
    print("\n3. CLASS IMBALANCE & BOUNDARY CASES:")
    print("   - Bus occupancy varies sharply between off-peak and peak hours.")
    print("   - 'Moderate' (50% to 79%) occupancy often suffers the most misclassifications")
    print("     because small 2-3 passenger fluctuations cross the 50% or 80% boundary.")
    print("\n4. SMALL DATASET REALITY:")
    print("   - In an undergraduate mini-project (300-500 observations), weather changes,")
    print("     delays, or events can cause outliers. High test accuracy on a small sample")
    print("     does not imply universal perfection across every bus route in the city.")
    print("=" * 70)


if __name__ == "__main__":
    main()
