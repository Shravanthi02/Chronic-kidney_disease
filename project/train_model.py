# train_model.py
# ─────────────────────────────────────────────────────────────────────────────
# Stand-alone script to train, evaluate, and save the best CKD classifier.
# Run:  python train_model.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (accuracy_score, precision_score, recall_score,
                                      f1_score, confusion_matrix, classification_report)
from sklearn.model_selection  import train_test_split, cross_val_score

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(BASE_DIR, "dataset", "Training_CKD_dataset.csv")
TEST_PATH   = os.path.join(BASE_DIR, "dataset", "Testing_CKD_dataset.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "model",   "kidney_model.pkl")
ASSETS_DIR  = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ─── Add project root to sys.path ────────────────────────────────────────────
sys.path.insert(0, BASE_DIR)
from ckd_utils.preprocessing import preprocess_pipeline, check_data_quality, load_data


def train_and_evaluate():
    print("=" * 70)
    print("  Chronic Kidney Disease - Model Training Pipeline")
    print("=" * 70)

    # ── 1. Load & Quality Check ───────────────────────────────────────────
    print("\n[1/5] Loading datasets ...")
    train_raw = load_data(TRAIN_PATH)
    test_raw  = load_data(TEST_PATH)

    full_data = pd.concat([train_raw, test_raw], ignore_index=True)
    train_split, test_split = train_test_split(
        full_data, test_size=0.2, random_state=42, stratify=full_data["Target"]
    )

    q = check_data_quality(train_split)
    print(f"  Training set  : {q['shape'][0]:,} rows × {q['shape'][1]} cols")
    print(f"  Missing values: {q['total_missing']}")
    print(f"  Duplicate rows: {q['duplicate_rows']}")

    # ── 2. Full Preprocessing ─────────────────────────────────────────────
    print("\n[2/5] Preprocessing ...")
    X_train, X_test, y_train, y_test, encoders, scaler, class_names = \
        preprocess_pipeline(train_split, test_split)

    print("\n===== FEATURE CHECK =====")
    print(f"Target column in X_train: {'Target' in X_train.columns}")
    print(f"Number of features: {X_train.shape[1]}")
    print("Feature names:")
    for col in X_train.columns:
        print(f"  - {col}")
    print()

    print(f"  Classes  : {class_names}")

    # ── 3. Define Models ──────────────────────────────────────────────────
    print("\n[3/5] Training classifiers ...")
    models = {
        "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree":        DecisionTreeClassifier(max_depth=10, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results      = {}
    best_name    = None
    best_f1      = -1
    best_model   = None

    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results[name] = {
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1_score":  round(f1,   4),
            "model":     clf,
            "y_pred":    y_pred,
            "time_s":    round(time.time() - t0, 2),
        }
        print(f"  {name:<25}  Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  ({results[name]['time_s']}s)")

        if f1 > best_f1:
            best_f1    = f1
            best_name  = name
            best_model = clf

    # ── 4. Save Best Model Bundle ─────────────────────────────────────────
    print(f"\n[4/5] Best model -> {best_name}  (F1={best_f1:.4f})")

    print("\nRunning 5-Fold Cross Validation...")
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    print("\nCross Validation Scores:")
    print(cv_scores)

    print(f"\nAverage CV Accuracy : {cv_scores.mean():.4f}")
    print(f"Standard Deviation  : {cv_scores.std():.4f}\n")

    # Build comparison metrics (exclude model/y_pred objects)
    comparison_metrics = {
        name: {k: v for k, v in res.items() if k not in ("model", "y_pred")}
        for name, res in results.items()
    }

    # Full classification report for the best model
    best_y_pred = results[best_name]["y_pred"]
    report_str  = classification_report(y_test, best_y_pred,
                                        target_names=class_names,
                                        zero_division=0)
    cm = confusion_matrix(y_test, best_y_pred)

    bundle = {
        "model":              best_model,
        "model_name":         best_name,
        "encoders":           encoders,
        "scaler":             scaler,
        "classes":            class_names,
        "features":           list(X_train.columns),
        "cv_scores":          cv_scores.tolist(),
        "cv_mean_accuracy":   float(cv_scores.mean()),
        "cv_std_accuracy":    float(cv_scores.std()),
        "metrics": {
            "accuracy":       results[best_name]["accuracy"],
            "precision":      results[best_name]["precision"],
            "recall":         results[best_name]["recall"],
            "f1_score":       results[best_name]["f1_score"],
        },
        "comparison_metrics": comparison_metrics,
        "confusion_matrix":   cm,
        "classification_report": report_str,
        "class_distribution": train_split["Target"].value_counts().to_dict(),
        "feature_names":      list(X_train.columns),
        "train_shape":        train_split.shape,
        "test_shape":         test_split.shape,
    }

    joblib.dump(bundle, MODEL_PATH)
    print(f"  Saved -> {MODEL_PATH}")

    # ── 5. Save Assets ────────────────────────────────────────────────────
    print("\n[5/5] Generating plots...")

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title(f"Confusion Matrix – {best_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(ASSETS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix -> {cm_path}")

    # Model Comparison Bar Chart
    metric_df = pd.DataFrame(
        {n: {k: v for k, v in r.items() if k in ("accuracy","precision","recall","f1_score")}
         for n, r in results.items()}
    ).T
    fig, ax = plt.subplots(figsize=(10, 5))
    metric_df.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="white", width=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    comp_path = os.path.join(ASSETS_DIR, "model_comparison.png")
    fig.savefig(comp_path, dpi=150)
    plt.close(fig)
    print(f"  Model comparison -> {comp_path}")

    # Feature Importance (if applicable)
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15] # Top 15
        top_features = [list(X_train.columns)[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_importances, y=top_features, ax=ax, palette="viridis")
        ax.set_title("Top 15 Feature Importances", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        fi_path = os.path.join(ASSETS_DIR, "feature_importance.png")
        fig.savefig(fi_path, dpi=150)
        plt.close(fig)
        print(f"  Feature importance -> {fi_path}")

    print("\n" + "=" * 70)
    print("  Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    train_and_evaluate()
