from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import pandas as pd
import numpy as np
import matplotlib as plt
import os

# CONFIGURATION
DATA_DIRS = {
    "Faults": "Data/filtered/Faults",
    "Nominal": "Data/filtered/Nominal",
}

FEATURES = [
    "T_filtered",
    "P_filtered",
    "dT_filtered_dt",
    "dP_filtered_dt",
]

TARGET = "label"

TEST_SIZE = 0.15
VAL_SIZE = 0.15

RANDOM_STATE = 42

MODEL_OUTPUT_PATH = "outputs/models/random_forest_baseline.joblib"


# LOADING DATA
def load_scenarios(scenario_table):
    X_list = []
    y_list = []

    for file_path in scenario_table["path"]:
        df = pd.read_csv(file_path)

        required_cols = FEATURES + [TARGET]

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Fichier ignoré : {file_path}\nColonnes manquantes : {missing_cols}")
            continue

        # On garde uniquement les variables nécessaires
        data = df[FEATURES + [TARGET]].copy()

        # Suppression éventuelle des NaN / inf
        data = data.replace([np.inf, -np.inf], np.nan)

        data = data.dropna()

        X = data[FEATURES]
        y = data[TARGET].astype(int)

        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise ValueError("Aucun scénario valide n'a pu être chargé.")

    X = pd.concat(X_list, ignore_index=True)

    y = pd.concat(y_list, ignore_index=True)

    return X, y


# EVALUATE RF
def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)

    y_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)

    precision = precision_score(y, y_pred, zero_division=0)

    recall = recall_score(y, y_pred, zero_division=0)

    f1 = f1_score(y, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # ROC-AUC uniquement si les deux classes sont présentes
    if len(np.unique(y)) == 2:
        auc = roc_auc_score(y, y_proba)

        print(f"ROC-AUC   : {auc:.4f}")

    print("\nClassification report:")

    print(classification_report(y, y_pred, digits=4, zero_division=0))

    print("Confusion matrix:")

    print(confusion_matrix(y, y_pred))


def plot_confusion_matrix(model, X, y, dataset_name):
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax, values_format="d")

    ax.set_title(f"Matrice de confusion - {dataset_name}")

    plt.tight_layout()
    PLOTS_DIR = "outputs/plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.savefig(
        os.path.join(PLOTS_DIR, f"confusion_matrix_{dataset_name.lower()}.png"), dpi=300
    )

    plt.show()
    plt.close()


def plot_roc_curve(model, X, y, dataset_name):
    if len(np.unique(y)) != 2:
        print(f"ROC non tracée pour {dataset_name}: une seule classe présente.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_estimator(model, X, y, ax=ax)

    ax.set_title(f"Courbe ROC - {dataset_name}")

    plt.tight_layout()
    PLOTS_DIR = "outputs/plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"roc_{dataset_name.lower()}.png"), dpi=300)

    plt.show()
    plt.close()


def plot_probability_distribution(model, X, y, dataset_name):
    y_proba = model.predict_proba(X)[:, 1]

    plt.figure(figsize=(8, 5))

    plt.hist(y_proba[y.to_numpy() == 0], bins=50, alpha=0.6, label="Label 0")

    plt.hist(y_proba[y.to_numpy() == 1], bins=50, alpha=0.6, label="Label 1")

    plt.xlabel("Probabilité prédite de la classe 1")
    plt.ylabel("Nombre d'échantillons")

    plt.title(f"Distribution des probabilités - {dataset_name}")

    plt.legend()

    plt.tight_layout()

    PLOTS_DIR = "outputs/plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(PLOTS_DIR, f"probabilities_{dataset_name.lower()}.png"), dpi=300
    )

    plt.show()
    plt.close()
