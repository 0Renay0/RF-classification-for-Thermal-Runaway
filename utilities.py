from RFClassification import FEATURES, TARGET
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix,roc_auc_score)
import pandas as pd 
import numpy as np 

# LOADING DATA 
def load_scenarios(scenario_table):

    X_list = []
    y_list = []

    for file_path in scenario_table["path"]:

        df = pd.read_csv(file_path)

        required_cols = FEATURES + [TARGET]

        missing_cols = [
            col
            for col in required_cols
            if col not in df.columns
        ]

        if missing_cols:
            print(
                f"Fichier ignoré : {file_path}\n"
                f"Colonnes manquantes : {missing_cols}"
            )
            continue

        # On garde uniquement les variables nécessaires
        data = df[FEATURES + [TARGET]].copy()

        # Suppression éventuelle des NaN / inf
        data = data.replace(
            [np.inf, -np.inf],
            np.nan
        )

        data = data.dropna()

        X = data[FEATURES]
        y = data[TARGET].astype(int)

        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise ValueError(
            "Aucun scénario valide n'a pu être chargé."
        )

    X = pd.concat(
        X_list,
        ignore_index=True
    )

    y = pd.concat(
        y_list,
        ignore_index=True
    )

    return X, y

# EVALUATE RF
def evaluate_model(
    model,
    X,
    y,
    dataset_name
):

    y_pred = model.predict(X)

    y_proba = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        y_pred
    )

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    # ROC-AUC uniquement si les deux classes sont présentes
    if len(np.unique(y)) == 2:

        auc = roc_auc_score(
            y,
            y_proba
        )

        print(f"ROC-AUC   : {auc:.4f}")


    print("\nClassification report:")

    print(
        classification_report(
            y,
            y_pred,
            digits=4,
            zero_division=0
        )
    )


    print("Confusion matrix:")

    print(
        confusion_matrix(
            y,
            y_pred
        )
    )
