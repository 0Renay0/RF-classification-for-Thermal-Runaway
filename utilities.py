from RFClassification import FEATURES, TARGET
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