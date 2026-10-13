import json 
import os 
from pathlib import Path 

import joblib
import numpy as np 
import optuna 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, precision_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from utilities import DATA_DIRS, FEATURES, TEST_SIZE, VAL_SIZE, RANDOM_STATE, load_scenarios


N_TRIALS = 50
OUTPUT_DIR = Path("outputs/optuna")

def collect_scenarios():
    rows = []
    for kind in ('faults', 'nominal'):
        for path in sorted(Path(DATA_DIRS[kind]).glob("*.csv")):
            rows.append({'path': str(path), 'type': kind})
    df = pd.DataFrame(rows, columns=['path', 'type'])
    if df.empty:
        raise ValueError("No scenarios found in the specified directories.")
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = collect_scenarios()
    train_s, temp_s = train_test_split(scenarios, test_size=VAL_SIZE + TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=scenarios['type'])
    # Split the remaining data into validation and test sets
    test_fraction_in_temp = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    val_s, test_s = train_test_split(temp_s, test_size=test_fraction_in_temp, random_state=RANDOM_STATE, shuffle=True, stratify=temp_s['type'])

    for name, group in [('train', train_s), ('val', val_s), ('test', test_s)]:
        group.to_csv(OUTPUT_DIR / f"scenarios_{name}.csv", index=False)
        print(f'{name}: {len(group)} scenarios')

    X_train, y_train = load_scenarios(train_s)
    X_val, y_val = load_scenarios(val_s)
    X_test, y_test = load_scenarios(test_s)
    if y_train.nunique() != 2 or y_val.nunique() != 2 or y_test.nunique() != 2:
        raise ValueError("The target variable must have exactly two classes for binary classification.")


