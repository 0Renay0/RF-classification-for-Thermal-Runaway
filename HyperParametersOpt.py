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


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600, step=50),
        'max_depth': trial.suggest_int('max_depth', [None, 5, 10, 15, 20, 30, 40]),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
    }

    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)
    return float(f1_score(y_val, pred, pos_label=1, zero_division=0))


study = optuna.create_study(
    study_name="RandomForestClassifier_Optimization",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    storage=f'sqlite:///{OUTPUT_DIR / "study.db".resolve().as_posix()}',
    load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
print(f'best F1 score: study.best_value')
print(f'best params: {study.best_params}')
study.trials_dataframe().to_csv(OUTPUT_DIR / "trials.csv", index=False)

# the model uses train and validation sets, the test set is used for final evaluation
X_fit = pd.concat([X_train, X_val], ignore_index=True)
y_fit = pd.concat([y_train, y_val], ignore_index=True)
final_model = RandomForestClassifier(**study.best_params, random_state=RANDOM_STATE, n_jobs=-1)
final_model.fit(X_fit, y_fit)
pred = final_model.predict(X_test)
proba = final_model.predict_proba(X_test)[:, list(final_model.classes_).index(1)]
metrics = {
    'accuracy': float(accuracy_score(y_test, pred)),
    'precision': float(precision_score(y_test,pred, pos_label=1, zero_division=0)),
    'recall': float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
    'f1': float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
    'roc_auc': float(roc_auc_score(y_test, pred)), 
}

print('\nMetrics TEST:', json.dumps(metrics, indent=2))
print(classification_report(y_test, pred, zero_division=0))
print('Confusion Matrix:\n', confusion_matrix(y_test, pred))
joblib.dump(final_model, OUTPUT_DIR / 'random_forest_optuna.joblib')
(OUTPUT_DIR / 'results.json').write_text(
    json.dumps({
        'Validation_best_f1': study.best_value,
        'best_params': study.best_params,
        'test_metrics': metrics,
        'features': FEATURES,
        'n_train_scenarios': len(train_s),
        'n_validation_scenarios': len(val_s),
        'n_test_scenarios': len(test_s)},
        indent=2),
        encoding='utf-8'
)


if __name__ == '__main__':
    main()