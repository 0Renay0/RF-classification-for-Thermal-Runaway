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

