import os
import glob
import pandas as pd
import matplotlib as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from sklearn.tree import plot_tree
from utilities import (
    load_scenarios,
    evaluate_model,
    DATA_DIRS,
    FEATURES,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    MODEL_OUTPUT_PATH,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_probability_distribution,
)

import joblib

# RECUPERATION DES SCENARIOS
fault_files = glob.glob(os.path.join(DATA_DIRS["Faults"], "*.csv"))

nominal_files = glob.glob(os.path.join(DATA_DIRS["Nominal"], "*.csv"))


print("=" * 60)
print("Nombre de scénarios")
print("=" * 60)

print(f"Faults  : {len(fault_files)}")
print(f"Nominal : {len(nominal_files)}")

# CREATION DE LA LISTE DES SCENARIOS
scenarios = []

for file_path in fault_files:
    scenarios.append({"path": file_path, "type": "Fault"})

for file_path in nominal_files:
    scenarios.append({"path": file_path, "type": "Nominal"})


scenario_df = pd.DataFrame(scenarios)

# SPLIT TRAIN/VAL/TEST >>>> SPLIT PAR SCENARIO
# 70% TRAIN
# 30% VAL/TEST

train_scenarios, temp_scenarios = train_test_split(
    scenario_df,
    test_size=TEST_SIZE + VAL_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=scenario_df["type"],
)

val_scenarios, test_scenarios = train_test_split(
    temp_scenarios,
    test_size=0.5,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=temp_scenarios["type"],
)


print("\n" + "=" * 60)
print("SPLIT PAR SCÉNARIO")
print("=" * 60)

print(f"Train      : {len(train_scenarios)} scénarios")
print(f"Validation : {len(val_scenarios)} scénarios")
print(f"Test       : {len(test_scenarios)} scénarios")


print("\nRépartition TRAIN")
print(train_scenarios["type"].value_counts())

print("\nRépartition VALIDATION")
print(val_scenarios["type"].value_counts())

print("\nRépartition TEST")
print(test_scenarios["type"].value_counts())


# LOAD DATA
X_train, y_train = load_scenarios(train_scenarios)

X_val, y_val = load_scenarios(val_scenarios)

X_test, y_test = load_scenarios(test_scenarios)


print("\n" + "=" * 60)
print("NOMBRE D'ÉCHANTILLONS")
print("=" * 60)

print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")


print("\nDistribution des labels - TRAIN")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))


print("\nDistribution des labels - VALIDATION")
print(y_val.value_counts())
print(y_val.value_counts(normalize=True))


print("\nDistribution des labels - TEST")
print(y_test.value_counts())
print(y_test.value_counts(normalize=True))

# RANDOM FOREST
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    # Utile si label 1 est moins fréquent que label 0
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ENTRAINEMENT
print("\n" + "=" * 60)
print("ENTRAINEMENT RANDOM FOREST")
print("=" * 60)

model.fit(X_train, y_train)

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# EVALUATION
evaluate_model(model, X_train, y_train, "TRAIN")

evaluate_model(model, X_val, y_val, "VALIDATION")

evaluate_model(model, X_test, y_test, "TEST")

# IMPORTANCE DES VARIABLES
feature_importance = pd.DataFrame(
    {"feature": FEATURES, "importance": model.feature_importances_}
)

feature_importance = feature_importance.sort_values(by="importance", ascending=False)


print("\n" + "=" * 60)
print("IMPORTANCE DES VARIABLES")
print("=" * 60)

print(feature_importance)

plt.figure(figsize=(8, 5))

plt.barh(feature_importance["feature"], feature_importance["importance"])

plt.xlabel("Importance")
plt.ylabel("Variable")
plt.title("Importance des variables - Random Forest")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=300)

plt.show()
plt.close()

plot_confusion_matrix(model, X_train, y_train, "TRAIN")
plot_confusion_matrix(model, X_val, y_val, "VALIDATION")
plot_confusion_matrix(model, X_test, y_test, "TEST")

plot_roc_curve(model, X_train, y_train, "TRAIN")
plot_roc_curve(model, X_val, y_val, "VALIDATION")
plot_roc_curve(model, X_test, y_test, "TEST")

tree_0 = model.estimators_[0]
plt.figure(figsize=(24, 12))

plot_tree(
    tree_0,
    feature_names=FEATURES,
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8,
)

plt.title("Exemple : arbre n°0 du Random Forest")

plt.tight_layout()

plt.savefig(os.path.join(PLOTS_DIR, "tree_example_0.png"), dpi=300)

plt.show()
plt.close()

plot_probability_distribution(model, X_test, y_test, "TEST")

# SAUVEGARDE
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

joblib.dump(model, MODEL_OUTPUT_PATH)

print(f"\nModèle sauvegardé : {MODEL_OUTPUT_PATH}")
