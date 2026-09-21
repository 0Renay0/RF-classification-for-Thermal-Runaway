import os
import glob

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


# RECUPERATION DES SCENARIOS
fault_files = glob.glob(
    os.path.join(DATA_DIRS["Faults"], "*.csv")
)

nominal_files = glob.glob(
    os.path.join(DATA_DIRS["Nominal"], "*.csv")
)


print("=" * 60)
print("Nombre de scénarios")
print("=" * 60)

print(f"Faults  : {len(fault_files)}")
print(f"Nominal : {len(nominal_files)}")