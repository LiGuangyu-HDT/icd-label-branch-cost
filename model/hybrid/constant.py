import os


# Data root. Resolved relative to this file, so the working directory does not matter.
# After preparing MIMIC-III, either set ICD_DATA_DIR=/path/to/data or edit the default below.
# It must contain: mimic3/ (per-split json and csv), mimic2/,
#                  D_ICD_DIAGNOSES.csv、D_ICD_PROCEDURES.csv、ICD9_descriptions
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("ICD_DATA_DIR", os.path.join(_HERE, "sample_data"))
MIMIC_2_DIR = os.path.join(DATA_DIR, "mimic2")
MIMIC_3_DIR = os.path.join(DATA_DIR, "mimic3")
