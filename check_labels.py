import glob
import os

import pandas as pd

from us_visa.constant import CURRENT_YEAR
from us_visa.pipeline.prediction_pipeline import USvisaClassifier

# Use the newest artifact folder that has a test.csv
paths = glob.glob(os.path.join("artifact", "*", "data_ingestion", "ingested", "test.csv"))
if not paths:
    raise SystemExit("No test.csv found in artifact/. Run demo.py first.")
test_path = max(paths, key=os.path.getmtime)
print("Using:", test_path)

df = pd.read_csv(test_path)
df["company_age"] = CURRENT_YEAR - df["yr_of_estab"]

x = df.drop(columns=["case_status"])
preds = USvisaClassifier().predict(x)

# Model output: 0 = Certified (approved), 1 = Denied
print()
print(pd.crosstab(df["case_status"], preds, rownames=["real"], colnames=["model"]))

correct = ((df["case_status"] == "Denied") == (preds == 1)).mean()
print()
print(f"accuracy: {correct:.3f}")