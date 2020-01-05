from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss, roc_auc_score
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

import weight_of_evidence

data = pd.read_csv("~/Downloads/application_train.csv")

EXCLUDE_COLS = [
    "SK_ID_CURR",
    "TARGET",
    "CODE_GENDER",
    "ORGANIZATION_TYPE",
    "NAME_FAMILY_STATUS",
]

CATERORICAL_COLS = data.drop(columns=EXCLUDE_COLS).select_dtypes("O").columns

NUMERIC_COLS = data.drop(columns=EXCLUDE_COLS).select_dtypes("int64").columns

data[CATERORICAL_COLS] = data[CATERORICAL_COLS].fillna("MISSING")

combined_results = pd.DataFrame()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

X = data.drop(columns=EXCLUDE_COLS)
y = data.TARGET


woebin_logit = Pipeline(
    steps=[
        ("tree_bin", weight_of_evidence.TreeBinner()),
        ("woe_scale", weight_of_evidence.LogitScaler()),
        ("standard_scale", StandardScaler()),
        ("log_reg_classifier", LogisticRegression(solver="lbfgs", max_iter=1e6)),
    ]
)

woebin_logit.fit(X, y)
