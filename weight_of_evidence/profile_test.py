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
import seaborn as sns
import matplotlib.pyplot as plt

import weight_of_evidence

"""
Code profile script, for tuning purposes
"""


DATA_PATH = "data/german_credit_data.csv"

COLUMN_NAMES = [
    "chk_acct",
    "duration",
    "credit_his",
    "purpose",
    "amount",
    "saving_acct",
    "present_emp",
    "installment_rate",
    "sex",
    "other_debtor",
    "present_resid",
    "property",
    "age",
    "other_install",
    "housing",
    "n_credits",
    "job",
    "n_people",
    "telephone",
    "foreign",
    "response",
]

data = pd.read_csv(DATA_PATH, sep=" ", names=COLUMN_NAMES)

data["response"] = data["response"] - 1

EXCLUDE_COLS = [
    "response",
]

X = data.drop(columns=EXCLUDE_COLS)
y = data["response"]


woebin_logit = Pipeline(
    steps=[
        ("tree_bin", weight_of_evidence.TreeBinner()),
        ("woe_scale", weight_of_evidence.LogitScaler()),
        ("standard_scale", StandardScaler()),
        ("log_reg_classifier", LogisticRegression(solver="lbfgs", max_iter=1e6)),
    ]
)

woebin_logit.fit(X, y)
