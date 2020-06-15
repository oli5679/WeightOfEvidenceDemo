# Weight of evidence Demo

Simple implementation of 'weight of evidence binning' feature engineering method to improve perfomance of linear models in binary clasificaiton tasks.

## Prerequesites

Download Anaconda [here](https://docs.anaconda.com/anaconda/install/)

## Setup

- install & activateenvironment

    conda env create -f environment.yml
    conda activate weightOfEvidence

- download test data 'German Credit'

    cd weight_of_evidence
    bash get_dataset.sh

- run unit tests

    cd weight_of_evidence
    pytest .

## Examples

- fit sklearn pipeline on dataset

    from sklearn.base import BaseEstimator, TransformerMixin
    import pandas as pd
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    import weight_of_evidence
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

- create plots explaining fitted model

    var_importance = weight_of_evidence.plot_feature_importance(
        X.columns, woebin_logit["log_reg_classifier"].coef_[0], n=5)

 
    top_5 = var_importance.tail(5).var_names[::-1]

    weight_of_evidence.plot_bins(
        X[top_5], y, woebin_logit["tree_bin"].splits_, space="log-odds"
    )