import pytest
import weight_of_evidence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

X_SORTED = pd.Series([1, 1, 2, 2, 3, 3, 3, 3, 3, 5, 10, 20,])

X_DF = pd.DataFrame(data=X_SORTED, columns=["company_age"])

Y = pd.Series([0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1,])

X_CAT = [
    "young",
    "medium",
    "young",
    "medium",
    "young",
    "medium",
    "old",
    "medium",
    "old",
    "old",
    "young",
    "young",
]

X_CAT_DF = pd.DataFrame(data=X_CAT, columns=["company_age"])

EXPECTED_GINI_DECREASES = pd.Series(
    [
        np.inf,
        0.061868687,
        0.002777778,
        0.041666667,
        0.111111111,
        0.209920635,
        0.125,
        0.067063492,
        0.027777778,
        0.00462963,
        0.069444444,
        0.031565657,
    ]
)

EXPECTED_LOGIT_VALUES = pd.Series(
    [
        -0.405465108,
        1.098612289,
        -0.405465108,
        1.098612289,
        -0.405465108,
        1.098612289,
        0.693147181,
        1.098612289,
        0.693147181,
        0.693147181,
        -0.405465108,
        -0.405465108,
    ]
)


@pytest.fixture
def single_var_decision_tree():
    return weight_of_evidence.SingleVariableDecisionTreeClassifier(
        min_samples_per_node=1, max_depth=1
    )


@pytest.fixture
def logit_scaler():
    return weight_of_evidence.LogitScaler()


@pytest.fixture
def tree_binner():
    return weight_of_evidence.TreeBinner(min_samples_per_node=1, max_depth=1)


def test_single_variable_decision_tree_classifier_gini(single_var_decision_tree):
    y_1 = np.array([1, 5, 20, 10, 10])
    y_count = np.array([2, 20, 50, 10, 10])

    assert (
        single_var_decision_tree._gini(y_1, y_count)
        == np.array([0.5, 0.375, 0.48, 0, 0])
    ).all()


def test_single_variable_decision_tree_classifier_find_gini_decreases(
    single_var_decision_tree,
):
    gini_decreases, _, _ = single_var_decision_tree._find_gini_decreases(Y)
    assert np.abs((gini_decreases - EXPECTED_GINI_DECREASES)).max() < 1e-6


def test_single_variable_decision_tree_classifier_best_split(single_var_decision_tree):

    assert single_var_decision_tree._best_split(X_SORTED, Y) == 3


def test_single_variable_decision_tree_classifier_grow_tree_unpack_splits(
    single_var_decision_tree,
):
    single_var_decision_tree.splits_ = []
    tree = single_var_decision_tree._grow_tree(X_SORTED, Y)
    single_var_decision_tree._unpack_splits(tree)
    assert single_var_decision_tree.splits_ == [3]


def test_single_variable_decision_tree_classifier_fit(single_var_decision_tree):
    single_var_decision_tree.fit(X_SORTED, Y)
    assert single_var_decision_tree.splits_ == [
        -np.inf,
        3,
        np.inf,
    ]


def test_tree_binner_fit(tree_binner):
    tree_binner.fit(X_DF, Y)
    assert tree_binner.splits_ == {"company_age": [-np.inf, 3, np.inf,]}


def test_tree_binner_transform(tree_binner):
    X_binned = tree_binner.fit_transform(X_DF, Y)
    assert sorted(list(X_binned["company_age"].unique())) == [
        "(-inf, 3.0]",
        "(3.0, inf]",
    ]


def test_logit_scaler_fit(logit_scaler):
    logit_scaler.fit(X_CAT_DF, Y)
    assert logit_scaler.logit_values_["company_age"]["young"] == pytest.approx(
        -0.405465108
    )
    assert logit_scaler.logit_values_["company_age"]["medium"] == pytest.approx(
        1.098612289
    )
    assert logit_scaler.logit_values_["company_age"]["old"] == pytest.approx(
        0.693147181
    )


def test_logit_scaler_transform(logit_scaler):
    X_scaled = logit_scaler.fit_transform(X_CAT_DF, Y)
    assert np.abs((X_scaled.company_age - EXPECTED_LOGIT_VALUES)).max() < 1e-6
