from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import scorecardpy as sc
import scipy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


class Node:
    """
    Decision tree Node

    Attributes:
        threshold (numeric): split chosen to maximise Gini
            NOTE - None if cannot find any that satisfies min_gini_decrease & min_samples_per_node
        left (Node): Decision tree node for values < threshold, None if this node is child node
        right (Node): Decision tree node for values >= threshold, None if this node is child node
        is_leaf (Bool): Is this node a leaf node?
        
    """

    def __init__(self):
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True


class SingleVariableDecisionTreeClassifier:
    """
    Single variable Decision tree classifier

    NOTE - only works for binary decision trees

    Attributes:
        max_depth (int): Maximum number of layers to tree
        min_gini_decrease (numeric): Minimum decrease in Gini for split to be accepted
        min_samples_per_node (numeric): Minimum samples in left and right, for split to be accepted

    Methods:
        fit: finds best splits
    """

    def __init__(self, max_depth=5, min_gini_decrease=1e-4, min_samples_per_node=10):
        self.max_depth = max_depth
        self.min_gini_decrease = min_gini_decrease
        self.min_samples_per_node = min_samples_per_node

    def _gini(self, y_1, y_count):
        """
        Finds Gini Impurity values for series of y values

        Formula = 1 - p^2 - (1-p)^2

        Args:
            y_1 (series): Number of observations with y == 1
            y_count (series): Total Numer of observations

        Returns:
            gini_impurity (series): Gini Impurity values for series
        """
        p = y_1 / y_count
        return 1.0 - (p ** 2) - ((1.0 - p) ** 2)

    def _find_gini_decreases(self, y):
        """
        Finds how much avg. Gini impurity would decrease if split into left & right:
            left = before point
            right = including & after point
            decrease = y Gini - population weighted avg. Gini of left & right
        
        Args:
            y (series): series to calc. Gini decreases

        Returns:
            gini_decreases (series): decrease in Gini from splitting before/after every point
            y_count_left (series): count of cumul datapoints. before each point
            y_count_right (series): count of cumul datapoints. including & after each point
        """
        y_count_total = y.size
        y_1_total = np.sum(y == 1)

        # y_1_left is sum of cumulative 1s BEFORE point
        y_1_left = (y == 1).cumsum() - y
        # y_1_right is sum of cumulative 1s INCLUDING & AFTER point
        y_1_right = y_1_total - y_1_left

        y_count_left = (y < 2).cumsum() - 1
        y_count_right = y_count_total - y_count_left

        baseline_gini = self._gini(y_1_total, y_count_total)
        gini_left = self._gini(y_1_left, y_count_left)
        gini_right = self._gini(y_1_right, y_count_right)

        # compare unsplit Gini with weighted average of Gini of left & right node
        gini_decreases = baseline_gini - (
            ((gini_left * y_count_left) + (gini_right * y_count_right)) / y_count_total
        )

        return (gini_decreases, y_count_left, y_count_right)

    def _best_split(self, X, y):
        """
        Finds split for X that has lowest average impurity of the two children, weighted by 
        population.

        Args:
            X (series): variable to be split
            y (series): target

        Returns:
            split (numeric): variable to be split

        NOTE - X & y must be sorted by X values

        NOTE - returns None if no split found that satisfies min_gini_decrease & min_samples_per_node
        """
        gini_decreases, y_count_left, y_count_right = self._find_gini_decreases(y)

        #  only consider candidate splits where:
        #    (a) X value has changed
        #    (b) At least min_samples_per_node in both left & right splits
        #    (c) Gini decrease of at least min_gini_decrease

        gini_valid = gini_decreases[
            (X != X.shift())
            & (y_count_left >= self.min_samples_per_node)
            & (y_count_right >= self.min_samples_per_node)
            & (gini_decreases >= self.min_gini_decrease)
        ]

        # If no candidate splits satisfy a-c, return None
        if gini_valid.shape[0] == 0:
            return None

        # Else return best split
        else:
            return X[gini_valid.idxmax()]

    def _grow_tree(self, X, y, depth=0):
        """
        Greedily creates tree with largest impurity decrease best splits that satisfy:
            min_gini_decrease
            min_samples_per_node
            max_depth

        Args:
            X (series): variable to be split
            y (series): target to split on

        Returns:
            Node object recording split threshold, and left & right children with possible further splits
        """
        node = Node()
        if depth < self.max_depth:
            split_threshold = self._best_split(X, y)
            if split_threshold is not None:
                indices_left = X < split_threshold

                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.threshold = split_threshold
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
                if node.left.threshold is not None and node.right.threshold is not None:
                    node.is_leaf = False
        return node

    def _unpack_splits(self, node):
        """
        Recursively unpacks splits from node, and appends leaf nodes' thresholds to self.splits_

        Args:
            node (Node): node to be unpacked
        """
        if node.left is not None:
            self._unpack_splits(node.left)
        if node.right is not None:
            self._unpack_splits(node.right)

        if node.is_leaf and node.threshold is not None:
            self.splits_.append(node.threshold)

    def fit(self, X, y):
        """
        Finds best splits for X, given target y

        Args:
            X (Series): Varaible to find splits
            y (Series): Binary target

        saves splits to self.splits_

        NOTE - only works for numeric variables
        """
        self.splits_ = [-np.inf, np.inf]

        # Sort X & y by X values
        sort_idx = X[X.notnull()].sort_values().index
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        # fit tree & unpack splits
        self.tree_ = self._grow_tree(X_sorted, y_sorted)
        self._unpack_splits(self.tree_)
        self.splits_ = sorted(self.splits_)


class TreeBinner(BaseEstimator, TransformerMixin):
    """
    Autobinning tool - automated tree binning by fitting single variable decision trees

    Attributes:
        max_depth (int): Maximum number of layers to tree
        min_gini_decrease (numeric): Minimum decrease in Gini for split to be accepted
        min_samples_per_node (numeric): Minimum samples in left and right, for split to be accepted

    Methods:
        fit: finds bin-thresholds for columns X given target y 
        transform: bins X according to found bin-thresholds
        fit_transform: fit & transform
    """

    def __init__(self, max_depth=5, min_gini_decrease=1e-4, min_samples_per_node=10):
        self.single_var_decision_tree = SingleVariableDecisionTreeClassifier(
            max_depth=max_depth,
            min_gini_decrease=min_gini_decrease,
            min_samples_per_node=min_samples_per_node,
        )

    def fit(self, X, y):
        """
        Finds bin-thresholds for numeric columns in X given target y 
        
        Args
            X (Dataframe): columns to be binned (if numeric)
            y (Series): binary target variable used for binning
        
        Returns:
            self (BaseEstimator): fitted Treebinner
        """
        self.splits_ = {}

        for col in X.select_dtypes(include=["int64", "float64"]).columns:
            self.single_var_decision_tree.fit(X[col], y)
            self.splits_[col] = self.single_var_decision_tree.splits_

        return self

    def transform(self, X, y=None):
        """
        Returns data binned by discovered splits
        
        Args: 
            X (Dataframe):
                data to be binned
        
        Returns:
            X_binned (Dataframe) binned dataframe
        """
        _X = X.copy()
        for feature in self.splits_.keys():
            breaks = self.splits_[feature]
            labels = pd.IntervalIndex.from_breaks(breaks).astype(str)
            _X[feature] = pd.cut(X[feature], breaks, labels=labels).astype("str")
        _X[X.isna()] = "missing"
        return _X


class LogitScaler(BaseEstimator, TransformerMixin):
    """
    Calculates Logit values for all categorical variables

    natural log of odds of target  rate in bin

    Clips to be between max & min, to avoid infinity issues.

    NOTE - leaves numeric columns unchanged

    NOTE - does not scale by population bad-rate

    Methods:
        fit: finds quasi-woe-values for each category, for each categorical varaible
        transform: maps values to fitted logit values of bad rate for each category 
        fit_transform: fit & transform
    """

    def __init__(self, clip_thresh=1e5):
        self.clip_thresh = clip_thresh

    def fit(self, X, y):
        """
        Args:
            X (dataframe): data to be Woe scaled

            y (series): target

        Returns:
            self (BaseEstimator): fitted transformer
        """
        self.logit_values_ = {}
        _data = X.copy()
        _data["target"] = y
        for col in X.select_dtypes(include=["object"]).columns:
            agg = _data.groupby(col)[["target"]].mean()
            clipped_logit_values = np.clip(
                -self.clip_thresh, scipy.special.logit(agg["target"]), self.clip_thresh
            )
            self.logit_values_[col] = clipped_logit_values.to_dict()
        return self

    def transform(self, X, y=None):
        """
        Args
        X (dataframe) : data to be logit-scaled

        NOTE - unseen categories are conservatively scaled with maximum WOE value
        
        Returns
        X (Dataframe) logit-scaed data
        """
        _X = X.copy()
        for var in self.logit_values_.keys():
            logit_val = X[var].map(self.logit_values_[var])
            # Fill unseen values with highest logit == most risky
            logit_max = max(self.logit_values_[var].values())
            logit_val[logit_val.isna()] = logit_max
            _X[var] = logit_val

        return _X


def plot_bins(X, y, splits, space="%"):
    """
    Plots target rates & counts for bins of splits

    Numeric column, split by 'splits' thresholds

    Categorical column, split by category 

    Args:
        X (dataframe): columns to be plotted
        y (target): target series
        splits (dictionary): splits to be applied to X
        space (string): space to plot target rates in
            NOTE - must be '%' or 'log-odds'
    """
    assert space in ["%", "log-odds"]
    data = X.copy()
    data["target"] = y
    for col in X.columns:
        if data[col].dtype == "O":
            data[f"{col}_binned"] = data[col]
        else:
            data[f"{col}_binned"] = pd.cut(data[col], bins=splits[col]).astype("str")

        data["obs_count"] = 1
        agg = data.groupby(f"{col}_binned").agg({"target": "mean", "obs_count": "sum"})
        # hack - add bin for missings, preserving order
        agg["sorter"] = agg.index
        agg["sorter"] = agg.sorter.apply(lambda x: x.split(", ")[0].replace("(", ""))
        agg["sorter"] = pd.to_numeric(agg.sorter, errors="coerce")
        agg = agg.sort_values(by="sorter")

        agg["target rate %"] = agg.target * 100
        agg["target rate log-odds"] = scipy.special.logit(agg.target)

        ax = agg["obs_count"].plot.bar(alpha=0.5, color="grey")
        ax.legend(["obs count"])
        plt.ylabel("obs count")
        plt.xlabel("bin group")

        ax2 = agg[f"target rate {space}"].plot.line(secondary_y=True, ax=ax)
        ax2.legend([f"target rate {space}"])
        plt.ylabel(f"target rate {space}")
        plt.title(f"Target rate vs. binned {col} \n")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.show(
