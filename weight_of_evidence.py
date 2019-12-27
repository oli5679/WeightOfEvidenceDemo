
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import scorecardpy as sc
import scipy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Node:
    def __init__(self):
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True


class SingleVariableDecisionTreeClassifier:
    def __init__(self, max_depth=5, min_gini_decrease=1e-4, min_samples_per_leaf=10):
        self.max_depth = max_depth
        self.min_gini_decrease=min_gini_decrease
        self.min_samples_per_leaf = min_samples_per_leaf

    def fit(self, X, y):
        self.splits_ = [-np.inf, np.inf]
        sort_idx = X[X.notnull()].sort_values().index
        
        X_sorted = X[sort_idx].reset_index(drop=True)
        y_sorted = y[sort_idx].reset_index(drop=True)
    
        self.tree_ = self._grow_tree(X_sorted,y_sorted)
        self._unpack_splits(self.tree_)
        self.splits_ = sorted(self.splits_)

    def _gini(self, y_0, y_1):
        y_count = y_0 + y_1
        y_0_freq = y_0 / y_count
        y_1_freq = y_1 / y_count
        return (y_0_freq*y_1_freq) *2
        
    def _best_split(self, X, y):
        y_count = y.size
        if y_count <= 1:
            return None
        y_0_total = np.sum(y == 0)
        y_1_total = np.sum(y == 1)

        y_1_left = (y==1).cumsum() 
        y_0_left = (y==0).cumsum() 
        
        y_1_right = y_1_total- y_1_left
        y_0_right = y_0_total- y_0_left
        
        y_left_count = (y < 2).cumsum()
        y_right_count = y_count - y_left_count
        
        baseline_gini = self._gini(y_0_total, y_1_total)
        gini_left = self._gini(y_0_left, y_1_left)    
        gini_right = self._gini(y_0_right, y_1_right)
        
        
        gini_combined = ((gini_left*y_left_count) + (gini_right*y_right_count))/y_count
        gini_valid = gini_combined[(X!=X.shift()) & 
                                   (y_left_count > self.min_samples_per_leaf) &
                                  (y_right_count > self.min_samples_per_leaf)]
        if gini_valid.shape[0] == 0:
            return None
        min_idx = gini_valid.idxmin()
        min_gini = gini_valid.min()
        
        if baseline_gini - min_gini >self.min_gini_decrease:
            split =  X[min_idx]
            return split
        else:
            return None

    def _grow_tree(self, X, y, depth=0):
        node = Node()
        if depth < self.max_depth:
            thr = self._best_split(X, y)
            if thr is not None:
                indices_left = (X < thr)
               
                
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
                if node.left.threshold is not None and node.right.threshold is not None:
                    node.is_leaf = False
        return node
                 
    def _unpack_splits(self,node):
        """
        Unpacks splits for single feature from self.tree_, and saves to self.splits
        
        Params
        ------
        feature : str
            feature to be unpacked
        """
        if node.left is not None:
            self._unpack_splits(node.left)
        if node.right is not None:
            self._unpack_splits(node.right)
            
        if node.is_leaf and node.threshold is not None:            
            self.splits_.append(
                    node.threshold
                )

class TreeBinner(BaseEstimator, TransformerMixin):
    """
    Autobinning tool - automated feature binning for regression models by fitting single-variable trees
    
    Methods
    -------
        fit
            finds bin-thresholds for columns X given target y 
        transform
            bins X according to found bin-thresholds
        fit_transform
            fit & transform
    """

    def __init__(self):
        self.single_var_decision_tree = SingleVariableDecisionTreeClassifier()

    def fit(self, X, y):
        """
        Find splits for all cols in X which exceed ROC-AUC threshold
        
        Params
        ------
            X : dataframe
                dataframe of numeric columns to be binned
            y : series
                binary target variable used for binning
        
        Returns 
        -------
            self : object
        """
        self.splits_ = {}

        for col in X.select_dtypes(include=["int64", "float64"]).columns:
            self.single_var_decision_tree.fit(X[col],y)
            self.splits_[col] = self.single_var_decision_tree.splits_
        
        return self


    def transform(self, X, y=None):
        """
        Returns data binned by discovered splits
        
        Params
        ------
        X : dataframe
            data to be binned
        
        Returns
        -------
        X_binned : dataframe
            binned dataframe
        """
        _X = X.copy()
        for feature in self.splits_.keys():
            binned_col = pd.cut(_X[feature], self.splits_[feature]).astype('str')
            binned_col[_X[feature].isna()] = "missing"
            _X[feature] = binned_col
        return _X

class WoeScaler(BaseEstimator, TransformerMixin):
    """
    Automatically calculates Quasi-WOE bin values according to pre-specified bin thresholds

    quasi-woe = logit of average bad rate in bin, will make data linear in log-odds of bad rate.

    Methods
    -------
        fit
            finds quasi-woe-values for each bin
        transform
            maps values to logit of bad rate for bin
        fit_transform
            fit & transform
    """
    def __init__(self, clip_thresh= 1e5):
        self.clip_thresh =clip_thresh

    def fit(self, X, y):
        """
        Parameters
        ----------
            X : dataframe
                data to be encoded

            y : target

        Returns
        ------
            self : BaseEstimator
                fitted transformer
        """
        self.woe_values_ = {}
        _data = X.copy()
        _data['target'] = y
        for col in X.select_dtypes(include=["object"]).columns:
            agg = _data.groupby(col)[['target']].mean()
            woe_values = scipy.special.logit(agg["target"])
            clipped_woe_values = np.clip(-self.clip_thresh, woe_values, self.clip_thresh)
            self.woe_values_[col] = clipped_woe_values.to_dict()
        return self

    def transform(self, X, y=None):
        """
        Parameters
        -----------
            X : dataframe
                data to be encoded
        Returns
        -------
            X : dataframe
                encoded data
        """
        _X = X.copy()
        for var in self.woe_values_.keys():
            woe_vals = X[var].map(self.woe_values_[var])
            woe_max = max(self.woe_values_[var].values())
            _X[var] = woe_vals.fillna(woe_max)
        return _X
    


class ScorecardPyEncoder(BaseEstimator, TransformerMixin):
    """
    Automatically filters and woebins data using scorecardpy

    Methods
    -------
        fit
            finds preprocessing strategies for all cols
        transform
            preprocesses all  cols
        fit_transform
            fit & transform
    """

    def __init__(self, filter_flag=False, manual_bins={}):
        self.filter_flag = filter_flag
        self.manual_bins = manual_bins

    def fit(self, X, y):
        """
        Parameters
        ----------
            X : dataframe
                data to be encoded

            y : target

        Returns
        ------
            self : BaseEstimator
                fitted transformer
        """
        data = X.copy()
        data["bad_flag"] = y

        if self.filter_flag:
            filtered_results = sc.var_filter(data, y="bad_flag", return_rm_reason=True)

            self.filter_reasons_ = filtered_results["rm"]
            self.dropped_cols_ = [
                c for c in data.columns if c not in filtered_results["dt"].columns
            ]
            woebin_data = filtered_results["dt"]
        else:
            woebin_data = data

        self.woe_bins_ = sc.woebin(woebin_data, y="bad_flag")

        # manual overwrite, based on business intuitiion
        for k in self.manual_bins:
            self.woe_bins_[k] = self.manual_bins[k]

        return self

    def transform(self, X, y=None):
        """
        Parameters
        -----------
            X : dataframe
                data to be encoded
        Returns
        -------
            X : dataframe
                encoded data
        """
        if self.filter_flag:
            X_woebin = X.copy().drop(columns=self.dropped_cols_)
        else:
            X_woebin = X.copy()

        return sc.woebin_ply(X_woebin, self.woe_bins_)


