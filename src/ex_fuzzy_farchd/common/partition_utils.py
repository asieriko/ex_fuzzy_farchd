#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom partition generation utilities for FARCHD that handle categorical variables properly.
"""

import numpy as np
import pandas as pd
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.utils as utils


def construct_partitions_minmax(X, fuzzy_type=fs.FUZZY_SETS.t1, categorical_mask=None, n_partitions=3, shape="triangular"):
    """
    Construct fuzzy partitions based on min/max limits instead of quantiles.

    For continuous variables: Uses min/max values to create partitions
    For categorical variables: Uses all unique categories found in the data

    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        The input data
    fuzzy_type : ex_fuzzy.FUZZY_SETS
        Type of fuzzy sets to create
    categorical_mask : list of bool, optional
        Boolean mask indicating which columns are categorical
    n_partitions : int, default=3
        Number of partitions for continuous variables
    shape : str, default="triangular"
        Shape of membership functions for continuous variables

    Returns:
    --------
    list
        List of fuzzy variables (partitions)
    """
    if isinstance(X, pd.DataFrame):
        # Convert DataFrame to numpy array but keep column info
        columns = X.columns.tolist()
        X_array = X.values
    else:
        X_array = X
        columns = [f"attr_{i}" for i in range(X_array.shape[1])]

    if categorical_mask is None:
        # If no mask provided, assume all numeric columns are continuous
        if isinstance(X, pd.DataFrame):
            from .dataframe_utils import get_categorical_mask_smart
            categorical_mask = get_categorical_mask_smart(X)
        else:
            categorical_mask = [False] * X_array.shape[1]

    # Calculate limits for continuous variables
    attribs_lims = []
    categorical_values = []

    for i in range(X_array.shape[1]):
        if categorical_mask[i]:
            # For categorical variables, get unique values
            unique_vals = np.unique(X_array[:, i])
            categorical_values.append(unique_vals.tolist())
            attribs_lims.append((0, len(unique_vals) - 1))  # Dummy limits for categorical
        else:
            # For continuous variables, use min/max
            col_min = float(np.min(X_array[:, i]))
            col_max = float(np.max(X_array[:, i]))
            # Add small margin to avoid boundary issues
            margin = (col_max - col_min) * 0.01
            attribs_lims.append((col_min - margin, col_max + margin))
            categorical_values.append(None)

    # Create synthetic dataset with min/max values for continuous variables
    synthetic_data = create_synthetic_minmax_data(X_array, categorical_mask, attribs_lims, categorical_values)

    # Use the standard construct_partitions function with synthetic data
    if isinstance(X, pd.DataFrame):
        synthetic_df = pd.DataFrame(synthetic_data, columns=columns)
        partitions = utils.construct_partitions(synthetic_df, fuzzy_type,
                                               categorical_mask=categorical_mask,
                                               n_partitions=n_partitions)
    else:
        partitions = utils.construct_partitions(synthetic_data, fuzzy_type,
                                               categorical_mask=categorical_mask,
                                               n_partitions=n_partitions)

    return partitions


def create_synthetic_minmax_data(X_array, categorical_mask, attribs_lims, categorical_values):
    """
    Create synthetic data that represents the min/max bounds for continuous variables
    and includes all categorical values for categorical variables.

    Parameters:
    -----------
    X_array : numpy.ndarray
        Original data array
    categorical_mask : list of bool
        Boolean mask for categorical variables
    attribs_lims : list of tuples
        Min/max limits for each attribute
    categorical_values : list
        List of unique categorical values for each categorical attribute

    Returns:
    --------
    numpy.ndarray
        Synthetic dataset with appropriate boundary values
    """
    n_features = X_array.shape[1]

    # Calculate how many rows we need
    continuous_rows = 2  # min and max for continuous variables
    categorical_rows = max([len(vals) if vals is not None else 0
                           for vals in categorical_values] + [0])

    # We need enough rows to represent both continuous bounds and all categorical values
    n_rows = max(continuous_rows, categorical_rows)
    if categorical_rows > 0:
        n_rows = max(n_rows, categorical_rows)

    synthetic_data = np.zeros((n_rows, n_features))

    for i in range(n_features):
        if categorical_mask[i]:
            # For categorical variables, cycle through all unique values
            unique_vals = categorical_values[i]
            for j in range(n_rows):
                synthetic_data[j, i] = unique_vals[j % len(unique_vals)]
        else:
            # For continuous variables, use min/max pattern
            min_val, max_val = attribs_lims[i]
            for j in range(n_rows):
                if j % 2 == 0:
                    synthetic_data[j, i] = min_val
                else:
                    synthetic_data[j, i] = max_val

    return synthetic_data


def construct_partitions_explicit_limits(attribs_lims, fuzzy_type=fs.FUZZY_SETS.t1, categorical_mask=None,
                                        categorical_values=None, n_partitions=3, shape="triangular"):
    """
    Construct fuzzy partitions from explicit attribute limits and categorical values.

    This is useful when you already know the limits and don't want to derive them from data.

    Parameters:
    -----------
    attribs_lims : list of tuples
        List of (min, max) tuples for each attribute
    fuzzy_type : ex_fuzzy.FUZZY_SETS
        Type of fuzzy sets to create
    categorical_mask : list of bool, optional
        Boolean mask indicating which attributes are categorical
    categorical_values : list of lists, optional
        List of possible values for each categorical attribute
    n_partitions : int, default=3
        Number of partitions for continuous variables
    shape : str, default="triangular"
        Shape of membership functions

    Returns:
    --------
    list
        List of fuzzy variables (partitions)
    """
    n_features = len(attribs_lims)

    if categorical_mask is None:
        categorical_mask = [False] * n_features

    if categorical_values is None:
        categorical_values = [None] * n_features

    # Create synthetic data
    synthetic_data = create_synthetic_from_limits(attribs_lims, categorical_mask, categorical_values)

    # Use standard construct_partitions function
    partitions = utils.construct_partitions(synthetic_data, fuzzy_type,
                                           categorical_mask=categorical_mask,
                                           n_partitions=n_partitions)

    return partitions


def create_synthetic_from_limits(attribs_lims, categorical_mask, categorical_values):
    """
    Create synthetic data from explicit limits and categorical values.
    """
    n_features = len(attribs_lims)

    # Determine number of rows needed
    max_categorical_size = 0
    for i, is_cat in enumerate(categorical_mask):
        if is_cat and categorical_values[i] is not None:
            max_categorical_size = max(max_categorical_size, len(categorical_values[i]))

    n_rows = max(2, max_categorical_size)  # At least 2 rows for min/max

    synthetic_data = np.zeros((n_rows, n_features))

    for i in range(n_features):
        if categorical_mask[i] and categorical_values[i] is not None:
            # For categorical variables
            cat_vals = categorical_values[i]
            for j in range(n_rows):
                synthetic_data[j, i] = cat_vals[j % len(cat_vals)]
        else:
            # For continuous variables
            min_val, max_val = attribs_lims[i]
            for j in range(n_rows):
                synthetic_data[j, i] = min_val if j % 2 == 0 else max_val

    return synthetic_data
