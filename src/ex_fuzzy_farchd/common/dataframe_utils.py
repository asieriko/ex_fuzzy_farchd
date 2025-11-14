#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for working with pandas DataFrames in the context of fuzzy systems.
"""

import pandas as pd
import numpy as np


def get_categorical_mask_from_dataframe(df):
    """
    Generate a categorical mask from a DataFrame based on column dtypes.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze

    Returns:
    --------
    list of bool
        Boolean list where True indicates the column is categorical.
        Length matches the number of columns in the DataFrame.

    Notes:
    ------
    This function considers the following dtypes as categorical:
    - 'object' (string columns)
    - 'category' (pandas categorical columns)
    - 'bool' (boolean columns)

    Numeric dtypes (int, float) are considered continuous.
    """
    mask = []

    for column in df.columns:
        dtype = df[column].dtype

        # Check if the column is categorical
        is_categorical = (
            dtype == 'object' or           # String columns
            dtype.name == 'category' or    # Pandas categorical
            dtype == 'bool'                # Boolean columns
        )

        mask.append(is_categorical)

    return mask


def get_categorical_mask_smart(df, max_unique_ratio=0.1, max_unique_count=10):
    """
    Generate a categorical mask using both dtype and heuristics.

    This function is more intelligent and can detect categorical variables
    that are stored as numeric types but have few unique values.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    max_unique_ratio : float, default=0.1
        Maximum ratio of unique values to total values to consider categorical
    max_unique_count : int, default=10
        Maximum number of unique values to consider categorical (for small datasets)

    Returns:
    --------
    list of bool
        Boolean list where True indicates the column is categorical.
    """
    mask = []

    for column in df.columns:
        dtype = df[column].dtype

        # First check obvious categorical types
        if dtype == 'object' or dtype.name == 'category' or dtype == 'bool':
            mask.append(True)
            continue

        # For numeric columns, use heuristics
        if pd.api.types.is_numeric_dtype(dtype):
            unique_count = df[column].nunique()
            total_count = len(df[column])
            unique_ratio = unique_count / total_count if total_count > 0 else 0

            # Consider categorical if:
            # 1. Few unique values relative to total, OR
            # 2. Very few unique values in absolute terms
            is_categorical = (
                unique_ratio <= max_unique_ratio or
                unique_count <= max_unique_count
            )

            mask.append(is_categorical)
        else:
            # For other dtypes, default to categorical
            mask.append(True)

    return mask


def print_column_analysis(df):
    """
    Print an analysis of DataFrame columns for categorical detection.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to analyze
    """
    print("Column Analysis:")
    print("-" * 60)
    print(f"{'Column':<20} {'Dtype':<15} {'Unique':<8} {'Ratio':<8} {'Categorical'}")
    print("-" * 60)

    basic_mask = get_categorical_mask_from_dataframe(df)
    smart_mask = get_categorical_mask_smart(df)

    for i, column in enumerate(df.columns):
        dtype = df[column].dtype
        unique_count = df[column].nunique()
        total_count = len(df[column])
        unique_ratio = unique_count / total_count if total_count > 0 else 0

        basic_cat = "Yes" if basic_mask[i] else "No"
        smart_cat = "Yes" if smart_mask[i] else "No"

        print(f"{column:<20} {str(dtype):<15} {unique_count:<8} {unique_ratio:<8.3f} {basic_cat}/{smart_cat}")
