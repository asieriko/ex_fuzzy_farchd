#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 11:20:26 2023

@author: asier.urio

Implementation of the 2011 paper:
A Fuzzy Association Rule-Based Classification Model for High-Dimensional
Problems with Genetic Rule Selection and Lateral Tuning

From:
    Jesús Alcalá-Fdez,
    Rafael Alcalá, and
    Francisco Herrera, Member, IEEE

IEEE TRANSACTIONS ON FUZZY SYSTEMS

DOI: 10.1109/TFUZZ.2011.2147794
"""
import numpy as np

from src.ex_fuzzy_farchd.common.fuzzy_partitions import (
    generate_fuzzy_partitions_lims,
    )
from src.ex_fuzzy_farchd.farchd_functions.stage3 import stage3
from src.ex_fuzzy_farchd.farchd_functions.stage2 import stage2
from src.ex_fuzzy_farchd.farchd_functions.stage1 import stage1


class FARCHD:
    """
    FARCHD class to implement the FARCHD algorithm for fuzzy rule-based classification.

    A Fuzzy Association Rule-Based Classification Model for High-Dimensional
    Problems with Genetic Rule Selection and Lateral Tuning
    """
    # [ex_fuzzy.fuzzy_sets.fuzzyVaraible]
    def __init__(self, partitions: list=None, classes_dict: dict=None,
                 n_labels: int=5, max_depth: int=3, min_support: float=0.05,
                 maxconf: float=0.8, kt: int=2, pop: int=50,
                 evaluations: int=15, BITSGENE: int=30, delta: float=0.2):

        # Algorithm parameters
        self.n_labels = n_labels
        self.max_depth = max_depth
        self.min_support = min_support
        self.maxconf = maxconf
        self.kt = kt
        self.pop = pop
        self.evaluations = evaluations
        self.BITSGENE = BITSGENE
        self.delta = delta

        self.partitions = partitions
        self.rules = None
        self.classes_dict = classes_dict

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        # FIXME: deep=False for us?? we have no sub-estimators
        return {
            'partitions': self.partitions,
            'classes_dict': self.classes_dict,
            'n_labels': self.n_labels,
            'max_depth': self.max_depth,
            'min_support': self.min_support,
            'maxconf': self.maxconf,
            'kt': self.kt,
            'pop': self.pop,
            'evaluations': self.evaluations,
            'BITSGENE': self.BITSGENE,
            'delta': self.delta
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters:
        -----------
        **params : dict
            Estimator parameters.

        Returns:
        --------
        self : estimator instance
            Estimator instance.
        """
        for parameter, value in params.items():
            if hasattr(self, parameter):
                setattr(self, parameter, value)
            else:
                raise ValueError(f"Invalid parameter {parameter} for estimator {type(self).__name__}")
        return self

    def fit(self, x, y):
        """
        Fit the FARCHD model to the data.

        Parameters:
        x (numpy.ndarray): Input data.
        y (numpy.ndarray): Target labels.

        Returns:
        tuple: A tuple containing the rules and partitions.
        """
        if self.partitions is None:
            # FIXME: self.attribs_lims is not defined, check also categorical variables
            attribs_lims = [(x[:, i].min(), x[:, i].max()) for i in range(x.shape[1])]
            self.partitions = generate_fuzzy_partitions_lims(attribs_lims,
                                                 n_labels=self.n_labels)
        # FIXME: update to ex_fuzzy partition generator, remember that farchd uses triangulars, equally partitioned
        if self.classes_dict is None:
            self.classes_dict = {c: i for i,c in enumerate(np.unique(y))}

        print("Stage 1:", end=" ")
        items = stage1(x, y, self.partitions, self.min_support, self.maxconf, self.max_depth)
        print(len(items), "rules")
        print("Stage 2:", end=" ")
        items = stage2(items, x, y, self.partitions, self.kt)
        print(len(items), "rules")
        print("Stage 3:")
        self.rules = stage3(items, x, y, self.partitions, self.classes_dict, self.pop, self.evaluations, self.BITSGENE, self.delta)
        print(sum([len(x) for x in self.rules]), "rules")
        return self.rules

    def predict(self, x):
        """
        Predict the class labels for the input data using the trained rules.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Predicted class labels.
        """
        if x.ndim == 1:
            return [self.predict_one(np.expand_dims(x, axis=0))]
        elif x.ndim == 2:
            return self.predict_one(x)


    def predict_one(self, x):
        """
        Predict the class labels for the input data using the trained rules.

        Parameters:
        x (numpy.ndarray): Input data.

        Returns:
        numpy.ndarray: Predicted class labels.
        """
        return self.rules.predict(x)
