#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:48:43 2023

@author: asier


Stage 1. Fuzzy Association Rule Extraction for Classification



Search tree -> list all possible fuzzy item set of a class
Root (level 0) -> empty set
Attributes ordered by appearance in training data
one-item sets corresponding to attributes are listed in the first level
(an item for each linguistic term)
Children -> tow-item that include parent
(if an attribute has j > 2 outcomes -> replaced by j binary variables)


frequent item set = support > minimum
if support < minimum -> no extend
if confidence > maximum -> reached quality -> no extend

Support = sum memberships in node /all elements¿?
Confidence = sum memberships in node / sum memberships ¿?


from frequent sets -> rules -> items sets antecedent and class consequent
(repeat for each class)

minimumSupport_Class_i = minSup * fci (fci pattern ratio of the class i)

Depth limited: Dmax

"""


import numpy as np
import pandas as pd
from itertools import combinations
from src.ex_fuzzy_farchd.farchd_functions.Item import Item

# Functions for stage1
# [ex_fuzzy.fuzzySets.fuzzyVaraible]
def generateL1(x, y, class_j, partitions: list, min_support_cj: float, maxconf: float):
    L1 = []
    Rules = []
    num_variables = len(partitions)
    N = len(y)

    class_data = x[y == class_j]

    # 2. Create the levels 0 and 1 of the tree
    for i in range(num_variables):
        num_labels = len(partitions[i].linguistic_variables)
        for j in range(num_labels):
            # Get membership values for class data using the specific linguistic variable
            class_memberships = partitions[i].linguistic_variables[j].membership(class_data[:, i])

            # Calculate support for this specific linguistic term
            support = class_memberships.sum() / N

            if support > min_support_cj:
                # Get membership values for all data using the same linguistic variable
                all_memberships = partitions[i].linguistic_variables[j].membership(x[:, i])

                # Calculate confidence
                confidence = support * N / all_memberships.sum()
                new_item = Item([[i, j]], support, confidence, class_j)

                # From Keel implementation
                if confidence <= maxconf:
                    L1.append(new_item)
                if confidence > 0.4:
                    Rules.append(new_item)

    return L1, Rules


def apriori_gen_old(L, maxconf):
    Li = []
    for i in range(len(L)-1):
        itemi = L[i].values
        if L[i].confidence > maxconf:
            continue
        for j in range(1, len(L)):
            itemj = L[j].values
            # TODO: Check maxconf¿?
            compatible = True
            for a, b in zip(itemi[:-1], itemj[:-1]):
                if a != b:
                    compatible = False
            # FIXME: Check if subsets exists in previous level before add
            if compatible and itemi[-1][0] < itemj[-1][0]:
                new_item = Item(itemi+[itemj[-1]], class_j=L[i].class_j)
                Li.append(new_item)
    return Li


def apriori_gen(L, maxconf):
    Li = []
    pairs = combinations(L, 2)
    for itemi, itemj in pairs:
        if itemi.confidence <= maxconf and \
            itemj.confidence <= maxconf and \
                itemi.compatible(itemj):
            new_item = itemi + itemj
            Li.append(new_item)
    return Li


def generateLi(Lclass_i, x, y, class_i, partitions, min_support_cj, maxconf=0.85):
    # FIXME:Not used
    # 3. Create a new level in the tree
    Li = apriori_gen(Lclass_i[-1], maxconf)
    # 4. Prune nodes
    Li = prune(Li, x, y, class_i, partitions, min_support_cj, maxconf=0.85)

    # 6. Generate the rules with class c_i on the right-hand side
    return Li


def ItemSubsets(item):
    values = item.values
    Nrep = len(values) - 1
    i = 0
    while i < Nrep:
        x = values[0:Nrep-i-1] + values[Nrep-i:]
        i += 1
        yield x


def prune(L, L_1):
    # Remove items if any of their L-1 subsets is not in L_1
    # TODO: I think it doesn't change the result. Maybe it avoids some
    # computations evaluating memberships
    for i in range(len(L)-1, 0-1):
        for subset in ItemSubsets(L[i]):
            if Item(subset, class_j=L[i].class_j) not in L_1:
                del L[i]
                break
    return L


def compute_support(L, x, y, class_i, partitions, min_support_cj, maxconf=0.85):
    # Count the support, outside prune?
    N = len(y)
    Li = []
    Rules = []
    for item in L:
        variables, labels = item.get_variables_labels()

        # Calculate membership values for all data
        all_memberships = []
        for var_idx, label_idx in zip(variables, labels):
            membership_values = partitions[var_idx].linguistic_variables[label_idx].membership(x[:, var_idx])
            all_memberships.append(membership_values)

        # Convert to numpy array and calculate product across variables
        all_memberships = np.array(all_memberships).T  # Transpose to get shape (n_samples, n_variables)
        conf_divisor = np.prod(all_memberships, axis=1).sum()

        # Calculate membership values for class data
        class_data = x[y == class_i]
        class_memberships = []
        for var_idx, label_idx in zip(variables, labels):
            membership_values = partitions[var_idx].linguistic_variables[label_idx].membership(class_data[:, var_idx])
            class_memberships.append(membership_values)

        # Convert to numpy array and calculate product across variables
        class_memberships = np.array(class_memberships).T  # Transpose to get shape (n_class_samples, n_variables)
        support = np.prod(class_memberships, axis=1).sum() / N

        if conf_divisor == 0:
            confidence = 0
        else:
            confidence = support * N / conf_divisor

        item.support = support
        item.confidence = confidence
        if support > min_support_cj:
            # Li.append(item)
            # From keel
            if confidence <= maxconf:
                Li.append(item)
            if confidence > 0.4:
                Rules.append(item)
    return Li, Rules

# [ex_fuzzy.fuzzy_sets.fuzzyVaraible]
def stage1(x, y, partitions: list, min_support: float, maxconf: float, max_depth: int):
    depth_max = max_depth
    classes = np.unique(y)
    N = len(y)
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()


    Rules = []
    # Fuzzy Association Rule Extraction For Classification
    # Foreach class
    for class_i in classes:
        Lclass_i = []
        N_class = len(x[y == class_i])
        min_support_cj = min_support * N_class/N
        # 2. Create the levels 0 and 1 of the tree
        L1, Rules_i = generateL1(x, y, class_i, partitions,
                                 min_support_cj, maxconf)
        Rules += Rules_i

        Lclass_i.append(L1)
        # 5. If there are more than two nodes in the new level, and the depth
        # of the tree is less than dept_max -> goto step 3
        for _ in range(depth_max - 1):
            if len(Lclass_i[-1]) < 2:
                break
            # 3. Create a new level in the tree
            Li = apriori_gen(Lclass_i[-1], maxconf)
            # 4. Prune nodes
            Li = prune(Li, Lclass_i[-1])
            Li, Rules_i = compute_support(Li, x, y, class_i, partitions,
                                          min_support_cj, maxconf)
            # 6. Generate the rules with class c_i on the right-hand side
            Rules += Rules_i
            Lclass_i.append(Li)

    return Rules
