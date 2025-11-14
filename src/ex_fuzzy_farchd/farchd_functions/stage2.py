#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:50:15 2023

@author: asier
"""
import numpy as np
import pandas as pd

# Functions for stage2

def calculate_all_memberships(x, partitions, variables, labels):
    all_memberships = []
    for var_idx, label_idx in zip(variables, labels):
        membership_values = partitions[var_idx].linguistic_variables[label_idx].membership(x[:, var_idx])
        # FIXME: How to deal with IT2, it returns  2 membership values (lower, upper)
        _, t = membership_values.shape
        if t == 2:
            membership_values = np.mean(membership_values,axis=1)

        all_memberships.append(membership_values)

    # Convert to numpy array and calculate product across variables
    all_memberships = np.array(all_memberships).T  # Transpose to get shape (n_samples, n_variables)
    return all_memberships

def wWRAcc_f(rule, x, y, partitions, pattern_weight):
    N = len(y)
    variables, labels = rule.get_variables_labels()
    class_j = rule.class_j

    # Calculate membership values using ex_fuzzy variables
    all_memberships = calculate_all_memberships(x, partitions, variables, labels)
    x_var_lab = np.prod(all_memberships, axis=1)

    prod = (x_var_lab * pattern_weight)
    nA = prod.sum()
    nAC = prod[y == class_j].sum()
    nC = pattern_weight[y == class_j].sum()
    NC = sum(y == class_j)

    # FIXME: Check 0 divisiona and assign -1 to wWRAcc
    if nC == 0 or nA == 0:
        wWRAcc = -1
    else:
        wWRAcc = nAC/nC * (nAC/nA - NC/N)
    # NOTE: From KEEL source
    wWRAcc *= rule.confidence

    return wWRAcc


# Stage 2. Candidate Rule Prescreening
def stage2(rules, x, y, partitions, kt: int):
    selected_rules = []
    classes = np.unique(y)
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    for class_j in classes:
        candidate_rule_set = [rule for rule in rules if rule.class_j == class_j]
        # TODO: Maybe only the patterns of the given class y[y==class_j]...
        # x_j = x[y==class_j]
        # y_j = x[y==class_j]
        pattern_count = np.zeros(y.shape)
        # 7. Set the weight of the patterns as 1
        pattern_weight = np.ones(y.shape)
        # 11. If any pattern has been covered less than kt times and there are
        # more rules in the candidate rule set
        while np.any(pattern_count[y == class_j] < kt) and \
                (len(candidate_rule_set) > 0):
            best_wWRAcc = -1
            best_idx = 0
            # 8. Calculate the wWRAcc value for each rule
            for i, rule in enumerate(candidate_rule_set):
                wWRAcc = wWRAcc_f(rule, x, y, partitions, pattern_weight)
                if wWRAcc > best_wWRAcc:
                    best_wWRAcc = wWRAcc
                    best_idx = i
            # 9. Select the best rule as part of the initial RB for Stage 3 and
            selected_rules.append(candidate_rule_set[best_idx])
            selected_rules[-1].wwracc = best_wWRAcc
            # remove it from the candidate rule set.
            candidate_rule_set.pop(best_idx)
            # Decreaste the weight of the patterns covered by the selected rule
            # FIXME: Repeated code from wWRAcc_f and prune and generateL!
            variables, labels = selected_rules[-1].get_variables_labels()
            x_class = x  # [y == class_j]

            # Calculate membership values using ex_fuzzy variables
            all_memberships = calculate_all_memberships(x_class, partitions, variables, labels)
            ACprod = np.prod(all_memberships, axis=1)
            covered = ACprod > 0

            # FIXME: count is not updated properly
            # pattern_count[np.where(y == class_j)[0][covered]] += 1
            pattern_count = np.where(covered, pattern_count+1, pattern_count)
            # pattern_weight = 1 / (pattern_count + 1)
            pattern_weight = np.where(pattern_count>=kt, 0, 1 / (pattern_count + 1))

    return selected_rules
