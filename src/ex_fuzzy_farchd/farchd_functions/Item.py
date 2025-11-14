#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 13:55:49 2023

@author: asier
"""
import numpy as np
import ex_fuzzy.rules as rl
import ex_fuzzy.fuzzy_sets as fs

class Item:
    def __init__(self, values, support=0, confidence=0, class_j=None):
        self.values = values
        self.support = support
        self.confidence = confidence
        self.class_j = class_j
        self.variables, self.labels = self.get_variables_labels()
        self.wwracc = 0

    def get_variables_labels(self):
        npitem = np.array(self.values)
        return np.swapaxes(npitem, 1, 0)

    def compatible(self, other):
        if len(self.values) == 1:
            return self.values[0][0] != other.values[0][0]
        for a, b in zip(self.values[:-1], other.values[:-1]):
            if a != b:
                return False
        if self.values[-1][0] >= other.values[-1][0]:
            return False
        return True

    def __eq__(self, other):
        # NOTE: Only to check if an item exist in a list
        return self.values == other.values and self.class_j == other.class_j

    def __add__(self, other):
        # NOTE: this addition only adds the last component of the other item
        # they should be compatible
        return Item(self.values + [other.values[-1]], 0, 0, self.class_j)

    def __str__(self):
        s = "IF "
        for variable, label in self.values:
            s += f"{variable=} is {label=} AND "
        s = s[:-4] + f" THEN class is {self.class_j}"
        return s


def items_to_master_rule_base(items: list[Item], classes_dict: dict, partitions):
    rule_dict = {c: [] for c in classes_dict.values()}
    for rule in items:
        rule_simple = to_rule_simple_with_consequent(rule, partitions, classes_dict)
        rule_dict[rule_simple.consequent].append(rule_simple)

    fuzzy_type = partitions[0].fs_type

    if fuzzy_type == fs.FUZZY_SETS.t1:
        rule_base_type = rl.RuleBaseT1
    elif fuzzy_type == fs.FUZZY_SETS.t2:
        rule_base_type = rl.RuleBaseT2
    elif fuzzy_type == fs.FUZZY_SETS.gt2:
        rule_base_type = rl.RuleBaseGT2


    rule_bases = []
    for consequent in rule_dict.keys():
        rule_base = rule_base_type(antecedents=partitions, rules=rule_dict[consequent])
        rule_bases.append(rule_base)
    master_rule_base = rl.MasterRuleBase(rule_base=rule_bases, consequent_names=list(classes_dict.keys()))
    return master_rule_base

def to_rule_simple_with_consequent(item, partitions, classes_dict):
    antecedents = [-1 for _ in partitions]
    for variable, label in item.values:
        antecedents[variable] = label
    consequent = classes_dict[item.class_j]
    score = item.support
    rule_simple = rl.RuleSimple(antecedents,consequent)
    rule_simple.score = score
    return rule_simple

def to_rule_simple(item, partitions):
    antecedents = [-1 for _ in partitions]
    for variable, label in item.values:
        antecedents[variable] = label
    score = item.support
    rule_simple = rl.RuleSimple(antecedents)
    rule_simple.score = score
    return rule_simple