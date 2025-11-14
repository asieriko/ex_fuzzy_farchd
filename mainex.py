#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 13:14:51 2022

@author: asier.urio
"""
from pathlib import Path, PurePath
import numpy as np
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.utils as utils

from src.ex_fuzzy_farchd.common.dat_loader import dat_loader
from src.ex_fuzzy_farchd.FARCHD import FARCHD



train_dataset_name = "crx_train.dat"
test_dataset_name = "crx_test.dat"

train_dataset_name = "iris-10-1tra.dat"
test_dataset_name = "iris-10-1tst.dat"

w_dir = Path(__file__).resolve().parent
data_path = PurePath(w_dir, 'datasets', train_dataset_name)
data_info = dat_loader(data_path)
X = data_info.get_x_dataframe()
y = data_info.get_y_series()
cat_mask = data_info.get_categorical_mask()
attributes = list(X.columns)
# attribs_lims = [(X[:, i].min(), X[:, i].max()) for i in range(X.shape[1])]
# fuzzy_variables_tri = utils.construct_partitions(attribs_lims, fs.FUZZY_SETS.t1, categorical_mask = cat_mask,
#                                                  n_partitions=3, shape="triangular")



# Create linguistic variables from the data
fuzzy_variables = utils.construct_partitions(X, fs.FUZZY_SETS.t2, categorical_mask = cat_mask, n_partitions=3)
classes_dict = {c: i for i,c  in enumerate(np.unique(y))}

# Train
print("Train")
farchd = FARCHD(partitions=fuzzy_variables, classes_dict=classes_dict,
                n_labels=5, max_depth=3, min_support=0.05, maxconf=0.8,
                kt=2, pop=50, evaluations=15, BITSGENE=30, delta=0.2)
rules = farchd.fit(X, y)

print("Discovered rule base with rules for each class")
rules.print_rules()

# Test
print("Test")

data_path = PurePath(w_dir, 'datasets', test_dataset_name)
data_info = dat_loader(data_path)
X_test = data_info.get_x_dataframe()
y_test = data_info.get_y_series()

y_predicted = farchd.predict(X_test)

accuracy = sum(y_predicted == y_test.map(classes_dict))/len(y_test)
print(f"Test Accuracy class = {accuracy*100:.2f}")
