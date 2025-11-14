#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 11:59:29 2023

@author: asier
"""
import unittest
import numpy as np

from src.ex_fuzzy_farchd.farchd_functions.stage2 import wWRAcc_f
from src.ex_fuzzy_farchd.farchd_functions.Item import Item


class wWRAccTestCase(unittest.TestCase):

    def test_wwracc_1(self):
        # Example from FARCHD paper
        # with confindence = 1 to match keel implementation
        x = np.array([[0, 10], [2.5, 4.0], [3.2, 1.0],
                      [9.0, 5.0], [2.5, 10.0]])
        y = np.array(["C1", "C2", "C2", "C2", "C1"])
        pattern_weight = np.array([1.0, 1.0, 0.0, 1.0, 0.5])
        rule = Item([[0, 0], [1, 2]], confidence=1, class_j="C1")
        partitions = np.array([
            [[-5, 0, 5], [0, 5, 10], [5, 10, 15]],
            [[-5, 0, 5], [0, 5, 10], [5, 10, 15]]])
        result = wWRAcc_f(rule, x, y, partitions, pattern_weight)
        np.testing.assert_almost_equal(result, 0.5)


if __name__ == '__main__':
    unittest.main()
