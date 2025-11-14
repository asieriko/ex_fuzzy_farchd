#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 09:44:25 2022

@author: asier.urio
"""

import unittest
import numpy as np

from src.ex_fuzzy_farchd.common.fuzzy_partitions import (
    generate_fuzzy_partitions_array,
    generate_fuzzy_partitions_lims
    )
from src.ex_fuzzy_farchd.common.fuzzy_partitions import label_for_data_array
from src.ex_fuzzy_farchd.common.fuzzy_partitions import membership_for_data_array


class MembershipTestCase(unittest.TestCase):

    def test_generate_fuzzy_partitions_array0(self):
        data = np.array([[1., 5.], [3., 2.], [2., 4.], [2., 8.]])
        partitions = generate_fuzzy_partitions_array(data, n_labels=3)
        expected_partitions = np.array([[[0.,  1.,  2.],
                                         [1.,  2.,  3.],
                                         [2.,  3.,  4.]],
                                        [[-1.,  2.,  5.],
                                         [2.,  5.,  8.],
                                         [5.,  8., 11.]]])
        np.testing.assert_equal(partitions, expected_partitions)

    def test_generate_fuzzy_partitions_array(self):
        # Iris data
        data = np.array([[4.3, 2.0], [7.9, 4.4]])
        partitions = generate_fuzzy_partitions_array(data, n_labels=3)
        expected_partitions = np.array([[[2.5, 4.3, 6.1],
                                         [4.3, 6.1, 7.9],
                                         [6.1, 7.9, 9.7]],
                                        [[0.8, 2.0, 3.2],
                                         [2.0, 3.2, 4.4],
                                         [3.2, 4.4, 5.6]]])
        np.testing.assert_almost_equal(partitions, expected_partitions)

    def test_generate_fuzzy_partitions_lims(self):
        # Iris data
        attrib_lims = np.array([[2.0, 4.4], [1.0, 6.9]])
        partitions = generate_fuzzy_partitions_lims(attrib_lims, n_labels=3)
        expected_partitions = np.array([[[0.8, 2.0, 3.2],
                                         [2.0, 3.2, 4.4],
                                         [3.2, 4.4, 5.6]],
                                        [[-1.95, 1.0, 3.95],
                                         [1.0, 3.95, 6.9],
                                         [3.95, 6.9, 9.85]]])
        np.testing.assert_almost_equal(partitions, expected_partitions)

    def test_label_for_data(self):
        # Iris data
        data = np.array([[5.1, 3.5]])
        partitions = np.array([[[2.5, 4.3, 6.1],
                                [4.3, 6.1, 7.9],
                                [6.1, 7.9, 9.7]],
                               [[0.8, 2.0, 3.2],
                                [2.0, 3.2, 4.4],
                                [3.2, 4.4, 5.6]]])
        expected_labels = np.array([[0, 1]])
        labels = label_for_data_array(data, partitions)
        np.testing.assert_equal(labels, expected_labels)

    def test_membership_for_data_array(self):
        # Iris data
        data = np.array([[7.9,2.0]])
        labels = np.array([2,0])
        partitions = np.array([[[2.5,  4.3,  6.1],
                             [4.3,  6.1,  7.9],
                             [6.1,  7.9,  9.7]],
                            [[0.8,  2.0,  3.2],
                             [2.0,  3.2,  4.4],
                             [3.2,  4.4, 5.6]]])
        expected_membership =np.array([[1,1]])
        memberships = membership_for_data_array(data, labels, partitions)
        np.testing.assert_equal(memberships, expected_membership)

    def test_membership_for_data(self):
        # Iris data
        data = np.array([[3.3]])
        labels = np.array([0])
        partitions = np.array([[[2.5,  4.3,  6.1]]])
        expected_membership =np.array([[0.4444444]])
        memberships = membership_for_data_array(data, labels, partitions)
        np.testing.assert_almost_equal(memberships, expected_membership)


    def test_membership_for_data_1(self):
        # Iris data
        data = np.array([[4.3]])
        labels = np.array([0])
        partitions = np.array([[[2.5,  4.3,  6.1]]])
        expected_membership =np.array([[1]])
        memberships = membership_for_data_array(data, labels, partitions)
        np.testing.assert_equal(memberships, expected_membership)

    def test_membership_for_data_0(self):
        # Iris data
        data = np.array([[2.5]])
        labels = np.array([0])
        partitions = np.array([[[2.5,  4.3,  6.1]]])
        expected_membership =np.array([[0]])
        memberships = membership_for_data_array(data, labels, partitions)
        np.testing.assert_equal(memberships, expected_membership)

    def test_membership_for_data_lower_0(self):
        # Iris data
        data = np.array([[2.]])
        labels = np.array([0])
        partitions = np.array([[[2.5,  4.3,  6.1]]])
        expected_membership =np.array([[0]])
        memberships = membership_for_data_array(data, labels, partitions)
        np.testing.assert_equal(memberships, expected_membership)

    def test_membership_for_data_upper_0(self):
        # Iris data
        data = np.array([[6.2]])
        labels = np.array([0])
        partitions = np.array([[[2.5,  4.3,  6.1]]])
        expected_membership =np.array([[0]])
        memberships = membership_for_data_array(data, labels, partitions)
        np.testing.assert_equal(memberships, expected_membership)

if __name__ == '__main__':
    unittest.main()
