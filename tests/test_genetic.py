#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 10:27:42 2023

@author: asier
"""
import unittest
import numpy as np

from src.stage3 import real_to_gray, gray_distance, crossover


class GeneticTestCase(unittest.TestCase):

    def test_real_to_gray_lower(self):
        number = -0.5
        gray = real_to_gray(number, 30)
        expected_gray = np.zeros(30)
        np.testing.assert_equal(gray, expected_gray)

    def test_real_to_gray_middle(self):
        number = 0.0
        gray = real_to_gray(number, 30)
        expected_gray = np.zeros(30)
        expected_gray[0] = 1
        expected_gray[1] = 1
        np.testing.assert_equal(gray, expected_gray)

    def test_real_to_gray_upper(self):
        number = 0.5
        gray = real_to_gray(number, 30)
        expected_gray = np.zeros(30)
        expected_gray[0] = 1
        np.testing.assert_equal(gray, expected_gray)


    def test_real_to_gray_step_01(self):
        number = -0.5 + 1/2**30
        gray = real_to_gray(number, 30)
        expected_gray = np.zeros(30)
        expected_gray[-1] = 1
        np.testing.assert_equal(gray, expected_gray)

    def test_real_to_gray_step_10(self):
        number = 0.5 - 1/2**30
        gray = real_to_gray(number, 30)
        expected_gray = np.zeros(30)
        expected_gray[0] = 1
        expected_gray[-1] = 1
        np.testing.assert_equal(gray, expected_gray)

    def test_gray_distance_0(self):
        c1 = [np.ones(5), np.array([0.25, 0.45])]
        c2 = [np.ones(5), np.array([0.25, 0.45])]
        dist = gray_distance(c1, c2, 30)
        expected_dist = 0
        self.assertEqual(dist, expected_dist)

    def test_gray_distance_part1(self):
        c1 = [np.ones(5), np.array([0.25, 0.45])]
        c2 = [np.ones(5), np.array([0.25, 0.45])]
        c2[0][0] = 0
        c2[0][-1] = 0
        dist = gray_distance(c1, c2, 30)
        expected_dist = 2
        self.assertEqual(dist, expected_dist)

    def test_gray_distance_part2(self):
        c1 = [np.ones(5), np.array([0.5, 0.45])]
        c2 = [np.ones(5), np.array([-0.5, 0.45])]
        dist = gray_distance(c1, c2, 30)
        expected_dist = 1
        self.assertEqual(dist, expected_dist)

    def test_gray_distance_part22(self):
        c1 = [np.ones(5), np.array([0.25, 0.45])]
        c2 = [np.ones(5), np.array([0.0, 0.0])]
        dist = gray_distance(c1, c2, 30)
        expected_dist = 15
        self.assertEqual(dist, expected_dist)

    def test_crossover_hux(self):
        p1 = [np.ones(5), np.array([0.25, 0.45])]
        p2 = [np.ones(5), np.array([0.25, 0.45])]
        p2[0][0] = 0
        p2[0][1] = 0
        o1, o2 = crossover(p1, p2, 1)
        np.testing.assert_equal(o1[0][2:], o2[0][2:])
        np.testing.assert_equal(o1[0][2:], p1[0][2:])
        np.testing.assert_equal(o1[0][2:], p2[0][2:])
        np.testing.assert_equal(o1[1], o2[1])
        np.testing.assert_equal(o1[1], p1[1])
        np.testing.assert_equal(o1[1], p2[1])
        self.assertNotEqual(o1[0][0], o2[0][0])
        self.assertNotEqual(o1[0][1], o2[0][1])
        np.testing.assert_equal((o1[0][0] == p1[0][0]) ^ (o1[0][0] == p2[0][0]), True)
        np.testing.assert_equal((o2[0][0] == p1[0][0]) ^ (o2[0][0] == p2[0][0]), True)

    def test_crossover_pcblx(self):
        p1 = [np.ones(5), np.array([0.3, 0.45])]
        p2 = [np.ones(5), np.array([0.2, 0.45])]
        o1, o2 = crossover(p1, p2, 1)
        np.testing.assert_equal(o1[0], o2[0])
        np.testing.assert_equal(o1[0], p1[0])
        np.testing.assert_equal(o1[0], p2[0])
        self.assertGreaterEqual(o1[1][0], p2[1][0])
        self.assertLessEqual(o1[1][0], p1[1][0])


if __name__ == '__main__':
    unittest.main()
