#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:02:23 2023

@author: asier.urio
"""

import unittest
from src.ex_fuzzy_farchd.farchd_functions.Item import Item


class ItemTestCase(unittest.TestCase):

    def test_item_size_1_incomp(self):
        itemi = Item([[0, 1]])
        itemj = Item([[0, 2]])
        self.assertFalse(itemi.compatible(itemj))

    def test_item_size_1_comp(self):
        itemi = Item([[0, 1]])
        itemj = Item([[1, 2]])
        self.assertTrue(itemi.compatible(itemj))

    def test_item_size_2_incomp(self):
        itemi = Item([[0, 1], [1, 1]])
        itemj = Item([[0, 2], [2, 1]])
        self.assertFalse(itemi.compatible(itemj))

    def test_item_size_2_comp(self):
        itemi = Item([[0, 1], [1, 1]])
        itemj = Item([[0, 1], [2, 3]])
        self.assertTrue(itemi.compatible(itemj))


if __name__ == '__main__':
    unittest.main()
