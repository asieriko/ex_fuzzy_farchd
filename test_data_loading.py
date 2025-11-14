#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ex_fuzzy_farchd.Data import Data

def test_categorical_data():
    print("Testing categorical data loading with crx_train.dat...")

    # Test with crx dataset that has categorical attributes
    data_path = "/home/asier/Ikerketa/Projects/ex_fuzzy_farchd/datasets/crx_train.dat"
    data_loader = Data(data_path)

    try:
        info, x_data, y_data = data_loader.get_data()

        print("Dataset loaded successfully!")
        print("Dataset name:", info.get('dataset'))
        print("Number of attributes:", data_loader.get_number_of_attributes())
        print("Classes:", data_loader.get_classes())
        print("Data shape:", x_data.shape)
        print("First 3 rows of data:")
        print(x_data[:3])
        print("First 3 class labels:", y_data[:3])

        # Show attribute information
        print("\nAttribute information:")
        for attr_name, attr_info in info['attributes'].items():
            print("  {}: {}".format(attr_name, attr_info))

        # Show categorical mappings
        print("\nCategorical mappings:")
        mappings = data_loader.get_categorical_mappings()
        for attr_name, mapping in mappings.items():
            print("  {}: {}".format(attr_name, mapping))

    except Exception as e:
        print("Error loading data:", e)
        import traceback
        traceback.print_exc()

def test_numeric_data():
    print("\n" + "="*50)
    print("Testing numeric data loading with iris dataset...")

    # Test with iris dataset that should have mostly numeric attributes
    data_path = "/home/asier/Ikerketa/Projects/ex_fuzzy_farchd/datasets/iris-10-1tra.dat"
    data_loader = Data(data_path)

    try:
        info, x_data, y_data = data_loader.get_data()

        print("Dataset loaded successfully!")
        print("Dataset name:", info.get('dataset'))
        print("Number of attributes:", data_loader.get_number_of_attributes())
        print("Classes:", data_loader.get_classes())
        print("Data shape:", x_data.shape)
        print("First 3 rows of data:")
        print(x_data[:3])
        print("First 3 class labels:", y_data[:3])

        # Show attribute information
        print("\nAttribute information:")
        for attr_name, attr_info in info['attributes'].items():
            print("  {}: {}".format(attr_name, attr_info))

    except Exception as e:
        print("Error loading data:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_categorical_data()
    test_numeric_data()
