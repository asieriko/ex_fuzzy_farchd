import numpy as np
import pandas as pd


class dat_loader:

    # *.dat file loader

    def __init__(self, datasetTST):
        self.dataset_path = datasetTST
        self.info = {}
        self.x_data = None
        self.y_data = None
        self.categorical_mappings = {}  # Store mappings for categorical attributes

    def print_data(self):
        print(np.concatenate(
            (np.expand_dims(list(self.info['attributes'].keys())+['class'],
                            axis=0),
             np.concatenate(
                 (self.x_data,
                  np.expand_dims(self.y_data, axis=1)), axis=1
                 )))
            )

    def get_number_of_attributes(self):
        return len(self.info.get('attributes', []))

    # Receive a dataset.. after an @data need to return the related information
    def load_data(self):
        self.info = {}
        self.info['attributes'] = {}
        self.categorical_mappings = {}

        with open(self.dataset_path) as f:
            line = f.readline()
            lineData = line.split()
            N = 1
            while lineData[0] != '@data':
                if (lineData[0] == '@relation'):
                    self.info['dataset'] = lineData[1]
                if (lineData[0] == '@attribute'):
                    if lineData[1] == 'class' or lineData[1] == 'Class':
                        self.info['classes'] = "".join(lineData[2:]).replace("{","").replace("}","").split(',')
                    else:
                        attr_name = lineData[1]
                        attr_definition = " ".join(lineData[2:])

                        if attr_definition.startswith('{') and attr_definition.endswith('}'):
                            # Categorical attribute
                            categories = attr_definition[1:-1].split(',')
                            categories = [cat.strip() for cat in categories]
                            self.info['attributes'][attr_name] = {'type': 'categorical', 'values': categories}
                            # Create mapping from category to numeric value
                            self.categorical_mappings[attr_name] = {cat: i for i, cat in enumerate(categories)}
                        elif 'real' in attr_definition or 'integer' in attr_definition:
                            # Numeric attribute
                            if '[' in attr_definition and ']' in attr_definition:
                                range_part = attr_definition[attr_definition.find('['):attr_definition.find(']')+1]
                                range_values = range_part[1:-1].split(',')
                                min_val = float(range_values[0].strip())
                                max_val = float(range_values[1].strip())
                                attr_type = 'real' if 'real' in attr_definition else 'integer'
                                self.info['attributes'][attr_name] = {'type': attr_type, 'range': [min_val, max_val]}
                            else:
                                # Default to real if no range specified
                                self.info['attributes'][attr_name] = {'type': 'real', 'range': [float('-inf'), float('inf')]}
                        else:
                            # Default to categorical if format is unclear
                            self.info['attributes'][attr_name] = {'type': 'unknown'}

                line = f.readline()
                lineData = line.split()
                N += 1

        # Load data
        data = np.genfromtxt(self.dataset_path, comments='@', delimiter=",",
                             dtype=str, autostrip=True)

        # Process x_data: convert categorical to numeric, keep numeric as float
        self.x_data = np.zeros((data.shape[0], data.shape[1] - 1), dtype=float)
        attr_names = list(self.info['attributes'].keys())

        for i, attr_name in enumerate(attr_names):
            col_data = data[:, i]
            attr_info = self.info['attributes'][attr_name]

            if attr_info.get('type') == 'categorical':
                # Convert categorical values to numeric using mapping
                for j, value in enumerate(col_data):
                    if value in self.categorical_mappings[attr_name]:
                        self.x_data[j, i] = self.categorical_mappings[attr_name][value]
                    else:
                        # Handle unknown categories
                        print("Warning: Unknown category '{}' for attribute '{}', assigning value -1".format(value, attr_name))
                        self.x_data[j, i] = -1
            else:
                # Numeric attribute - convert to float
                try:
                    self.x_data[:, i] = col_data.astype(float)
                except ValueError as e:
                    print("Error converting attribute '{}' to float: {}".format(attr_name, e))
                    # Try to handle as categorical if conversion fails
                    unique_values = np.unique(col_data)
                    mapping = {val: i for i, val in enumerate(unique_values)}
                    for j, value in enumerate(col_data):
                        self.x_data[j, i] = mapping[value]

        self.y_data = data[:, -1]

    def get_data(self):
        if self.x_data is None or self.y_data is None:
            self.load_data()
        return self.info, self.x_data, self.y_data

    def get_classes(self):
        return self.info.get('classes', [])

    def get_attributes(self):
        return self.info.get('attributes', [])

    def get_examples_per_class(self):
        if self.x_data is None or self.y_data is None:
            self.load_data()
        instances_per_class = {k: [] for k in self.info['classes']}
        for x_values, y_class in zip(self.x_data, self.y_data):
            instances_per_class[y_class].append(x_values)
        return instances_per_class

    def get_categorical_mappings(self):
        """Returns the mapping used to convert categorical values to numeric ones"""
        return self.categorical_mappings

    def get_categorical_mask(self):
        """
        Returns a boolean mask indicating which attributes are categorical.

        Returns:
        --------
        list of bool
            Boolean list where True indicates the attribute is categorical.
            Length matches the number of attributes.
        """
        if self.x_data is None or self.y_data is None:
            self.load_data()

        attr_names = list(self.info['attributes'].keys())
        mask = []
        for attr_name in attr_names:
            attr_info = self.info['attributes'][attr_name]
            mask.append(attr_info.get('type') == 'categorical')
        return mask

    def get_x_dataframe(self, use_original_categorical=True):
        """
        Returns only the feature data (X) as a pandas DataFrame, excluding the class column.

        Parameters:
        -----------
        use_original_categorical : bool, default=True
            If True, categorical attributes will show their original string values.
            If False, categorical attributes will show their numeric encoded values.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with only the feature attributes (no class column).
        """
        if self.x_data is None or self.y_data is None:
            self.load_data()

        attr_names = list(self.info['attributes'].keys())

        if use_original_categorical:
            # Load raw data again to get original categorical values
            raw_data = np.genfromtxt(self.dataset_path, comments='@', delimiter=",",
                                   dtype=str, autostrip=True)

            # Process each column to keep categoricals as strings and convert numerics to float
            df_data = {}

            for i, attr_name in enumerate(attr_names):
                attr_info = self.info['attributes'][attr_name]
                col_data = raw_data[:, i]

                if attr_info.get('type') == 'categorical':
                    # Keep categorical values as strings
                    df_data[attr_name] = col_data
                else:
                    # Convert numeric values to float
                    try:
                        df_data[attr_name] = col_data.astype(float)
                    except ValueError:
                        # If conversion fails, keep as string
                        df_data[attr_name] = col_data
        else:
            # Use the already processed numeric data
            df_data = {}
            for i, attr_name in enumerate(attr_names):
                df_data[attr_name] = self.x_data[:, i]

        return pd.DataFrame(df_data)

    def get_y_series(self):
        """
        Returns the class column as a pandas Series.

        Returns:
        --------
        pandas.Series
            Series containing the class labels.
        """
        if self.x_data is None or self.y_data is None:
            self.load_data()

        return pd.Series(self.y_data, name='class')

    def get_dataframe(self, use_original_categorical=True):
        """
        Returns the dataset as a pandas DataFrame.

        Parameters:
        -----------
        use_original_categorical : bool, default=True
            If True, categorical attributes will show their original string values.
            If False, categorical attributes will show their numeric encoded values.

        Returns:
        --------
        pandas.DataFrame
            DataFrame with attribute names as columns and the class column.
        """
        if self.x_data is None or self.y_data is None:
            self.load_data()

        # Get attribute names and create column names
        attr_names = list(self.info['attributes'].keys())
        column_names = attr_names + ['class']

        # Create the dataframe data
        if use_original_categorical:
            # Load raw data again to get original categorical values
            raw_data = np.genfromtxt(self.dataset_path, comments='@', delimiter=",",
                                   dtype=str, autostrip=True)

            # Process each column to keep categoricals as strings and convert numerics to float
            df_data = {}

            for i, attr_name in enumerate(attr_names):
                attr_info = self.info['attributes'][attr_name]
                col_data = raw_data[:, i]

                if attr_info.get('type') == 'categorical':
                    # Keep categorical values as strings
                    df_data[attr_name] = col_data
                else:
                    # Convert numeric values to float
                    try:
                        df_data[attr_name] = col_data.astype(float)
                    except ValueError:
                        # If conversion fails, keep as string
                        df_data[attr_name] = col_data

            # Add class column
            df_data['class'] = raw_data[:, -1]

        else:
            # Use the already processed numeric data
            df_data = {}
            for i, attr_name in enumerate(attr_names):
                df_data[attr_name] = self.x_data[:, i]
            df_data['class'] = self.y_data

        return pd.DataFrame(df_data)
