"""
This script is based on the following source code:
    https://gitlab.citius.gal/efren.rama/pmdlcompararator/-/tree/master?ref_type=heads

We took the necessary parts and adjusted them to efficiently use it in our study.
"""


# Import necessary libraries
import os
import pandas as pd
from processtransformer.processtransformer.data.processor import LogsDataProcessor

class DataPreprocessor:

    """
    A class to preprocess event log data for different approaches such as 
    ProcessTransformer, Tax, and Mauro preprocessing methods.
    """

    def __init__(self, dataset_name, data_dir='data'):

        """
        Initialize the DataPreprocessor with the dataset name and directory paths.

        Args:
            dataset_name: Name of the dataset to be processed.
            data_dir: Directory where the dataset is located. Default is 'data'.
        """

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, dataset_name)

    def preprocess_all_subfolders(self, approach):

        """
        Preprocess all subfolders within the main dataset directory using the specified approach.

        Args:
            approach: The preprocessing approach to use ('processtransformer', 'tax', or 'mauro').
        """
        
        print(f"Approach: {approach}")  

        # Loop through all subfolders in the dataset directory
        for subfolder_name in os.listdir(self.dataset_path):
            subfolder_path = os.path.join(self.dataset_path, subfolder_name)

            # Check if the path is a directory
            if os.path.isdir(subfolder_path):
                print(f"Processing folder: {subfolder_name}")
                self._preprocess_subfolder(subfolder_name, subfolder_path, approach)

    def _preprocess_subfolder(self, subfolder_name, subfolder_path, approach):

        """
        Preprocess a specific subfolder based on the selected approach.

        Args:
            subfolder_name: Name of the subfolder to preprocess.
            subfolder_path: Path to the subfolder.
            approach: The preprocessing approach to use ('processtransformer', 'tax', or 'mauro').
        """

        # Load the datasets for the subfolder
        df_orig = pd.read_csv(os.path.join(subfolder_path, f'{subfolder_name}.csv'))
        df_train = pd.read_csv(os.path.join(subfolder_path, f'train_{subfolder_name}.csv'))
        df_test = pd.read_csv(os.path.join(subfolder_path, f'test_{subfolder_name}.csv'))
        df_val = pd.read_csv(os.path.join(subfolder_path, f'val_{subfolder_name}.csv'))

        # List of dataframes to process
        df_list = [
            (df_orig, f'{subfolder_name}'),
            (df_train, f'train_{subfolder_name}'),
            (df_test, f'test_{subfolder_name}'),
            (df_val, f'val_{subfolder_name}')
        ]

        # Preprocess according to the selected approach
        if approach == 'processtransformer':

            # Rename dataset for furhter use of ProcessTransformer
            df_list = [
                (df_orig, f'full_{subfolder_name}'),
                (df_train, f'train_{subfolder_name}'),
                (df_test, f'test_{subfolder_name}'),
                (df_val, f'val_{subfolder_name}')
            ]
            self._processtransformer_preprocessing(df_list, subfolder_name)

        elif approach == 'tax':
            self._tax_preprocessing(df_list, subfolder_name)

        elif approach == 'mauro':
            self._mauro_preprocessing(df_list, subfolder_name)


    def _processtransformer_preprocessing(self, df_list, subfolder_name):

        """
        Preprocess data for ProcessTransformer.

        Parameters:
        - df_list: List of tuples containing dataframes and their respective names.
        - subfolder_name: Name of the subfolder being processed.
        """

        for df, df_name in df_list:

            # Convert activity names to lowercase and replace spaces with underscores
            df["concept:name"] = df["concept:name"].str.lower().str.replace(" ", "_")

            # Format timestamp
            df["time:timestamp"] = pd.to_datetime(df["time:timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")

            # Add end-of-case [EOC] activity and save as csv file
            final_dataframe = self._augment_end_activity_to_csv(df)

            os.makedirs(f'processtransformer\\datasets\\{subfolder_name}', exist_ok=True)
            final_dataframe.to_csv(f'processtransformer\\datasets\\{subfolder_name}\\{df_name}.csv', sep=",", index=False)

        # Preprocess logs using the LogsDataProcessor from the ProcessTransformer library
        data_processor = LogsDataProcessor(name=f'{subfolder_name}', 
                                            filepath=f'processtransformer\\datasets\\{subfolder_name}\\full_{subfolder_name}.csv', 
                                            columns=["case:concept:name", "concept:name", "time:timestamp"],
                                            dir_path=f'processtransformer\\datasets', pool=1)
        data_processor.process_logs()

    def _tax_preprocessing(self, df_list, subfolder_name):

        """
        Preprocess data for the Tax approach.

        Args:
            df_list: List of tuples containing dataframes and their respective names.
            subfolder_name: Name of the subfolder being processed.
        """

        # Generate a dictionary to map activity names to integer categories
        category_dict = self._generate_category_dict('concept:name', df_list[0][0])

        for df, df_name in df_list:

            # Map activity names to integers
            df['concept:name'] = df['concept:name'].map(category_dict).astype(int)

            # Format timestamp
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Save preprocessed dataframes to csv
            os.makedirs(f'tax\\data\\{subfolder_name}', exist_ok=True)
            df.to_csv(f'tax\\data\\{subfolder_name}\\{df_name}.csv', sep=",", index=False)

    def _mauro_preprocessing(self, df_list, subfolder_name):

        """
        Preprocess data for the Mauro approach.

        Args:
            df_list: List of tuples containing dataframes and their respective names.
            subfolder_name: Name of the subfolder being processed.
        """

        for df, df_name in df_list:

            # Format timestamp
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True).apply(lambda ts: ts.strftime('%Y/%m/%d %H:%M:%S.%f'))

            # Add end-of-case [EOC] activity and save as csv file
            final_dataframe = self._augment_end_activity_to_csv(df)
            final_dataframe.to_csv(f'nnpm\\data\\{df_name}.csv', sep=",", index=False)


    def _augment_end_activity_to_csv(self, df):

        """
        Add an end-of-case [EOC] activity to each case in the dataframe.

        Args:
            df: DataFrame containing the event log.

        Returns:
            df: DataFrame with the EOC activity added to each case.
        """

        groups = [pandas_df for _, pandas_df in df.groupby('case:concept:name', sort=False)]
        for i, group in enumerate(groups):

            # Create a copy of the last row and mark it as the end-of-case activity
            last_rows = group[-1:].copy()
            last_rows['concept:name'] = "[EOC]"
            groups[i] = pd.concat([group, last_rows])

        return pd.concat(groups, sort=False).reset_index(drop=True)

    def _generate_category_dict(self, column, df):

        """
        Generate a dictionary to map unique values in a column to integers.

        Args:
            column: The column to map.
            df: DataFrame containing the data.

        Returns:
            category_dict: Dictionary mapping column values to integers.
        """

        category_list = df[column].astype("category").cat.categories.tolist()
        category_dict = {c: i for i, c in enumerate(category_list)}
        print("Activity assignment: ", category_dict)
        
        return category_dict
