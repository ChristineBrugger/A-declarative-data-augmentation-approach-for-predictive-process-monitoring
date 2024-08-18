import os
import json
import pandas as pd
import numpy as np
import datetime
from multiprocessing import  Pool

# from ..constants import Task
from pathlib import Path
import re

class LogsDataProcessor:
    def __init__(self, name, filepath, columns, dir_path = "./datasets/processed", pool = 1):
        """Provides support for processing raw logs.
        Args:
            name: str: Dataset name
            filepath: str: Path to raw logs dataset
            columns: list: name of column names
            dir_path:  str: Path to directory for saving the processed dataset
            pool: Number of CPUs (processes) to be used for data processing
        """
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = dir_path
        if not os.path.exists(f"{dir_path}/{self._name}/processed"):
            os.makedirs(f"{dir_path}/{self._name}/processed")
        self._dir_path = f"{self._dir_path}/{self._name}/processed"
        self._pool = pool

        # Get the base name without prefix
        self.base_name = Path(self._filepath).stem.split("_", 1)[-1]
        self.directory_name = Path(self._filepath).parent

        self.train_fold = self.directory_name / f"train_{self.base_name}.csv"
        self.val_fold = self.directory_name / f"val_{self.base_name}.csv"
        self.test_fold = self.directory_name / f"test_{self.base_name}.csv"
        self.full_log = self.directory_name / f"full_{self.base_name}.csv"


    def _load_df(self, filepath, sort_temporally = False):
        df = pd.read_csv(filepath)
        df = df[self._org_columns]
        df.columns = ["case:concept:name", 
            "concept:name", "time:timestamp"]
        df["time:timestamp"] = df["time:timestamp"].str.replace("/", "-")
        df["time:timestamp"]= pd.to_datetime(df["time:timestamp"],  
            dayfirst=True).map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        if sort_temporally:
            df.sort_values(by = ["time:timestamp"], inplace = True)
        return df

    def _extract_logs_metadata(self, df):
        keys = ["[PAD]", "[UNK]"]
        activities = list(df["concept:name"].unique())
        keys.extend(activities)
        val = range(len(keys))

        coded_activity = dict({"x_word_dict":dict(zip(keys, val))})
        code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

        coded_activity.update(code_activity_normal)
        print("Coded activity: ", coded_activity)
        coded_json = json.dumps(coded_activity)
        with open(f"{self._dir_path}/metadata.json", "w") as metadata_file:
            metadata_file.write(coded_json)

    def _next_activity_helper_func(self, df, min_prefix_length=1):
        case_id, case_name = "case:concept:name", "concept:name"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "next_act"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()

            # only take prefixes of minimal length
            for i in range(len(act) - 1):
                if i + 1 < min_prefix_length:
                    continue
                
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))
                next_act = act[i+1]
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "next_act"] = next_act
                idx += 1
        return processed_df

    def _process_next_activity(self, df, train_list, test_list, val_list):

        df_split = np.array_split(df, self._pool)

        with Pool(processes=self._pool) as pool:
            from functools import partial
            process_func = partial(self._next_activity_helper_func, min_prefix_length=2)
            processed_df = pd.concat(pool.imap_unordered(process_func, df_split))
        print('processed_df: ', processed_df)

        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        val_df = processed_df[processed_df["case_id"].isin(val_list)]

        train_df.to_csv(f"{self._dir_path}/next_activity_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/next_activity_test.csv", index=False)
        val_df.to_csv(f"{self._dir_path}/next_activity_val.csv", index=False)


    def process_logs(self, sort_temporally = False):

        # full_df = self._load_df(os.path.join(self.directory_name, self.full_log), False)
        # train_df = self._load_df(os.path.join(self.directory_name, self.train_fold), False)
        # test_df = self._load_df(os.path.join(self.directory_name, self.test_fold), False)
        # val_df = self._load_df(os.path.join(self.directory_name, self.val_fold), False)
        full_df = self._load_df(self.full_log, sort_temporally)
        train_df = self._load_df(self.train_fold, sort_temporally)
        test_df = self._load_df(self.test_fold, sort_temporally)
        val_df = self._load_df(self.val_fold, sort_temporally)
        self._extract_logs_metadata(full_df)
        train_list = train_df["case:concept:name"].unique()
        test_list = test_df["case:concept:name"].unique()
        val_list = val_df["case:concept:name"].unique()

        print('Train list:', train_list)
        print('Test list:', test_list)
        print('Val list:', val_list)

        self._process_next_activity(full_df, train_list, test_list, val_list)
