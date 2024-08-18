# Import necessary libraries and modules
import os
import calendar
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import pm4py
from itertools import product

# Due to changes in the code we need the functions from the code repository
from declare4py_2_2_0.Declare4Py.ProcessModels.DeclareModel import DeclareModel
from declare4py_2_2_0.Declare4Py.ProcessMiningTasks.Discovery.DeclareMiner import DeclareMiner
from declare4py_2_2_0.Declare4Py.D4PyEventLog import D4PyEventLog
from declare4py_2_2_0.Declare4Py.ProcessMiningTasks.ASPLogGeneration.asp_generator import AspGenerator

from pix_framework.discovery import case_arrival
from pix_framework.io.event_log import EventLogIDs

# Ignore warning
import warnings
warnings.simplefilter('ignore')

class EventLogProcessor:

    """
    A class to execute the declarative data augmentation for predictive process monitoring.
    The included functions process event logs for training and testing purposes, 
    including log splitting, model discovery, synthetic log generation, 
    and data augmentation.

    """

    def __init__(self, dataset_name, data_dir='data'):

        """
        Initialize the EventLogProcessor with the dataset name and directory paths.

        Args:
            dataset_name: Name of the dataset to be processed.
            data_dir: Directory where the dataset is located.

        """

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.dataset_path = os.path.join(data_dir, dataset_name)
        self.original_dir = os.path.join(self.dataset_path, f"{dataset_name}_orig")
        os.makedirs(self.original_dir, exist_ok=True)

        # Paths and names for augmented datasets and models
        self.augm_dataset_path = None
        self.augm_dataset_name = None
        self.syn_path = None
        self.model_path = None


    def train_test_split(self, split_ratio=0.8):

        """
        Split the dataset into training and testing sets based on the provided split ratio.

        Args:
            split_ratio: Ratio of training data to the total data. Default is 0.8 (80% training, 20% testing).
        """
        
        # Load original data set
        df = pd.read_csv(os.path.join(self.original_dir, f'{self.dataset_name}.csv'))
        df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
        df = df[['case:concept:name', 'concept:name', 'time:timestamp']]

        # Sort temporally and split the data into training and testing sets based on complete cases
        unique_cases_sorted = df.sort_values(by=['time:timestamp', 'case:concept:name'])['case:concept:name'].unique()
        split_index = int(len(unique_cases_sorted) * split_ratio)
        train_list = unique_cases_sorted[:split_index]
        test_list = unique_cases_sorted[split_index:]

        train_df = df[df['case:concept:name'].isin(train_list)]
        test_df = df[df['case:concept:name'].isin(test_list)]

        # Reset indices and save the split data
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        train_df.to_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_complete.csv'), index=False)
        test_df.to_csv(os.path.join(self.original_dir, f'test_{self.dataset_name}_orig.csv'), index=False)


    def generate_train_log(self):

        """
        Convert the training data into an event log format and save it as a XES file.
        """

        # Load training data
        train_df = pd.read_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_complete.csv'))

        train_df['time:timestamp'] = pd.to_datetime(train_df['time:timestamp'])
        train_df['case:concept:name'] = train_df['case:concept:name'].astype(str)
        train_df['concept:name'] = train_df['concept:name'].astype(str)

        # Convert the DataFrame to an event log and save it as a XES file
        train_log = pm4py.convert_to_event_log(train_df)
        pm4py.write_xes(train_log, file_path=os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_log.xes'))


    def declare_discovery(self, consider_vacuity:bool, min_support:float, itemsets_support:float, max_declare_cardinality:int):
        """
        Discover a declarative process model from the event log and save it in the appropriate directory.

        Args:
            consider_vacuity: Whether to consider vacuously satisfied traces as satisfied (True) or violated (False) otherwise in the discovery process.
            min_support: Minimum support for discovered constraints.
            itemsets_support: Minimum support for frequent itemsets.
            max_declare_cardinality: Maximum cardinality for Exactly, Existence and Absence templates.
        """

        # Generate the subfolder name based on the parameters (including '_augm' suffix)
        vacuity_str = 'T' if consider_vacuity else 'F'
        subfolder_name = f"{self.dataset_name}_cv{vacuity_str}_{min_support}_{itemsets_support}_dc{max_declare_cardinality}_augm"

        # Create the directory if it doesn't exist
        subfolder_path = os.path.join(self.dataset_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # Define path to save the discovered model without the '_augm' suffix in the file name
        model_file_name = subfolder_name.replace('_augm', '') + ".decl"
        self.model_path = os.path.join(subfolder_path, model_file_name)

        # Update the augmented dataset name and path
        self.augm_dataset_name = subfolder_name.replace('_augm', '')
        self.augm_dataset_path = subfolder_path

        # Load the event log
        event_log: D4PyEventLog = D4PyEventLog(case_name="case:concept:name")
        event_log.parse_xes_log(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_log.xes'))

        # Discover the declarative process model
        discovery = DeclareMiner(log=event_log, consider_vacuity=consider_vacuity, min_support=min_support, itemsets_support=itemsets_support, 
                                 max_declare_cardinality=max_declare_cardinality)
        discovered_model: DeclareModel = discovery.run()

        # Show discovered activities and constraints
        print("Model activities:")
        print("-----------------")
        for idx, act in enumerate(discovered_model.get_model_activities()):
            print(idx, act)
        print("\n")

        print("Model constraints:")
        print("-----------------")
        for idx, constr in enumerate(discovered_model.get_decl_model_constraints()):
            print(idx, constr)

        # Save the discovered model without the '_augm' suffix in the file name
        discovered_model.to_file(self.model_path)

    
    def get_syn_log_parameters(self):

        """
        Calculate parameters for the process simulation.

        Returns:
            min_events: Minimum number of events per case.
            max_events: Maximum number of events per case.
            num_of_cases: Number of unique cases.
            orig_prob: Original case length distribution of original data.
        """

        # Load data and compute first parameters
        df = pd.read_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_complete.csv'))
        min_events = int(df.groupby(['case:concept:name'])['concept:name'].count().min())
        max_events = int(df.groupby(['case:concept:name'])['concept:name'].count().max())
        num_of_cases = df['case:concept:name'].nunique()

        # Calculate the distribution of cases based on event count
        event_count_sorted = (
            df.groupby('case:concept:name')
            .size()
            .value_counts()
            .reset_index()
            .rename(columns={'index': 'Number of Events', 'count': 'Number of Cases'})
            .sort_values(by='Number of Events')
            .reset_index(drop=True)
        )

        print(event_count_sorted)

        number_cases_per_event = list(event_count_sorted['Number of Cases'])

        orig_prob = [0] * (max_events - min_events + 1)

        # Calculate the orginal case length distribution
        for idx in range(len(event_count_sorted)):

            current_events = event_count_sorted['Number of Events'].iloc[idx]   

            # Ensure to take correct index         
            cor_idx = current_events - min_events
            orig_prob[cor_idx] = number_cases_per_event[idx] / num_of_cases

        # Round all probabilities to 10 decimal places and adjust the last element to ensure the sum is exactly 1
        orig_prob = [round(prob, 10) for prob in orig_prob]
        orig_prob[-1] += (1 - sum(orig_prob))

        return min_events, max_events, num_of_cases, orig_prob


    def generate_syn_log(self, num_of_cases, num_min_events, num_max_events, orig_prob):

        """
        Generate the synthetic event log based on the discovered Declare model.

        Args:
            num_of_cases: Number of synthetic cases to generate.
            num_min_events: Minimum number of events per case.
            num_max_events: Maximum number of events per case.
            orig_prob: Original probability distribution for event generation.
        """

        syn_log_path = os.path.join(self.augm_dataset_path, f"{self.augm_dataset_name}_syn_log.xes")
        model = DeclareModel().parse_from_file(self.model_path)

        asp_gen = AspGenerator(model, num_of_cases, num_min_events, num_max_events)
        asp_gen.set_distribution(distributor_type="custom", custom_probabilities=orig_prob)
        asp_gen.run()

        asp_gen.to_xes(syn_log_path)


    def convert_syn_log_to_df(self):

        """
        Convert the synthetic event log to a DataFrame and save it as a CSV file.
        """

        # Load synthetic log
        syn_log_path = os.path.join(self.augm_dataset_path, f"{self.augm_dataset_name}_syn_log.xes")

        # Convert log to DataFrame
        syn_log = pm4py.read_xes(syn_log_path)
        syn_df = pm4py.convert_to_dataframe(syn_log)
        self.syn_path = os.path.join(self.augm_dataset_path, f"{self.augm_dataset_name}_syn.csv")

        # Save DataFrame as csv
        syn_df.to_csv(self.syn_path, index=False)
        

    def generate_timedelta(self, req_num_mean:int):

        """
        Calculate the time differences between activities in the original dataset.

        Args:
            req_num_mean: Required number of means in other combinations to calculate the durations based on first or second activity.

        Returns:
            time_delta: Dictionary with durations of original data and estimated ones.
        """

        # Load data
        df_orig = pd.read_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_complete.csv'))
        df_syn = pd.read_csv(self.syn_path)

        df_orig['time:timestamp'] = pd.to_datetime(df_orig['time:timestamp'])
        df_syn['time:timestamp'] = pd.to_datetime(df_syn['time:timestamp'])

        # Create dictionary with all possible combinations
        activities = list(df_orig['concept:name'].unique())
        combo = product(activities, repeat=2)
        time_delta = {combination: {'durations': [], 'mean': None, 'std': None} for combination in combo}

        # Calculate the time differences between consecutive activities
        for case_id in df_orig['case:concept:name'].unique():
            case_df = df_orig[df_orig['case:concept:name'] == case_id]
            for i in range(len(case_df) - 1):
                activity1 = case_df.iloc[i]['concept:name']
                activity2 = case_df.iloc[i + 1]['concept:name']
                time_diff = (case_df.iloc[i + 1]['time:timestamp'] - case_df.iloc[i]['time:timestamp']).total_seconds()
                time_delta[(activity1, activity2)]['durations'].append(time_diff)

        # Calculate mean and standard deviation for each activity pair
        for key, values in time_delta.items():
            durations = values['durations']
            if durations:
                time_delta[key]['mean'] = round(np.mean(durations), 2)
                time_delta[key]['std'] = round(np.std(durations), 2)

        # Update missing values in the time delta
        time_delta = self._update_missing_values_timedelta(time_delta, req_num_mean)

        return time_delta


    def _update_missing_values_timedelta(self, time_delta, req_num_mean):

        """
        Update missing mean and standard deviation values in the time_delta dictionary.

        Args:
            time_delta: Dictionary containing mean durations between activities, with probably empty values.
            req_num_mean: Required number of means in other combinations to calculate the durations based on first or second activity.

        Returns:
            time_delta: Updated time_delta.
        """

        # Loop over all combinations
        for key in time_delta.keys():

            # Check if mean and std values are present for current combination
            if not self._has_valid_values(key, time_delta):

                first_activity, second_activity = key

                # Generate list with all combinations that have the same first activity as current combination and valid mean and std values
                related_keys_first = [k for k in time_delta.keys() if k[0] == first_activity and self._has_valid_values(k, time_delta)]

                # Calculate mean and std values if enough combinations are found
                if len(related_keys_first) > req_num_mean:
                    mean, std = self._compute_mean_std(related_keys_first, time_delta)

                else:
                    # Generate list with all combinations that have the same second activity as current combination and valid mean and std values
                    related_keys_second = [k for k in time_delta.keys() if k[1] == second_activity and self._has_valid_values(k, time_delta)]
                    if len(related_keys_second) > req_num_mean:
                        mean, std = self._compute_mean_std(related_keys_second, time_delta)

                    # Compute overall mean and std values for all combinations with valid values
                    else:
                        all_keys_with_durations = self._get_keys_with_durations(time_delta)
                        mean, std = self._compute_mean_std(all_keys_with_durations, time_delta)

                # Update mean and std values
                time_delta[key]["mean"] = mean
                time_delta[key]["std"] = std
                
        return time_delta


    def _has_valid_values(self, key, time_delta):

        """
        Check if the time_delta dictionary has valid mean and standard deviation values for a given activity combination.

        Args:
            key: The key (activity combination) in the time_delta dictionary.
            time_delta: Dictionary containing durations between activities.
        """

        return time_delta[key].get("mean") is not None and time_delta[key].get("std") is not None


    def _compute_mean_std(self, keys, time_delta):

        """
        Compute the mean of the mean and std values for a list of keys in the time_delta dictionary.

        Args:
            keys: List of keys (activity combinations) in the time_delta dictionary.
            time_delta: Dictionary containing durations between activities.

        Returns:
            mean: computed mean for activity combination
            std: computed mean value of std for activity combination
        """

        # Compute mean over the considered mean and std values
        mean_values = [time_delta[k]["mean"] for k in keys]
        std_values = [time_delta[k]["std"] for k in keys]
        mean = np.mean(mean_values)
        std = np.mean(std_values)

        return mean, std


    def _get_keys_with_durations(self, time_delta):

        """
        Get a list of keys in the time_delta dictionary that have valid mean and standard deviation values.

        Args:
            time_delta: Dictionary containing durations between activities.

        Returns:
            List of keys in the time_delta dictionary that have valid mean and std values.
        """

        return [k for k, v in time_delta.items() if v.get("mean") is not None and v.get("std") is not None]


    def get_inter_case_dist(self):

        """
        Discover the inter-arrival time distribution between cases in the original event log.

        Returns:
            inter_case_dist: Discovered distribution of inter-arrival times between cases in original data.
        """

        # Load data
        df_orig = pd.read_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_complete.csv'))
        df_orig['time:timestamp'] = pd.to_datetime(df_orig['time:timestamp'])
        
        custom_log_ids = {
            "case": "case:concept:name",
            "activity": "concept:name",
            "start_time": "time:timestamp"
        }

        # Discover inter-arrival distribution
        event_log_ids = EventLogIDs.from_dict(custom_log_ids)
        inter_case_dist = case_arrival.discover_inter_arrival_distribution(df_orig, event_log_ids)
        print(inter_case_dist)

        return inter_case_dist


    def generate_first_timestamps(self, inter_case_dist):

        """
        Generate the initial timestamps for synthetic cases based on the inter-arrival distribution.

        Args:
            inter_case_dist: Discovered distribution of inter-arrival times between cases in original data.

        Returns:
            df_syn: Synthetic DataFrame with initial timestamps for the cases. 
        """

        # Set random seed and load data
        np.random.seed(42)
        df_syn = pd.read_csv(self.syn_path)
        df_syn['time:timestamp'] = pd.to_datetime(df_syn['time:timestamp'])
        df_syn["syn_timestamp"] = None

        # Get indices of initial activities of all cases
        first_activity_indices = df_syn.groupby("case:concept:name").head(1).index
        df_syn.loc[0, "syn_timestamp"] = df_syn.loc[0, "time:timestamp"]

        # Add random number based on discovered distribution for intial activities
        for i in range(1, len(first_activity_indices)):
            
            if inter_case_dist['distribution_name'] == 'gamma':
                mean = inter_case_dist['distribution_params'][0]['value']
                var = inter_case_dist['distribution_params'][1]['value']
                min = inter_case_dist['distribution_params'][2]['value']
                max = inter_case_dist['distribution_params'][3]['value']

                shape = (mean ** 2) / var
                scale = var / mean

                inter_arrival_time = np.random.gamma(shape, scale)
                inter_arrival_time = np.clip(inter_arrival_time, min, max)

            elif inter_case_dist['distribution_name'] == 'exponential':
                mean = inter_case_dist['distribution_params'][0]['value']
                min = inter_case_dist['distribution_params'][1]['value']
                max = inter_case_dist['distribution_params'][2]['value']

                inter_arrival_time = np.random.exponential(mean)
                inter_arrival_time = np.clip(inter_arrival_time, min, max)

            elif inter_case_dist['distribution_name'] == 'normal':
                mean = inter_case_dist['distribution_params'][0]['value']
                std = inter_case_dist['distribution_params'][1]['value']
                min = inter_case_dist['distribution_params'][2]['value']
                max = inter_case_dist['distribution_params'][3]['value']

                inter_arrival_time = np.random.normal(mean, std)
                inter_arrival_time = np.clip(inter_arrival_time, min, max)

            elif inter_case_dist['distribution_name'] == 'uniform':
                min = inter_case_dist['distribution_params'][0]['value']
                max = inter_case_dist['distribution_params'][1]['value']

                inter_arrival_time = np.random.uniform(min, max)

            elif inter_case_dist['distribution_name'] == 'log_normal':
                mean = inter_case_dist['distribution_params'][0]['value']
                variance = inter_case_dist['distribution_params'][1]['value']
                min = inter_case_dist['distribution_params'][2]['value']
                max = inter_case_dist['distribution_params'][3]['value']

                std = np.sqrt(variance)
                inter_arrival_time = np.random.lognormal(mean, std)
                inter_arrival_time = np.clip(inter_arrival_time, min, max)

            elif inter_case_dist['distribution_name'] == 'fixed':
                inter_arrival_time = inter_case_dist['distribution_params'][0]['value']

            current_idx = first_activity_indices[i]
            previous_idx = first_activity_indices[i-1]
            df_syn.loc[current_idx, "syn_timestamp"] = df_syn.loc[previous_idx, "time:timestamp"] + pd.Timedelta(seconds=inter_arrival_time)

        return df_syn


    def generate_timestamps(self, df_syn, time_delta, consider_hour: bool, consider_day: bool, start_h=8, end_h=20):

        """
        Generate synthetic timestamps for the synthetic DataFrame based on time deltas.

        Args:
            df_syn: DataFrame containing the synthetic log.
            time_delta: Dictionary containing the durations between activities.
            consider_hour: Whether to consider business hours when generating timestamps.
            consider_day: Whether to consider weekdays when generating timestamps.
            start_h: Start of business hours. Default is 8 AM.
            end_h: End of business hours. Default is 8 PM.
        """
        
        # Set seed
        np.random.seed(42)

        # Generate timestamps for each synthetic dase
        for case_id, case_df in df_syn.groupby("case:concept:name"):

            for i in range(1, len(case_df)):

                previous_activity_timestamp = case_df.iloc[i-1]["syn_timestamp"]

                previous_activity = case_df.iloc[i - 1]["concept:name"]
                current_activity = case_df.iloc[i]["concept:name"]

                mean_duration = time_delta[(previous_activity, current_activity)]["mean"]
                std_duration = time_delta[(previous_activity, current_activity)]["std"]

                random_seconds = max(math.pow(10,-4), np.random.normal(mean_duration, std_duration))

                new_timestamp = previous_activity_timestamp + pd.Timedelta(seconds=random_seconds)

                # Adjust timestamps based on business hours and weekdays
                if consider_hour:

                    new_timestamp = self._check_business_hours(new_timestamp, start=start_h, end=end_h)

                if consider_day:

                    new_timestamp = self._check_weekday(new_timestamp)
                
                case_df.at[case_df.index[i], "syn_timestamp"] = new_timestamp

            df_syn.loc[case_df.index] = case_df

        # Save updated synthetic data
        self._save_new_syn_time(df_syn)


    def _check_business_hours(self, new_timestamp, start=8, end=20):

        """
        Adjust the timestamp to ensure it falls within business hours.

        Args:
            new_timestamp: The timestamp to be adjusted.
            start: Start of business hours. Default is 8 AM.
            end: End of business hours. Default is 8 PM.

        Returns:
            new_timestamp: The updated timestamp that is within the business hours.
        """

        # Check if hour is before the start of the business day
        if new_timestamp.hour < start:

            new_timestamp = new_timestamp.replace(hour=start)

        # Check if hour is after the end of the business day
        elif new_timestamp.hour > end:

            # Check if the current day is the last day of the month
            if new_timestamp.day + 1 > calendar.monthrange(new_timestamp.year, new_timestamp.month)[1]:

                # Adjust year if the month is December
                if new_timestamp.month == 12: 
                    new_timestamp = new_timestamp.replace(year=new_timestamp.year + 1, month=1, day=1, hour=start)
                   
                # Move timestamp to the first day of the next month
                else:
                    new_timestamp = new_timestamp.replace(month=new_timestamp.month + 1, day=1, hour=start)

            # Move timestamp to the next day
            else:
                new_timestamp = new_timestamp.replace(day=new_timestamp.day + 1, hour=start)

        return new_timestamp


    def _check_weekday(self, new_timestamp):

        """
        Adjust the new synthetic timestamp to ensure it falls on a weekday (Monday to Friday).

        Args:
            new_timestamp: The timestamp to be adjusted.

        Returns:
            new_timestamp: The updated timestamp that is within the business days.
        """

        # Check if the day is on a weekend (4 rperesents Friday)
        if new_timestamp.weekday() > 4:  

            # Adjust timestamp if it is on a Saturday
            if new_timestamp.weekday() == 5: 

                # Adjust month if necessary
                if new_timestamp.day + 2 > calendar.monthrange(new_timestamp.year, new_timestamp.month)[1]:

                    # Adjust year if the month is December
                    if new_timestamp.month == 12: 
                        new_timestamp = new_timestamp.replace(year=new_timestamp.year + 1, month=1, day=(2 - (calendar.monthrange(new_timestamp.year, new_timestamp.month)[1] - new_timestamp.day)))
                    
                    # Move to the first or second day of the next month
                    else:
                        new_timestamp = new_timestamp.replace(month=new_timestamp.month + 1, day=(2 - (calendar.monthrange(new_timestamp.year, new_timestamp.month)[1] - new_timestamp.day)))
                
                # Add two days to the timestamp
                else:
                    new_timestamp = new_timestamp.replace(day=new_timestamp.day + 2)

            # Adjust timestamp if it is on a Sunday
            elif new_timestamp.weekday() == 6:  

                # Adjust month if necessary
                if new_timestamp.day + 1 > calendar.monthrange(new_timestamp.year, new_timestamp.month)[1]:

                    # Adjust year if the month is December
                    if new_timestamp.month == 12:  
                        new_timestamp = new_timestamp.replace(year=new_timestamp.year + 1, month=1, day=1)

                    # Move to the first day of the next month
                    else:
                        new_timestamp = new_timestamp.replace(month=new_timestamp.month + 1, day=1)

                # Add one days to the timestamp
                else:
                    new_timestamp = new_timestamp.replace(day=new_timestamp.day + 1)

        return new_timestamp


    def _save_new_syn_time(self, df_syn):

        """
        Save the synthetic log with updated timestamps.

        Args:
            df_syn: DataFrame containing the synthetic log with updated timestamps.
        """

        # Load data and replace old timestamp column with new synthetic one
        df_syn = df_syn.drop(columns='time:timestamp')
        df_syn = df_syn.rename(columns={"syn_timestamp": "time:timestamp"})
        df_syn['time:timestamp'] = pd.to_datetime(df_syn['time:timestamp'])

        df_syn.to_csv(self.syn_path, index=False)


    def get_most_frequent_common_variants(self):

        """
        Identify the most frequent common variants between the synthetic and original data.

        Returns:
            common_variants_list: Sorted list with common variants, including their frequency and proportion.
        """

        # Load data
        df_syn = pd.read_csv(self.syn_path)
        df_orig = pd.read_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_complete.csv'))

        df_syn['time:timestamp'] = pd.to_datetime(df_syn['time:timestamp'])
        df_orig['time:timestamp'] = pd.to_datetime(df_orig['time:timestamp'])

        df_syn[['case:concept:name', 'concept:name']] = df_syn[['case:concept:name', 'concept:name']].astype(str)
        df_orig[['case:concept:name', 'concept:name']] = df_orig[['case:concept:name', 'concept:name']].astype(str)

        # Get variants from both logs
        variants_syn = pm4py.get_variants_as_tuples(df_syn)
        variants_orig = pm4py.get_variants_as_tuples(df_orig)

        # Identify common variants between the original and synthetic log
        common_variants = set(variants_syn.keys()).intersection(set(variants_orig.keys()))
        print('Number of common variants:', len(common_variants))

        total_cases_syn = df_syn['case:concept:name'].nunique()
        total_cases_orig = df_orig['case:concept:name'].nunique()

        # Calculate the frequency and proportion of common variants
        common_variants_dict = {
            variant: {
                'frequency_orig': variants_orig[variant],
                'frequency_syn': variants_syn[variant],
                'proportion_orig': (variants_orig[variant] / total_cases_orig),
                'proportion_syn': (variants_syn[variant] / total_cases_syn)
            }
            for variant in common_variants
        }

        # Sort the common variants by frequency in the original log
        common_variants_list = [(variant, data) for variant, data in common_variants_dict.items()]
        common_variants_list.sort(key=lambda x: x[1]['frequency_orig'])
        print('Sorted common variants:', common_variants_list)

        return common_variants_list


    def common_variants_to_remove(self, common_variants, method, threshold):

        """
        Identify common variants to remove based on a specified method and threshold.

        Args:
            common_variants: List of common variants.
            method: Method for selecting variants to remove ('common_ratio' or 'orig_proportion').
            threshold: Threshold for the selected method.

        Returns:
            variants_to_remove: Set of variants that should be removed.
        """

        if method == 'common_ratio':
            # Define variants that should be removed based on the common ratio
            num_to_remove = round(len(common_variants) * threshold)
            variants_to_remove_list = [variant for variant, data in common_variants[-num_to_remove:]]
            variants_to_remove = {variant for variant in variants_to_remove_list}

        elif method == 'orig_proportion':
            # Define variants that should be removed based on the proportion in the original data
            variants_to_remove_list = [variant for variant, data in common_variants if data['proportion_orig'] > threshold]
            variants_to_remove = {variant for variant in variants_to_remove_list}
        
        else:
            print('This method is not implemented. No variants will be defined to remove.')
            return set()
        
        return variants_to_remove


    def save_filtered_variants(self, most_frequent_common_variants):

        """
        Save the synthetic log with the most frequent common variants removed.

        Args:
            most_frequent_common_variants: List of common variants to remove.
        """

        # Load synthetic data and convert it to an event log
        df_syn = pd.read_csv(self.syn_path)
        df_syn['time:timestamp'] = pd.to_datetime(df_syn['time:timestamp'])
        df_syn[['case:concept:name', 'concept:name']] = df_syn[['case:concept:name', 'concept:name']].astype(str)
        log_syn = pm4py.convert_to_event_log(df_syn)

        # Filter and save the synthetic log without the defined most frequent common variants
        syn_filtered_log = pm4py.filter_log(lambda trace: tuple(event["concept:name"] for event in trace) not in most_frequent_common_variants, log_syn)
        df_syn_filtered = pm4py.convert_to_dataframe(syn_filtered_log)
        
        # Rename the dataset and save it
        self.syn_path = self.syn_path.replace("_syn.csv", "_syn_filtered.csv")
        df_syn_filtered.to_csv(self.syn_path, index=False)


    def train_val_split(self, split_ratio=0.8):

        """
        Split the training data into training and validation sets based on the provided split ratio.

        Args:
            split_ratio: Ratio of training data to the total training data. Default is 0.8 (80% training, 20% validation).
        """

        # Load original train data
        df_train_orig = pd.read_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig_complete.csv'))    

        # Sort temporally and split the data into training and testing sets based on complete cases
        unique_cases_sorted = df_train_orig.sort_values(by=['time:timestamp', 'case:concept:name'])['case:concept:name'].unique()

        split_index = int(len(unique_cases_sorted) * split_ratio)
        train_list = unique_cases_sorted[:split_index]
        val_list = unique_cases_sorted[split_index:]

        df_train_orig_80 = df_train_orig[df_train_orig['case:concept:name'].isin(train_list)]
        df_val = df_train_orig[df_train_orig['case:concept:name'].isin(val_list)]

        # Save train and test data
        df_train_orig_80.to_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig.csv'), index=False)
        df_val.to_csv(os.path.join(self.original_dir, f'val_{self.dataset_name}_orig.csv'), index=False)


    def augment_new_variants(self):

        """
        Augment the original training data (without the validation data) with synthetic variants.
        """

        # Load datasets
        df_train_orig_80 = pd.read_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig.csv'))
        df_train_syn = pd.read_csv(self.syn_path)

        df_orig = pd.read_csv(os.path.join(self.original_dir, f'{self.dataset_name}.csv'))
        df_orig['case:concept:name'] = df_orig['case:concept:name'].astype(int)

        # Separate case IDs of synthetic data
        df_train_syn = df_train_syn[['case:concept:name', 'concept:name', 'time:timestamp']]

        df_train_syn = self._separate_case_ids(df_train_syn)
        
        # Increase case IDs by maximum case ID
        df_train_syn = self._adjust_case_ids(df_base=df_orig, df_adjust=df_train_syn)
        df_train_syn.to_csv(self.syn_path, index=False)
        
        # Create augmented training data
        df_train_augm = pd.concat([df_train_orig_80, df_train_syn]).reset_index(drop=True)

        # Save data
        df_train_augm.to_csv(os.path.join(self.augm_dataset_path, f"train_{self.augm_dataset_name}_augm.csv"), index=False)


    def save_total_data(self, type):

        """
        Save the full dataset (original or augmented) by concatenating training, validation, and test sets.

        Args:
            type: Specify whether to save the 'original' or 'augmented' dataset.
        """

        # Load relevent training data and define path
        if type == 'original':
            df_train = pd.read_csv(os.path.join(self.original_dir, f'train_{self.dataset_name}_orig.csv'))
            total_augm_path = os.path.join(self.original_dir, f'{self.dataset_name}_orig.csv')

        elif type == 'augmented':
            df_train = pd.read_csv(os.path.join(self.augm_dataset_path, f"train_{self.augm_dataset_name}_augm.csv"))
            total_augm_path = os.path.join(self.augm_dataset_path, f"{self.augm_dataset_name}_augm.csv")
        
        # Concatenate and save the full dataset
        df_val = pd.read_csv(os.path.join(self.original_dir, f'val_{self.dataset_name}_orig.csv'))
        df_test = pd.read_csv(os.path.join(self.original_dir, f'test_{self.dataset_name}_orig.csv'))

        df_total = pd.concat([df_train, df_val, df_test]).reset_index(drop=True)

        df_total.to_csv(total_augm_path, index=False)


    def _separate_case_ids(self, df):

        """
        Separate numeric case IDs from a mixed-format column.

        Args:
            df: DataFrame containing the event log.

        Returns:
            df: DataFrame containing only the numeric case IDs
        """

        df['case:concept:name'] = df['case:concept:name'].str.extract(r'(\d+)').astype(int)

        return df


    def _adjust_case_ids(self, df_base, df_adjust):

        """
        Adjust case IDs in the synthetic log to avoid duplication with the original log.

        Args:
            df_base: DataFrame containing the original log.
            df_adjust: DataFrame containing the synthetic log to be adjusted.
        
        Returns:
            df_adjust: DataFrame containing the synthetic log with adjusted case IDs.
        """

        # Get maximum case ID of base DataFrame
        max_ID_base = df_base['case:concept:name'].max()

        # Increase the case IDs of df_adjust by maximum case ID of df_base
        df_adjust['case:concept:name'] = df_adjust['case:concept:name'] + max_ID_base + 1

        return df_adjust