"""
This script is of the following source code:
    https://github.com/mkaep/pbpm-ssl-suite/tree/976d6e4efd82b4188d3bf9b820456e3025663654
We just kept some parts that are necessary to efficiently use it in our study.
"""

import os
import typing
import json as js
import pm4py

from augm_baseline.ml.core import model, loader
from augm_baseline.ml.augmentation import augmentation_strategy
from augm_baseline.ml import pipeline

from pm4py.objects.log.obj import EventLog


def build_augmentation_strategies_from_config(
        augmentation_strategies_configs: typing.List[model.AugmentationStrategyConfig]):
    augmentation_strategies = []
    for augmentation_strategy_config in augmentation_strategies_configs:
        augmentation_strategies.append(augmentation_strategy.AugmentationStrategyBuilder(
            configuration=augmentation_strategy_config).build())

    return augmentation_strategies


def run_pipeline(experiment: model.AugmentationExperiment, verbose=False) -> None:
    experiment_dir = os.path.join(experiment.run_dir, experiment.name)
    pipeline.create_basic_folder_structure(experiment, verbose)

    jobs: typing.List[model.Job] = []
    for dataset in experiment.event_logs:
        event_log = loader.Loader.load_event_log(dataset.file_path, verbose)

        augmentation_strategies = build_augmentation_strategies_from_config(experiment.augmentation_strategies)

        assert event_log is not None

        # Augment training data
        if verbose:
            print('Start augmentation...')
        augmented_files = {}
        for strategy in augmentation_strategies:
            aug_dir = os.path.join(experiment.data_dir, strategy.name)
            os.makedirs(aug_dir, exist_ok=True)

            print("BEFORE FIT:")
            for aug in strategy.augmentors:
                print(aug.to_string())
            strategy.fit(event_log)
            print("AFTER FIT:")
            for aug in strategy.augmentors:
                print(aug.to_string())
            aug_event_log, aug_count, aug_record, aug_duration = strategy.augment(event_log,
                                                                                    record_augmentation=True,
                                                                                    verbose=verbose)
            # Store the aug_count and aug_record
            with open(os.path.join(aug_dir, 'aug_count.json'), 'w', encoding='utf8') as f:
                f.write(js.dumps(aug_count))
            with open(os.path.join(aug_dir, 'aug_record.json'), 'w', encoding='utf8') as f:
                f.write(js.dumps(aug_record))
            with open(os.path.join(aug_dir, 'aug_time.json'), 'w', encoding='utf8') as f:
                f.write(js.dumps({
                    'augmentation_duration': aug_duration
                }))

            augmented_train_file = os.path.join(aug_dir, 'train_' + strategy.name + '_augm.csv')
            aug_df = pm4py.convert_to_dataframe(aug_event_log)
            # pm4py.write_xes(aug_event_log, augmented_train_file)
            aug_df.to_csv(augmented_train_file, index=False)
            augmented_files[strategy.name] = augmented_train_file
