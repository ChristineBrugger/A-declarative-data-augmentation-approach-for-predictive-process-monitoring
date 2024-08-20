# A declarative data augmentation approach for predictive process monitoring

## Introduction
This is the support code of the master thesis "A declarative data augmentation approach for predictive process monitoring."

![augmentation approach](https://github.com/user-attachments/assets/6c6fb37b-e013-4c95-992d-86c2e648bed9)

The codes and experiments are adapted from the work of Efren Rama-Maneiro, Juan Vidal and Manuel Lama: [_Deep Learning for Predictive Business Process Monitoring: Review and Benchmark_]([url](https://gitlab.citius.gal/efren.rama/pmdlcompararator))

We used their implementation of the following three approaches and just adapted it at some parts to efficiently use it in our study. Therefore we only kept the necessary parts for our experiments.

## Implemented approaches

| Author | Paper | Original Code repository |
| --------------- | --------------- | --------------- |
| Tax et al.    | [Link]([url](https://arxiv.org/abs/1612.02130))     | [Code]([url](https://github.com/verenich/ProcessSequencePrediction))     |
| Mauro et al.    | [Link]([url](https://openreview.net/forum?id=OxYPkm8nGEq))     | [Code]([url](https://github.com/nicoladimauro/nnpm))     |
| Bukhsh et al.    | [Link]([url](https://arxiv.org/abs/2104.00721))    | [Code]([url](https://github.com/Zaharah/processtransformer))     |

## Setup
In order to execute our scripts we provide the necessary environment data in _decl_data_augm.yml_ This envirionment can be used for our declarative data augmentation approach as well as the data preprocessing and the analysis of the test results. As we need a different pm4py version to execute the codes of the augmentation baseline (https://github.com/mkaep/pbpm-ssl-suite) we also need a different environment. This environment is defined at the _csbdeep.yml_ file.


## Prepare datasets

In order to prepare the augmented datasets we provide the necessary Jupyter Notebooks:
* _Generate Baseline Data_: Notebook to generate the baseline data based on the work of Martin Käppel and Stefan Jablonski _Model-Agnostic Event Log Augmentation for Predictive Process Monitoring_ with the source code https://github.com/mkaep/pbpm-ssl-suite
* _Declarative Data Augmentation_: Notebook to apply our declarative data augmentation approach.
  * Note: We provide the datasets that we used for the predictive process monitoring (PPM) approaches and for which we also documented the results in our thesis. If this notebook is executed for the same original datasets, our generated data wil be overwritten as with every execution a new synthetic dataset will be created.

## Preprocess datasets
In order to preprocess the datasets for the PPM approaches, please execute the provided Notebook: _Preprocess Data_

## Run the experiments
The necessary steps for executing the experiments are taken from https://gitlab.citius.gal/efren.rama/pmdlcompararator and are just repeated here.

### Setup of environment
Use anaconda to install some dependencies and activate the environment:

    conda create -n "tf_2.0_ppm" python=3.6 tensorflow-gpu=2.1.0
    conda activate tf_2.0_ppm

Install additional dependencies:

    python -m pip install pm4py==1.2.12 hyperopt==0.2.3 jellyfish==0.7.2 distance==0.1.3 strsim==0.0.3 pyyaml==5.3.1 nltk==3.5 swifter==0.304 py4j==0.10.9

### Tax (tax)
Run the training procedure and next event prediction with the following command (inside the "code" folder). Each "--option" indicates the task to perform.

    python train.py --dataset ..\data\[FOLDER]\[DATASET] --train --test
    
Example:

	python train.py --dataset ..\data\helpdesk_orig\helpdesk_orig.csv --train --test


### Mauro (nnpm)
Run the experiments with the following command:

    python deeppm_act.py --dataset data/[DATASET] --train --test

Where fold_dataset refers to the split dataset and full dataset refers to the whole dataset. Example:

	python deeppm_act.py --dataset data/helpdesk_orig.csv --train --test

### Bukhsh (processtransformer)
As for this approach a new separate environment is needed (also mentioned in https://gitlab.citius.gal/efren.rama/pmdlcompararator) we adpated this. Unfortunately we could not execute the steps as mentioned, so we needed to setup the required environment based on the work of https://github.com/CSBDeep/CSBDeep/tree/main/extras#conda-environment

Therefore the following step is necessary to setup the required environment:

	conda env create -f https://raw.githubusercontent.com/CSBDeep/CSBDeep/main/extras/environment-gpu-py3.8-tf2.4.yml

As we also want to use this environment for the generation of the baseline data we further need to install the two following packages. All details about the environment can also be found at _csbdeep.yml_

 	pip install pm4py==2.2.20.1 click


Run the experimentation as follows

	python next_activity.py --dataset [DATASET] --epoch 100 --learning_rate 0.001

Example:

	python next_activity.py --dataset helpdesk_orig.csv --epoch 100 --learning_rate 0.001


## Analyze Results
To analyze the result, regarding the class-wise accuracy we provide the notebook _Analysis of Test Results_

