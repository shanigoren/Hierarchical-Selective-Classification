# Hierarchical-Selective-Classification

## Description
This repository contains the official implementation of the paper "Hierarchical Selective Classification" currently under review for UAI-2024.


## Setup

1. Clone the repository

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
## Usage

1. To evaluate hierarchical selective inference rules (Section 3 and 5 results) on all 1117 models run:
    ```bash
    python arch_comparison.py --data_dir  #path to the ImageNet validation folder#
    ```
    Additional arguments for different configurations such as evaluating a subset of the models are present in ```arch_comparison.py```.

2. To evaluate the optimal threshold algorithm (Section 4 results) on all 1117 models:
   First, the outputs of all evaluated models must be saved to the ```resources``` directory. This can be done by running ```arch_comparison.py``` with ```--save-y-scores``` as follows:
    ```bash
    python arch_comparison.py --save-y-scores --data_dir  #path to the ImageNet validation folder#
    ```
   After saving the desired results, run:
    ```bash
    python threshold_algorithm.py
    ```
    Additional arguments for different configurations such as evaluating a subset of the models are present in ```threshold_algorithm.py```.


