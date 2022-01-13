# Machine learning-based library design improves packaging and diversity of adeno-associated virus (AAV) libraries

## Overview

`aav_clean` contains Python sourcecode for training the supervised machine learning (ML) models in [our manuscript](https://www.biorxiv.org/content/10.1101/2021.11.02.467003v2) and for using these models to design AAV5 insertion libraries. We include a small demo file demonstrating the format of the raw experimental results for an insertion library, as well as the tools required for performing all pre-processing and filtering steps. We also include Python scripts for the evaluations and visualizations included in the paper.

## System Requirements

`aav_clean` can be run on a standard computer, and supports both GPU and CPU-only model training. The code has been tested on a Linux (Ubuntu 18.04.6 LTS) system with Python 3.7.9 and 3.9.5. Relevant Python dependencies include:
* BioPython (1.76+)
* matplotlib (3.3.2+)
* numpy (1.19.4+)
* pandas (1.1.4+)
* scikit-learn (0.22.1+)
* scipy (1.7.0+)
* seaborn (0.11.0+)
* tensorflow (2.0.0+)

## Installation

To install from GitHub:
```
git clone https://github.com/dhbrookes/aav_clean.git
cd aav_clean
```

## Demo
TODO:
* Instructions to run on data
* Expected Output
* Expected run time for demo on normal computer

## Instructions for use

TODO:
 * How to run softward on your data
