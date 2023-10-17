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

## Usage

### Data Processing: Demo

To pre-precess the demo sequencing data, run:
```
cd data
python ../src/pre_process.py demo pre -r -c -p 2
python ../src/pre_process.py demo post -r -c -p 2
```
Pre-processing the 'pre-' and 'post-' selection demo data takes ~0.8 sec each on a standard computer. The expected outputs of these two steps the following four Pandas DataFrames in CSV file format:
 * `reads/demo_pre_reads.csv`
 * `reads/demo_post_reads.csv`
 * `counts/demo_pre_counts.csv`
 * `counts/demo_post_counts.csv`

### Modeling

Given a pre-processed library, a supervised model can be trained using a command of the form:
```
python src/keras_models_run.py library_name model_type
```
where `model_type` specifies whether a linear (`'linear'`) or neural network (`'ann'`) model is trained. Information about additional arguments that may be specified (such as model hyperparameters) can be obtained by running
```
python src/keras_models_run.py --help
```
On a standard computer, it takes ~10 minute to train a standard linear model (CPU-only) on an AAV5 insertion library, and around ~30 minutes to train a reasonable-sized neural network model (CPU-only) on the same library, although train times for the latter are highly hyperparameter-dependent. The expected outputs of a training run are:
 * a saved Keras model
 * `{modelname}_test_pred.npy` containing a numpy array of model test predictions

### Entropy Optimization

Given a trained Keras model, entropy optimization can be performed using a command of the form:
```
python src/entropy_opt.py model_path save_path --min_lambda 1 --max_lambda 2.5 --num_lambda 150 --num_iter 3000 --learning_rate 0.01 --num_samples 1000
```
Information about additional arguments can be obtained by running
```
python src/entropy_opt.py --help
```
On a standard computer, it takes ~30 minutes to perform the optimization (CPU-only) for a single `lambda` value. The expected output of an `entropy_opt.py` run is a `{save_path}.npy` file which contains a dictionary containing metadata about the optimization run and, for each `lambda` value, a tuple `(entropy, mean_predict_log_enrichment, optimized_distribution)`.

### Extracting designed library probabilities

The position-wise nucleotide probabilities required to synthesize any of the libraries designed in [our manuscript](https://www.biorxiv.org/content/10.1101/2021.11.02.467003v2) are stored in `results` and can be easily extracted using the utility provided in `opt_analysis`. For example,
```
libraries, metadata = opt_analysis.load_designed_library_probabilities('../results/opt_results_nuc_1.npy')
libraries[0.25]
```
outputs the position-wise nucleotide probabilities for the library designed using our ML-guided approach with the diversity trade-off parameter `lambda=0.25`.