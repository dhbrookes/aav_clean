import sys
import os
sys.path.append("../src")
from tensorflow import keras
import modeling
import data_prep
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from functools import reduce

from matplotlib import rcParams
rcParams['figure.dpi'] = 200
rcParams['savefig.dpi'] = 300
rcParams['lines.linewidth'] = 1.0
rcParams['axes.grid'] = True
rcParams['axes.spines.right'] = True
rcParams['axes.spines.top'] = True
rcParams['grid.color'] = 'gray'
rcParams['grid.alpha'] = 0.2
rcParams['axes.linewidth'] = 0.5
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')


"""
This script makes plots comparing predictive models 
based on the culled correlation metrics
"""


def calculate_culled_correlation(ypred, ytest, fracs):
    """
    Calculates the pearson correlation between predictions and true fitness values
    among subsets of test data. In particular, test data is succesively culled to only
    include the largest true fitness values.
    """
    spears = []
    n_test = len(ypred)
    y_test = ytest[:n_test]
    sorted_test_idx = np.argsort(y_test)
    for i in range(len(fracs)):
        num_frac = int(n_test * fracs[i])
        idx = sorted_test_idx[num_frac:]
        ypred_frac = ypred[idx]
        ytest_frac = ytest[idx]
        spear = pearsonr(ypred_frac, ytest_frac)[0]
        spears.append(spear)
    return spears


def make_culled_corr_plot(model_lbls, fracs, cs_dict, plot_params, savefile=None):
    fig, ax = plt.subplots()
    for lbl in model_lbls:
        cs = cs_dict[lbl]
        prms = plot_params[lbl]
        ax.plot(fracs, cs, marker=None,ms=1,lw=1, **prms)
    ax.set_xlabel("Fraction of top test sequences")
    ax.set_ylabel("Pearson correlation")
    fracs2 = [fracs[i] for i in range(len(fracs)) if i % 10 == 0]
    ax.set_xticks(fracs2)
    ax.set_xticklabels(["%.1f" % (1-f) for f in fracs2])
    ax.legend()
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile, dpi=300)
    plt.show()
    
    
# Define some plotting params for different models
cm = plt.cm.tab20c
plot_params = {'linear_is': {'label': 'Linear, IS', 'c':cm(0.11)},
              'linear_neighbors': {'label': 'Linear, Neighbors', 'c':cm(0.06)},
              'linear_pairwise': {'label': 'Linear, Pairwise', 'c':cm(0.01)},
              'linear_pairwise_unweighted': {'label': 'Linear, Pairwise (unweighted)', 'c':cm(0.01), 'ls':'--'},
              'ann_1000_is': {'label': 'NN, 1000', 'c': cm(0.21)},
              'ann_500_is': {'label': 'NN, 500', 'c': cm(0.26)},
              'ann_200_is': {'label': 'NN, 200', 'c': cm(0.31)},
              'ann_100_is': {'label': 'NN, 100', 'c': cm(0.36)},
              'ann_200_is_unweighted': {'label': 'NN, 200 (unweighted)', 'c': cm(0.31), 'ls': '--'},
              'ann_100_is_unweighted': {'label': 'NN, 100 (unweighted)', 'c': cm(0.31), 'ls': '--'},
              }
model_lbls = [k for k in plot_params.keys()]

# Load indices of test sequences and fitness values
lib = 'old_nnk'
test_idx = np.load("../models/%s_test_idx.npy" % lib)
data_df = data_prep.load_data(lib,   
                              pre_file ="../data/counts/old_nnk_pre_counts_old.csv",   # these older files contain the correctly index test values
                              post_file="../data/counts/old_nnk_post_counts_old.csv",
                              seq_column='aa_seq',
                              count_column='counts')
seqs, en_scores = data_prep.prepare_data(data_df)
X, y, weights = data_prep.featurize_and_transform(seqs, en_scores)
y_test = y[test_idx]
sorted_test_idx = np.argsort(y_test)


# Calculate culled correlation values
culled_spear = {}
fracs = np.arange(0, 1, 0.01)
for lbl in model_lbls:
    ypred = np.load("../models/%s_%s_test_pred.npy" % (lib, lbl))
    print(lbl)
    cs = calculate_culled_correlation(ypred, y_test, fracs)
    culled_spear[lbl] = cs
    
    
# Make plot comparing models with weighted loss
make_culled_corr_plot(['ann_100_is', 'ann_200_is', 'ann_500_is', 'ann_1000_is',
                       'linear_is', 'linear_neighbors', 'linear_pairwise'],
                        fracs, culled_spear, plot_params, savefile="plots/%s_update_models_comparison.png" % lib)
                       
# Make plot comparing models with weighted and unweighted losses                   
make_culled_corr_plot(['ann_100_is', 'ann_100_is_unweighted', 'linear_pairwise', 'linear_pairwise_unweighted'], 
                       fracs, culled_spear, plot_params, 
                       savefile="plots/%s_update_loss_comparison.png" % lib)