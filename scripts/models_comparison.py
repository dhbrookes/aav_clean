import sys
import os
sys.path.append("../src")
from tensorflow import keras
import modeling
import data_prep
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import pandas as pd
from functools import reduce

from matplotlib import rcParams
rcParams['figure.dpi'] = 200
rcParams['savefig.dpi'] = 300
rcParams['lines.linewidth'] = 1.0
rcParams['axes.grid'] = True
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
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
    fig, ax = plt.subplots(figsize=(3, 3))
    for lbl in model_lbls:
        cs = cs_dict[lbl]
        prms = plot_params[lbl]
        ax.plot(fracs, cs, lw=2, **prms)
    ax.set_xlabel("K (Fraction of top test sequences)")
    ax.set_ylabel("Pearson correlation")
    fracs2 = [fracs[i] for i in range(len(fracs)) if i % 10 == 0]
    ax.set_xticks(fracs2)
    ax.set_xticklabels(["%.1f" % (1-f) for f in fracs2])
#     ax.legend(loc='lower left', fontsize=7)
    ax.legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.close()


def make_paired_plot(ypred, ytest, plot_params, plot_type='histplot', savefile=None):
    n_test = len(ypred)
    y_test = ytest[:n_test]
    sorted_test_idx = np.argsort(y_test)
    idx = sorted_test_idx[0:]
    ypred = ypred[idx]
    ytest = ytest[idx]
    fig, ax = plt.subplots(figsize=(1.75, 1.75))
    if plot_type == 'histplot':
        sns.histplot(x=ypred, y=ytest, stat='density', color=plot_params['c'], edgecolor='none', bins=50, cbar=True, cbar_kws=dict(shrink=.5), ax=ax, pmax=0.1)
    elif plot_type == 'scatter':
        ax.scatter(ypred, ytest, s=5, alpha=0.2, edgecolor='none', **plot_params)
        ax.set_xlim(ax.get_xlim()[0] - 0.25, ax.get_xlim()[1] + 0.25)
    ax.set_xlabel("\"{}\" Predicted Log Enrichment".format(plot_params['label']), fontsize=7)
    ax.xaxis.set_tick_params(labelsize=7)
    ax.set_ylabel("Observed Log Enrichment", fontsize=7)
    ax.yaxis.set_tick_params(labelsize=7)
    ax.figure.axes[-1].yaxis.set_tick_params(labelsize=6)
    sns.regplot(x=ypred, y=ytest, scatter=False, ci=None, color='k', line_kws={'lw': 1, 'ls':'--'}, ax=ax)
    r, p = pearsonr(ypred, ytest)
    plt.figtext(0.6, 0.3, 'Pearson={:0.3f} (p={:0.2f})'.format(r, p), ha='center', fontsize=5, color='k') #bbox={'facecolor': 'gray', 'alpha': 0.3, 'pad': 5}
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile, dpi=300)
    plt.close()


def make_histogram(preds, plot_params, n_bins='auto', savefile=None):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    sns.histplot(x=preds, color=plot_params['c'], ax=ax, legend=False,
                 bins=n_bins, log_scale=(False, True), edgecolor='none', linewidth=0)
    ax.set_xlabel("\"{}\" Predicted Log Enrichment".format(plot_params['label']), fontsize=10)
    plt.tight_layout()
    if savefile is not None:
        plt.savefig(savefile, dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
    plt.close()
    
    
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
                              pre_file ="/storage/akosua/aav_clean/data/counts/old_nnk_pre_counts_old.csv",   # these older files contain the correctly index test values
                              post_file="/storage/akosua/aav_clean/data/counts/old_nnk_post_counts_old.csv",
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
    # Make paired plot comparing model predictions to observed log-enrichment.
    make_paired_plot(ypred, y_test, plot_params[lbl], savefile="plots/%s_%s_paired_plot.png" % (lib, lbl))
    # Make paired plot comparing observed and predicted log-enrichment of 5 sampled variants.
    q_idx = np.argsort(ypred)[[int(q * (len(ypred) - 1)) for q in [0, 0.25, 0.5, 0.75, 1.0]]]
    make_paired_plot(ypred[q_idx], y_test[q_idx], plot_params[lbl],
                     plot_type='scatter', savefile="plots/%s_%s_paired_plot_quantile_sample.png" % (lib, lbl))
    # Make histogram of model predictions.
    make_histogram(ypred, plot_params[lbl], savefile="plots/%s_%s_predictions_histogram.png" % (lib, lbl))

# Make plot comparing models with weighted loss
make_culled_corr_plot(['ann_100_is', 'ann_200_is', 'ann_500_is', 'ann_1000_is',
                       'linear_is', 'linear_neighbors', 'linear_pairwise'],
                        fracs, culled_spear, plot_params, savefile="plots/%s_update_models_comparison.png" % lib)
                       
# Make plot comparing models with weighted and unweighted losses                   
make_culled_corr_plot(['ann_100_is', 'ann_100_is_unweighted', 'linear_pairwise', 'linear_pairwise_unweighted'], 
                       fracs, culled_spear, plot_params, 
                       savefile="plots/%s_update_loss_comparison.png" % lib)
