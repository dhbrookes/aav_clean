import sys
sys.path.append("../src")
import numpy as np
import pandas as pd
import pre_process
from Bio.Seq import Seq
import opt_analysis
import entropy_opt
import seaborn as sns
import data_prep
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from matplotlib.ticker import NullFormatter
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


def calculate_dist(s1, s2):
    return np.sum([1 if si1 != si2 else 0 for si1, si2 in zip(s1, s2)])


def calc_edist_aa_from_seqs(seqs_int, aa=False):
    """
    Calculate the average amino acid distance between a set of sequences
    represented as integers
    """
    seqs_str = []
    N, L = seqs_int.shape
    for i in range(N):
        si = seqs_int[i]
        if aa:
            ss = "".join([pre_process.AA_ORDER[i] for i in si])
        else:
            nss =  "".join([pre_process.NUC_ORDER[i] for i in si])
            ss = str(Seq(nss).translate())
        seqs_str.append(ss)
        
    num = 0
    ed = 0
    for i in range(N):
        for j in range(i+1, N):
            ed += calculate_dist(seqs_str[i], seqs_str[j])
            num += 1
    return ed / num


def calc_sampling_stats_for_plotting(results_dict, savefile=None, aa=False):
    """
    Calculates the relevant plotting quantities given a results dictionary.
    """
    stats = ['expected_aa_dist']
    plot_data = {s: [] for s in stats}
    plot_data['mean_enrichment'] = []
    plot_data['lambda'] = []
    if savefile is not None:
        plot_data['savefile'] = []
    
    lambdas = results_dict.keys()
    for l in lambdas:
        sequences, energies = results_dict[l]
        plot_data['lambda'].append(l)
        plot_data['mean_enrichment'].append(np.mean(energies))
        if savefile is not None:
            plot_data['savefile'].append(savefile)
        for s in stats:
            if s == 'expected_aa_dist':
                val = calc_edist_aa_from_seqs(sequences, aa=aa)
            plot_data[s].append(val)
    return plot_data


def load_sampling_data_for_plotting(savefile):
    """
    Loads data from sampling run and calculate desired plotting statistics.
    """
    results = np.load(savefile, allow_pickle=True).item()
    meta_data = results.pop('meta')
    aa = meta_data['aa']
    plot_data = calc_sampling_stats_for_plotting(results, aa=aa, savefile=savefile)
    return plot_data, meta_data, plot_data


fancy_labels = {'expected_aa_dist': "Expected Pairwise Distance (Amino Acids)",
                'expected_nuc_dist': "Expected Pairwise Distance (Nucleotides)",
                'nuc_entropy': 'Entropy (nucletides)',
                'aa_entropy': 'Entropy (amino acids)',
                'mean_enrichment': 'Mean predicted enrichment'}

# Load optimization results
all_results = {}
include = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]
plot_data = None
meta_data = None
for i in include:
    fi = "../results/opt_results_nuc_%i.npy" % i
    pdi, mdi, results_i = opt_analysis.load_data_for_plotting(fi)
    all_results[fi] = results_i
    if meta_data is None:
        meta_data = mdi
    if plot_data is None:
        plot_data = pdi
    for key in plot_data.keys():
        plot_data[key] += pdi[key]
        
# Calculate plotting statistics for NNK
if meta_data['encoding'] == 'pairwise':
    enc = data_prep.encode_one_plus_pairwise
elif meta_data['encoding'] == 'is':
    enc = data_prep.one_hot_encode
elif meta_data['encoding'] == 'neighbors':
    enc = data_prep.encode_one_plus_neighbors
model_path = meta_data['model_path']
if model_path == "../models/aav5_ann_100_is":
    model_path = "../models/old_nnk_ann_100_is"
nnk_stats = opt_analysis.calculate_nnk_stats(model_path, enc, n_samples=10000)

# Load sampling results
all_samp_results = {}
include = [1, 2, 3]
samp_plot_data = None
samp_meta_data = None
for i in include:
    fi = "../results/sampling_results_nuc_%i.npy" % i
    pdi, mdi, results_i = load_sampling_data_for_plotting(fi)
    all_results[fi] = results_i
    if samp_meta_data is None:
        samp_meta_data = mdi
    if samp_plot_data is None:
        samp_plot_data = pdi
    for key in samp_plot_data.keys():
        samp_plot_data[key] += pdi[key]
        
# Make plot
fig, ax = plt.subplots(figsize=(4, 4))
x_key='expected_aa_dist'
y_key='mean_enrichment'
sc = ax.scatter(plot_data[x_key], plot_data[y_key], 
                edgecolor='none', s=20, label='Combinatorial', alpha=0.5
                )

sc = ax.scatter(samp_plot_data[x_key], samp_plot_data[y_key], 
                edgecolor='none', s=20, label='Specified', alpha=0.5
                )

ax.scatter(nnk_stats[x_key], nnk_stats[y_key], c='k', marker='x', label='NNK')
ax.set_xlabel(fancy_labels[x_key], fontsize=12)
ax.set_ylabel(fancy_labels[y_key], fontsize=12)
ax.legend()
ax.set_xlim([0, 7])
plt.savefig('plots/specified_vs_combinatorial.png', dpi=300, facecolor='white', transparent=False)