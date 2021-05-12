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

fancy_labels = {'expected_aa_dist': "Expected Pairwise Distance (Amino Acids)",
                'expected_nuc_dist': "Expected Pairwise Distance (Nucleotides)",
                'nuc_entropy': 'Entropy (nucletides)',
                'aa_entropy': 'Entropy (amino acids)',
                'mean_enrichment': 'Mean predicted enrichment'}

# Load results:
all_results = {}
include = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]
# include = [4, 11]
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


# Make panel
fig, ax = plt.subplots(figsize=(5.5, 5))
x_key='expected_aa_dist'
y_key='mean_enrichment'
scatter_cmap = sns.color_palette("flare", as_cmap=True)
heatmap_cmap = sns.color_palette("Blues", as_cmap=True)
color_vmax = 0.30
sc = ax.scatter(plot_data[x_key], plot_data[y_key], 
                c=plot_data['lambda'], vmin=0, edgecolor='none',
                vmax=color_vmax, s=10, 
                cmap = scatter_cmap
                )
ax.scatter(nnk_stats[x_key], nnk_stats[y_key], c='k', marker='x', label='NNK')

chosen = [(4, 0.12125, 'b'), (11, 0.53,'c')]


j = 0
for i, l, lbl in chosen:
    f = "../results/opt_results_nuc_%i.npy" % i
    chosen_dict = {l: all_results[f][l]}
    chosen_plot_data = opt_analysis.calc_stats_for_plotting(chosen_dict, aa=meta_data['aa'])
    ax.scatter(chosen_plot_data[x_key], chosen_plot_data[y_key], 
                    edgecolor='k', s=20, 
                    linewidth=1, facecolors='none')

    ax.text(chosen_plot_data[x_key][0]+0.02, chosen_plot_data[y_key][0]+0.2, lbl, fontsize=16)
    j += 1

ax.set_xlabel(fancy_labels[x_key], fontsize=14)
ax.set_ylabel(fancy_labels[y_key], fontsize=14)
ax.legend()
ax.set_xlim([0, 7])

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("$\lambda$", rotation=0, fontsize=12)
ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
cbar.set_ticks(ticks)
ticks[-1] = '>%s' % color_vmax
cbar.set_ticklabels(ticks)
plt.savefig('plots/opt_curve.png', dpi=300)
    