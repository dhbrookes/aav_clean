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
                'mean_enrichment': 'Mean predicted log enrichment'}

# Load results:
all_results = {}
include = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]
# include = [4, 11]
plot_data = None
meta_data = None
for i in include:
    print(i)
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
nnk_stats = opt_analysis.calculate_nnk_stats(model_path, enc, n_samples=int(1e4))
nsc_stats = opt_analysis.calculate_no_stop_codon_stats(model_path, enc, n_samples=int(1e4))


# Make panel
fig = plt.figure(constrained_layout=True, figsize=(8.5, 4.5))
gs = fig.add_gridspec(3, 2)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 1])
cm = plt.cm.get_cmap('plasma')
x_key='expected_aa_dist'
y_key='mean_enrichment'
scatter_cmap = sns.color_palette("flare", as_cmap=True)
heatmap_cmap = sns.color_palette("Blues", as_cmap=True)
color_vmax = 0.30
sc = ax1.scatter(plot_data[x_key], plot_data[y_key], 
                c=plot_data['lambda'], vmin=0, edgecolor='none',
                vmax=color_vmax, s=10, 
                cmap = scatter_cmap
                )
ax1.scatter(nnk_stats[x_key], nnk_stats[y_key], c='k', marker='x', label='NNK')
print('NNK', nnk_stats[x_key], nnk_stats[y_key])
ax1.scatter(nsc_stats[x_key], nsc_stats[y_key], c='c', marker='x', label='Filtered Uniform')
print('Filtered uniform', nsc_stats[x_key], nsc_stats[y_key])
ax1.text(-1, 5, 'a', fontsize=20)

chosen = [(1, 0.095, ax2, 'D1', 0.02, 0.2), (4, 0.12125, ax3, 'D2', 0.02, 0.2), (11, 0.53,ax4, 'D3', 0.15, 0.0)]
plot_lbls = ['b', 'c', 'd']

j = 0
for i, l, ax, lbl, pos1, pos2 in chosen:
    f = "../results/opt_results_nuc_%i.npy" % i
    chosen_dict = {l: all_results[f][l]}
    chosen_plot_data = opt_analysis.calc_stats_for_plotting(chosen_dict, aa=meta_data['aa'])
    ax1.scatter(chosen_plot_data[x_key], chosen_plot_data[y_key], 
                    edgecolor='k', s=20, 
                    linewidth=1, facecolors='none')
    print(lbl, chosen_plot_data[x_key], chosen_plot_data[y_key])

    ax1.text(chosen_plot_data[x_key][0]+0.02, chosen_plot_data[y_key][0]+0.2, lbl, fontsize=16)
    
    thetal = chosen_dict[l][2]
    pl = entropy_opt.normalize_theta(thetal)
    if not meta_data['aa']:
        aa_p = opt_analysis.aa_probs_from_nuc_probs(pl).T
    else:
        aa_p = pl
    aa_p = aa_p[::-1]
    im = ax.imshow(aa_p, vmin=0, vmax=1, cmap=heatmap_cmap, aspect=1)
    ax.text(-4.5, 6.5, plot_lbls[j], fontsize=20)
    # Major ticks
    ax.set_yticks(np.array(range(7)))
    ax.set_yticklabels(range(7, 0, -1), fontsize=12)
    ax.set_xticks(np.array(range(len(pre_process.AA_ORDER))))
    ax.set_xticklabels([a for a in pre_process.AA_ORDER], fontsize=12)
    ax.set_yticks(np.arange(-.5, 7.5, 1), minor=True)
    ax.set_xticks(np.arange(-.5, len(pre_process.AA_ORDER)+0.5, 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=1, alpha=0.2)
    ax.grid(which='major', color='none')

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    
    ax.set_ylim([-0.5, 6.5])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax, use_gridspec=True)
    cbar.set_ticks([0, 0.5, 1])
    if j == 0:
        ax.set_ylabel("Position", fontsize=14)
        cbar.set_label('Probability', rotation=270, labelpad=14, fontsize=12)
    if j == 2:
        ax.set_xlabel("Amino Acid", fontsize=14)
    j += 1

ax1.set_xlabel(fancy_labels[x_key], fontsize=14)
ax1.set_ylabel(fancy_labels[y_key], fontsize=14)
ax1.legend()
plt.setp(ax1.get_legend().get_texts(), fontsize='12')
ax1.set_xlim([0, 7])

cbar = fig.colorbar(sc, ax=ax1, aspect=50)
cbar.set_label("$\lambda$", rotation=0, fontsize=16)
ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
cbar.set_ticks(ticks)
ticks[-1] = '>%s' % color_vmax
cbar.set_ticklabels(ticks)
plt.savefig('plots/opt_curve.png', dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    
