import sys
import os
import argparse
sys.path.append("../src")
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from seqtools import SequenceTools
import seaborn as sns
from matplotlib import rcParams, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Define plot aesthetics.
plt.style.use('seaborn-deep')
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


# Determine data location and libraries to include on plots.
parser = argparse.ArgumentParser()
parser.add_argument('library_names', help='names of libraries to include on plots', nargs='+', type=str)
parser.add_argument('data_directory', help='full path to directory containing counts files', type=str)
args = parser.parse_args()

data_dir = '../data/counts/' if args.data_directory is None else args.data_directory

names_to_labels = {'lib_b': 'Library D2', 'lib_c': 'Library D3', 'old_nnk': 'NNK'}
names = ['lib_b', 'lib_c', 'old_nnk'] if args.library_names is None else args.library_names
labels = [names_to_labels[name] for name in names]

include_cell_specific = False

aa_order = [k.upper() for k in SequenceTools.protein2codon_.keys()]


def load_counts(filename):
    counts_file = os.path.join(data_dir, 'counts', filename)
    counts = pd.read_csv(counts_file)
    counts = counts.loc[~counts['seq'].str.contains('X')]
    counts = counts.reset_index()
    return counts


def calc_entropy(counts_df):
    probs = np.array(counts_df['count'] / counts_df['count'].sum())
    entropy = -np.sum(probs*np.log(probs))
    return entropy


def calc_avg_pairwise_dist(counts_df):
    result = 0.
    aas = aa_order
    for i in range(len(counts_df['seq'][0])):
        for aa in aas:
            seq_i = counts_df['seq'].str.get(i)
            result += counts_df.loc[(seq_i == aa), 'count'].sum() ** 2
    result = len(counts_df['seq'][0]) - result / (counts_df['count'].sum() ** 2)
    return result


def calc_marginal_counts(counts_df, unique_seq=False):
    aas = aa_order
    result = np.zeros((len(counts_df['seq'][0]), len(aas)))
    for i in range(result.shape[0]):
        seq_i = counts_df['seq'].str.get(i)
        for j, aa in enumerate(aas):
            if unique_seq:
                result[i][j] = np.sum(1 * (seq_i == aa))
            else:
                result[i][j] = counts_df.loc[(seq_i == aa), 'count'].sum()
    return result


def calc_marginal_distributions(counts_df, unique_seq=False):
    result = calc_marginal_counts(counts_df, unique_seq)
    result = result / np.sum(result, axis=1).reshape(len(counts_df['seq'][0]), 1)
    return result


def calc_marginal_entropies(counts_df):
    p = calc_marginal_distributions(counts_df)
    p_ma = np.ma.masked_where(p==0, p)
    logp = np.log(p_ma)
    H = -np.sum(p_ma * logp, axis=1)
    return H


ents_df = defaultdict(list)
for name in names:
    i = names.index(name)
    lbl = labels[i]
    for s in ['pre', 'post']:
        fname = "%s_%s_counts.csv" % (name, s)
        counts = load_counts(fname)
        ent = calc_entropy(counts)
        marginal_ent = calc_marginal_entropies(counts)
        marginal_dist = calc_marginal_distributions(counts, unique_seq=True)
        ents_df['Library'].append(lbl)
        c = 'initial library' if s == 'pre' else 'packaging selection'
        ents_df['Condition'].append(c.capitalize())
        ents_df['Entropy'].append(ent)
        ents_df['Marginal Distributions'].append(marginal_dist)
        ents_df['Marginal Entropies'].append(marginal_ent)
    
    # Incorporate post-infection data where possible.
    if name in ['lib_b', 'old_nnk']:
        cs = ['brain', 'neuron', 'microglia', 'glia'] if include_cell_specific else ['brain']
        for c in cs:
            fname = '%s_%s_post_counts.csv' % (c, name.split('_')[1])
            counts = load_counts(fname)
            ent = calc_entropy(counts)
            marginal_ent = calc_marginal_entropies(counts)
            marginal_dist = calc_marginal_distributions(counts, unique_seq=True)
            ents_df['Library'].append(lbl)
            ents_df['Condition'].append('%s infection' % c.capitalize())
            ents_df['Entropy'].append(ent)
            ents_df['Marginal Distributions'].append(marginal_dist)
            ents_df['Marginal Entropies'].append(marginal_ent)

ents_df = pd.DataFrame(data=ents_df)
ents_df['Effective N'] = np.exp(ents_df['Entropy'])
print(ents_df)


# Entropy bar plot.
fig, ax = plt.subplots(figsize=(6, 3))
sns.barplot(x='Library', y='Entropy', hue='Condition', data=ents_df, order=labels, palette='colorblind', ax=ax)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set_xlabel('')
ax.grid(False)
plt.tight_layout()
plt.savefig('plots/library_entropies.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
plt.close()


# Entropy dot plot.
fig, ax = plt.subplots(figsize=(3, 3))
colors = sns.color_palette('colorblind', n_colors=3)
for i, nm in enumerate(names):
    lbl = labels[i]
    ent_pre = ents_df.loc[(ents_df['Library'] == lbl) & (ents_df['Condition'] == 'Initial library'), 'Entropy'].values
    ent_post = ents_df.loc[(ents_df['Library'] == lbl) & (ents_df['Condition'] == 'Packaging selection'), 'Entropy'].values
    if nm in ['lib_b', 'old_nnk']:
        ent_infection = ents_df.loc[(ents_df['Library'] == lbl) & (ents_df['Condition'] == 'Brain infection'), 'Entropy'].values
    if i == 0:
        ax.scatter([i], [ent_pre], c=colors[0], label='Initial library', s=30)
        ax.scatter([i], [ent_post], c=colors[1], label='Packaging selection', s=30)
    else:
        ax.scatter([i], [ent_post], c=colors[1], s=30)
        ax.scatter([i], [ent_pre], c=colors[0], s=30)
#     if nm == 'old_nnk':
#         ax.scatter([i], [ent_infection], c=colors[2], s=30)
#         ax.scatter([i], [ent_post], edgecolor='k', facecolor='none', s=40)
#     if nm == 'lib_b':
#         ax.scatter([i], [ent_infection], c=colors[2], label='Brain infection', s=30)
#         ax.scatter([i], [ent_pre], edgecolor='k', facecolor='none', s=40)
ax.set_ylabel("Entropy")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, ha='center', fontsize=8)
ax.legend(fontsize='small')
ax.set_title('b', fontfamily='serif', loc='left', fontsize='medium')
ax.grid(False)
plt.tight_layout()
plt.savefig('plots/library_entropies_dots.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
plt.close()


# Figure 6.
fig = plt.figure(constrained_layout=True, figsize=(8, 4))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
heatmap_cmap = sns.color_palette("Blues", as_cmap=True)
ax = ax1
inf_df = ents_df[ents_df['Condition'] == 'Brain infection'].reset_index(drop=True)
sns.barplot(x='Library', y='Effective N', hue='Condition', data=inf_df, order=labels, palette='colorblind', ax=ax)
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.get_legend().remove()
ax.set_xlabel('')
ax.set_ylabel('Effective Number of Distinct Sequences', fontsize=16)
plt.setp(ax.get_xticklabels(), Fontsize=16)
plt.setp(ax.get_yticklabels(), Fontsize=16)
ax.grid(False)

ax = ax2
marginal_dist = ents_df[(ents_df['Library'] == 'Library D2') & (ents_df['Condition'] == 'Brain infection')].reset_index(drop=True)['Marginal Distributions'][0]
vmax = 1 # vmax = np.amax(marginal_counts)
im = ax.imshow(marginal_dist, vmin=0, vmax=vmax, cmap=heatmap_cmap, aspect=1)
#ax.text(-4.5, 6.5, 'b', fontsize=20)
ax.set_yticks(np.arange(7))
ax.set_yticklabels(range(7, 0, -1), fontsize=8)
ax.set_xticks(np.arange(len(aa_order)))
ax.set_xticklabels(aa_order)
ax.set_yticks(np.arange(-.5, 7.5, 1), minor=True)
ax.set_xticks(np.arange(-.5, len(aa_order)+0.5, 1), minor=True)
ax.grid(which='minor', color='grey', linestyle='-', linewidth=1, alpha=0.2)
ax.grid(which='major', color='none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_ylim([-0.5, 6.5])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='4%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, use_gridspec=True)
cbar.set_ticks([0, 0.5*vmax, vmax])
ax.set_ylabel('Position', fontsize=14)
cbar.set_label('Empirical Probability', rotation=270, labelpad=14, fontsize=12)
ax.set_xlabel('Amino Acid', fontsize=14)
ax.set_title('Brain Infection', fontsize=14)

ax = ax3
marginal_dist = ents_df[(ents_df['Library'] == 'Library D2') & (ents_df['Condition'] == 'Packaging selection')].reset_index(drop=True)['Marginal Distributions'][0]
im = ax.imshow(marginal_dist, vmin=0, vmax=vmax, cmap=heatmap_cmap, aspect=1)
#ax.text(-4.5, 6.5, 'c', fontsize=20)
ax.set_yticks(np.arange(7))
ax.set_yticklabels(range(7, 0, -1), fontsize=8)
ax.set_xticks(np.arange(len(aa_order)))
ax.set_xticklabels(aa_order)
ax.set_yticks(np.arange(-.5, 7.5, 1), minor=True)
ax.set_xticks(np.arange(-.5, len(aa_order)+0.5, 1), minor=True)
ax.grid(which='minor', color='grey', linestyle='-', linewidth=1, alpha=0.2)
ax.grid(which='major', color='none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_ylim([-0.5, 6.5])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='4%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, use_gridspec=True)
cbar.set_ticks([0, 0.5*vmax, vmax])
ax.set_ylabel('Position', fontsize=14)
cbar.set_label('Empirical Probability', rotation=270, labelpad=14, fontsize=12)
ax.set_xlabel('Amino Acid', fontsize=14)
ax.set_title('Packaging Selection')

plt.tight_layout()
plt.savefig('plots/figure_6.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
plt.close()


# NNK Heatmaps.
fig = plt.figure(constrained_layout=True, figsize=(8, 4))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])
ax = ax3
marginal_dist = ents_df[(ents_df['Library'] == 'NNK') & (ents_df['Condition'] == 'Packaging selection')].reset_index(drop=True)['Marginal Distributions'][0]
im = ax.imshow(marginal_dist, vmin=0, vmax=vmax, cmap=heatmap_cmap, aspect=1)
#ax.text(-4.5, 6.5, 'c', fontsize=20)
ax.set_yticks(np.arange(7))
ax.set_yticklabels(range(7, 0, -1), fontsize=8)
ax.set_xticks(np.arange(len(aa_order)))
ax.set_xticklabels(aa_order)
ax.set_yticks(np.arange(-.5, 7.5, 1), minor=True)
ax.set_xticks(np.arange(-.5, len(aa_order)+0.5, 1), minor=True)
ax.grid(which='minor', color='grey', linestyle='-', linewidth=1, alpha=0.2)
ax.grid(which='major', color='none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_ylim([-0.5, 6.5])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='4%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, use_gridspec=True)
cbar.set_ticks([0, 0.5*vmax, vmax])
ax.set_title('Packaging Selection', fontsize=14)
ax.set_ylabel('Position', fontsize=14)
cbar.set_label('Empirical Probability', rotation=270, labelpad=14, fontsize=12)
ax.set_xlabel('Amino Acid', fontsize=14)

ax = ax2
marginal_dist = ents_df[(ents_df['Library'] == 'NNK') & (ents_df['Condition'] == 'Brain infection')].reset_index(drop=True)['Marginal Distributions'][0]
im = ax.imshow(marginal_dist, vmin=0, vmax=vmax, cmap=heatmap_cmap, aspect=1)
#ax.text(-4.5, 6.5, 'c', fontsize=20)
ax.set_yticks(np.arange(7))
ax.set_yticklabels(range(7, 0, -1), fontsize=8)
ax.set_xticks(np.arange(len(aa_order)))
ax.set_xticklabels(aa_order)
ax.set_yticks(np.arange(-.5, 7.5, 1), minor=True)
ax.set_xticks(np.arange(-.5, len(aa_order)+0.5, 1), minor=True)
ax.grid(which='minor', color='grey', linestyle='-', linewidth=1, alpha=0.2)
ax.grid(which='major', color='none')
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
ax.set_ylim([-0.5, 6.5])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='4%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, use_gridspec=True)
cbar.set_ticks([0, 0.5*vmax, vmax])
ax.set_title('Brain Infection', fontsize=14)
ax.set_ylabel('Position', fontsize=14)
cbar.set_label('Empirical Probability', rotation=270, labelpad=14, fontsize=12)
ax.set_xlabel('Amino Acid', fontsize=14)

plt.tight_layout()
plt.savefig('plots/nnk_heatmaps.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
plt.close()