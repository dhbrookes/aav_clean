import sys
import os
import argparse
sys.path.append("../src")
import data_prep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rcParams


# Define plot aestherics.
plt.style.use('seaborn-deep')
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


# Determine data location and libraries to include on plots.
parser = argparse.ArgumentParser()
parser.add_argument('--library_names', help='names of libraries to include on plots', nargs='+', type=str)
parser.add_argument('--data_directory', help='full path to directory containing counts files', type=str)
parser.add_argument('--prop_threshold', help='top proportion of variants to be colored', default=0.8, type=float)
args = parser.parse_args()

data_dir = '../data' if args.data_directory is None else args.data_directory

names_to_labels = {'lib_b': 'Library D2', 'lib_c': 'Library D3', 'old_nnk': 'NNK'}
names = ['old_nnk', 'lib_b'] if args.library_names is None else args.library_names
labels = [names_to_labels[name] for name in names]

include_cell_specific = False


def get_counts_path(filename):
    return os.path.join(data_dir, 'counts', filename)


def load_counts(filename):
    counts_file = get_counts_path(filename)
    counts = pd.read_csv(counts_file)
    counts = counts.loc[~counts['seq'].str.contains('X')]
    counts = counts.reset_index()
    return counts


def merge_counts(pre_df, post_df):
    merged_df = pd.merge(pre_df, post_df, how='outer', on='seq', suffixes=('_pre', '_post')).fillna(0)
    merged_df = merged_df.rename(columns={'count_pre': 'count'})
    merged_df = merged_df.reset_index()
    merged_df.sort_values('count', ascending=False, inplace=True)
    return merged_df


def get_frequencies(counts, pseudocount=1):
    f = np.array(counts) + pseudocount
    return f / f.sum()


def get_top_fraction(counts, threshold, pseudocount=1):
    i_sorted = np.argsort(np.array(counts))[::-1]
    cum_freq = np.array(counts)[i_sorted] + pseudocount
    cum_freq = np.cumsum(cum_freq) / np.sum(cum_freq)
    top_fraction = np.zeros_like(cum_freq)
    top_fraction[i_sorted] = cum_freq <= threshold
    return top_fraction.astype(bool)


def get_n_distinct(counts, threshold, pseudocount=1):
    cum_freq = np.sort(np.array(counts) + pseudocount)[::-1]
    cum_freq = np.cumsum(cum_freq) / np.sum(cum_freq)
    return np.argmin(cum_freq <= threshold)


def get_count_annotation_by_color(counts_df, color_name, x_fn=np.mean, y_fn=np.amax):
    n = (counts_df['color'] == color_name).sum()
    x = x_fn(counts_df.loc[counts_df['color'] == color_name, 'freq_pack'].values)
    y = y_fn(counts_df.loc[counts_df['color'] == color_name, 'freq_brain'].values)
    return n, x, y


def sci_notation(n, sig_fig=2):
    if n > 9999:
        fmt_str = '{0:.{1:d}e}'.format(n, sig_fig)
        n, exp = fmt_str.split('e')
        return r'${n:s} \times 10^{{{e:d}}}$'.format(n=n, e=int(exp))
    return str(n)


# Visualize effects of selection.
# Visualization inspired by https://insight.jci.org/articles/view/135112/pdf
fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 3), sharex=True, sharey=True)
p_threshold = args.prop_threshold
for i, name in enumerate(names):
    counts_df = load_counts('%s_post_counts.csv' % name)
    brain_df = load_counts('brain_%s_post_counts.csv' % name.split('_')[1])
    counts_df = merge_counts(counts_df, brain_df)
    counts_df = counts_df.rename(columns={'count_post': 'count_brain'})
    counts_df['freq_pack'] = get_frequencies(counts_df['count'])
    counts_df['freq_brain'] = get_frequencies(counts_df['count_brain'])
    counts_df['color'] = '_lightgrey'
    pack_condition = get_top_fraction(counts_df['count'], p_threshold)
    print('{} {} has {:.3e} ({:.3e} fraction) in top {} fraction'.format(labels[i], 'packaging', np.sum(pack_condition), np.mean(pack_condition), p_threshold))
    brain_condition = get_top_fraction(counts_df['count_brain'], p_threshold, pseudocount=0)
    print('{} {} has {:.3e} ({:.3e} fraction) in top {} fraction'.format(labels[i], 'brain', np.sum(brain_condition), np.mean(brain_condition), p_threshold))
    counts_df.loc[pack_condition, 'color'] = 'blue'
    counts_df.loc[brain_condition, 'color'] = 'yellow'
    pack_brain_condition = np.logical_and(pack_condition, brain_condition)
    print('{} {} has {:.3e} ({:.3e} fraction) in top {} fraction'.format(labels[i], 'BOTH brain & packaging', np.sum(pack_brain_condition), np.mean(pack_brain_condition), p_threshold))
    counts_df.loc[pack_brain_condition, 'color'] = 'green'
    ax = axes[i]
    legend = (i == 1)
    splot = sns.scatterplot(x='freq_pack', y='freq_brain', hue='color', data=counts_df, palette=['lightgrey', 'blue', 'gold', 'green'], hue_order=['_lightgrey', 'blue', 'yellow', 'green'], alpha=.2, legend=legend, ax=ax, linewidth=0, edgecolor='none', s=20)
    splot.set(xscale='log', yscale='log')
    ax.set_xlabel('Post-Packaging Frequency')
    ax.set_ylabel('Post-Infection Frequency')
    ax.set_title('{}'.format(labels[i]))
    n_blue, x_blue, y_blue = get_count_annotation_by_color(counts_df, 'blue', np.amax, np.mean)
    n_yellow, x_yellow, y_yellow = get_count_annotation_by_color(counts_df, 'yellow', np.mean, np.amax)
    n_green, x_green, y_green = get_count_annotation_by_color(counts_df, 'green', np.mean, np.amax)
    ax.annotate(sci_notation(n_blue), (x_blue, y_blue), textcoords='offset pixels', xytext=(0, 10), fontsize=10, color='blue', ha='center')
    ax.annotate(sci_notation(n_yellow), (x_yellow, y_yellow), textcoords='offset pixels', xytext=(0, 6), fontsize=10, color='gold', ha='center')
    ax.annotate(sci_notation(n_green), (x_green, y_green), textcoords='offset pixels', xytext=(0, 3), fontsize=10, color='green', ha='center')
    ax.grid(False)
    print('{} has {:.3e} blue, {:.3e} yellow, and {:.3e} green'.format(labels[i], n_blue, n_yellow, n_green))
ls = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.).get_texts()
ls[0].set_text('Most prevalent post-packaging')
ls[1].set_text('Most prevalent post-infection')
ls[2].set_text('Overlap in prevalent variants')
plt.setp(ls, fontsize='12')
plt.savefig('plots/selection_scatterplots_{}.png'.format(p_threshold), dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
plt.close()
