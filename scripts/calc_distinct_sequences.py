import sys
import os
import argparse
sys.path.append("../src")
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict


# Plot aesthetics.
plt.style.use('seaborn-deep')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.2
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'


# Determine data location and libraries to include on plots.
parser = argparse.ArgumentParser()
parser.add_argument('library_names', help='names of libraries to include on plots', nargs='+', type=str)
parser.add_argument('data_directory', help='full path to directory containing counts files', type=str)
args = parser.parse_args()

data_dir = '../data' if args.data_directory is None else args.data_directory

names_to_labels = {'lib_b': 'Library D2', 'lib_c': 'Library D3', 'old_nnk': 'NNK'}
names = ['old_nnk', 'lib_b'] if args.library_names is None else args.library_names
labels = [names_to_labels[name] for name in names]


def get_n_distinct(counts_df, threshold, pseudocount=True):
    counts = counts_df['count'].values
    if pseudocount and counts_df['count'].min() < 1:
        counts = counts + 1
    cum_probs = np.cumsum(counts) / np.sum(counts)
    return np.argmin(cum_probs <= threshold)


def merge_counts(pre_df, post_df):
    merged_df = pd.merge(pre_df, post_df, how='outer', on='seq', suffixes=('_pre', '_post')).fillna(0)
    merged_df = merged_df.rename(columns={'count_pre': 'count'})
    merged_df = merged_df.reset_index()
    merged_df.sort_values('count', ascending=False, inplace=True)
    return merged_df


ts = np.linspace(0.1, 1., 1000, endpoint=False)

# Compile experimental results into DataFrame.
results = defaultdict(list)
for name in names:
    i = names.index(name)

    pre_counts_file = os.path.join(data_dir, "counts/%s_pre_counts.csv" % name)
    post_counts_file = os.path.join(data_dir, "counts/%s_post_counts.csv" % name)
        
    pre_counts = pd.read_csv(pre_counts_file)
    post_counts = pd.read_csv(post_counts_file)

    pre_counts = pre_counts.loc[~pre_counts['seq'].str.contains('X')]
    pre_counts = pre_counts.reset_index()

    post_counts = post_counts.loc[~post_counts['seq'].str.contains('X')]
    post_counts = post_counts.reset_index()
    
    # Incorporate post-infection data where possible.
    infection_counts = None
    if name in ['lib_b', 'old_nnk']:
        infection_file = os.path.join(data_dir, "counts/brain_%s_post_counts.csv" % name.split('_')[1])
        infection_counts = pd.read_csv(infection_file)
        infection_counts = infection_counts.loc[~infection_counts['seq'].str.contains('X')]
        infection_counts = infection_counts.reset_index()
    
    pre_counts = merge_counts(pre_counts, post_counts)
    if infection_counts is not None:
        pre_counts = merge_counts(pre_counts, infection_counts)
        post_counts = merge_counts(post_counts, infection_counts)
    
    # Store results for 'pre' library.
    results['Library'].extend([labels[i]] * ts.size)
    results['Condition'].extend(['Initial'] * ts.size)
    results['threshold'].extend(list(ts))
    results['n_distinct_seq'].extend([get_n_distinct(pre_counts, t) for t in ts])
    print('{} {} overall distinct sequences: {} (or {} w/o merging)'.format(labels[i], 'pre', len(pre_counts), np.sum(pre_counts['count'].values > 0)))
    
    # Store results for 'post' library.
    results['Library'].extend([labels[i]] * ts.size)
    results['Condition'].extend(['Post-Packaging'] * ts.size)
    results['threshold'].extend(list(ts))
    results['n_distinct_seq'].extend([get_n_distinct(post_counts, t) for t in ts])
    print('{} {} overall distinct sequences: {} (or {} w/o merging)'.format(labels[i], 'post', len(post_counts), np.sum(post_counts['count'].values > 0)))
    
    # Store results for post-infection library.
    if infection_counts is not None:
        results['Library'].extend([labels[i]] * ts.size)
        results['Condition'].extend(['Post-Brain Infection'] * ts.size)
        results['threshold'].extend(list(ts))
        results['n_distinct_seq'].extend([get_n_distinct(infection_counts, t) for t in ts])
        print('{} {} overall distinct sequences: {}'.format(labels[i], 'post-infection', len(infection_counts)))

results = pd.DataFrame(results)


# Visualize the number of distinct sequences in each library.
fig, ax = plt.subplots(figsize=(6, 3))
colors = sns.color_palette('colorblind', n_colors=len(names))
g = sns.lineplot(x='threshold', y='n_distinct_seq', hue='Library', hue_order=labels[::-1], style='Condition', data=results, palette=colors, ax=ax)
g.set(yscale='log')
ax.set_ylabel('Number of Distinct Sequences', fontsize=14)
ax.set_xlabel('Fraction of Total Reads', fontsize=14)
ls = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.).get_texts()
ls[0].set_weight('bold')
ls[3].set_weight('bold')
ax.grid(False)
plt.tight_layout()
plt.savefig('plots/library_counts.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
plt.close()


# Visualize the fold change between post-infection libraries.
fig, ax = plt.subplots(figsize=(3, 3))
results.sort_values('threshold', ascending=True, inplace=True)
lib_b_infection = results.loc[(results['Library'] == 'Library D2') & (results['Condition'] == 'Post-Brain Infection'), 'n_distinct_seq']
nnk_infection = results.loc[(results['Library'] == 'NNK') & (results['Condition'] == 'Post-Brain Infection'), 'n_distinct_seq']
fold_changes = lib_b_infection.values / nnk_infection.values
sns.lineplot(x=ts, y=fold_changes, ax=ax)
ax.set_ylabel('Fold Change (Library D2 / NNK)')
ax.set_xlabel('Library Fraction')
ax.grid(False)
plt.title('Post-Infection Fold Change')
plt.tight_layout()
plt.savefig('plots/post-infection_fold_change.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
plt.close()
