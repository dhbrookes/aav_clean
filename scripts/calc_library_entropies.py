import sys
import os
sys.path.append("../src")
from tensorflow import keras
import modeling
from Bio.Seq import Seq
import pre_process
import data_prep
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
from seqtools import SequenceTools
from functools import reduce
from tensorflow.keras.models import load_model
import seaborn as sns
plt.style.use('seaborn-deep')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.2


names = ['lib_b', 'lib_c', 'new_nnk', 'old_nnk']
labels = ['Lib B', 'Lib C', 'New NNK', 'Old NNK']

lib_b_nuc = np.array([[0.12,0.04,0.39,0.45], [0.18,0.47,0.3,0.05], [0.21,0.19,0.28,0.32], [0.14,0.02,0.19,0.65],
[0.23,0.33,0.29,0.15],[0.28,0.24,0.25,0.23],[0.35,0.0,0.14,0.51],[0.13,0.17,0.36,0.34],
[0.21,0.31,0.31,0.17],[0.13,0.0,0.06,0.81],[0.26,0.12,0.22,0.4],[0.16,0.29,0.36,0.19],
[0.09,0.0,0.08,0.83],[0.36,0.12,0.37,0.15],[0.13,0.49,0.24,0.14],[0.22,0.0,0.13,0.65],
[0.29,0.08,0.24,0.39],[0.1,0.42,0.34,0.14],[0.16,0.01,0.09,0.74],[0.28,0.11,0.47,0.14],
[0.17,0.35,0.3,0.18]])

lib_c_nuc = np.array([[0.21,0.09,0.43,0.27],[0.22,0.25,0.37,0.16],[0.25,0.28,0.27,0.2],[0.27,0.12,0.13,0.48],
[0.2,0.35,0.27,0.18],[0.22,0.23,0.22,0.33],[0.22,0.08,0.16,0.54],[0.32,0.19,0.34,0.15],
[0.25,0.17,0.27,0.31],[0.24,0.07,0.43,0.26],[0.34,0.35,0.2,0.11],[0.22,0.26,0.28,0.24],
[0.28,0.06,0.17,0.49],[0.45,0.14,0.29,0.12],[0.2,0.27,0.3,0.23],[0.32,0.08,0.19,0.41],
[0.23,0.2,0.35,0.22],[0.23,0.27,0.36,0.14],[0.2,0.04,0.32,0.44],[0.39,0.17,0.3,0.14],
[0.26,0.25,0.24,0.25]])

nnk_nuc = data_prep.get_nnk_p()

nuc_probs = {'lib_b': lib_b_nuc, 'lib_c': lib_c_nuc, 'new_nnk': nnk_nuc, 'old_nnk': nnk_nuc}

def calc_entropy(counts_df):
    probs = np.array(counts_df['count'] / counts_df['count'].sum())
    entropy = -np.sum(probs*np.log(probs))
    return entropy

# name = 'old_nnk'
ents = {}
for name in names:
    i = names.index(name)
    # name = names[i]
    lbl = labels[i]

    pre_counts_file = "../data/counts/%s_pre_counts.csv" % name
    post_counts_file = "../data/counts/%s_post_counts.csv" % name
        
    pre_counts = pd.read_csv(pre_counts_file)
    post_counts = pd.read_csv(post_counts_file)
    merged_counts = data_prep.load_data("aav5",
                                        pre_file=pre_counts_file, 
                                        post_file=post_counts_file)

    pre_counts = pre_counts.loc[~pre_counts['seq'].str.contains('X')]
    pre_counts = pre_counts.reset_index()

    post_counts = post_counts.loc[~post_counts['seq'].str.contains('X')]
    post_counts = post_counts.reset_index()
    
    pre_ent = calc_entropy(pre_counts)
    post_ent = calc_entropy(post_counts)
    ents[name] = (pre_ent, post_ent)
    
    
titers = {
    'lib_b': (5.2e11, -1),
    'lib_c': (2.75e11, -1),
    'old_nnk': (1.82e11, 4.38e11),
    'new_nnk': (1.02e11, -1)
}


fig, ax = plt.subplots(figsize=(3, 3))
colors = sns.color_palette('Set1', n_colors=2)
for i, nm in enumerate(names):
    lbl = labels[i]
    ent_pre, ent_post = ents[nm]
    if i == 0:
        ax.scatter([i], [ent_pre], c=colors[1], label='Pre', s=30)
        ax.scatter([i], [ent_post], c=colors[0], label='Post', s=30)
    else:
        ax.scatter([i], [ent_post], c=colors[0], s=30)
        ax.scatter([i], [ent_pre], c=colors[1], s=30)
    if nm == 'old_nnk':
        ax.scatter([i], [ent_post], edgecolor='k', facecolor='none', s=40)
    if nm == 'lib_b':
        ax.scatter([i], [ent_pre], edgecolor='k', facecolor='none', s=40)
ax.set_ylabel("Entropy")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(labels, ha='center', fontsize=8)
ax.legend()
    
ax.grid(False)
plt.tight_layout()
plt.savefig('plots/library_entropies.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)
plt.close()

fig, ax = plt.subplots(figsize=(3, 3))
colors = sns.color_palette('Set1', n_colors=5)
for i, nm in enumerate(names):
    lbl = labels[i]
    ent_pre, ent_post = ents[nm]
    titer_pre, titer_post = titers[nm]
    ax.scatter(ent_pre, titer_pre, label="%s pre" % lbl, c = colors[i])
    if titer_post != -1:
        ax.scatter(ent_post, titer_post, label="%s post" % lbl, edgecolor=colors[i], facecolor='none', linewidth=1.5)
    
ax.set_xlabel("Entropy")
ax.set_ylabel("Viral Genome (vm/mL)")
# ax.set_xticks([0, 1, 2, 3])
# ax.set_xticklabels(labels, ha='center', fontsize=8)
ax.legend()
    
ax.grid(False)
plt.tight_layout()
plt.savefig('plots/entropies_titers.png', dpi=300, transparent=False, bbox_inches='tight', facecolor='white',)