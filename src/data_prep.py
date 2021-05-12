import os
import numpy as np
from itertools import combinations
import pre_process
import pandas as pd
from seqtools import SequenceTools


DATA_DIR = "../data/counts/"


def calculate_enrichment_scores(n_pre, n_post, N_pre, N_post):
    """
    Calculates the mean and approximate variance
    of enrichment scores. The variance is that implied
    by the ratio of two binomials.
    """
    f_pre = n_pre / N_pre
    f_post = n_post / N_post
    mean_true = np.log(f_post/f_pre)
    approx_sig = (1/n_post) * (1-f_post) + (1/n_pre) * (1-f_pre)
    return np.array([mean_true, approx_sig]).T


def one_hot_encode(seq):
    """
    Returns a one-hot encoded matrix representing
    the input sequence.
    """
    l = len(seq)
    m = len(pre_process.AA_ORDER)
    out = np.zeros((l, m))
    for i in range(l):
        out[i, pre_process.AA_IDX[seq[i]]] = 1
    return out.flatten()


def encode_neighbors(seq):
    """
    Returns a binary matrix where each entry represents
    the presence or absence of a particular pair of amino
    acid at a pair of neighboring positions
    """
    neighbors = [(i, i+1) for i in range(6)]
    m = len(pre_process.AA_ORDER)
    out = np.zeros((len(neighbors), m, m))
    for i, (c1, c2) in enumerate(neighbors):
        s1 = seq[c1]
        s2 = seq[c2]
        out[i, pre_process.AA_IDX[s1], pre_process.AA_IDX[s2]] = 1
    return out.flatten()


def encode_pairwise(seq):
    """
    Returns a binary matrix where each entry represents
    the presence or absence of a particular pair of amino
    acid at a pair of positions
    """
    combos = list(combinations(range(7), 2))
    m = len(pre_process.AA_ORDER)
    out = np.zeros((len(combos), m, m))
    for i, (c1, c2) in enumerate(combos):
        s1 = seq[c1]
        s2 = seq[c2]
        out[i, pre_process.AA_IDX[s1], pre_process.AA_IDX[s2]] = 1
    return out.flatten()


def encode_one_plus_pairwise(seq):
    """
    Combines the one-hot and pairwise encodings
    into a single binary vector
    """
    one_hot = one_hot_encode(seq)
    pairwise = encode_pairwise(seq)
    both = np.concatenate((one_hot, pairwise))
    return both


def encode_one_plus_neighbors(seq):
    """
    Combines the one-hot and neighbor encodings
    into a single binary vector
    """
    one_hot = one_hot_encode(seq)
    neighbors = encode_neighbors(seq)
    both = np.concatenate((one_hot, neighbors))
    return both


def get_example_encoding(encoding_function, nuc=False):
    """
    Returns an example of the possible sequence encodings,
    given the encoding function. Should be used for sizing
    purposes.
    """
    if nuc:
        seq = "".join(['A']*21)
    else:
        seq = "".join(['A']*7)
    return encoding_function(seq)


def load_data(lib_name, pre_file=None, post_file=None, 
              seq_column='seq', count_column='count', use_filtered=False):
    """Loads pre- and post- count databases and combine into one merged DataFrame"""
    if pre_file is None:
        fname = "%s_pre_counts" % lib_name
        if use_filtered:
            fname += "_filtered"
        fname += ".csv"
        pre_file = os.path.join(DATA_DIR, fname)
    if post_file is None:
        fname = "%s_post_counts" % lib_name
        if use_filtered:
            fname += "_filtered"
        fname += ".csv"
        post_file = os.path.join(DATA_DIR, fname)
        
    pre_df = pd.read_csv(pre_file)[[seq_column, count_column]]
    post_df = pd.read_csv(post_file)[[seq_column, count_column]]
    pre_groups = pre_df.groupby(seq_column)
    pre_df = pre_groups.sum().reset_index()
    post_groups = post_df.groupby(seq_column)
    post_df = post_groups.sum().reset_index()
    merged_df = pd.merge(pre_df, post_df, how='outer', on=seq_column, suffixes=('_pre', '_post')).fillna(0)
    merged_df = merged_df.rename(columns={count_column + "_pre": 'count_pre', 
                                          count_column + "_post": 'count_post', 
                                          seq_column: 'seq'})
    merged_df = merged_df.loc[~merged_df['seq'].str.contains('X')]
    merged_df = merged_df.reset_index()
    return merged_df


def prepare_data(merged_df):
    """
    Converts merged DataFrame of count data into list of sequences
    and (enrichment score, variance) pairs.
    """
    n_pre = np.array(merged_df['count_pre'] + 1)
    n_post = np.array(merged_df['count_post'] + 1)
    N_pre = n_pre.sum()
    N_post = n_post.sum()

    enrich_scores = calculate_enrichment_scores(n_pre, n_post, N_pre, N_post)
    sequences = list(merged_df['seq'])
    return sequences, enrich_scores


def featurize_and_transform(sequences, enrich_scores, 
                            encoding_func=encode_one_plus_pairwise):
    """
    Encodes sequences given an encoding function and calculates sample weights
    from (enrichment score, variance) pairs.
    """
    d = get_example_encoding(encoding_func).shape[0]
    X = np.zeros((len(sequences), d), dtype=np.int8)
    for i in range(len(X)):
        X[i] = encoding_func(sequences[i])

    y = enrich_scores[:, 0]
    sample_weights = 1/(2*enrich_scores[:, 1])
    return X, y, sample_weights


def get_nnk_p():
    """
    Get NNK nucleotide probabilities.
    """
    p_nnk = np.ones((3, 4))
    p_nnk[:2] *= 0.25
    p_nnk[2, pre_process.NUC_IDX['A']] = 0
    p_nnk[2, pre_process.NUC_IDX['C']] = 0
    p_nnk[2] *= 0.5
    p_nnk = np.tile(p_nnk.T, 7).T
    return p_nnk