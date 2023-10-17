import pandas as pd
import numpy as np
import pre_process
import entropy_opt
import modeling
from seqtools import SequenceTools
from Bio.Seq import Seq
from tensorflow import keras


def aa_probs_from_nuc_probs(nuc_probs):
    """
    Calculates amino acid probabilities given nucleotide probabilities.
    """
    l_nuc = nuc_probs.shape[0]
    aa_prob = pd.DataFrame(0., index=pre_process.AA_ORDER, 
                           columns=range(1, int(l_nuc/3)+1))
    for i in range(int(l_nuc / 3)):
        aa_to_cod = SequenceTools.protein2codon_
        for aa in pre_process.AA_ORDER:
            cods = aa_to_cod[aa.lower()]
            for cod in cods:
                p_cod = 1
                for j in range(3):
                    nuc_idx = pre_process.NUC_IDX[cod[j].upper()]
                    p_cod *= nuc_probs[i*3 + j, nuc_idx]
                aa_prob[i+1].loc[aa] += p_cod
    return aa_prob


def calculate_mean_enrichment(model, enc, p, n_samples=1000, aa=False):
    """
    Calculates the mean enrichment of a distribution given
    the predictive model, encoding function, and probabilties.
    """
    x = entropy_opt.sample_sequences(p, n_samples)
    xgen = entropy_opt.featurize(x, enc, aa=aa)
    fx = model.predict_generator(xgen).reshape(n_samples, 1, 1)
    return np.mean(fx)


def calculate_nnk_stats(model_path, enc, n_samples=1000):
    """
    Calculates the relevant plotting statistics for the NNK distribution.
    """
    model = keras.models.load_model(model_path)
    p_nnk = get_nnk_p(num_copies=7)
    L = p_nnk.shape[0]
    mean_enrich = calculate_mean_enrichment(model, enc, p_nnk, n_samples=n_samples, aa=False)
    
    edist_nuc = L - np.sum(p_nnk**2)
    p_nnk_aa = np.array(aa_probs_from_nuc_probs(p_nnk))
    
    edist_aa = int(L/3) - np.sum(p_nnk_aa**2)

    entropy_aa = entropy_opt.calc_entropy(p_nnk_aa)
    
    nnk_stats = {'mean_enrichment': mean_enrich,'expected_nuc_dist':edist_nuc, 
            'expected_aa_dist': edist_aa, 'aa_entropy': entropy_aa}
    return nnk_stats


def calculate_pre_dist_stats(model_path, enc, n_samples=1000):
    """
    Calculates the relevant plotting statistics for the NNK distribution.
    """
    model = keras.models.load_model(model_path)
    p_aa = np.array(pd.read_csv("../data/pre_library_dist.csv", index_col=0)).T
    L = p_aa.shape[0]
    mean_enrich = calculate_mean_enrichment(model, enc, p_aa, n_samples=n_samples, aa=True)
    
    edist_aa = L - np.sum(p_aa**2)

    entropy_aa = entropy_opt.calc_entropy(p_aa)
    pd_stats = {'mean_enrichment': mean_enrich,'expected_aa_dist': edist_aa, 'aa_entropy': entropy_aa}
    return pd_stats


def calculate_no_stop_codon_stats(model_path, enc, n_samples=1000):
    """
    Calculates the relevant plotting statistics for the distribution defined
    by sampling uniformly and then filtering stop codons.
    """
    model = keras.models.load_model(model_path)
    p_uniform = 0.25 * np.ones((21, 4))
    L = p_uniform.shape[0]
    x = entropy_opt.sample_sequences(p_uniform, n_samples)
    aa_seqs = []
    for i in range(x.shape[0]):
        seq = x[i]
        nuc_seq = "".join([pre_process.NUC_ORDER[seq[j]] for j in range(21)])
        aa_seq = str(Seq(nuc_seq).translate())
        if '*' not in aa_seq:
            aa_seqs.append(aa_seq)
    xgen = modeling.DataGenerator(aa_seqs, np.zeros((len(aa_seqs), 2)), list(range(len(aa_seqs))), enc, batch_size=len(aa_seqs), shuffle=False)
    fx = model.predict_generator(xgen).reshape(len(aa_seqs), 1, 1)
    mean_enrich = np.mean(fx)

    p_aa = pd.DataFrame(0., index=pre_process.AA_ORDER, columns=range(1, int(L/3)+1))
    aa_to_cod = SequenceTools.protein2codon_
    for i in range(int(L / 3)):
        for aa in pre_process.AA_ORDER:
            cods = aa_to_cod[aa.lower()]
            p_aa[i+1].loc[aa] = 0 if aa == '*' else len(cods)
    p_aa = np.array(p_aa / p_aa.sum())
    
    edist_aa = int(L/3) - np.sum(p_aa**2)
    
    entropy_aa = entropy_opt.calc_entropy(p_aa)

    return {'mean_enrichment': mean_enrich, 'expected_aa_dist': edist_aa, 'aa_entropy': entropy_aa}


def calc_marginal_counts(counts_df, seq_column='seq', count_column='count', unique_seq=False, aa=False):
    if aa:
        alphabet = pre_process.AA_ORDER
    else:
        alphabet = pre_process.NUC_ORDER
    result = np.zeros((len(counts_df[seq_column][0]), len(alphabet)))
    for i in range(result.shape[0]):
        seq_i = counts_df[seq_column].str.get(i)
        for j, aa in enumerate(alphabet):
            if unique_seq:
                result[i][j] = np.sum(1 * (seq_i == aa))
            else:
                result[i][j] = counts_df.loc[(seq_i == aa), count_column].sum()
    return result


def load_counts_df(counts_file, seq_column='seq'):
    df = pd.read_csv(counts_file)
    return df.loc[~df[seq_column].str.contains('X')]
    

def calculate_empirical_positionwise_stats(model_path, enc, n_samples=1000, threshold=None, pre_file=None, post_file=None):
    """
    Calculates the relevant plotting statistics for the distribution defined
    by sampling according to observed empirical position-wise log-enrichment.
    """
    if pre_file is None:
        pre_file = '/storage/akosua/aav_clean/data/counts/old_nnk_pre_counts_old.csv'
    if post_file is None:
        post_file = '/storage/akosua/aav_clean/data/counts/old_nnk_post_counts_old.csv'
    model = keras.models.load_model(model_path)
    
    
    position_post_counts = calc_marginal_counts(load_counts_df(post_file, seq_column='aa_seq'),
                                                seq_column='nuc_seq', count_column='counts')
    position_pre_counts = calc_marginal_counts(load_counts_df(pre_file, seq_column='aa_seq'),
                                               seq_column='nuc_seq', count_column='counts')
#     position_enrichment = np.log((position_post_counts / np.sum(position_post_counts, axis=1)[:, np.newaxis])) - np.log((position_pre_counts / np.sum(position_pre_counts, axis=1)[:, np.newaxis]))
    position_enrichment = (position_post_counts / np.sum(position_post_counts, axis=1)[:, np.newaxis]) / (position_pre_counts / np.sum(position_pre_counts, axis=1)[:, np.newaxis])
    
    if threshold is not None:
        p = np.ones_like(position_enrichment) * (position_enrichment > threshold)
    else:
        p = position_enrichment #- np.amin(position_enrichment)
    p = p / np.sum(p, axis=1)[:, np.newaxis]
    p_aa = np.array(aa_probs_from_nuc_probs(p))
    
    L = position_enrichment.shape[0]
    mean_enrich = calculate_mean_enrichment(model, enc, p, n_samples=n_samples, aa=False)
    edist_aa = int(L/3) - np.sum(p_aa**2)
    entropy_aa = entropy_opt.calc_entropy(p_aa)

    return {'mean_enrichment': mean_enrich, 'expected_aa_dist': edist_aa, 'aa_entropy': entropy_aa}


def calculate_no_hydrophobic_stats(model_path, enc, n_samples=1000):
    """
    Calculates the relevant plotting statistics for the
    nucleotide distributions that are optimized to minimize occurence
    of hydrophobic/disallowed amino acids.
    """
    model = keras.models.load_model(model_path)
    
    result = {}
    for f_name in ['nuc_probs_hydro.npy', 'nuc_probs_disallowed.npy']:
        p = np.load('../results/{}'.format(f_name))
        p = np.tile(p.T, 7).T
        
        L = p.shape[0]
        mean_enrich = calculate_mean_enrichment(model, enc, p, n_samples=n_samples, aa=False)
        edist_nuc = L - np.sum(p**2)
        
        p_aa = np.array(aa_probs_from_nuc_probs(p))
        edist_aa = int(L/3) - np.sum(p_aa**2)
        entropy_aa = entropy_opt.calc_entropy(p_aa)
        result[f_name] = {'mean_enrichment': mean_enrich,'expected_nuc_dist':edist_nuc, 
                          'expected_aa_dist': edist_aa, 'aa_entropy': entropy_aa}
    return result['nuc_probs_hydro.npy'], result['nuc_probs_disallowed.npy']


def get_nnk_p(num_copies=7):
    """
    Returns the probabilties of the degeneerate codon NNK.
    """
    p_nnk = np.ones((3, 4))
    p_nnk[:2] *= 0.25
    p_nnk[2, pre_process.NUC_IDX['A']] = 0
    p_nnk[2, pre_process.NUC_IDX['C']] = 0
    p_nnk[2] *= 0.5
    p_nnk = np.tile(p_nnk.T, num_copies).T
    return p_nnk


def calc_expected_pairwise_dist(p, calc_aa=False, input_is_aa=False):
    """
    Calculates the expected pairwise distance between sequences
    drawn from the input probability distribution
    """
    if not calc_aa and input_is_aa:
        raise ValueError("Cannot calculate expected nucleotide distance from amino acid probabilities")
    if calc_aa and not input_is_aa:
        p = np.array(aa_probs_from_nuc_probs(p)).T
    L = p.shape[0]
    e_dist = L - np.sum(p**2)
    return e_dist


def calc_expected_pairwise_dist_from_theta(theta, calc_aa=False, input_is_aa=False):
    """
    Calculates the expected pairwise distance between sequences
    given the unnormalized probabilities.
    """
    if not calc_aa and input_is_aa:
        raise ValueError("Cannot calculate expected nucleotide distance from amino acid probabilities")
    p = entropy_opt.normalize_theta(theta)
    if calc_aa and not input_is_aa:
        p = np.array(aa_probs_from_nuc_probs(p)).T
    L = p.shape[0]
    e_dist = L - np.sum(p**2)
    return e_dist


def calc_stats_for_plotting(results_dict, savefile=None, aa=False):
    """
    Calculates the relevant plotting quantities given a results dictionary.
    """
    stats = ['expected_aa_dist']
    plot_data = {s: [] for s in stats}
    plot_data['mean_enrichment'] = []
    if not aa:
        plot_data['nuc_entropy'] = []
        plot_data['aa_entropy'] = []
    else:
        plot_data['aa_entropy'] = []
    plot_data['lambda'] = []
    if savefile is not None:
        plot_data['savefile'] = []
    
    lambdas = results_dict.keys()
    for l in lambdas:
        Hl, fxl, thetal = results_dict[l]
        plot_data['lambda'].append(l)
        plot_data['mean_enrichment'].append(fxl)
        if aa:
            plot_data['aa_entropy'].append(Hl)
        else:
            plot_data['nuc_entropy'].append(Hl)
            p_aa = np.array(aa_probs_from_nuc_probs(entropy_opt.normalize_theta(thetal))).T
            plot_data['aa_entropy'].append(entropy_opt.calc_entropy(p_aa))
        if savefile is not None:
            plot_data['savefile'].append(savefile)
        for s in stats:
            if s == 'expected_aa_dist':
                val = calc_expected_pairwise_dist_from_theta(thetal, calc_aa=True, input_is_aa=aa)
            elif s == 'expected_nuc_dist':
                val = calc_expected_pairwise_dist_from_theta(thetal, calc_aa=False, input_is_aa=aa)
            plot_data[s].append(val)
    return plot_data


def load_data_for_plotting(savefile):
    """
    Load the optimization data at savefile and calculate the desired plotting
    quantities.
    """
    results = np.load(savefile, allow_pickle=True).item()
    meta_data = results.pop('meta')
    aa = meta_data['aa']
    plot_data = calc_stats_for_plotting(results, aa=aa, savefile=savefile)
    return plot_data, meta_data, results


def load_designed_library_probabilities(savefile):
    """
    Load the optimization data at savefile and return the designed
    position-wise nucleotide probabilities.
    """
    results = np.load(savefile, allow_pickle=True).item()
    meta_data = results.pop('meta')
    aa = meta_data['aa']
    
    libraries = {}
    lambdas = results.keys()
    for l in lambdas:
        _, _, thetal = results[l]
        libraries[l] = entropy_opt.normalize_theta(thetal)
    return libraries, meta_data


def round_percentages(per):
    """Rounds a list of percentages to nearest percent."""
    per = np.array(per)
    per_rem, per_int  = np.modf(per)
    num_add = int(100 - np.sum(per_int))
    top_rem = np.argsort(-per_rem)
    for i in range(num_add):
        idx = top_rem[i]
        rem = per_rem[idx]
        per_int[idx] += 1
    return per_int
