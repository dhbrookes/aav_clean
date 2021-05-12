import os.path
import numpy as np
import pre_process
import data_prep
import argparse
import entropy_opt
import opt_analysis
import time


def mutate_seqs(seqs, aa=False):
    N, L = seqs.shape
    if aa:
        K = len(pre_process.AA_ORDER)
    else:
        K = len(pre_process.NUC_ORDER)
    seqs_mut = np.copy(seqs)
    mut_pos = np.random.randint(L, size=N)
    for i in range(N):
        mp = mut_pos[i]
        options = [int(k) for k in range(K) if k != seqs[i, mp]]
        seqs_mut[i, mp] = np.random.choice(options)
    return seqs_mut


def metropolis_hastings(model, enc, lam, n_chain=1000, n_sample=1000, seqs=None,
                        n_burnin=1000, aa=False, print_every=200):

    if seqs is None:
        nnk = opt_analysis.get_nnk_p()
        if aa:
            nnk = np.array(opt_analysis.aa_probs_from_nuc_probs(nnk)).T
        seqs = entropy_opt.sample_sequences(nnk, n_chain)
    else:
        if seqs.shape[0] != n_chain:
            raise ValueError("Initial sequences wrong shape: {}".format(seqs))

    xgen = entropy_opt.featurize(seqs, enc, aa=aa)
    energies =  model.predict_generator(xgen).flatten()
    fx = np.mean(energies)
    t_start = time.time()
    print("Burn-in iter, f(x)      Time (s)")
    print("0            {:.4f}".format(fx))
    for i in range(1, n_burnin):
        seqs_mut = mutate_seqs(seqs, aa=aa)
        xgen_mut = entropy_opt.featurize(seqs_mut, enc, aa=aa)
        energies_mut = model.predict_generator(xgen_mut).flatten()
        dE = energies_mut - energies
        rands = np.random.rand(n_chain)
        log_prob_accept = dE / lam
        accept = (np.log(rands) < log_prob_accept)
        seqs = np.where(accept.reshape(n_chain, 1), seqs_mut, seqs)
        energies = np.where(accept, energies_mut, energies)
        
        if i % print_every == 0:
            fx = np.mean(energies)
            print("{:<13}{:.4f}{:>8}".format(i, fx, int(time.time() - t_start)))

    print("Sample iter, f(x)")
    samples_nxcxsa = []
    energies_n = []
    for i in range(1, n_sample + 1):
        seqs_mut = mutate_seqs(seqs, aa=aa)
        xgen_mut = entropy_opt.featurize(seqs_mut, enc, aa=aa)
        energies_mut = model.predict_generator(xgen_mut).flatten()
        dE = energies_mut - energies
        rands = np.random.rand(n_chain)
        log_prob_accept = dE / lam
        accept = (np.log(rands) < log_prob_accept)
        seqs = np.where(accept.reshape(n_chain, 1), seqs_mut, seqs)
        energies = np.where(accept, energies_mut, energies)
        samples_nxcxsa.append(seqs)
        energies_n.append(energies)

        if i % print_every == 0:
            fx = np.mean(energies)
            print("{:<12}{:.4f}".format(i, fx))
    samples_nxcxsa = np.stack(samples_nxcxsa)
    energies_n = np.hstack(energies_n)
    return samples_nxcxsa, energies_n


def run_lambda_set(results_dict, savefile, lambdas, model, enc, 
                   n_chain=1000, n_sample=1000, n_burnin=1000, warmstart=True, aa=False):
    """
    Runs the sampling for a given set of lambda values.
    """
    if warmstart:
        lambdas = np.sort(lambdas)[::-1]
        print("Running and warm-starting lambda values in decreasing order.")

    init_library = None
    for l in lambdas:  
        print("Lambda: {}".format(l))
        seqs, energies = metropolis_hastings(model, enc, l,
                                             seqs=init_library,  # warm-start sequences
                                             n_chain=n_chain,
                                             n_sample=n_sample,
                                             n_burnin=n_burnin, aa=aa)
        if warmstart:
            init_library = seqs[-1]
        results_dict[l] = (seqs, energies)
        np.save(savefile, results_dict)
    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to Keras predictive model")
    parser.add_argument("savefile", help="path to savefile")
    parser.add_argument("--max_lambda", help="maximum lambda", type=float)
    parser.add_argument("--min_lambda", help="minimum lambda", type=float)
    parser.add_argument("--num_lambda", help="Number of lambdas to test betweeen min and max", type=int)
    parser.add_argument("--num_iter", help="number of iterations in each optimization", type=int)
    parser.add_argument("--num_chains", help="number of independent MCMCs to run in parallel", type=int)
    parser.add_argument("--encoding", help="The model's sequence encoding function", type=str, default='pairwise')
    parser.add_argument("--aa", 
                        help="If true, optimizes amino acid probabilities directly", action='store_true')
    
    args = parser.parse_args()
    model_path = args.model_path
    savefile = args.savefile
    max_lambda = args.max_lambda
    min_lambda = args.min_lambda
    num_lambda = args.num_lambda
    niter = args.num_iter
    num_chains = args.num_chains
    aa = args.aa
    if args.encoding == 'pairwise':
        enc = data_prep.encode_one_plus_pairwise
    elif args.encoding == 'is':
        enc = data_prep.one_hot_encode
        alpha = 0
    elif args.encoding == 'neighbors':
        enc = data_prep.encode_one_plus_neighbors
        
    if os.path.exists(savefile):
        raise IOError("Savefile already exists at %s. Choose another name" % savefile)
        
    model = keras.models.load_model(model_path)
    
    results = {}
    results['meta'] = {
        'model_path': model_path,
        'encoding': args.encoding,
        'num_iter': niter,
        'aa': aa,
        'chains': num_chains
    }
    
    lambdas = np.linspace(min_lambda, max_lambda, num_lambda+1)
    
    run_lambda_set(results, savefile, lambdas, model, enc, 
                   chains=num_chains, niter=niter, aa=aa)
