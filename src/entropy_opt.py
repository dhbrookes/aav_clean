import os.path
import numpy as np
import data_prep
import pre_process
from Bio.Seq import Seq
import argparse
from tensorflow import keras
import modeling


def normalize_theta(theta):
    """
    Converts unnormalized probabilities of categorical distributions
    into the corresponding normalized probabilities.
    """
    L = theta.shape[0]
    p = np.exp(theta) / np.sum(np.exp(theta), axis=1).reshape(L, 1)
    return p


def sample_sequences(p, num):
    """
    Samples integer sequences given the probabilities 
    of each character at each position.
    """
    L, K = p.shape
    arr = np.array([np.random.choice(K, num, p=p[i]) for i in range(L)]).T
    return arr


def sample_sequences_from_theta(theta, num):
    """
    Samples integer sequences given unnormalized probabilities.
    """
    p = normalize_theta(theta)
    return sample_sequences(p, num)


def logp(x_onehot, theta):
    """
    Calculates the log probability of one hot encoded sequences given
    the unnormalized probability matrix.
    """
    p = normalize_theta(theta)
    logp = np.log(p)
    logp_x = np.sum(x_onehot * logp, axis=(1, 2))
    return logp_x


def featurize(x, encoding_func, aa=False):
    """
    Converts integer sequences into a DataGenerator object that
    can be input into a keras model.
    """
    aa_seqs = ["X" * 7] * len(x)
    for i in range(x.shape[0]):
        seq = x[i]
        if not aa:
            nuc_seq = "".join([pre_process.NUC_ORDER[seq[j]] for j in range(21)])
            aa_seqs[i] = str(Seq(nuc_seq).translate())
        else:
            aa_seqs[i] = "".join([pre_process.AA_ORDER[seq[j]] for j in range(7)])
    xgen = modeling.DataGenerator(aa_seqs, np.zeros((len(x), 2)), list(range(len(x))), 
                                  encoding_func, batch_size=len(x), shuffle=False)
    return xgen


def calc_entropy_from_theta(theta, aa=False):
    """
    Calculates entropy from unnormalized probabilities.
    """
    p = normalize_theta(theta)
    return calc_entropy(p)


def calc_entropy(p):
    """
    Calculates entropy from probabilities.
    """
    p_ma = np.ma.masked_where(p==0, p)
    logp = np.log(p_ma)
    H = -np.sum(p_ma * logp)
    return H


def opt_theta_entropy_sgd(theta0, model, encoding_func, 
                          entropy_reg, learning_rate=1., n_samples=1000, 
                          niter=1000, print_every=100, aa=False):
    """
    Performs entropy regularized optimization with SGD
    """
    theta = theta0
    print("iter", "fx", 'H')
    for j in range(niter):
        x = sample_sequences_from_theta(theta, n_samples)
        if not aa:
            x_onehot = np.eye(4)[x]
        else:
            x_onehot = np.eye(21)[x]
        xgen = featurize(x, encoding_func, aa=aa)
        fx = model.predict_generator(xgen).reshape(n_samples)
        logp_x = logp(x_onehot, theta)
        p = np.exp(theta) / np.sum(np.exp(theta), axis=1).reshape(theta.shape[0], 1)
        w = fx - entropy_reg * (1+logp_x)
        grad_logp = x_onehot - p.reshape(1, p.shape[0], p.shape[1])
        grad_theta = w.reshape(n_samples, 1, 1) * grad_logp
        theta = theta + learning_rate * np.mean(grad_theta, axis=0)
        if j % print_every == 0:
            print(j, np.mean(fx), theta.shape[0] - np.sum(p**2))
    
    return theta, np.mean(fx), calc_entropy_from_theta(theta, aa=aa)


def run_lambda_set(results_dict, savefile, lambdas, model, enc, 
                   learning_rate=1., niter=1000, n_samples=1000,
                   aa=False, random_start=False):
    """
    Runs the optimization for a given set of lambda values.
    """
    for l in lambdas:  
        print("lambda = %s" % l)
        if random_start:
            if aa:
                theta0 = np.random.randn(7, 21)
            else:
                theta0 = np.random.randn(21, 4)
        else:
            if aa:
                theta0 = np.ones((7, 21))
            else:
                theta0 = np.ones((21, 4))

        theta_opt, fx_opt, H_opt = opt_theta_entropy_sgd(theta0, model, 
                                                     enc, 
                                                     entropy_reg=l,
                                                     learning_rate=learning_rate,
                                                     n_samples=n_samples, 
                                                     niter=niter,  
                                                     print_every=100,
                                                     aa=aa)


        results_dict[l] = (H_opt, fx_opt, theta_opt)
        np.save(savefile, results_dict)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="path to Keras predictive model")
    parser.add_argument("savefile", help="path to savefile")
    parser.add_argument("--max_lambda", help="maximum lambda", type=float)
    parser.add_argument("--min_lambda", help="minimum lambda", type=float)
    parser.add_argument("--num_lambda", help="Number of lambdas to test betweeen min and max", type=int)
    parser.add_argument("--num_iter", help="number of iterations in each optimization", type=int)
    parser.add_argument("--learning_rate", help="learning rate for sgd", type=float)
    parser.add_argument("--num_samples", help="number of samples to take at each step of SGD", type=int)
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
    nsamples = args.num_samples
    learning_rate = args.learning_rate
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
        'num_samples': nsamples,
        'aa': aa,
        'learning_rate': learning_rate
    }
    
    lambdas = np.linspace(min_lambda, max_lambda, num_lambda+1)

    run_lambda_set(results, savefile, lambdas, model, enc, 
                   learning_rate=learning_rate, 
                   niter=niter, n_samples=nsamples,
                   aa=aa, random_start=True)
    
    
    