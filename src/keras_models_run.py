import sys
import argparse
import modeling
import data_prep
import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("library", help="name of library: 'old_nnk', 'new_nnk', 'lib_b', or 'lib_c'", type=str)
parser.add_argument("model_type", help="type of model: 'linear','ann'", type=str)
parser.add_argument("-e", "--encoding", default='pairwise', 
                    help="which sequence encoding to use: 'is', 'neighbors' or 'pairwise'")
parser.add_argument("-d", "--hidden_size", help="size of hidden layers in ann model", type=int)
parser.add_argument("-n", "--normalize", help="normalize enrichment scores", action='store_true')
parser.add_argument("-u", "--unweighted_loss", help="use an unweighted loss function", action='store_true')
parser.add_argument("-o", "--use_old_db", help="use the old count database", action='store_true')
parser.add_argument("-f", "--use_filtered_linkers", help="use count data where incorrect linkers are filtered out", action='store_true')

args = parser.parse_args()
lib = args.library
filtered = args.use_filtered_linkers
weighted_loss = True
if args.unweighted_loss:
    weighted_loss = False

if args.use_old_db:
    data_df = data_prep.load_old_data(lib, args.merge_data)
else:
    data_df = data_prep.load_data(lib, use_filtered=filtered)
    
seqs, en_scores = data_prep.prepare_data(data_df)
if args.normalize:
    mean_en, var_en = en_scores[:, 0], en_scores[:, 1]
    mu_en = np.mean(mean_en)
    sig_en = np.std(mean_en)
    mean_en = (mean_en - mu_en) / sig_en
    var_en = (np.sqrt(var_en) / sig_en)**2
    en_scores[:, 0] = mean_en
    en_scores[:, 1] = var_en
    
train_idx_fname = "%s_train_idx" % lib
test_idx_fname = "%s_test_idx" % lib
if filtered:
    train_idx_fname += "_filtered"
    test_idx_fname += "_filtered"
train_idx_fname += ".npy"
test_idx_fname += ".npy"
try:
    train_idx = np.load(train_idx_fname)
    test_idx = np.load(test_idx_fname)
except FileNotFoundError:
    train_idx, test_idx = train_test_split(range(len(seqs)), test_size=0.2, random_state=123) 
    np.save(train_idx_fname, train_idx)
    np.save(test_idx_fname, test_idx)

if args.encoding == 'pairwise':
    enc = data_prep.encode_one_plus_pairwise
    alpha = 0.0025
elif args.encoding == 'is':
    enc = data_prep.one_hot_encode
    alpha = 0
elif args.encoding == 'neighbors':
    enc = data_prep.encode_one_plus_neighbors
    alpha = 0.001

input_shape = data_prep.get_example_encoding(enc).shape
if args.model_type == 'linear':
    epochs = 5
    batch_size = 1000
    savefile = "%s_linear_%s" % (lib, args.encoding)
    model = modeling.make_linear_model(input_shape, l2_reg=alpha, 
                                       weighted_loss=weighted_loss)
    
elif args.model_type == 'ann':
    epochs = 10
    batch_size = 100
    savefile = "%s_ann_%s_%s" % (lib, args.hidden_size, args.encoding)
    model = modeling.make_ann_model(input_shape, num_hid=2, 
                                    hid_size=args.hidden_size, 
                                    weighted_loss=weighted_loss)
    
train_gen = modeling.DataGenerator(seqs, en_scores, train_idx,
                                   enc, batch_size=batch_size, shuffle=True)
test_gen = modeling.DataGenerator(seqs, en_scores, test_idx,
                                  enc, batch_size=1000, shuffle=False)
    
history_callback = model.fit_generator(generator=train_gen,
                                       epochs=epochs,
                                       use_multiprocessing=True,
                                       workers=4,
                                       verbose=2
                                       )

loss_history = np.array(history_callback.history["loss"])

pred = model.predict_generator(test_gen, verbose=2).flatten()

if not weighted_loss:
    savefile += "_unweighted"
if args.normalize:
    savefile += "_normalized"  
if filtered:
    savefile += "_filtered"

model_savefile = savefile + "_model"
loss_savefile = savefile + "_loss_history.npy"
test_pred_savefile = savefile + "_test_pred.npy"

model.save(savefile)
np.save(loss_savefile, loss_history)
np.save(test_pred_savefile, pred)






