import pre_process
import os
import data_prep
import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from itertools import combinations
from mpl_toolkits.axes_grid1 import make_axes_locatable
tfk = tf.keras
tfkl = tf.keras.layers


class DataGenerator(tfk.utils.Sequence):
    """
    Generates sequence/enrichment score data for a given sequence encoding 
    (additive, neighbors, pairwise, etc.)
    """
    
    def __init__(self, sequences, enrich_scores, ids, 
                 encoding_function, batch_size=1000, shuffle=True,
                nuc_encoding=False):
        self.sequences = sequences
        self.enrich_scores = enrich_scores
        self.ids = ids
        self.encoding_function = encoding_function
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.nuc_encoding = nuc_encoding 
        self.on_epoch_end()
        
    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))
    
    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        ids_temp = [self.ids[k] for k in indexes]

        # Generate data
        inputs = self._data_generation(ids_temp)

        return inputs
    
    def _data_generation(self, list_ids_temp):
        """
        Encodes sequences and returns the inputs to our model. Due to how 
        the models are constructed, the inputs are 
        [sequence, enrichment_mean, enrichment_variance]
        """
        length = data_prep.get_example_encoding(self.encoding_function, nuc=self.nuc_encoding).shape[0]
        X = np.zeros((self.batch_size, length))
        for i, idx in enumerate(list_ids_temp):
            X[i] = self.encoding_function(self.sequences[idx])
        y = self.enrich_scores[list_ids_temp]
        
        inputs = [X, y[:, 0], y[:, 1]]
        return inputs
    
    
def get_regularizer(l1_reg=0., l2_reg=0.):
    """
    Returns a keras regularizer object given 
    the l1 and l2 regularization parameters
    """
    if l1_reg > 0 and l2_reg > 0:
        reg = regularizers.l1_l2(l1=l1_reg, l2=l2_reg)
    elif l1_reg > 0:
        reg = tfk.regularizers.l1(l1_reg)
    elif l2_reg > 0:
        reg = tfk.regularizers.l2(l2_reg)
    else:
        reg = None

        
def make_linear_model(input_shape, l1_reg=0., l2_reg=0., weighted_loss=True):
    """
    Makes a linear keras model. If weighted_loss is True, than the loss 
    is Gaussian with variance determined ny what should be expectd from 
    the ratio of two binomials (see Katz, 1978 for details on this variance.).
    Otherwise, the loss is least-squares.
    """
    reg = get_regularizer(l1_reg, l2_reg)

    inp = tfkl.Input(shape=input_shape)
    output = tfkl.Dense(1, activation='linear', kernel_regularizer=reg, bias_regularizer=reg)(inp)
    
    label_mean = tfkl.Input((1,))
    label_sig = tfkl.Input((1,))
    #model = tfk.models.Model(inputs=[inp, label_mean, label_sig], outputs=output)
    if weighted_loss:
        loss = K.mean((1/(2*label_sig)) * (label_mean - output) ** 2)
    else:
        loss = K.mean((label_mean - output) ** 2)
    outputs = tfkl.Lambda(lambda x: x[0])((output, loss))
    model = tfk.models.Model(inputs=[inp, label_mean, label_sig], outputs=outputs)
    model.add_loss(loss)
    model.compile(optimizer='adam')
    return model


def make_ann_model(input_shape, num_hid=2, hid_size=100, weighted_loss=True):
    """
    Builds an artificial neural network model.
    """
    inp = tfkl.Input(shape=input_shape)
    z = inp
    for i in range(num_hid):
        z = tfkl.Dense(hid_size, activation='tanh')(z)
    out = tfkl.Dense(1, activation='linear')(z)
    
    label_mean = tfkl.Input((1,))
    label_sig = tfkl.Input((1,))
    #model = tfk.models.Model(inputs=[inp, label_mean, label_sig], outputs=out)
    if weighted_loss:
        loss = K.mean((1/(2*label_sig)) * (label_mean - out) ** 2)
    else:
        loss = K.mean((label_mean - out) ** 2)
    outputs = tfkl.Lambda(lambda x: x[0])((out, loss))
    model = tfk.models.Model(inputs=[inp, label_mean, label_sig], outputs=outputs)
    model.add_loss(loss)
    model.compile(optimizer='adam')
    return model


def calculate_culled_correlation(ypred, ytest, fracs):
    """
    Calculates the pearson correlation between predictions and true fitness values
    among subsets of test data. In particular, test data is succesively culled to only
    include the largest true fitness values.
    """
    spears = []
    n_test = len(ypred)
    y_test = ytest[:n_test]
    sorted_test_idx = np.argsort(y_test)
    for i in range(len(fracs)):
        num_frac = int(n_test * fracs[i])
        idx = sorted_test_idx[num_frac:]
        ypred_frac = ypred[idx]
        ytest_frac = ytest[idx]
#         print(np.mean(ytest_frac), np.mean(ypred_frac))
        spear = pearsonr(ypred_frac, ytest_frac)[0]
        spears.append(spear)
    return spears

