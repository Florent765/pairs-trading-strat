import os
import numpy as np
import pandas as pd
from sklearn.model_selection import NearestNeighbors
from scipy.stats import zscore, pearsonr

PRICES_CLEAN = "data/clean/prices_clean.csv"
PAIRS_FILE = "data/clean/scanned_pairs.csv"
OUT_DIR = "data/processed"
OUT_FILE = "pairs.csv"

# Hyperparameters
E = 2 # embedding dimension
TAU = 1 # delay
K = 8 # neighbors
R_TRESHOLD = 0.8 # minimum cross-map skill

def embed(x, E, tau=1):
    N = len(x)
    M = N - (E - 1) * tau
    X = np.empty((M, E))
    for i in range(M):
        X[i] = x[i: i + E * tau: tau]
    return X

def knn_indices(X, k):
    nbr = NearestNeighbors(n_neighbors=k).fit(X)
    return nbr.kneighbors(X, return_distance=False)

#def ccm_cross_map(x, y, E, tau, k):
#    X, Y = embed(x, E, tau), embed(y, E, tau)