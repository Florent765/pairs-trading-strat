import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import zscore, pearsonr

PRICES_CLEAN = "data/clean/train_prices_clean.csv"
PAIRS_FILE = "data/processed/scanned_pairs_train.csv"
OUT_DIR = "data/processed"
OUT_FILE = "pairs_train.csv"

# Hyperparameters
E = 4 # best embedding dimension
TAU = 1 # delay
K = 5 # best neighbors
R_TRESHOLD = 0.8 # minimum cross-map skill

def embed(x, E, tau=1):
    N = len(x)
    M = N - E * tau
    X = np.empty((M, E))
    for i in range(M):
        X[i] = x[i: i + E * tau: tau]
    return X

def knn_indices(X, k):
    nbr = NearestNeighbors(n_neighbors=k).fit(X)
    return nbr.kneighbors(X, return_distance=False)

def ccm_cross_map(x, y, E, tau, k):
    # Reconstruct each series into a time-delay embedding
    X, Y = embed(x, E, tau), embed(y, E, tau)

    # Normalize each embedding
    Xn, Yn = zscore(X, axis=1), zscore(Y, axis=1)

    # Handle NaNs that may arise from z-scoring constant rows (std_dev = 0)
    Xn = np.nan_to_num(Xn) # Replaces NaN with 0.0
    Yn = np.nan_to_num(Yn) # Replaces NaN with 0.0

    # Find nearest neighbors
    idx_X, idx_Y = knn_indices(Xn, k), knn_indices(Yn, k)

    M = X.shape[0]
    x_pred, y_pred = np.empty(M), np.empty(M)

    # Cross-map
    for i in range(M):
        # Use neighbors from X to predict y
        nbrs = idx_X[i]
        dist = np.linalg.norm(Xn[i] - Xn[nbrs], axis=1)

        w = 1/(dist + 1e-6) # Weight

        y_fut = y[nbrs + E * tau] # Future values of y
        x_pred[i] = np.dot(w, y_fut) / w.sum()

        # Use neighbors from Y to predict x
        nbrs = idx_Y[i]
        dist = np.linalg.norm(Yn[i] - Yn[nbrs], axis=1)

        w = 1/(dist + 1e-6) # Weight

        x_fut = x[nbrs + E * tau] # Future values of x
        y_pred[i] = np.dot(w, x_fut) / w.sum()
    
    return x_pred, y_pred

def cross_map_skill(true, pred):
    return pearsonr(true, pred)[0]

#if __name__ == "__main__":
    # Load
    #df_train = pd.read_csv(PRICES_CLEAN)
    #pairs = pd.read_csv(PAIRS_FILE)
def filter_ccm(df, pairs, E, tau, k, r_tresh):
    good = []

    for _, row in pairs.iterrows():
        t1, t2 = row["stock1"], row["stock2"]

        x_train, y_train = df[t1].values, df[t2].values

        x_pred, y_pred = ccm_cross_map(x_train, y_train, E, TAU, K)

        # Compute the offset due to embedding
        start = E * TAU
        end = start + len(x_pred)  
        true_x, true_y = x_train[start:end], y_train[start:end]

        # Compute cross-map skill
        r_x = cross_map_skill(true_x, x_pred)
        r_y = cross_map_skill(true_y, y_pred)

        # Keep pairs that exceed treshold
        if r_x > R_TRESHOLD and r_y > R_TRESHOLD:
            good.append((t1, t2, r_x, r_y))

    return good
    # Save
    #os.makedirs(OUT_DIR, exist_ok=True)
    #out = pd.DataFrame(good, columns=["stock1","stock2","r_xy","r_yx"])
    #out.to_csv(os.path.join(OUT_DIR, OUT_FILE), index=False)

    #print(f"In-sample CCM-filtered pairs saved to {OUT_DIR}/{OUT_FILE}")
    #print(out)