import pandas as pd
from cointegration import scan_pairs
from ccm_filter   import filter_ccm

# Load
df = pd.read_csv("data/clean/train_prices_clean.csv", index_col=0, parse_dates=True)

# Linear shortlist
pairs_lin = scan_pairs(df, p=0.05)
pairs_lin = pd.DataFrame(
    pairs_lin,
    columns=["stock1","stock2","pvalue"]
)
print(f"Engle–Granger found {len(pairs_lin)} pairs")

# Hyperparameter grid search on df_train
r0 = 0.8
best = {"score": -1}
for E in [2,3,4,5,6]:
    for k in [5,7,10,15]:
        candidates = filter_ccm(df, pairs_lin, E, tau=1, k=k, r_tresh=r0)
        # pick worst-case skill among them as a simple metric
        if not candidates:
             continue
        min_skill = min(min(r_xy, r_yx) for *_, r_xy, r_yx in candidates)
        if min_skill > best["score"]:
            best.update(E=E, k=k, r0=r0, score=min_skill, candidates=candidates)

print("Best CCM params:", best)
# Save your winning list & params
out_df = pd.DataFrame(best["candidates"], columns=["stock1","stock2","r_xy","r_yx"])
#out_df.to_csv("data/processed/universal_pairs.csv", index=False)
print(f"Saved {len(out_df)} pairs with E={best['E']}, k={best['k']}, r0={best['r0']}")