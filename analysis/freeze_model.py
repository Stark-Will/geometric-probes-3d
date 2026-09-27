"""Refit and freeze the milestone-1 GeomProbe-LR models (silver-holdout trained) for external use.
Identical procedure to analysis/main_comparison.py; saves pickles + JSON (features, medians, thresholds)."""
import json, os, pickle, hashlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_comparison as mc

OUT = f"{mc.ROOT}/results/frozen_models"; os.makedirs(OUT, exist_ok=True)
meta1 = json.load(open(f"{mc.ROOT}/results/main_comparison_meta.json"))
probes = mc.load_probes()
feats = mc.featurize(probes)
probes = pd.concat([probes[["object_id", "split"]], feats], axis=1)
ho = set(pd.read_parquet(f"{mc.DB}/vlm_predictions_silver_holdout.parquet")["object_id"].unique())
tr = probes[(probes.split == "silver") & probes.object_id.isin(ho) & ~probes.object_id.isin(mc.EXCLUDE_TRAIN)]
tr = tr.merge(pd.read_csv(f"{mc.DB}/silver_labels.csv"), on="object_id")
keep = meta1["features_used"]
med = tr[keep].median()
X = tr[keep].fillna(med).to_numpy()
frozen = dict(features=keep, medians=med.to_dict(), models={})
for d in mc.DEFECTS:
    y = tr[f"{d}_majority_vote"].to_numpy(int)
    _, t, _, m_oof, model = mc.fit_predict(X, y, X[:1])
    assert abs(t - meta1["thresholds"][f"{d}|GeomProbe-LR"]["threshold"]) < 1e-12, d
    pickle.dump(model, open(f"{OUT}/geomprobe_lr_{d}.pkl", "wb"))
    frozen["models"][d] = dict(threshold=t, silver_oof_mcc=m_oof,
                               coef=model[-1].coef_[0].tolist(), intercept=float(model[-1].intercept_[0]),
                               scaler_mean=model[0].mean_.tolist(), scaler_scale=model[0].scale_.tolist())
json.dump(frozen, open(f"{OUT}/geomprobe_lr_frozen.json", "w"), indent=1)
print(hashlib.sha256(open(f"{OUT}/geomprobe_lr_frozen.json", "rb").read()).hexdigest())
