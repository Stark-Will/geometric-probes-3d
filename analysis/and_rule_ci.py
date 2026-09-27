"""Paired asset-bootstrap CIs for the pre-registered training-free AND/OR fusion rules (probe_pred & VLM_pred)
vs VLM alone on golden agreement cells. Probe threshold = frozen silver threshold (no golden fitting)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_comparison as mc
from ablations import load
tr, te, keep = load()
vlm = pd.read_parquet(f"{mc.DB}/vlm_predictions_expert.parquet"); vlm = vlm[vlm.parse_ok.astype(bool)]
rng = np.random.default_rng(12345); rows = []
for d in mc.PRIMARY:
    ytr = tr[f"{d}_majority_vote"].to_numpy(int); cell = te[f"{d}_agreement_rate"].to_numpy() >= 1
    ids = te.loc[cell, "object_id"].to_numpy(); y = te.loc[cell, f"{d}_majority_vote"].to_numpy(int)
    s, t, _, _, _ = mc.fit_predict(tr[keep].to_numpy(), ytr, te.loc[cell, keep].to_numpy()); pp = (s >= t).astype(int)
    W = np.zeros((10000, len(y))); np.add.at(W, (np.arange(10000)[:, None], rng.integers(0, len(y), (10000, len(y)))), 1)
    for m in sorted(vlm.model_slug.unique()):
        v = vlm[(vlm.defect_name == d) & (vlm.model_slug == m)].set_index("object_id")["pred"].reindex(ids).to_numpy().astype(int)
        for rule, p in (("AND", v & pp), ("OR", v | pp)):
            dm = mc.w_mcc(W, y, p) - mc.w_mcc(W, y, v)
            rows.append(dict(defect=d, vlm=m, rule=rule, MCC_vlm=mc.mcc0(y, v), MCC_rule=mc.mcc0(y, p), dMCC=mc.mcc0(y, p) - mc.mcc0(y, v),
                             lo=np.percentile(dm, 2.5), hi=np.percentile(dm, 97.5), p_boot=min(1, 2 * min((dm <= 0).mean(), (dm >= 0).mean()))))
r = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
for rule in ("AND", "OR"):
    k = r.rule == rule; r.loc[k, "q_BH"] = multipletests(r.loc[k, "p_boot"], method="fdr_bh")[1]
r.to_csv(f"{mc.ROOT}/results/ablations/and_or_rule_bootstrap.csv", index=False)
pd.set_option("display.width", 200); print(r[r.rule == "AND"].round(3).to_string())
print(r.groupby(["rule", "defect"])[["MCC_vlm", "MCC_rule", "dMCC"]].mean().round(3))
print("AND sig improvements:", ((r.rule == "AND") & (r.q_BH < 0.05) & (r.dMCC > 0)).sum(), " sig worse:", ((r.rule == "AND") & (r.q_BH < 0.05) & (r.dMCC < 0)).sum())
