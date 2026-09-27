"""Pre-registered ablations (analysis/PREREGISTRATION_M2.md §3) on 3D-DefectBench.
LOFO / family-only, single-feature rules, silver-vs-expert training, fusion with repeated CV."""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_comparison as mc

OUT = f"{mc.ROOT}/results/ablations"; os.makedirs(OUT, exist_ok=True)
FAM = {
 "topology": ["n_components", "largest_component_area_frac", "n_small_components", "small_component_area_frac", "n_large_components", "genus_proxy"],
 "boundary_manifold": ["boundary_edge_frac", "boundary_length", "n_boundary_loops", "winding_inconsistent_edge_frac", "duplicate_face_frac"],
 "self_intersection": ["self_intersect_face_frac", "self_intersect_area_frac"],
 "hidden": ["hidden_surface_frac"],
 "thin": ["thickness_defined_frac", "thin_frac_0.005", "thin_frac_0.01", "thickness_median"],
 "roughness": ["dihedral_mean_deg", "dihedral_gt30_lenfrac", "dihedral_gt60_lenfrac", "dihedral_gt90_lenfrac", "angle_defect_mean", "angle_defect_p95"],
 "tri_quality": ["tri_quality_median", "tri_quality_p05", "sliver_face_frac", "sliver_area_frac", "edge_length_cv"],
 "size": ["n_faces"],
}
RULES = {"q_fused_incomplete": ["genus_proxy", "self_intersect_face_frac", "hidden_surface_frac"],
         "q_form_surface": ["thin_frac_0.005", "dihedral_mean_deg"],
         "q_extra_geometry": ["n_small_components", "small_component_area_frac"]}
N_BOOT = 2000  # ablation CIs (fewer resamples than the main table; stated in the report)


def load():
    meta = json.load(open(f"{mc.ROOT}/results/main_comparison_meta.json"))
    keep = meta["features_used"]
    probes = mc.load_probes(); feats = mc.featurize(probes)
    probes = pd.concat([probes[["object_id", "split"]], feats], axis=1)
    ho = set(pd.read_parquet(f"{mc.DB}/vlm_predictions_silver_holdout.parquet")["object_id"].unique())
    tr = probes[(probes.split == "silver") & probes.object_id.isin(ho) & ~probes.object_id.isin(mc.EXCLUDE_TRAIN)]
    tr = tr.merge(pd.read_csv(f"{mc.DB}/silver_labels.csv"), on="object_id").reset_index(drop=True)
    te = probes[probes.split == "golden"].sort_values("object_id").merge(pd.read_csv(f"{mc.DB}/golden_labels.csv"), on="object_id").reset_index(drop=True)
    med = tr[keep].median()
    for c in keep:
        tr[c] = tr[c].fillna(med[c]); te[c] = te[c].fillna(med[c])
    assert sorted(sum(FAM.values(), [])) == sorted(keep), set(keep) ^ set(sum(FAM.values(), []))
    return tr, te, keep


def boot_ci(y, s, p, seed=0):
    rng = np.random.default_rng(seed); A, M = [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        A.append(roc_auc_score(y[i], s[i])); M.append(mc.mcc0(y[i], p[i]))
    return np.percentile(A, [2.5, 97.5]).tolist(), np.percentile(M, [2.5, 97.5]).tolist()


def main():
    tr, te, keep = load()
    rows = []
    for d in mc.PRIMARY:
        ytr = tr[f"{d}_majority_vote"].to_numpy(int)
        cell = te[f"{d}_agreement_rate"].to_numpy() >= 1; y = te[f"{d}_majority_vote"].to_numpy(int)[cell]
        configs = [("all", keep)] + [(f"-{f}", [c for c in keep if c not in FAM[f]]) for f in FAM] + \
                  [(f"only:{f}", FAM[f]) for f in FAM] + [(f"rule:{r}", [r]) for r in RULES[d]]
        for name, cols in configs:
            s, t, oof, m_oof, _ = mc.fit_predict(tr[cols].to_numpy(), ytr, te[cols].to_numpy())
            s = s[cell]; p = (s >= t).astype(int)
            (alo, ahi), (mlo, mhi) = boot_ci(y, s, p)
            rows.append(dict(defect=d, config=name, n_feat=len(cols), AUROC=roc_auc_score(y, s), AUROC_lo=alo, AUROC_hi=ahi,
                             MCC=mc.mcc0(y, p), MCC_lo=mlo, MCC_hi=mhi, silver_oof_auc=roc_auc_score(ytr, oof), threshold=t))
    abl = pd.DataFrame(rows); abl.to_csv(f"{OUT}/lofo_family_rules.csv", index=False)

    # ---------------- silver vs expert training (repeated CV on golden agreement cells)
    sv = []
    for d in mc.PRIMARY:
        ytr = tr[f"{d}_majority_vote"].to_numpy(int)
        cell = te[f"{d}_agreement_rate"].to_numpy() >= 1
        Xg = te.loc[cell, keep].to_numpy(); yg = te.loc[cell, f"{d}_majority_vote"].to_numpy(int)
        s_sil, t_sil, _, _, _ = mc.fit_predict(tr[keep].to_numpy(), ytr, Xg)
        n_splits = min(5, yg.sum())
        for rep in range(50):
            skf = StratifiedKFold(n_splits, shuffle=True, random_state=rep)
            s_exp = np.zeros(len(yg)); p_exp = np.zeros(len(yg), int)
            for a, b in skf.split(Xg, yg):
                inner = min(5, yg[a].sum())
                # expert-trained, threshold from inner OOF of the training folds
                skf_i = StratifiedKFold(inner, shuffle=True, random_state=rep)
                oof = np.zeros(len(a))
                for ia, ib in skf_i.split(Xg[a], yg[a]):
                    oof[ib] = mc.lr().fit(Xg[a][ia], yg[a][ia]).predict_proba(Xg[a][ib])[:, 1]
                t_e, _ = mc.mcc_threshold(yg[a], oof)
                s_exp[b] = mc.lr().fit(Xg[a], yg[a]).predict_proba(Xg[b])[:, 1]
                p_exp[b] = (s_exp[b] >= t_e).astype(int)
            p_sil = (s_sil >= t_sil).astype(int)
            sv.append(dict(defect=d, rep=rep, n_splits=n_splits, AUROC_expert=roc_auc_score(yg, s_exp), AUROC_silver=roc_auc_score(yg, s_sil),
                           MCC_expert=mc.mcc0(yg, p_exp), MCC_silver=mc.mcc0(yg, p_sil)))
    sv = pd.DataFrame(sv); sv.to_csv(f"{OUT}/silver_vs_expert_cv.csv", index=False)
    # expert-trained on all golden agreement cells -> silver holdout
    rev = []
    for d in mc.PRIMARY:
        cell = te[f"{d}_agreement_rate"].to_numpy() >= 1
        Xg = te.loc[cell, keep].to_numpy(); yg = te.loc[cell, f"{d}_majority_vote"].to_numpy(int)
        ytr = tr[f"{d}_majority_vote"].to_numpy(int)
        s, t, _, _, _ = mc.fit_predict(Xg, yg, tr[keep].to_numpy())
        _, _, oof, m_oof, _ = mc.fit_predict(tr[keep].to_numpy(), ytr, tr[keep].to_numpy()[:1])
        rev.append(dict(defect=d, AUROC_expert_trained_on_silver=roc_auc_score(ytr, s), MCC_expert_trained_on_silver=mc.mcc0(ytr, (s >= t).astype(int)),
                        AUROC_silver_OOF=roc_auc_score(ytr, oof), MCC_silver_OOF=m_oof))
    pd.DataFrame(rev).to_csv(f"{OUT}/expert_trained_on_silver.csv", index=False)

    # ---------------- fusion with repeated CV on golden agreement cells
    vlm = pd.read_parquet(f"{mc.DB}/vlm_predictions_expert.parquet"); vlm = vlm[vlm.parse_ok.astype(bool)]
    fus = []
    for d in mc.PRIMARY:
        ytr = tr[f"{d}_majority_vote"].to_numpy(int)
        cell = te[f"{d}_agreement_rate"].to_numpy() >= 1
        ids = te.loc[cell, "object_id"].to_numpy(); yg = te.loc[cell, f"{d}_majority_vote"].to_numpy(int)
        s_sil, t_sil, _, _, _ = mc.fit_predict(tr[keep].to_numpy(), ytr, te.loc[cell, keep].to_numpy())
        lg = np.log(np.clip(s_sil, 1e-6, 1 - 1e-6) / (1 - np.clip(s_sil, 1e-6, 1 - 1e-6)))
        p_probe = (s_sil >= t_sil).astype(int)
        n_splits = min(5, yg.sum())
        for m in sorted(vlm.model_slug.unique()):
            v = vlm[(vlm.defect_name == d) & (vlm.model_slug == m)].set_index("object_id")["pred"].reindex(ids).to_numpy().astype(int)
            Z = np.c_[lg, v]
            base = dict(defect=d, vlm=m, MCC_vlm=mc.mcc0(yg, v), MCC_probe=mc.mcc0(yg, p_probe),
                        MCC_OR=mc.mcc0(yg, v | p_probe), MCC_AND=mc.mcc0(yg, v & p_probe))
            deltas = []
            for rep in range(50):
                skf = StratifiedKFold(n_splits, shuffle=True, random_state=rep)
                pf = np.zeros(len(yg), int)
                for a, b in skf.split(Z, yg):
                    inner = min(5, yg[a].sum()); skf_i = StratifiedKFold(inner, shuffle=True, random_state=rep)
                    oof = np.zeros(len(a))
                    for ia, ib in skf_i.split(Z[a], yg[a]):
                        oof[ib] = mc.lr().fit(Z[a][ia], yg[a][ia]).predict_proba(Z[a][ib])[:, 1]
                    t_f, _ = mc.mcc_threshold(yg[a], oof)
                    pf[b] = (mc.lr().fit(Z[a], yg[a]).predict_proba(Z[b])[:, 1] >= t_f).astype(int)
                deltas.append(mc.mcc0(yg, pf) - base["MCC_vlm"])
            deltas = np.array(deltas)
            fus.append({**base, "MCC_fusionCV_mean": base["MCC_vlm"] + deltas.mean(), "dMCC_mean": deltas.mean(),
                        "dMCC_p05": np.percentile(deltas, 5), "dMCC_p95": np.percentile(deltas, 95), "frac_reps_improved": (deltas > 0).mean()})
    fus = pd.DataFrame(fus); fus.to_csv(f"{OUT}/fusion_repeated_cv.csv", index=False)

    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
    print(abl.round(3).to_string())
    print(sv.groupby("defect")[["AUROC_expert", "AUROC_silver", "MCC_expert", "MCC_silver"]].agg(["mean", "std"]).round(3))
    print(pd.DataFrame(rev).round(3))
    print(fus.round(3).to_string())


if __name__ == "__main__":
    main()
