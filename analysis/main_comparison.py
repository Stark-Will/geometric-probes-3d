"""Pre-registered main comparison: GeomProbe-LR vs 12 VLM judges on 3D-DefectBench expert cells.
Implements analysis/PREREGISTRATION.md exactly. Deterministic (seeds fixed). Outputs -> results/.
"""
import json, os, sys, hashlib
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score
from statsmodels.stats.multitest import multipletests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = f"{ROOT}/data/3d-defectbench/data"
OUT = f"{ROOT}/results"
os.makedirs(OUT, exist_ok=True)
PRIMARY = ["q_fused_incomplete", "q_form_surface", "q_extra_geometry"]
CONTROLS = ["q_missing_parts", "q_pose_placement"]
DEFECTS = PRIMARY + CONTROLS
N_BOOT, BOOT_SEED, CV_SEED = 10_000, 12345, 0
EXCLUDE_TRAIN = {609}
DROP = {"orig_bbox_diag", "status", "path", "seconds", "peak_rss_mb", "n_edges", "n_vertices",
        "watertight", "error", "stderr", "trace", "split", "object_id"}


# ------------------------------------------------------------------ data
def load_probes():
    df = pd.read_csv(f"{ROOT}/results/probes/defectbench.csv")
    df["split"] = df["path"].str.extract(r"glb/(golden|silver)/")[0]
    df["object_id"] = df["path"].str.extract(r"/(\d+)\.glb$")[0].astype(int)
    return df


def featurize(df):
    X = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c in DROP or df[c].dtype == object:
            continue
        if c == "euler_characteristic":
            X["genus_proxy"] = np.log1p(np.maximum(0, 2 * df["n_components"] - df[c]) / 2)
        else:
            X[c] = np.log1p(df[c].astype(float).clip(lower=0))
    return X


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))


def mcc_threshold(y, p):
    best_t, best_m = 0.5, -2
    for t in np.unique(p):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, float(t)
    return best_t, float(best_m)


def fit_predict(Xtr, ytr, Xte):
    """OOF on train -> MCC-optimal threshold; refit on all train; score test."""
    skf = StratifiedKFold(5, shuffle=True, random_state=CV_SEED)
    oof = np.zeros(len(ytr))
    for a, b in skf.split(Xtr, ytr):
        oof[b] = lr().fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
    t, m_oof = mcc_threshold(ytr, oof)
    model = lr().fit(Xtr, ytr)
    pte = model.predict_proba(Xte)[:, 1]
    return pte, t, oof, m_oof, model


# ------------------------------------------------------------------ weighted metrics for bootstrap
def w_conf(W, y, p):
    tp = W @ ((y == 1) & (p == 1)); tn = W @ ((y == 0) & (p == 0))
    fp = W @ ((y == 0) & (p == 1)); fn = W @ ((y == 1) & (p == 0))
    return tp, tn, fp, fn


def w_mcc(W, y, p):
    tp, tn, fp, fn = w_conf(W, y, p)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (tp * tn - fp * fn) / den, 0.0)


def w_f1(W, y, p):
    tp, tn, fp, fn = w_conf(W, y, p)
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / den, 0.0)


def w_bacc(W, y, p):
    tp, tn, fp, fn = w_conf(W, y, p)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))


def w_auc(W, y, s):
    pos, neg = y == 1, y == 0
    M = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
    Wp, Wn = W[:, pos], W[:, neg]
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((Wp @ M) * Wn).sum(1) / (Wp.sum(1) * Wn.sum(1))


def ci(v):
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def mcc0(y, p):
    return 0.0 if (len(np.unique(y)) < 2 and len(np.unique(p)) < 2) else float(matthews_corrcoef(y, p))


# ------------------------------------------------------------------ main
def main():
    probes = load_probes()
    bad = probes[probes["status"] != "ok"]
    gold_lab = pd.read_csv(f"{DB}/golden_labels.csv")
    silv_lab = pd.read_csv(f"{DB}/silver_labels.csv")
    manifest = pd.read_csv(f"{DB}/manifest.csv")
    vlm = pd.read_parquet(f"{DB}/vlm_predictions_expert.parquet")
    vlm = vlm[vlm["parse_ok"].astype(bool)]
    vlm_s = pd.read_parquet(f"{DB}/vlm_predictions_silver_holdout.parquet")
    holdout_ids = set(vlm_s["object_id"].unique())

    feats = featurize(probes)
    probes = pd.concat([probes[["object_id", "split", "status"]], feats], axis=1)
    tr = probes[(probes.split == "silver") & probes.object_id.isin(holdout_ids) & ~probes.object_id.isin(EXCLUDE_TRAIN)]
    te = probes[probes.split == "golden"].sort_values("object_id").reset_index(drop=True)
    tr = tr.merge(silv_lab, on="object_id", how="left")
    te = te.merge(gold_lab, on="object_id", how="left")
    mv = manifest.set_index("object_id")["model_version"]
    assert (tr["model_version"].to_numpy() == mv.reindex(tr["object_id"]).to_numpy()).all()
    assert (te["model_version"].to_numpy() == mv.reindex(te["object_id"]).to_numpy()).all()
    assert len(te) == 129 and te[[c for c in te.columns if c.endswith("_majority_vote")]].notna().all().all()
    fcols = list(feats.columns)
    med = tr[fcols].median()
    keep = [c for c in fcols if tr[c].fillna(med[c]).std() > 0]
    dropped_zero_var = sorted(set(fcols) - set(keep))
    Xtr_all = tr[keep].fillna(med[keep]).to_numpy(); Xte_all = te[keep].fillna(med[keep]).to_numpy()
    fc = keep.index("n_faces")
    gen_tr = (tr["model_version"] == "model B").astype(float).to_numpy()[:, None]
    gen_te = (te["model_version"] == "model B").astype(float).to_numpy()[:, None]

    # bootstrap weights over golden assets (shared across all predictors / defects)
    rng = np.random.default_rng(BOOT_SEED)
    nA = len(te)
    idx = rng.integers(0, nA, size=(N_BOOT, nA))
    Wfull = np.zeros((N_BOOT, nA)); np.add.at(Wfull, (np.arange(N_BOOT)[:, None], idx), 1.0)
    aid = te["object_id"].to_numpy()

    rows, paired, thresholds, feat_auc, secondary, within_gen, fusion = [], [], {}, [], [], [], []
    vlm_models = sorted(vlm["model_slug"].unique())
    for d in DEFECTS:
        ytr = tr[f"{d}_majority_vote"].to_numpy(int)
        cell = te[f"{d}_agreement_rate"].to_numpy() >= 1.0
        y = te[f"{d}_majority_vote"].to_numpy(int)
        preds = {}
        for name, Xa, Xb in [("GeomProbe-LR", Xtr_all, Xte_all), ("FaceCount-LR", Xtr_all[:, [fc]], Xte_all[:, [fc]]),
                             ("Generator-only", gen_tr, gen_te)]:
            s, t, oof, m_oof, model = fit_predict(Xa, ytr, Xb)
            preds[name] = (s, (s >= t).astype(int))
            thresholds[f"{d}|{name}"] = dict(threshold=t, silver_oof_mcc=m_oof, n_train=int(len(ytr)),
                                             train_pos=int(ytr.sum()), silver_oof_auc=float(roc_auc_score(ytr, oof)))
            if name == "GeomProbe-LR":
                coefs = dict(zip(keep, model[-1].coef_[0].round(4).tolist()))
                thresholds[f"{d}|{name}"]["coef"] = coefs
                oof_probe = oof
        # VLMs: align to golden asset order
        vd = vlm[vlm.defect_name == d]
        for m in vlm_models:
            ser = vd[vd.model_slug == m].set_index("object_id")["pred"]
            p = ser.reindex(aid).to_numpy()
            preds[m] = (None, p)
        # sanity: reference labels in VLM file == ours
        ref = vd.drop_duplicates("object_id").set_index("object_id")["reference_label"]
        assert set(ref.index) == set(aid[cell]), d
        assert (ref.reindex(aid[cell]).to_numpy() == y[cell]).all(), d

        W = Wfull[:, cell]; yc = y[cell]
        boot_mcc = {}
        for name, (s, p) in preds.items():
            pc = p[cell]
            if np.isnan(pc.astype(float)).any():
                raise RuntimeError(f"missing VLM preds {name} {d}")
            pc = pc.astype(int)
            bm = w_mcc(W, yc, pc); boot_mcc[name] = bm
            r = dict(defect=d, role="primary" if d in PRIMARY else "neg_control", predictor=name,
                     n_cells=int(cell.sum()), n_pos=int(yc.sum()), pred_pos=int(pc.sum()),
                     MCC=mcc0(yc, pc), F1=float(f1_score(yc, pc, zero_division=0)))
            r["MCC_lo"], r["MCC_hi"] = ci(bm)
            r["F1_lo"], r["F1_hi"] = ci(w_f1(W, yc, pc))
            if s is not None and len(np.unique(yc)) == 2:
                r["AUROC"] = float(roc_auc_score(yc, s[cell])); r["AUROC_type"] = "score"
                r["AUROC_lo"], r["AUROC_hi"] = ci(w_auc(W, yc, s[cell]))
            elif len(np.unique(yc)) == 2:
                r["AUROC"] = float(0.5 * (((pc == 1) & (yc == 1)).sum() / (yc == 1).sum() + ((pc == 0) & (yc == 0)).sum() / (yc == 0).sum()))
                r["AUROC_type"] = "binary(=balanced acc.)"
                r["AUROC_lo"], r["AUROC_hi"] = ci(w_bacc(W, yc, pc))
            rows.append(r)
        # paired tests: GeomProbe-LR vs each VLM (and vs baselines)
        pg = preds["GeomProbe-LR"][1][cell].astype(int)
        for m in vlm_models + ["FaceCount-LR", "Generator-only"]:
            pm = preds[m][1][cell].astype(int)
            cg, cm = pg == yc, pm == yc
            b, c = int((cg & ~cm).sum()), int((~cg & cm).sum())
            pval = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
            dm = boot_mcc["GeomProbe-LR"] - boot_mcc[m]
            lo, hi = ci(dm)
            paired.append(dict(defect=d, role="primary" if d in PRIMARY else "neg_control", comparator=m,
                               MCC_probe=mcc0(yc, pg), MCC_comp=mcc0(yc, pm), dMCC=mcc0(yc, pg) - mcc0(yc, pm),
                               dMCC_lo=lo, dMCC_hi=hi, probe_only_correct=b, comp_only_correct=c, mcnemar_p=float(pval)))
        # within-generator AUROC (probe score), agreement cells
        s = preds["GeomProbe-LR"][0]
        for g in ["model A", "model B"]:
            mk = cell & (te["model_version"].to_numpy() == g)
            if len(np.unique(y[mk])) == 2:
                within_gen.append(dict(defect=d, generator=g, n=int(mk.sum()), n_pos=int(y[mk].sum()),
                                       AUROC=float(roc_auc_score(y[mk], s[mk]))))
        # secondary: union target (all 129 assets), probes only
        yu = (te[f"{d}_mean_value"].to_numpy() > 0).astype(int)
        for name in ["GeomProbe-LR", "FaceCount-LR", "Generator-only"]:
            s_, p_ = preds[name]
            secondary.append(dict(defect=d, predictor=name, n=nA, n_pos=int(yu.sum()), MCC=mcc0(yu, p_),
                                  F1=float(f1_score(yu, p_, zero_division=0)),
                                  AUROC=float(roc_auc_score(yu, s_)) if len(np.unique(yu)) == 2 else np.nan))
        # exploratory: per-feature AUROC on agreement cells (raw feature, positive direction reported)
        if len(np.unique(yc)) == 2:
            for j, f in enumerate(keep):
                a = roc_auc_score(yc, Xte_all[cell, j])
                n1, n0 = yc.sum(), (1 - yc).sum()
                # Mann-Whitney p-value via scipy
                from scipy.stats import mannwhitneyu
                pv = mannwhitneyu(Xte_all[cell, j][yc == 1], Xte_all[cell, j][yc == 0], alternative="two-sided").pvalue
                feat_auc.append(dict(defect=d, feature=f, AUROC=float(a), p=float(pv), n_pos=int(n1), n_neg=int(n0)))
        # exploratory: late fusion per VLM (probe OOF logit + VLM vote), trained on silver holdout
        vs = vlm_s[(vlm_s.defect_name == d) & vlm_s.parse_ok.astype(bool)]
        for m in vlm_models:
            vtr = vs[vs.model_slug == m].set_index("object_id")["pred"].reindex(tr["object_id"]).to_numpy()
            ok = ~np.isnan(vtr.astype(float))
            lg = np.log(np.clip(oof_probe, 1e-6, 1 - 1e-6) / (1 - np.clip(oof_probe, 1e-6, 1 - 1e-6)))
            Ztr = np.c_[lg, vtr][ok]
            ste = preds["GeomProbe-LR"][0]
            Zte = np.c_[np.log(np.clip(ste, 1e-6, 1 - 1e-6) / (1 - np.clip(ste, 1e-6, 1 - 1e-6))), preds[m][1].astype(float)]
            sf, tf, _, _, _ = fit_predict(Ztr, ytr[ok], Zte[cell])
            pf = (sf >= tf).astype(int)
            fusion.append(dict(defect=d, vlm=m, MCC_vlm=mcc0(yc, preds[m][1][cell].astype(int)), MCC_fusion=mcc0(yc, pf),
                               MCC_probe=mcc0(yc, pg), silver_vlm_mcc=mcc0(ytr[ok], vtr[ok].astype(int))))

    res = pd.DataFrame(rows); pr = pd.DataFrame(paired)
    # BH over the 36 primary McNemar tests (probe vs 12 VLMs x 3 primary defects); controls separately
    for role in ["primary", "neg_control"]:
        mk = (pr.role == role) & pr.comparator.isin(vlm_models)
        pr.loc[mk, "q_BH"] = multipletests(pr.loc[mk, "mcnemar_p"], method="fdr_bh")[1]
    fa = pd.DataFrame(feat_auc)
    if len(fa):
        ok = np.isfinite(fa["p"])
        fa.loc[ok, "q_BH"] = multipletests(fa.loc[ok, "p"], method="fdr_bh")[1]
    # macro geometry MCC over the 5 scored defects (paper headline format)
    macro = res.groupby("predictor")["MCC"].mean().rename("macro_geom_MCC_5").sort_values(ascending=False)
    macro3 = res[res.role == "primary"].groupby("predictor")["MCC"].mean().rename("macro_MCC_primary3")

    res.to_csv(f"{OUT}/main_comparison_expert_agreement.csv", index=False)
    pr.to_csv(f"{OUT}/paired_tests_probe_vs_vlm.csv", index=False)
    pd.DataFrame(secondary).to_csv(f"{OUT}/secondary_union_target_probes.csv", index=False)
    pd.DataFrame(within_gen).to_csv(f"{OUT}/within_generator_auroc.csv", index=False)
    fa.to_csv(f"{OUT}/exploratory_feature_auroc.csv", index=False)
    pd.DataFrame(fusion).to_csv(f"{OUT}/exploratory_fusion.csv", index=False)
    pd.concat([macro, macro3], axis=1).to_csv(f"{OUT}/macro_mcc.csv")
    meta = dict(prereg_sha256=hashlib.sha256(open(f"{ROOT}/analysis/PREREGISTRATION.md", "rb").read()).hexdigest(),
                n_train=int(len(tr)), n_test_assets=int(nA), probe_failures=bad[["path", "status"]].to_dict("records"),
                features_used=keep, dropped_zero_variance=dropped_zero_var, n_boot=N_BOOT, boot_seed=BOOT_SEED,
                cv_seed=CV_SEED, excluded_train=sorted(EXCLUDE_TRAIN), thresholds=thresholds)
    json.dump(meta, open(f"{OUT}/main_comparison_meta.json", "w"), indent=1)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print(res[["defect", "predictor", "n_cells", "n_pos", "pred_pos", "MCC", "MCC_lo", "MCC_hi", "F1", "AUROC", "AUROC_lo", "AUROC_hi"]].round(3).to_string())
    print(pd.concat([macro, macro3], axis=1).round(3))
    print(pr[pr.comparator.isin(vlm_models) & (pr.role == "primary")].round(4).to_string())


if __name__ == "__main__":
    main()
