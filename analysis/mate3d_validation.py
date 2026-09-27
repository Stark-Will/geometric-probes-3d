"""Pre-registered MATE-3D external validation (analysis/PREREGISTRATION_M2.md §1).
Usage: python analysis/mate3d_validation.py [probe_csv] [tag]"""
import json, os, pickle, re, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_comparison as mc

ROOT = mc.ROOT
CSV = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/probes/mate3d_dec10k.csv"
TAG = sys.argv[2] if len(sys.argv) > 2 else "dec10k"
OUT = f"{ROOT}/results/mate3d"; os.makedirs(OUT, exist_ok=True)
NB, BSEED, NP, PSEED = 10_000, 2026, 10_000, 2027
PRIMARY = ["thin_frac_0.005", "genus_proxy", "dihedral_mean_deg", "self_intersect_face_frac", "hidden_surface_frac",
           "P_form_surface", "P_fused_incomplete", "P_extra_geometry", "P_defect_mean"]


def key(s): return re.sub(r"[^A-Za-z0-9]+", "_", s.strip()).strip("_").lower()


def load():
    mos = pd.read_excel(f"{ROOT}/data/mate-3d/MOS_MATE_3D.xlsx", sheet_name="ALL")
    mos["k"] = mos.prompt.map(key)
    pr = pd.read_csv(CSV)
    rel = pr.path.str.replace(r".*meshes/", "", regex=True)
    pr["generator"] = rel.str.split("/").str[0]; pr["k"] = rel.str.split("/").str[1].map(key)
    bad = pr[pr.status != "ok"][["path", "status"]]
    feats = mc.featurize(pr.drop(columns=["generator", "k"]))
    fz = json.load(open(f"{ROOT}/results/frozen_models/geomprobe_lr_frozen.json"))
    X = feats[fz["features"]].fillna(pd.Series(fz["medians"])).to_numpy()
    for d in ["q_form_surface", "q_fused_incomplete", "q_extra_geometry"]:
        m = pickle.load(open(f"{ROOT}/results/frozen_models/geomprobe_lr_{d}.pkl", "rb"))
        pr[f"P_{d[2:]}"] = m.predict_proba(X)[:, 1]
    pr["P_defect_mean"] = pr[["P_form_surface", "P_fused_incomplete", "P_extra_geometry"]].mean(axis=1)
    pr["genus_proxy"] = np.maximum(0, 2 * pr["n_components"] - pr["euler_characteristic"]) / 2
    pr["log_orig_n_faces"] = np.log(pr["orig_n_faces"])
    df = pr.merge(mos.rename(columns={"model": "generator"}), on=["generator", "k"], how="inner")
    assert len(df) == len(pr[pr.status == "ok"]), (len(df), len(pr))
    return df, bad


def wg_ranks(v, g):
    out = np.empty(len(v))
    for gg in np.unique(g):
        m = g == gg; out[m] = rankdata(v[m]) / m.sum()
    return out


def rowcorr(A, B):
    A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / np.sqrt((A ** 2).sum(1) * (B ** 2).sum(1))


def main():
    df, bad = load()
    gens = sorted(df.generator.unique()); prompts = sorted(df.k.unique())
    df = df.sort_values(["k", "generator"]).reset_index(drop=True)
    # balanced layout: (prompt, generator) grid
    P, G = len(prompts), len(gens)
    assert len(df) == P * G, "unbalanced after failures; handled below only if balanced"
    y = df["Geometry"].to_numpy().reshape(P, G)
    feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool and c not in
             ("Alignment", "Geometry", "Texture", "Overall", "seconds", "peak_rss_mb", "orig_bbox_diag")]
    rng = np.random.default_rng(BSEED); BI = rng.integers(0, P, size=(NB, P))
    prng = np.random.default_rng(PSEED)
    PERM = np.stack([np.stack([prng.permutation(P) for _ in range(G)], 1) for _ in range(NP)])  # NP x P x G
    rows, pergen = [], []
    for f in feats:
        x = df[f].to_numpy(dtype=float).reshape(P, G)
        if np.nanstd(x) == 0 or np.isnan(x).any():
            continue
        r = dict(feature=f, primary=f in PRIMARY)
        # A pooled
        r["A_spearman"] = spearmanr(x.ravel(), y.ravel())[0]; r["A_kendall"] = kendalltau(x.ravel(), y.ravel())[0]
        # B within-generator (ranks within generator column)
        rx = np.apply_along_axis(rankdata, 0, x) / P; ry = np.apply_along_axis(rankdata, 0, y) / P
        r["B_wg_spearman"] = np.corrcoef(rx.ravel(), ry.ravel())[0, 1]
        # C within-prompt Kendall across the 8 generators
        taus = np.array([kendalltau(x[i], y[i])[0] for i in range(P)])
        r["C_wp_kendall_mean"] = np.nanmean(taus)
        # D two-way FE on global ranks
        gx = rankdata(x.ravel()).reshape(P, G); gy = rankdata(y.ravel()).reshape(P, G)
        fe = lambda a: a - a.mean(0, keepdims=True) - a.mean(1, keepdims=True) + a.mean()
        r["D_twoway_fe"] = np.corrcoef(fe(gx).ravel(), fe(gy).ravel())[0, 1]
        # bootstrap over prompts (chunks)
        bA, bB, bC, bD = [], [], [], []
        for c0 in range(0, NB, 1000):
            I = BI[c0:c0 + 1000]; xb, yb = x[I], y[I]  # nb x P x G
            gxb = rankdata(xb.reshape(len(I), -1), axis=1); gyb = rankdata(yb.reshape(len(I), -1), axis=1)
            bA.append(rowcorr(gxb, gyb))
            rxb = rankdata(xb, axis=1); ryb = rankdata(yb, axis=1)
            bB.append(rowcorr(rxb.reshape(len(I), -1), ryb.reshape(len(I), -1)))
            bC.append(np.nanmean(taus[I], axis=1))
            gxb = gxb.reshape(xb.shape); gyb = gyb.reshape(yb.shape)
            fe3 = lambda a: a - a.mean(1, keepdims=True) - a.mean(2, keepdims=True) + a.mean((1, 2), keepdims=True)
            bD.append(rowcorr(fe3(gxb).reshape(len(I), -1), fe3(gyb).reshape(len(I), -1)))
        for nm, b in zip("ABCD", [bA, bB, bC, bD]):
            b = np.concatenate(b); r[f"{nm}_lo"], r[f"{nm}_hi"] = np.percentile(b, [2.5, 97.5])
        # permutation p for B: permute MOS within generator
        ryv = ry
        null = []
        for c0 in range(0, NP, 1000):
            Pm = PERM[c0:c0 + 1000]  # n x P x G
            ryp = np.take_along_axis(np.broadcast_to(ryv, Pm.shape), Pm, axis=1)
            null.append(rowcorr(np.broadcast_to(rx.ravel(), (len(Pm), P * G)), ryp.reshape(len(Pm), -1)))
        null = np.concatenate(null)
        r["B_perm_p"] = (1 + (np.abs(null) >= abs(r["B_wg_spearman"])).sum()) / (NP + 1)
        rows.append(r)
        if f in PRIMARY:
            for j, g in enumerate(gens):
                pergen.append(dict(feature=f, generator=g, spearman=spearmanr(x[:, j], y[:, j])[0],
                                   mean_feature=np.mean(x[:, j]), mean_MOS=np.mean(y[:, j])))
    res = pd.DataFrame(rows)
    prim = res.primary
    res.loc[prim, "B_q_BH_primary"] = multipletests(res.loc[prim, "B_perm_p"], method="fdr_bh")[1]
    res.loc[~prim, "B_q_BH_exploratory"] = multipletests(res.loc[~prim, "B_perm_p"], method="fdr_bh")[1]

    def verdict(r):
        if not r.primary: return ""
        sigB = r.B_q_BH_primary < 0.05 and (r.B_hi < 0 or r.B_lo > 0)
        if sigB and r.B_wg_spearman < 0: return "replicates"
        if sigB and r.B_wg_spearman > 0: return "reverses"
        others = [(r[f"{a}_hi"] < 0) for a in "ACD"]
        return "partial" if any(others) else "fails"
    res["verdict"] = res.apply(verdict, axis=1)
    pg = pd.DataFrame(pergen)
    gl = pg.groupby("feature").apply(lambda t: spearmanr(t.mean_feature, t.mean_MOS)[0], include_groups=False).rename("generator_level_spearman_n8")
    res = res.merge(gl, left_on="feature", right_index=True, how="left")
    res.to_csv(f"{OUT}/mate3d_{TAG}_correlations.csv", index=False)
    pg.to_csv(f"{OUT}/mate3d_{TAG}_per_generator.csv", index=False)
    json.dump(dict(n_meshes=len(df), failures=bad.to_dict("records"), n_boot=NB, boot_seed=BSEED, n_perm=NP, perm_seed=PSEED,
                   generator_mean_MOS=df.groupby("generator").Geometry.mean().round(3).to_dict()),
              open(f"{OUT}/mate3d_{TAG}_meta.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    cols = ["feature", "A_spearman", "A_lo", "A_hi", "B_wg_spearman", "B_lo", "B_hi", "B_perm_p", "B_q_BH_primary", "C_wp_kendall_mean", "C_lo", "C_hi", "D_twoway_fe", "D_lo", "D_hi", "generator_level_spearman_n8", "verdict"]
    print(res[prim][cols].round(3).to_string())
    print(res[~prim].sort_values("B_perm_p")[["feature", "A_spearman", "B_wg_spearman", "B_lo", "B_hi", "B_q_BH_exploratory", "C_wp_kendall_mean", "D_twoway_fe"]].round(3).to_string())
    print(pg.pivot(index="feature", columns="generator", values="spearman").round(2).to_string())
    print(df.groupby("generator").Geometry.mean().round(2).to_dict())


if __name__ == "__main__":
    main()
