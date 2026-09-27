"""Pre-registered Hi3DBench within-generator scale-up (analysis/PREREGISTRATION_M3_hi3dbench.md, git c003104).
SECONDARY evidence: targets are Hi3DEval MLLM pseudo-labels, not human ratings.
Usage: python analysis/hi3dbench_validation.py"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau
from statsmodels.stats.multitest import multipletests
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/results/hi3dbench"; os.makedirs(OUT, exist_ok=True)
NB, NB_D, BSEED, NP, PSEED = 10_000, 2_000, 2029, 10_000, 2030
PRED = ["thin_frac_0.005", "genus_proxy", "self_intersect_face_frac", "n_components"]
GENS = ["spard", "trellis", "triposr", "instant-mesh", "hunyuan", "crm"]
TARGETS = ["geometry_score", "geo_detail_score"]  # primary, secondary


def load():
    lab = json.load(open(f"{ROOT}/data/hi3dbench/object-level.json"))
    pr = pd.read_csv(f"{ROOT}/results/probes/hi3dbench_dec10k.csv")
    rel = pr.path.str.replace(r".*meshes/", "", regex=True)
    pr["generator"] = rel.str.split("/").str[0]
    pr["prompt"] = rel.str.split("/").str[1].str.replace(r"\.(glb|obj)$", "", regex=True).astype(int)
    pr["key"] = "image2shape_" + pr.generator + "_" + pr.prompt.astype(str)
    pr = pr[pr.generator.isin(GENS) & pr.key.isin(lab.keys())].copy()
    n_labelled = sum(1 for k in lab if k.rsplit("_", 1)[0].replace("image2shape_", "") in GENS)
    bad = pr[pr.status != "ok"][["path", "status"]]
    pr = pr[pr.status == "ok"].copy()
    pr["genus_proxy"] = np.maximum(0, 2 * pr["n_components"] - pr["euler_characteristic"]) / 2
    for t in TARGETS: pr[t] = pr.key.map(lambda k: lab[k][t])
    return pr.reset_index(drop=True), bad, n_labelled


def wg_rank(v, g):
    out = np.empty(len(v))
    for gg in np.unique(g):
        m = g == gg; out[m] = rankdata(v[m]) / m.sum()
    return out


def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def twoway_resid(v, g, p, iters=200, tol=1e-10):
    """Residualise v on generator + prompt dummies (unbalanced) by alternating projections."""
    r = v - v.mean()
    gi = np.unique(g, return_inverse=True)[1]; pi = np.unique(p, return_inverse=True)[1]
    ng, npr = gi.max() + 1, pi.max() + 1
    cg, cp = np.bincount(gi, minlength=ng), np.bincount(pi, minlength=npr)
    for _ in range(iters):
        r0 = r.copy()
        r = r - (np.bincount(gi, r, ng) / cg)[gi]
        r = r - (np.bincount(pi, r, npr) / cp)[pi]
        if np.abs(r - r0).max() < tol: break
    return r


def stats(x, y, g, p):
    A = spearmanr(x, y)[0]
    B = corr(wg_rank(x, g), wg_rank(y, g))
    return A, B


def main():
    df, bad, n_labelled = load()
    prompts = np.sort(df.prompt.unique())
    rows_by_prompt = {q: np.flatnonzero(df.prompt.to_numpy() == q) for q in prompts}
    g = df.generator.to_numpy(); p = df.prompt.to_numpy()
    rng = np.random.default_rng(BSEED)
    BI = rng.integers(0, len(prompts), size=(NB, len(prompts)))
    prng = np.random.default_rng(PSEED)
    gidx = {gg: np.flatnonzero(g == gg) for gg in np.unique(g)}
    res, pergen = [], []
    for tgt in TARGETS:
        y = df[tgt].to_numpy(float)
        ry = wg_rank(y, g)
        for f in PRED:
            x = df[f].to_numpy(float)
            r = dict(target=tgt, feature=f, n=len(df))
            r["A_spearman"], r["B_wg_spearman"] = stats(x, y, g, p)
            # C: within-prompt Kendall tau-b across generators (prompts with >= 3 generators)
            taus = {}
            for q, ix in rows_by_prompt.items():
                if len(ix) >= 3:
                    t = kendalltau(x[ix], y[ix])[0]
                    taus[q] = t
            r["C_wp_kendall_mean"] = float(np.nanmean(list(taus.values()))); r["C_n_prompts"] = len(taus)
            # D: two-way FE on global ranks
            gx, gy = rankdata(x), rankdata(y)
            r["D_twoway_fe"] = corr(twoway_resid(gx, g, p), twoway_resid(gy, g, p))
            # prompt-cluster bootstrap
            bA, bB, bC, bD = [], [], [], []
            tau_arr = np.array([taus.get(q, np.nan) for q in prompts])
            for b in range(NB):
                sel = BI[b]
                ix = np.concatenate([rows_by_prompt[prompts[s]] for s in sel])
                xb, yb, gb = x[ix], y[ix], g[ix]
                bA.append(spearmanr(xb, yb)[0]); bB.append(corr(wg_rank(xb, gb), wg_rank(yb, gb)))
                bC.append(np.nanmean(tau_arr[sel]))
                if b < NB_D:
                    pb = np.concatenate([np.full(len(rows_by_prompt[prompts[s]]), k) for k, s in enumerate(sel)])  # copies = new clusters
                    bD.append(corr(twoway_resid(rankdata(xb), gb, pb), twoway_resid(rankdata(yb), gb, pb)))
            for nm, bb in zip("ABCD", [bA, bB, bC, bD]):
                bb = np.asarray(bb, float); bb = bb[np.isfinite(bb)]
                r[f"{nm}_lo"], r[f"{nm}_hi"] = np.percentile(bb, [2.5, 97.5])
            # permutation p for B: permute target within generator
            rx = wg_rank(x, g); null = np.empty(NP)
            for k in range(NP):
                ryp = ry.copy()
                for gg, ix in gidx.items(): ryp[ix] = ry[prng.permutation(ix)]
                null[k] = corr(rx, ryp)
            r["B_perm_p"] = (1 + (np.abs(null) >= abs(r["B_wg_spearman"])).sum()) / (NP + 1)
            res.append(r)
            for gg in GENS:
                m = g == gg
                if m.sum() > 2:
                    pergen.append(dict(target=tgt, feature=f, generator=gg, n=int(m.sum()), spearman=spearmanr(x[m], y[m])[0],
                                       mean_feature=float(x[m].mean()), mean_target=float(y[m].mean())))
            print(tgt, f, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
    res = pd.DataFrame(res)
    for tgt in TARGETS:
        m = res.target == tgt
        res.loc[m, "B_q_BH"] = multipletests(res.loc[m, "B_perm_p"], method="fdr_bh")[1]

    def verdict(r):
        sigB = r.B_q_BH < 0.05 and (r.B_hi < 0 or r.B_lo > 0)
        if sigB and r.B_wg_spearman < 0: return "replicates"
        if sigB and r.B_wg_spearman > 0: return "reverses"
        return "partial" if any(r[f"{a}_hi"] < 0 for a in "ACD") else "fails"
    res["verdict"] = res.apply(verdict, axis=1)
    res.to_csv(f"{OUT}/hi3dbench_correlations.csv", index=False)
    pd.DataFrame(pergen).to_csv(f"{OUT}/hi3dbench_per_generator.csv", index=False)
    json.dump(dict(n_meshes_analysed=len(df), n_labelled_in_scope=n_labelled, n_prompts=len(prompts),
                   per_generator_n=df.generator.value_counts().to_dict(), failures=bad.to_dict("records"),
                   n_boot=NB, n_boot_D=NB_D, boot_seed=BSEED, n_perm=NP, perm_seed=PSEED,
                   generator_mean_geometry_score=df.groupby("generator").geometry_score.mean().round(3).to_dict(),
                   label_source="Hi3DEval automated MLLM pipeline (pseudo-labels, not human ratings)"),
              open(f"{OUT}/hi3dbench_meta.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    print(res[["target", "feature", "A_spearman", "B_wg_spearman", "B_lo", "B_hi", "B_perm_p", "B_q_BH", "C_wp_kendall_mean", "D_twoway_fe", "D_lo", "D_hi", "verdict"]].round(3).to_string())


if __name__ == "__main__":
    main()
