"""Pre-registered GSO defect-injection analysis (analysis/PREREGISTRATION_M2.md §2)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_comparison as mc
ROOT = mc.ROOT; OUT = f"{ROOT}/results/gso"; os.makedirs(OUT, exist_ok=True)
NB, SEED = 10_000, 2028
TARGET = {"holes": ["boundary_edge_frac"], "floaters": ["n_small_components"],
          "interpenetrating": ["self_intersect_face_frac", "hidden_surface_frac"], "thin_fin": ["thin_frac_0.005"],
          "noise": ["dihedral_mean_deg", "angle_defect_mean"], "crossing_sheet": ["self_intersect_face_frac"],
          "hidden_shell": ["hidden_surface_frac"]}
CHECKS = ["boundary_edge_frac", "n_boundary_loops", "n_small_components", "n_components", "self_intersect_face_frac",
          "hidden_surface_frac", "thin_frac_0.005", "thin_frac_0.01", "dihedral_mean_deg", "angle_defect_mean",
          "genus_proxy", "winding_inconsistent_edge_frac", "nonmanifold_edge_frac", "sliver_face_frac"]
STRICT_OK = {"boundary_edge_frac", "n_boundary_loops", "n_small_components", "self_intersect_face_frac",
             "hidden_surface_frac", "thin_frac_0.005", "thin_frac_0.01", "winding_inconsistent_edge_frac", "nonmanifold_edge_frac"}


def add_genus(d):
    d["genus_proxy"] = np.maximum(0, 2 * d["n_components"] - d["euler_characteristic"]) / 2
    return d


def main():
    g = add_genus(pd.read_csv(f"{ROOT}/results/probes/gso.csv"))
    rel = g.path.str.replace(r".*gso_meshes/", "", regex=True)
    g["obj"] = rel.str.split("/").str[0]; g["variant"] = rel.str.split("/").str[1].str.replace(".ply", "")
    fails = g[g.status != "ok"][["path", "status"]]
    g = g[g.status == "ok"]
    # benchmark-calibrated thresholds: P95 over silver-holdout assets with no crowd geometry defect
    db = add_genus(mc.load_probes())
    sl = pd.read_csv(f"{mc.DB}/silver_labels.csv")
    ho = set(pd.read_parquet(f"{mc.DB}/vlm_predictions_silver_holdout.parquet")["object_id"].unique())
    geo = [f"{d}_majority_vote" for d in mc.DEFECTS]
    clean_ids = set(sl[sl.object_id.isin(ho) & (sl[geo].sum(1) == 0)].object_id)
    ref = db[(db.split == "silver") & db.object_id.isin(clean_ids)]
    tau_c = {c: float(np.percentile(ref[c], 95)) for c in CHECKS}
    json.dump(dict(n_ref_assets=len(ref), tau_c=tau_c), open(f"{OUT}/tau_calibrated.json", "w"), indent=1)

    wide = g.pivot(index="obj", columns="variant")
    objs = wide.index.to_numpy(); n = len(objs)
    rng = np.random.default_rng(SEED); W = np.zeros((NB, n))
    np.add.at(W, (np.arange(NB)[:, None], rng.integers(0, n, (NB, n))), 1.0)
    rows = []
    variants = sorted(v for v in g.variant.unique() if v != "clean")
    for c in CHECKS:
        clean = wide[(c, "clean")].to_numpy(float)
        for v in variants:
            inj = wide[(c, v)].to_numpy(float)
            ok = ~np.isnan(inj) & ~np.isnan(clean)
            M = (inj[:, None] > clean[None, :]) + 0.5 * (inj[:, None] == clean[None, :])
            M = np.where(ok[:, None] & ok[None, :], M, 0.0)
            Wk = W * ok
            auc_b = ((Wk @ M) * Wk).sum(1) / (Wk.sum(1) ** 2)
            auc = M[ok][:, ok].mean()
            defect, sev = v.rsplit("_S", 1)
            r = dict(check=c, defect=defect, severity=int(sev), n=int(ok.sum()), is_target=c in TARGET[defect],
                     AUROC=auc, AUROC_lo=np.percentile(auc_b, 2.5), AUROC_hi=np.percentile(auc_b, 97.5),
                     paired_increase_rate=float(np.mean(inj[ok] > clean[ok])),
                     median_clean=float(np.median(clean[ok])), median_injected=float(np.median(inj[ok])))
            for nm, tau in (("strict", 0.0 if c in STRICT_OK else None), ("calib", tau_c[c])):
                if tau is None: continue
                sens_b = (Wk @ (inj > tau)) / Wk.sum(1); spec_b = (Wk @ (clean <= tau)) / Wk.sum(1)
                r[f"sens_{nm}"] = float(np.mean(inj[ok] > tau)); r[f"spec_{nm}"] = float(np.mean(clean[ok] <= tau))
                r[f"sens_{nm}_lo"], r[f"sens_{nm}_hi"] = np.percentile(sens_b, [2.5, 97.5])
                r[f"spec_{nm}_lo"], r[f"spec_{nm}_hi"] = np.percentile(spec_b, [2.5, 97.5])
                r[f"tau_{nm}"] = tau
            rows.append(r)
    res = pd.DataFrame(rows); res.to_csv(f"{OUT}/gso_detection.csv", index=False)
    meta = [json.load(open(f"{ROOT}/results/gso_meshes/{o}/meta.json")) for o in objs]
    json.dump(dict(n_objects=n, probe_failures=fails.to_dict("records"), skipped=[(m["name"], m["skipped"]) for m in meta if m["skipped"]],
                   clean_faces_median=float(np.median([m["clean_faces"] for m in meta])),
                   clean_watertight_frac=float(wide[("watertight", "clean")].astype(bool).mean())),
              open(f"{OUT}/gso_meta.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    t = res[res.is_target]
    print(t[["check", "defect", "severity", "n", "AUROC", "AUROC_lo", "AUROC_hi", "paired_increase_rate", "sens_strict", "spec_strict", "sens_calib", "spec_calib", "tau_calib"]].round(3).to_string())
    X = res.pivot_table(index="check", columns=["defect", "severity"], values="AUROC")
    X.round(2).to_csv(f"{OUT}/gso_crosstalk_auroc.csv")
    print(res[res.severity == 3].pivot(index="check", columns="defect", values="AUROC").round(2).to_string())
    print(json.load(open(f"{OUT}/gso_meta.json")))
    print({k: round(v, 5) for k, v in tau_c.items()}, len(ref))


if __name__ == "__main__":
    main()
