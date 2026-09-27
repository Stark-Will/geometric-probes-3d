"""Figure 1 (pipeline schematic + real failure-case gallery) and Figure 3a (GSO injection examples).
Cases are selected by fixed, documented rules (no hand-picking); selections are written to
paper/figures/gallery_cases.csv. Run from repo root: .venv/bin/python paper/scripts/fig_gallery.py"""
import json, os, pickle, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/analysis")
from geomcheck.visualize import prepare, face_flags, render, COLORS
import main_comparison as mc
FIG = f"{ROOT}/paper/figures"
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
PRIM = ["q_fused_incomplete", "q_form_surface", "q_extra_geometry"]


def frozen_scores(df):
    feats = mc.featurize(df)
    fz = json.load(open(f"{ROOT}/results/frozen_models/geomprobe_lr_frozen.json"))
    X = feats[fz["features"]].fillna(pd.Series(fz["medians"])).to_numpy()
    out = {}
    for d in PRIM:
        m = pickle.load(open(f"{ROOT}/results/frozen_models/geomprobe_lr_{d}.pkl", "rb"))
        out[d] = m.predict_proba(X)[:, 1]
    return out, {d: fz["models"][d]["threshold"] for d in PRIM}


def select_cases():
    rows = []
    # --- 3D-DefectBench expert set (golden), unanimous-agreement cells only
    p = mc.load_probes(); g = p[p.split == "golden"].reset_index(drop=True)
    sc, thr = frozen_scores(g)
    lab = pd.read_csv(f"{mc.DB}/golden_labels.csv").set_index("object_id").loc[g.object_id].reset_index()
    for d in PRIM: g[f"P_{d}"] = sc[d]
    def cell(d): return lab[f"{d}_agreement_rate"].to_numpy() >= 1.0
    def y(d): return lab[f"{d}_majority_vote"].to_numpy(int)
    rules = [("DB-TP-fused", "q_fused_incomplete", cell("q_fused_incomplete") & (y("q_fused_incomplete") == 1), "max",
              "3D-DefectBench: experts unanimous 'fused/incomplete'; highest probe score (true positive)"),
             ("DB-TP-extra", "q_extra_geometry", cell("q_extra_geometry") & (y("q_extra_geometry") == 1), "max",
              "3D-DefectBench: experts unanimous 'extra geometry'; highest probe score (true positive)"),
             ("DB-FP-fused", "q_fused_incomplete", np.all([cell(d) & (y(d) == 0) for d in PRIM], 0), "max",
              "3D-DefectBench: experts unanimous 'no defect' on all 3 primary defects; highest fused score (false positive)"),
             ("DB-FN-form", "q_form_surface", cell("q_form_surface") & (y("q_form_surface") == 1), "min",
              "3D-DefectBench: experts unanimous 'form/surface defect'; lowest probe score (false negative)")]
    used = set()
    for cid, d, mask, how, desc in rules:
        idx = [i for i in np.flatnonzero(mask) if i not in used]; s = g.loc[idx, f"P_{d}"]  # distinct assets per panel
        i = s.idxmax() if how == "max" else s.idxmin(); used.add(i)
        rows.append(dict(case=cid, dataset="3D-DefectBench", path=g.loc[i, "path"], rule=desc,
                         info=f"P={g.loc[i, f'P_{d}']:.2f} (thr {thr[d]:.2f}), {lab.loc[i, 'model_version']}"))
    # --- MATE-3D (human geometry MOS)
    import mate3d_validation as mv
    df, _ = mv.load()
    lo, hi = df.Geometry.quantile(0.1), df.Geometry.quantile(0.9)
    i = df[df.Geometry <= lo]["thin_frac_0.005"].idxmax()
    rows.append(dict(case="MATE-lowMOS-thin", dataset="MATE-3D", path=df.loc[i, "path"],
                     rule="MATE-3D: bottom-decile geometry MOS; highest thin_frac_0.005",
                     info=f"MOS={df.loc[i, 'Geometry']:.2f}; thin={df.loc[i, 'thin_frac_0.005']:.3f}\n{df.loc[i, 'generator']}"))
    i = df[df.Geometry >= hi]["P_defect_mean"].idxmax()
    rows.append(dict(case="MATE-highMOS-probeFP", dataset="MATE-3D", path=df.loc[i, "path"],
                     rule="MATE-3D: top-decile geometry MOS; highest frozen GeomProbe-LR mean defect score (probe false alarm on a human-preferred mesh)",
                     info=f"MOS={df.loc[i, 'Geometry']:.2f}; mean P={df.loc[i, 'P_defect_mean']:.2f}\n{df.loc[i, 'generator']}"))
    # --- GSO clean scans: real scans that already fail strict checks
    gs = pd.read_csv(f"{ROOT}/results/probes/gso.csv"); gs = gs[gs.path.str.endswith("/clean.ply") & (gs.status == "ok")].reset_index(drop=True)
    i = gs.self_intersect_face_frac.idxmax()
    rows.append(dict(case="GSO-clean-selfint", dataset="GSO (clean scan)", path=gs.loc[i, "path"],
                     rule="GSO clean scans: highest self_intersect_face_frac (strict check fires on an undamaged scan)",
                     info=f"self-int={gs.loc[i, 'self_intersect_face_frac']:.3f}"))
    i = gs.hidden_surface_frac.idxmax()
    rows.append(dict(case="GSO-clean-hidden", dataset="GSO (clean scan)", path=gs.loc[i, "path"],
                     rule="GSO clean scans: highest hidden_surface_frac",
                     info=f"hidden={gs.loc[i, 'hidden_surface_frac']:.3f}"))
    t = pd.DataFrame(rows); t.to_csv(f"{FIG}/gallery_cases.csv", index=False)
    return t


def panel(ax, path, title, xray=None):
    V, F = prepare(f"{ROOT}/{path}")
    fl = face_flags(V, F)
    if xray is None: xray = fl["hidden"].mean() > 0.01
    render(ax, V, F, fl, alpha_unflagged=0.25 if xray else 1.0)
    items = [f"{ {'self_intersect':'self-int','small_component':'small','thin':'thin','hidden':'hidden','boundary':'boundary'}[k]} {100 * v.mean():.1f}%" for k, v in fl.items() if v.mean() > 0]
    frac = ", ".join(items[:2]) + (",\n" + ", ".join(items[2:]) if len(items) > 2 else "")
    ax.set_title(title + ("\nflagged: " + frac if frac else "\nno faces flagged"), fontsize=5.6, pad=0)


def legend(fig, y=0.0):
    h = [Patch(color=COLORS[k], label=l) for k, l in [("self_intersect", "self-intersecting face"), ("small_component", "small component (<1% area)"),
                                                     ("thin", "thin wall (<0.005 diag)"), ("hidden", "hidden (not visible from 64 dirs)"), ("boundary", "boundary (hole) edge")]]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, y))


def fig1_pipeline():
    rt = pd.read_csv(f"{ROOT}/results/runtime_summary.csv")  # written by fig_results.py
    tmed = f"{rt.sec_median.min():.1f}–{rt.sec_median.max():.1f} s"; rss = f"{rt.rss_max_mb.max():.0f} MB"
    fig, ax = plt.subplots(figsize=(7.4, 1.8)); ax.set_axis_off(); ax.set_xlim(0, 10.7); ax.set_ylim(0, 2)
    specs = [(1.3, "Generated mesh\n(GLB / OBJ / PLY)"),
             (2.0, "Normalise to unit\nbbox diagonal; weld\nvertices; optional\nQEM decimation to\n10k faces (topology-\npreserving)"),
             (2.6, "Deterministic CPU probes\n8 families, 30 features:\ntopology, boundary/manifold,\nself-intersection, hidden\nsurface, thin wall, roughness,\ntriangle quality, size"),
             (1.9, "Per-asset features\nmedian " + tmed + "\nper mesh; peak RSS\n\u2264 " + rss + "; no GPU"),
             (1.9, "Evaluated against:\nexpert defect labels\nhuman geometry MOS\ninjected defects\nMLLM pseudo-labels")]
    x = 0.05
    for i, (w, t) in enumerate(specs):
        ax.add_patch(FancyBboxPatch((x, 0.15), w, 1.7, boxstyle="round,pad=0.03", fc="#f2f2f2", ec="k", lw=0.6))
        ax.text(x + w / 2, 1.0, t, ha="center", va="center", fontsize=5.6, linespacing=1.25)
        if i < len(specs) - 1:
            ax.annotate("", xy=(x + w + 0.15, 1.0), xytext=(x + w + 0.02, 1.0), arrowprops=dict(arrowstyle="->", lw=0.8))
        x += w + 0.17
    fig.savefig(f"{FIG}/fig1a_pipeline.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/fig1a_pipeline.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig1_gallery(t):
    fig = plt.figure(figsize=(7.4, 5.2))
    for k, r in t.reset_index(drop=True).iterrows():
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        panel(ax, r.path, f"({chr(97 + k)}) {r.case}\n{r['info']}")
    legend(fig); fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0.06, wspace=0.0, hspace=0.25)
    fig.savefig(f"{FIG}/fig1b_gallery.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/fig1b_gallery.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig3a_gso_examples():
    gs = pd.read_csv(f"{ROOT}/results/probes/gso.csv")
    gs["obj"] = gs.path.str.split("/").str[-2]
    okobj = sorted(o for o, d in gs.groupby("obj") if (d.status == "ok").all() and len(d) == 22)
    obj = okobj[0]  # rule: alphabetically first object with all 22 variants probed OK
    vs = ["clean", "holes_S3", "floaters_S3", "interpenetrating_S3", "crossing_sheet_S3", "hidden_shell_S3", "thin_fin_S3", "noise_S3"]
    fig = plt.figure(figsize=(7.2, 3.4))
    for k, v in enumerate(vs):
        ax = fig.add_subplot(2, 4, k + 1, projection="3d")
        V, F = prepare(f"{ROOT}/results/gso_meshes/{obj}/{v}.ply"); fl = face_flags(V, F)
        render(ax, V, F, fl, alpha_unflagged=0.3 if v == "hidden_shell_S3" else 1.0, zoom=1.9)
        ax.set_title(v.replace("_S3", " (S3)").replace("_", " "), fontsize=6)
    legend(fig, y=0.0); fig.subplots_adjust(left=0, right=1, top=0.93, bottom=0.08, wspace=0, hspace=0.1)
    fig.savefig(f"{FIG}/fig3a_gso_examples.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/fig3a_gso_examples.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    json.dump(dict(object=obj, rule="alphabetically first GSO object with all 22 variants probed OK"), open(f"{FIG}/gso_example_object.json", "w"))


if __name__ == "__main__":
    fig1_pipeline(); print("pipeline ok")
    t = select_cases(); print(t[["case", "path", "info"]].to_string())
    fig1_gallery(t); print("gallery ok")
    fig3a_gso_examples(); print("gso ok")
