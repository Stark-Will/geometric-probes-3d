"""Figures 2-9 and Table 1 (datasets/runtime) generated ONLY from committed result files.
Run from repo root: .venv/bin/python paper/scripts/fig_results.py"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/analysis")
R = f"{ROOT}/results"; FIG = f"{ROOT}/paper/figures"; TAB = f"{ROOT}/paper/tables"
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
DEF = {"q_fused_incomplete": "Fused / incomplete", "q_form_surface": "Form / surface", "q_extra_geometry": "Extra geometry"}
NICE = {"claude_haiku_4_5": "Claude Haiku 4.5", "claude_opus_4_7": "Claude Opus 4.7", "claude_sonnet_4_6": "Claude Sonnet 4.6",
        "gemini_2_5_pro": "Gemini 2.5 Pro", "gemini_3_1_flash_lite": "Gemini 3.1 Flash-Lite", "gemini_3_1_pro": "Gemini 3.1 Pro",
        "gpt_4o": "GPT-4o", "gpt_5_4": "GPT-5.4", "gpt_5_mini": "GPT-5 mini", "mistral_small_3_1_24b": "Mistral Small 3.1 24B",
        "qwen2_5_vl_7b": "Qwen2.5-VL 7B", "qwen_3_5_397b_a17b": "Qwen3.5 397B-A17B"}
def nice(s): return NICE.get(s, s)
def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: forest plot (expert cells)
def fig2():
    d = pd.read_csv(f"{R}/main_comparison_expert_agreement.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.6), sharex=True)
    for ax, (k, lab) in zip(axs, DEF.items()):
        t = d[d.defect == k].sort_values("MCC").reset_index(drop=True)
        y = np.arange(len(t))
        for i, r in t.iterrows():
            c = "#d62728" if r.predictor == "GeomProbe-LR" else ("#7f7f7f" if r.predictor in ("FaceCount-LR", "Generator-only") else "#1f77b4")
            ax.errorbar(r.MCC, i, xerr=[[r.MCC - r.MCC_lo], [r.MCC_hi - r.MCC]], fmt="o", ms=3, color=c, elinewidth=0.8, capsize=0)
        ax.set_yticks(y); ax.set_yticklabels([nice(p) for p in t.predictor], fontsize=6.5)
        n = int(t.n_cells.iloc[0]); npos = int(t.n_pos.iloc[0])
        ax.set_title(f"{lab}\n(n={n}, {npos} positive)")
        ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("MCC (95% bootstrap CI)")
    fig.tight_layout(); save(fig, "fig2_forest_mcc")


# ---------------------------------------------------------------- Fig 3: GSO severity curves
TARGET = {"holes": ["boundary_edge_frac"], "floaters": ["n_small_components"],
          "interpenetrating": ["self_intersect_face_frac", "hidden_surface_frac"], "thin_fin": ["thin_frac_0.005"],
          "noise": ["dihedral_mean_deg", "angle_defect_mean"], "crossing_sheet": ["self_intersect_face_frac"],
          "hidden_shell": ["hidden_surface_frac"]}
def fig3():
    g = pd.read_csv(f"{R}/gso/gso_detection.csv")
    fig, axs = plt.subplots(1, 7, figsize=(7.4, 1.9), sharey=True)
    for ax, (dft, checks) in zip(axs, TARGET.items()):
        for j, c in enumerate(checks):
            t = g[(g.defect == dft) & (g.check == c)].sort_values("severity")
            ax.errorbar(t.severity + 0.06 * j, t.AUROC, yerr=[t.AUROC - t.AUROC_lo, t.AUROC_hi - t.AUROC], marker="o", ms=3,
                        lw=1, capsize=0, label=c.replace("_face_frac", "").replace("_frac", "").replace("_", " "))
        ax.set_title(dft.replace("_", " "), fontsize=7.5); ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["S1", "S2", "S3"])
        ax.axhline(0.5, color="k", lw=0.5, ls=":"); ax.set_ylim(0.4, 1.02); ax.legend(fontsize=5, loc="lower right", frameon=False)
    axs[0].set_ylabel("AUROC (injected vs clean)")
    fig.tight_layout(); save(fig, "fig3_gso_severity")


# ---------------------------------------------------------------- Fig 4: cross-talk heatmap
def fig4():
    g = pd.read_csv(f"{R}/gso/gso_detection.csv")
    g = g[g.severity == 2]
    checks = ["boundary_edge_frac", "n_boundary_loops", "n_small_components", "n_components", "self_intersect_face_frac",
              "hidden_surface_frac", "thin_frac_0.005", "thin_frac_0.01", "dihedral_mean_deg", "angle_defect_mean", "genus_proxy", "sliver_face_frac"]
    defects = list(TARGET)
    M = g.pivot(index="check", columns="defect", values="AUROC").reindex(index=checks, columns=defects)
    T = g.pivot(index="check", columns="defect", values="is_target").reindex(index=checks, columns=defects).fillna(False)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    im = ax.imshow(M.to_numpy(float), cmap="RdBu_r", vmin=0, vmax=1)
    for i in range(len(checks)):
        for j in range(len(defects)):
            v = M.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, fontweight="bold" if T.iat[i, j] else "normal",
                    color="w" if abs(v - 0.5) > 0.35 else "k")
            if T.iat[i, j]: ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="k", lw=1.2))
    ax.set_xticks(range(len(defects))); ax.set_xticklabels([d.replace("_", " ") for d in defects], fontsize=6.5, rotation=40, ha="right")
    ax.set_yticks(range(len(checks))); ax.set_yticklabels(checks, fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.04, label="AUROC (severity S2)")
    ax.set_title("Check $\\times$ injected defect (boxed = pre-registered target)")
    fig.tight_layout(); save(fig, "fig4_gso_crosstalk")


# ---------------------------------------------------------------- Fig 5: MATE-3D A/B/C/D
MATE_FEATS = ["thin_frac_0.005", "genus_proxy", "self_intersect_face_frac", "dihedral_mean_deg", "hidden_surface_frac",
              "P_form_surface", "P_fused_incomplete", "P_extra_geometry", "P_defect_mean", "n_components"]
def fig5():
    m = pd.read_csv(f"{R}/mate3d/mate3d_dec10k_correlations.csv").set_index("feature")
    est = [("A_spearman", "A_lo", "A_hi", "A: pooled Spearman"), ("B_wg_spearman", "B_lo", "B_hi", "B: within-generator Spearman (primary)"),
           ("C_wp_kendall_mean", "C_lo", "C_hi", "C: within-prompt Kendall"), ("D_twoway_fe", "D_lo", "D_hi", "D: two-way FE (prompt+generator)")]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cols = ["#9467bd", "#d62728", "#2ca02c", "#1f77b4"]
    for j, (e, lo, hi, lab) in enumerate(est):
        y = np.arange(len(MATE_FEATS)) + (j - 1.5) * 0.18
        v = m.loc[MATE_FEATS, e].to_numpy(float)
        ax.errorbar(v, y, xerr=[v - m.loc[MATE_FEATS, lo], m.loc[MATE_FEATS, hi] - v], fmt="o", ms=2.5, color=cols[j], elinewidth=0.8, label=lab)
    labels = [f + ("  (exploratory)" if f == "n_components" else "") + ("  [" + str(m.loc[f, "verdict"]) + "]" if isinstance(m.loc[f, "verdict"], str) else "") for f in MATE_FEATS]
    ax.set_yticks(range(len(MATE_FEATS))); ax.set_yticklabels(labels, fontsize=6.5); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.5); ax.set_xlabel("Association with human geometry MOS (negative = more defect, lower MOS)")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, frameon=False)
    ax.set_title("MATE-3D (1,280 meshes, 8 generators, 160 prompts); decimated to 10k faces")
    fig.tight_layout(); save(fig, "fig5_mate3d_estimands")


# ---------------------------------------------------------------- Fig 6: per-generator heatmap
def fig6():
    p = pd.read_csv(f"{R}/mate3d/mate3d_dec10k_per_generator.csv")
    feats = [f for f in MATE_FEATS if f in set(p.feature)]
    M = p.pivot(index="feature", columns="generator", values="spearman").reindex(feats)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    im = ax.imshow(M.to_numpy(float), cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M.iat[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(M.columns, rotation=35, ha="right", fontsize=6.5)
    ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(M.index, fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman $\\rho$ with geometry MOS (n=160 per generator)")
    fig.tight_layout(); save(fig, "fig6_mate3d_per_generator")


# ---------------------------------------------------------------- Fig 7: ECDFs crowd-clean DefectBench vs clean GSO
def fig7():
    import main_comparison as mc
    from gso_injection import add_genus
    db = add_genus(mc.load_probes())
    sl = pd.read_csv(f"{mc.DB}/silver_labels.csv")
    ho = set(pd.read_parquet(f"{mc.DB}/vlm_predictions_silver_holdout.parquet")["object_id"].unique())
    geo = [f"{d}_majority_vote" for d in mc.DEFECTS]
    clean_ids = set(sl[sl.object_id.isin(ho) & (sl[geo].sum(axis=1) == 0)].object_id)
    ref = db[(db.split == "silver") & db.object_id.isin(clean_ids)]
    g = pd.read_csv(f"{R}/probes/gso.csv"); g = g[g.path.str.endswith("/clean.ply") & (g.status == "ok")]
    tau = json.load(open(f"{R}/gso/tau_calibrated.json"))
    assert tau["n_ref_assets"] == len(ref)
    feats = ["self_intersect_face_frac", "hidden_surface_frac", "thin_frac_0.005", "n_small_components"]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9))
    for ax, f in zip(axs, feats):
        for data, lab, c in ((ref[f], f"3D-DefectBench, crowd defect-free (n={len(ref)})", "#d62728"), (g[f], f"GSO clean scans (n={len(g)})", "#1f77b4")):
            x = np.sort(data.to_numpy(float)); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, lw=1, label=lab)
        ax.axvline(tau["tau_c"][f], color="k", ls="--", lw=0.6)
        hi = max(np.percentile(ref[f], 99), np.percentile(g[f], 99), tau["tau_c"][f]) * 1.15
        ax.set_xlim(-0.02 * hi, hi)
        ax.set_title(f, fontsize=7); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("ECDF"); axs[0].legend(fontsize=5.5, loc="lower right", frameon=False)
    fig.tight_layout(); save(fig, "fig7_ecdf_clean_reference")


# ---------------------------------------------------------------- Fig 8: ablation bars
def fig8():
    a = pd.read_csv(f"{R}/ablations/lofo_family_rules.csv")
    cfgs = ["all"] + [c for c in a.config.unique() if c.startswith("only:")] + [c for c in a.config.unique() if c.startswith("-")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, (k, lab) in zip(axs, DEF.items()):
        t = a[a.defect == k].set_index("config").reindex(cfgs)
        y = np.arange(len(cfgs))
        col = ["#d62728" if c == "all" else ("#2ca02c" if c.startswith("only") else "#7f7f7f") for c in cfgs]
        ax.barh(y, t.AUROC - 0.5, left=0.5, color=col, height=0.7)
        ax.errorbar(t.AUROC, y, xerr=[t.AUROC - t.AUROC_lo, t.AUROC_hi - t.AUROC], fmt="none", ecolor="k", elinewidth=0.6)
        ax.axvline(0.5, color="k", lw=0.5); ax.set_title(lab); ax.set_xlabel("Expert-cell AUROC")
        ax.set_yticks(y); ax.set_yticklabels(cfgs, fontsize=6.5); ax.invert_yaxis(); ax.set_xlim(0.2, 1.0)
    fig.suptitle("Feature-family ablation (silver-trained; test-set observation, not used for model selection)", fontsize=8)
    fig.tight_layout(); save(fig, "fig8_ablation_families")


# ---------------------------------------------------------------- Fig 9: AND / OR rule dMCC
def fig9():
    b = pd.read_csv(f"{R}/ablations/and_or_rule_bootstrap.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.4), sharex=True)
    for ax, (k, lab) in zip(axs, DEF.items()):
        t = b[(b.defect == k) & (b.rule == "AND")].sort_values("MCC_vlm").reset_index(drop=True)
        for i, r in t.iterrows():
            c = "#d62728" if r.q_BH < 0.05 else "#1f77b4"
            ax.errorbar(r.dMCC, i, xerr=[[r.dMCC - r.lo], [r.hi - r.dMCC]], fmt="o", ms=3, color=c, elinewidth=0.8)
        ax.set_yticks(range(len(t))); ax.set_yticklabels([f"{nice(v)} ({m:.2f})" for v, m in zip(t.vlm, t.MCC_vlm)], fontsize=6)
        ax.axvline(0, color="k", lw=0.5); ax.set_title(lab); ax.set_xlabel("$\\Delta$MCC: (VLM AND probe) $-$ VLM")
    fig.suptitle("Training-free AND rule (red: BH q<0.05; label shows the VLM's own MCC)", fontsize=8)
    fig.tight_layout(); save(fig, "fig9_and_rule")


# ---------------------------------------------------------------- Fig 10: Hi3DBench (secondary, MLLM pseudo-labels)
def fig10():
    f = f"{R}/hi3dbench/hi3dbench_correlations.csv"
    if not os.path.exists(f): print("no Hi3DBench results; skip"); return
    h = pd.read_csv(f); pg = pd.read_csv(f"{R}/hi3dbench/hi3dbench_per_generator.csv")
    feats = ["thin_frac_0.005", "genus_proxy", "self_intersect_face_frac", "n_components"]
    est = [("A_spearman", "A_lo", "A_hi", "A: pooled"), ("B_wg_spearman", "B_lo", "B_hi", "B: within-generator (primary)"),
           ("C_wp_kendall_mean", "C_lo", "C_hi", "C: within-prompt Kendall"), ("D_twoway_fe", "D_lo", "D_hi", "D: two-way FE")]
    cols = ["#9467bd", "#d62728", "#2ca02c", "#1f77b4"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    gens = ["crm", "hunyuan", "instant-mesh", "spard", "trellis", "triposr"]
    for k, tgt in enumerate(["geometry_score", "geo_detail_score"]):
        ax = axs[0, k]; t = h[h.target == tgt].set_index("feature").reindex(feats)
        for j, (e, lo, hi, lab) in enumerate(est):
            y = np.arange(len(feats)) + (j - 1.5) * 0.18; v = t[e].to_numpy(float)
            ax.errorbar(v, y, xerr=[v - t[lo], t[hi] - v], fmt="o", ms=2.5, color=cols[j], elinewidth=0.8, label=lab)
        ax.set_yticks(range(len(feats))); ax.set_yticklabels([f"{f}\n[{t.loc[f, 'verdict']}]" for f in feats], fontsize=6)
        ax.invert_yaxis(); ax.axvline(0, color="k", lw=0.5); ax.set_xlim(-0.3, 0.3)
        ax.set_title(f"{tgt} ({'primary' if k == 0 else 'secondary'})", fontsize=7.5)
        ax.set_xlabel("association (negative = more signal, lower score)", fontsize=6.5)
        if k == 0: ax.legend(fontsize=5.5, loc="lower right", frameon=True, framealpha=0.9)
        M = pg[pg.target == tgt].pivot(index="feature", columns="generator", values="spearman").reindex(index=feats, columns=gens)
        ax2 = axs[1, k]; im = ax2.imshow(M.to_numpy(float), cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax2.text(j, i, f"{M.iat[i, j]:.2f}", ha="center", va="center", fontsize=6)
        ax2.set_xticks(range(len(gens))); ax2.set_xticklabels([g + (" (n=247)" if g == "crm" else "") for g in gens], rotation=30, ha="right", fontsize=6)
        ax2.set_yticks(range(len(feats))); ax2.set_yticklabels(feats, fontsize=6)
        ax2.set_title(f"per-generator Spearman with {tgt}", fontsize=7.5)
    fig.colorbar(im, ax=axs[1, :], fraction=0.03, label="Spearman $\\rho$")
    fig.suptitle("Hi3DBench, 2,797 meshes (secondary evidence: targets are MLLM pseudo-labels, not human ratings)", fontsize=7.5)
    save(fig, "fig10_hi3dbench")


# ---------------------------------------------------------------- Table 1: datasets / licences / runtime
def table1():
    rows = []
    spec = [("3D-DefectBench (expert + silver)", "probes/defectbench.csv", "CC BY-NC 4.0 (non-commercial research use; no redistribution of GLBs)", "expert + crowd defect labels"),
            ("MATE-3D", "probes/mate3d_dec10k.csv", "CC BY 4.0", "human MOS (geometry)"),
            ("GSO + injected defects", "probes/gso.csv", "CC BY 4.0 (scans); injections ours", "known injected defect")]
    hp = f"{R}/probes/hi3dbench_dec10k.csv"
    if os.path.exists(hp):
        spec.append(("Hi3DBench (secondary)", "probes/hi3dbench_dec10k.csv", "MIT (HF dataset card)", "MLLM pseudo-labels (geometry plausibility); 2,797 labelled meshes analysed"))
    for name, f, lic, lab in spec:
        p = pd.read_csv(f"{R}/{f}"); ok = p[p.status == "ok"]
        rows.append(dict(dataset=name, n_meshes=len(p), n_ok=len(ok), labels=lab, licence=lic,
                         sec_median=ok.seconds.median(), sec_p95=ok.seconds.quantile(.95),
                         rss_median_mb=ok.peak_rss_mb.median(), rss_max_mb=ok.peak_rss_mb.max(),
                         faces_median=ok.n_faces.median()))
    t = pd.DataFrame(rows); t.to_csv(f"{R}/runtime_summary.csv", index=False)
    with open(f"{TAB}/tab_datasets.tex", "w") as fh:
        fh.write("% generated by paper/scripts/fig_results.py from results/runtime_summary.csv\n")
        fh.write("\\begin{tabular}{p{3.2cm}rrp{3.0cm}p{3.4cm}rrr}\n\\toprule\n")
        fh.write("Dataset & Meshes & Probed OK & Labels & Licence & Time/mesh (s) med [P95] & Peak RSS (MB) med [max] & Faces (med.)\\\\\n\\midrule\n")
        for r in rows:
            fh.write(f"{r['dataset']} & {r['n_meshes']:,} & {r['n_ok']:,} & {r['labels']} & {r['licence']} & "
                     f"{r['sec_median']:.2f} [{r['sec_p95']:.2f}] & {r['rss_median_mb']:.0f} [{r['rss_max_mb']:.0f}] & {r['faces_median']:,.0f}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(t.round(3).to_string())


if __name__ == "__main__":
    for f in (fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, table1):
        f(); print("ok", f.__name__)
