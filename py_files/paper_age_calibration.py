

import os
import time

import numpy as np
import pandas as pd

import paper_calibration_study as cs
import paper_predictions as pp
import paper_run_full as rf

MODELS = ["GLM", "Tweedie GLM", "EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Neural GA2M+", "Tweedie-CANN"]
AGE_GROUPINGS = ["DrivAge band", "DrivAge x BonusMalus", "BonusMalus x VehAge",
                 "BonusMalus band", "DrivAge x VehAge", "DrivAge x VehPower", "DrivAge band x Region"]
VARIANT_GROUPINGS = ["DrivAge band", "DrivAge x BonusMalus", "BonusMalus x VehAge",
                     "BonusMalus band", "DrivAge x VehAge", "DrivAge band x Region"]
K_VARIANT = [100, 500, 1000, 2000, 5000, 10000]
K_PAPER = 2000                      # credibility constant of the proposed calibrator, used in every table and figure
K_CURVE = ["100", "250", "500", "1000", "2000", "5000", "10000"]
LABEL = {"DrivAge band": "DrivAge",
         "DrivAge x BonusMalus": "DrivAge×BM", "BonusMalus x VehAge": "BM×VehAge", "BonusMalus band": "BM",
         "DrivAge x VehAge": "DrivAge×VehAge", "DrivAge x VehPower": "DrivAge×VehPower",
         "DrivAge band x Region": "DrivAge×Region"}
TARGET, EXPOSURE = "PurePremium", "Exposure"
VARIANT_FILE = "paper_cells_on_A.csv"


def _deviance(y, mu, e, p):
    """Exposure-weighted mean Tweedie deviance (= paper_adjust_clean_plus.tweedie_deviance)."""
    mu = np.maximum(mu, 1e-10)
    d = 2 * (np.where(y > 0, y ** (2 - p), 0.0) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p)
             + mu ** (2 - p) / (2 - p))
    return float(np.sum(e * d) / np.sum(e))


# =============================================================================================================
# 1. Proposed calibrator from the stored predictions (resumable per seed)
# =============================================================================================================
def compute_variants(df_freq, df_sev, pred_dirs, out_dir="v2_age_calibration", seeds=range(41, 91), models=MODELS,
                     groupings=VARIANT_GROUPINGS, ks=K_VARIANT):
    """Paper cell factors on top of A, for every (seed, model, grouping, k), plus A itself (k = -1) as reference.
    Columns: seed, model, grouping, k, test_B, deviance, gini, share_clipped. About 10-15 minutes for 50 seeds."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, VARIANT_FILE)
    done = set(pd.read_csv(path, usecols=["seed"]).seed) if os.path.exists(path) else set()
    power = pp.power()
    df, num_cols, cat_cols = pp.prepare(df_freq, df_sev)
    df = cs.add_bands(df, num_cols)
    t0 = time.time()
    for seed in [int(s) for s in seeds]:
        if seed in done:
            continue
        tr, va, te = pp.split(df, seed)
        yv, ev = va[TARGET].to_numpy(float), va[EXPOSURE].to_numpy(float)
        yt, et = te[TARGET].to_numpy(float), te[EXPOSURE].to_numpy(float)
        obs_train = float(np.sum(tr[EXPOSURE].to_numpy(float) * tr[TARGET].to_numpy(float)))
        obs_test = float(np.sum(et * yt))
        codes = {}
        for g in groupings:
            (iv, it), n = cs._cell_codes([va, te], cs.GROUPINGS[g])
            codes[g] = (iv, it, n)
        rows = []
        for m in models:
            d = pp.load_predictions(pred_dirs, m, seed)
            for s, frame in (("valid", va), ("test", te)):
                if d["idhash"][s] != pp.id_hash(frame["IDpol"].to_numpy(np.int64)):
                    raise RuntimeError(f"{m} seed {seed}: stored {s} policies differ from the regenerated split")
            mv, mt = d["valid"], d["test"]
            c = obs_train / d["train_sum_e_mu"]
            for g, (iv, it, n) in codes.items():
                E = np.bincount(iv, weights=ev, minlength=n)
                O = np.bincount(iv, weights=ev * yv, minlength=n)
                P = np.bincount(iv, weights=ev * mv, minlength=n)
                seen = np.bincount(iv, minlength=n) > 0
                mu_A = c * mt
                rows.append(dict(seed=seed, model=m, grouping=g, k=-1, test_B=float(np.sum(et * mu_A)) / obs_test - 1,
                                 deviance=_deviance(yt, mu_A, et, power), gini=rf.gini_exposure(yt, mu_A, et),
                                 share_clipped=np.nan))
                for k in ks:
                    Z = np.where(seen, E / (E + k), 0.0)
                    f = np.where(seen, np.clip(1 + Z * ((O + 1.0) / (c * P + 1.0) - 1), 0.95, 1.15), 1.0)
                    mu = mu_A * f[it]
                    rows.append(dict(seed=seed, model=m, grouping=g, k=k,
                                     test_B=float(np.sum(et * mu)) / obs_test - 1,
                                     deviance=_deviance(yt, mu, et, power), gini=rf.gini_exposure(yt, mu, et),
                                     share_clipped=float(np.mean((f[seen] <= 0.95 + 1e-9) | (f[seen] >= 1.15 - 1e-9)))))
        rf._append(rows, path)
        print(f"seed {seed}: done | {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.read_csv(path)


# =============================================================================================================
# 2. Tables
# =============================================================================================================
def _cell_vs_A(r, groupings=AGE_GROUPINGS):
    A = r[r["mode"] == "A"].set_index(["seed", "model", "grouping"])
    x = r[(r["mode"] == "cell_only_A") & r.grouping.isin(groupings)].join(
        A[["deviance", "gini"]], on=["seed", "model", "grouping"], rsuffix="_A")
    x["dd"], x["dg"] = x.deviance - x.deviance_A, 1000 * (x.gini - x.gini_A)
    return x


def _variants(r, var, groupings=VARIANT_GROUPINGS):
    """Two variants (clipped and unclipped cell factors after global rescaling), same columns: dd (deviance - A), dg (Gini - A, x1000), test_B, pred / obs totals."""
    A = var[var.k == -1].set_index(["seed", "model", "grouping"])
    obs = r[r["mode"] == "A"].set_index(["seed", "model", "grouping"]).obs_total
    new = var[var.k > 0].copy()
    new["variant"] = "paper cells on A"
    old = []
    for mode, lab in (("cell_only_A", "cell factors on A, no clip"),):
        y = r[(r["mode"] == mode) & r.grouping.isin(groupings) & r.k.isin([str(k) for k in K_VARIANT])].copy()
        y["k"] = y.k.astype(int)
        y["variant"] = lab
        old.append(y[["seed", "model", "grouping", "k", "test_B", "deviance", "gini", "variant"]])
    v = pd.concat([new[["seed", "model", "grouping", "k", "test_B", "deviance", "gini", "variant"]]] + old,
                  ignore_index=True)
    v = v.join(A[["deviance", "gini"]].rename(columns=lambda c: c + "_A"), on=["seed", "model", "grouping"])
    v["obs_total"] = np.asarray(pd.MultiIndex.from_frame(v[["seed", "model", "grouping"]]).map(obs), float)
    v["pred_total"] = v.obs_total * (1 + v.test_B)
    v["dd"], v["dg"] = v.deviance - v.deviance_A, 1000 * (v.gini - v.gini_A)
    return v


def _summ(d):
    return pd.Series({"mean |B| (%)": 100 * d.test_B.abs().mean(),
                      "pooled bias (%)": 100 * (d.pred_total.sum() / d.obs_total.sum() - 1),
                      "deviance - A, median": d.dd.median(), "deviance - A, mean": d.dd.mean(),
                      "worse than A (%)": 100 * (d.dd > 0).mean(), "Gini - A (x1000)": d.dg.mean()})


def tables(cal_dir="v2_calibration", age_dir="v2_age_calibration"):
    r, h = cs.load(cal_dir)
    var = pd.read_csv(os.path.join(age_dir, VARIANT_FILE))
    # check: A recomputed from the predictions = A of the calibration study
    A1 = var[var.k == -1].set_index(["seed", "model", "grouping"])
    A2 = r[r["mode"] == "A"].set_index(["seed", "model", "grouping"])
    j = A1.join(A2[["deviance", "gini"]], rsuffix="_study", how="inner")
    diff = max((j.deviance - j.deviance_study).abs().max(), (j.gini - j.gini_study).abs().max())
    print(f"check: correction A recomputed vs calibration study, {len(j)} cases, max abs difference {diff:.2e}"
          + (" -> OK" if diff < 1e-6 else " -> NOT OK, stop and report"))
    x = _cell_vs_A(r)
    t = {}
    for k in ["1000", "5000", "10000", "BS"]:
        d = x[x.k == k]
        t[f"deviance - A, median, k={k}"] = d.pivot_table(index="model", columns="grouping", values="dd",
                                                          aggfunc="median").reindex(index=MODELS, columns=AGE_GROUPINGS)
        t[f"worse than A, k={k}"] = d.assign(w=d.dd > 0).pivot_table(index="model", columns="grouping", values="w",
                                                                    aggfunc="mean").reindex(index=MODELS, columns=AGE_GROUPINGS)
        t[f"Gini - A x1000, k={k}"] = d.pivot_table(index="model", columns="grouping", values="dg",
                                                    aggfunc="mean").reindex(index=MODELS, columns=AGE_GROUPINGS)
    hh = h[h.grouping.isin(AGE_GROUPINGS)]
    t["validation p<0.05"] = hh.assign(s=hh.p_heterogeneity < 0.05).pivot_table(
        index="model", columns="grouping", values="s", aggfunc="mean").reindex(index=MODELS, columns=AGE_GROUPINGS)
    t["validation finite k_hat"] = hh.assign(s=np.isfinite(hh.k_hat.astype(float))).pivot_table(
        index="model", columns="grouping", values="s", aggfunc="mean").reindex(index=MODELS, columns=AGE_GROUPINGS)
    t["k curve, deviance - A, median"] = x[x.k.isin(K_CURVE + ["BS"])].pivot_table(
        index=["grouping", "model"], columns="k", values="dd", aggfunc="median")[K_CURVE + ["BS"]]
    v = _variants(r, var)
    cl = v[v.variant == "paper cells on A"]
    for k in K_VARIANT:
        d = cl[cl.k == k]
        t[f"clipped: deviance - A, median, k={k}"] = d.pivot_table(index="model", columns="grouping", values="dd",
                                                                   aggfunc="median").reindex(index=MODELS, columns=VARIANT_GROUPINGS)
        t[f"clipped: worse than A, k={k}"] = d.assign(w=d.dd > 0).pivot_table(
            index="model", columns="grouping", values="w", aggfunc="mean").reindex(index=MODELS, columns=VARIANT_GROUPINGS)
        t[f"clipped: Gini - A x1000, k={k}"] = d.pivot_table(index="model", columns="grouping", values="dg",
                                                             aggfunc="mean").reindex(index=MODELS, columns=VARIANT_GROUPINGS)
    s = v.groupby(["grouping", "variant", "k"]).apply(_summ, include_groups=False).reset_index()
    t["variants, summary"] = s.sort_values(["grouping", "variant", "k"]).reset_index(drop=True)
    sel = [("paper cells on A", "DrivAge x BonusMalus"), ("paper cells on A", "DrivAge band"),
           ("cell factors on A, no clip", "DrivAge x BonusMalus")]
    rows = []
    for var_name, g in sel:
        d = v[(v.variant == var_name) & (v.grouping == g) & (v.k == 2000)]
        q = d.groupby("model").apply(_summ, include_groups=False).reindex(MODELS)
        q.insert(0, "grouping", g)
        q.insert(0, "variant", var_name)
        rows.append(q.reset_index())
    t["variants by model, k=2000"] = pd.concat(rows, ignore_index=True)
    return t, x, v


def write_excel(t, path):
    def _note(name):
        if name.startswith("clipped:"):
            return "proposed calibrator: clipped cell factors on top of A, test set, over 50 seeds (figure 1 uses k = 2000)"
        if name.startswith(("deviance - A", "worse than A", "Gini - A")):
            return "unclipped cell factors on top of A, test set, over 50 seeds"
        return {"validation p<0.05": "share of seeds with permutation p < 0.05 on the validation set (about 5% expected without heterogeneity)",
                "validation finite k_hat": "share of seeds with a finite Buhlmann-Straub k on the validation set",
                "k curve, deviance - A, median": "median test deviance minus A by k (unclipped cell factors on A)",
                "variants, summary": "cell variants after rescaling (clipped / unclipped) x groupings x k: mean |B|, pooled bias, deviance - A, share worse, Gini - A",
                "variants by model, k=2000": "data of figure 3 (k = 2000, by model)"}.get(name, "")
    notes = pd.DataFrame({"sheet": list(t), "content": [_note(n) for n in t]})
    with pd.ExcelWriter(path) as w:
        notes.to_excel(w, sheet_name="notes", index=False)
        for i, (name, tab) in enumerate(t.items(), 1):
            short = (name.replace("clipped: ", "clip ").replace("deviance - A, median", "dev-A med").replace("worse than A", "worse")
                     .replace("Gini - A x1000", "Gini-A").replace("validation p<0.05", "valid p lt 0.05")
                     .replace(",", ""))
            sheet = f"{i} {short}"
            if len(sheet) > 31:
                raise ValueError(f"sheet name too long: {sheet}")
            tab.round(4).to_excel(w, sheet_name=sheet, index=not name.startswith("variants"))
    print("tables written to", path)


# =============================================================================================================
# 3. Figures
# =============================================================================================================
INK, INK2 = "#1f1e1c", "#5f5e5a"
COL_LABEL = {"DrivAge band": "Driver\nage",
             "DrivAge x BonusMalus": "Driver age ×\nBonus–Malus", "BonusMalus x VehAge": "Bonus–Malus ×\nVehicle age",
             "BonusMalus band": "Bonus–\nMalus", "DrivAge x VehAge": "Driver age ×\nVehicle age",
             "DrivAge x VehPower": "Driver age ×\nVehicle power", "DrivAge band x Region": "Driver age ×\nRegion"}
MODEL_LABEL = {"Tweedie-CANN": "CANN"}


def _style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#8a8985", "axes.linewidth": 0.7,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "axes.grid": True,
                         "grid.color": "#ebe9e4", "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "legend.frameon": False})
    return plt


def _save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)


def _heat(ax, val, ann, cmap, norm, title, fmt, show_x):
    im = ax.imshow(val.to_numpy(float), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(val.shape[1]))
    ax.set_xticklabels([COL_LABEL[g].replace("–\n", "–").replace("\n", " ") for g in val.columns] if show_x else [], fontsize=7,
                       rotation=30, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(val.shape[0]))
    ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in val.index])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(val.shape[0]):
        for j in range(val.shape[1]):
            v = val.iat[i, j]
            if np.isnan(v):
                continue
            rgba = im.cmap(im.norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, fmt(v, None if ann is None else ann.iat[i, j]), ha="center", va="center", fontsize=6.3,
                    color="#ffffff" if lum < 0.5 else INK)
    ax.set_title(title, loc="center")
    return im


def figure1(t, out_dir, k=K_PAPER):
    """Choice of the grouping for the proposed calibrator (clipped credibility cells after global rescaling, constant k):
    (a) median change in test deviance and share of splits worse than global rescaling, (b) mean change in Gini,
    (c) residual heterogeneity on the validation sets."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
    plt = _style()
    div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f4f2ec", "#f2a07f", "#b03a1a"])
    seq = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), gridspec_kw={"hspace": 0.32})
    specs = [(t[f"clipped: deviance - A, median, k={k}"], t[f"clipped: worse than A, k={k}"], div,
              TwoSlopeNorm(0, -0.12, 0.12),
              "(a) Change in test deviance (median) and share of splits with higher deviance",
              lambda v, a: f"{v:+.3f}\n{100 * a:.0f}%", "deviance change"),
             (t[f"clipped: Gini - A x1000, k={k}"], None, div.reversed(), TwoSlopeNorm(0, -8, 8),
              "(b) Change in Gini index (×1000, mean)", lambda v, a: f"{v:+.1f}", "Gini change (×1000)"),
             (t["validation p<0.05"].reindex(columns=VARIANT_GROUPINGS), None, seq, Normalize(0, 0.8),
              "(c) Splits with significant residual heterogeneity on the validation set (p < 0.05)",
              lambda v, a: f"{100 * v:.0f}%", "share of splits")]
    for i, (ax, (val, ann, cmap, norm, title, fmt, clab)) in enumerate(zip(axes, specs)):
        im = _heat(ax, val, ann, cmap, norm, title, fmt, show_x=(i == 2))
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015, aspect=15)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label(clab, fontsize=7)
        cb.outline.set_visible(False)
    axes[2].set_xlabel("Cell grouping")
    _save(fig, out_dir, f"fig1_heatmaps_k{k}")
    plt.show()


def figure2(v, out_dir):
    """Median change in test deviance relative to global rescaling, by model and credibility constant k: clipped cells
    (the proposed calibrator) for three groupings, and unclipped cells on driver age x Bonus-Malus."""
    import matplotlib.lines as mlines
    plt = _style()
    series = [("paper cells on A", "DrivAge x BonusMalus", "#eb6834", "-", "o", "Driver age × Bonus–Malus"),
              ("paper cells on A", "DrivAge band", "#2a78d6", "-", "o", "Driver age"),
              ("paper cells on A", "BonusMalus x VehAge", "#1baf7a", "-", "o", "Bonus–Malus × Vehicle age"),
              ("cell factors on A, no clip", "DrivAge x BonusMalus", "#eb6834", "--", "", "Driver age × Bonus–Malus, unclipped")]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6), sharey=True, sharex=True,
                             gridspec_kw={"hspace": 0.32, "wspace": 0.10})
    for ax, m in zip(axes.flat, MODELS):
        for var_name, g, c, ls, mk, _ in series:
            med = v[(v.variant == var_name) & (v.grouping == g) & (v.model == m)].groupby("k").dd.median()
            med = med.reindex(K_VARIANT)
            ax.plot(range(len(K_VARIANT)), med.values, color=c, lw=1.5 if ls == "-" else 1.1, ls=ls, marker=mk or None,
                    ms=2.8)
        ax.axvline(K_VARIANT.index(K_PAPER), color="#8a8985", lw=0.7, ls=":")
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xticks(range(len(K_VARIANT)))
        ax.set_xticklabels(K_VARIANT, rotation=45, fontsize=6.5)
        ax.set_title(MODEL_LABEL.get(m, m), loc="center")
        ax.set_ylim(-0.32, 0.42)
    fig.supxlabel("Credibility constant k (dotted line: k = 2000, used in the paper)", fontsize=8, y=0.01)
    fig.supylabel("Change in test deviance (median over splits)", fontsize=8, x=0.03)
    handles = [mlines.Line2D([], [], color=c, ls=ls, marker=mk or None, ms=3, lw=1.5 if ls == "-" else 1.1, label=lab)
               for _, _, c, ls, mk, lab in series]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.95), ncol=2, handlelength=2.2,
               columnspacing=1.5)
    _save(fig, out_dir, "fig2_k_curves_by_model")
    plt.show()


def figure3(t, out_dir):
    """Cell variants after global rescaling, by model (k = 2000); all changes relative to global rescaling."""
    plt = _style()
    T = t["variants by model, k=2000"]
    series = [("paper cells on A", "DrivAge x BonusMalus", "Credibility cells, driver age × Bonus–Malus, clipped", "#2a78d6", "o"),
              ("cell factors on A, no clip", "DrivAge x BonusMalus", "Credibility cells, driver age × Bonus–Malus, unclipped", "#eb6834", "D"),
              ("paper cells on A", "DrivAge band", "Credibility cells, driver age, clipped", "#1baf7a", "^")]
    panels = [("pooled bias (%)", "(a) Pooled bias (%)", 0),
              ("deviance - A, median", "(b) Change in test deviance (median)", 0),
              ("worse than A (%)", "(c) Splits with higher deviance (%)", 50),
              ("Gini - A (x1000)", "(d) Change in Gini index (×1000)", 0)]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), sharey=True, gridspec_kw={"hspace": 0.28, "wspace": 0.08})
    Y = np.arange(len(MODELS))[::-1]
    off = np.linspace(-0.27, 0.27, len(series))
    for ax, (col, title, ref) in zip(axes.flat, panels):
        for (var, g, lab, c, mk), o in zip(series, off):
            d = T[(T.variant == var) & (T.grouping == g)].set_index("model").reindex(MODELS)
            ax.scatter(d[col].values, Y + o, color=c, marker=mk, s=22, edgecolor="white", linewidth=0.7, zorder=3,
                       label=lab)
        ax.axvline(ref, color=INK, lw=0.7, ls="--" if ref == 50 else "-")
        ax.set_title(title, loc="center")
        ax.set_yticks(Y)
        ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in MODELS])
        ax.grid(axis="y", visible=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.92), ncol=2, handletextpad=0.3, columnspacing=1.6)
    _save(fig, out_dir, "fig3_calibrator_variants")
    plt.show()


def report(cal_dir="v2_calibration", age_dir="v2_age_calibration"):
    """Tables (Excel + displayed) and the three figures, written to age_dir."""
    t, x, v = tables(cal_dir, age_dir)
    write_excel(t, os.path.join(age_dir, "age_calibration_tables.xlsx"))
    cs._display(t[f"clipped: deviance - A, median, k={K_PAPER}"].round(3))
    cs._display(t["variants by model, k=2000"].round(3))
    figure1(t, age_dir)
    figure2(v, age_dir)
    figure3(t, age_dir)
    return t


# =============================================================================================================
# 4. Cell exposures and active bounds of the proposed calibrator
# =============================================================================================================
def cell_diagnostics(df_freq, df_sev, pred_dirs, grouping="DrivAge x BonusMalus", ks=(2000, 5000),
                     seeds=range(41, 91), models=MODELS, out_dir="v2_age_calibration"):
    """Validation exposure per cell, median credibility weight, and the share of cells (and of validation exposure)
    whose factor 1 + Z((O+1)/(cP+1) - 1) reaches the bounds 0.95 / 1.15. Reads the stored predictions only."""
    df, num_cols, _ = pp.prepare(df_freq, df_sev)
    df = cs.add_bands(df, num_cols)
    rows, cells = [], []
    for seed in [int(s) for s in seeds]:
        tr, va, te = pp.split(df, seed)
        (iv, _), n = cs._cell_codes([va, te], cs.GROUPINGS[grouping])
        ev, yv = va[EXPOSURE].to_numpy(float), va[TARGET].to_numpy(float)
        E = np.bincount(iv, weights=ev, minlength=n)
        O = np.bincount(iv, weights=ev * yv, minlength=n)
        seen = np.bincount(iv, minlength=n) > 0
        cells += [{"seed": seed, "E": e} for e in E[seen]]
        obs_train = float(np.sum(tr[EXPOSURE].to_numpy(float) * tr[TARGET].to_numpy(float)))
        for m in models:
            d = pp.load_predictions(pred_dirs, m, seed)
            c = obs_train / d["train_sum_e_mu"]
            P = np.bincount(iv, weights=ev * d["valid"], minlength=n)
            for k in ks:
                Z = E / (E + k)
                f = 1 + Z * ((O + 1) / (c * P + 1) - 1)
                lo, hi = seen & (f <= 0.95), seen & (f >= 1.15)
                rows.append({"seed": seed, "model": m, "k": k, "cells": int(seen.sum()),
                             "share at 0.95": lo.sum() / seen.sum(), "share at 1.15": hi.sum() / seen.sum(),
                             "exposure at 0.95": E[lo].sum() / E.sum(), "exposure at 1.15": E[hi].sum() / E.sum(),
                             "median Z": float(np.median(Z[seen]))})
    r, c = pd.DataFrame(rows), pd.DataFrame(cells)
    q = c.E.quantile([0.1, 0.5, 0.9])
    print(f"{grouping}: validation exposure per cell p10 / median / p90 = {q.iloc[0]:.0f} / {q.iloc[1]:.0f} / "
          f"{q.iloc[2]:.0f}; share of cells < 500: {(c.E < 500).mean():.0%}, < 2000: {(c.E < 2000).mean():.0%}")
    summ = r.groupby("k")[["share at 0.95", "share at 1.15", "exposure at 0.95", "exposure at 1.15", "median Z"]].mean()
    cs._display(summ.round(3))
    os.makedirs(out_dir, exist_ok=True)
    r.to_csv(os.path.join(out_dir, "cell_diagnostics.csv"), index=False)
    return r, c
