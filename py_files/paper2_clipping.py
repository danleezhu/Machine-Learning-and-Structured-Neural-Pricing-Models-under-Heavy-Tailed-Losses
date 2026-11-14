"""
paper2_clipping.py — clipped versus unclipped credibility cells, and the sensitivity to the clipping bounds.

The proposed calibrator is
    mu_cal = c * mu * clip(1 + Z ((O + 1) / (c P + 1) - 1), LOW, HIGH),   Z = E / (E + k),
with c from the training set, the cell totals O, P, E of the validation set, cells driver age x Bonus-Malus,
k = 2000 and [LOW, HIGH] = [0.95, 1.15] (fixed before the test results were seen). "Unclipped" drops the clip.

What this module computes, from the stored predictions only (no refit), for every split and model:
  * test deviance, Gini and balance of global rescaling, of the clipped and of the unclipped calibrator for
    k in K_GRID, and of the calibrator for every pair of bounds in LOW_GRID x HIGH_GRID at k = 2000;
  * share of cells at each bound;
  * a validation cross-fit score for every pair of bounds (k = 2000): the validation set is halved at random
    R = 5 times; factors are estimated on one half and the change of deviance relative to global rescaling is
    measured on the other half (10 fits). Bounds chosen this way use no test data.

    import paper2_clipping as clp
    res = clp.run(df_freq, df_sev, [PRED_DIR, f"{FIX_DIR}/predictions"], out_dir="v2_clipping")   # ~20-30 min
    tabs = clp.summary(res, out_dir="v2_clipping")
    clp.tradeoff_figure(res, out_dir="v2_clipping", paper_dir="v2_paper")   # clipped vs unclipped
    clp.bounds_figure(res, out_dir="v2_clipping")                            # sensitivity to the bounds
"""
import os
import time

import numpy as np
import pandas as pd

import paper2_calibration_study as cs
import paper2_predictions as pp
import paper2_run_full as rf
from paper2_age_calibration import MODELS, MODEL_LABEL, TARGET, EXPOSURE, _deviance

GROUPING = "DrivAge x BonusMalus"
K_PAPER = 2000
LOW_PAPER, HIGH_PAPER = 0.95, 1.15
K_GRID = [100, 500, 1000, 2000, 5000, 10000]
LOW_GRID = [0.0, 0.80, 0.90, 0.95, 1.00]           # 0.0 = no lower bound
HIGH_GRID = [1.05, 1.10, 1.15, 1.25, 1.50, np.inf]  # inf = no upper bound
R_HALVINGS = 5
FILE = "clipping.csv"
C_CLIP, C_FREE, INK, INK2 = "#2a78d6", "#eb6834", "#1f1e1c", "#5f5e5a"


def _raw_factor(iv, e, y, mu_c, n, k):
    E = np.bincount(iv, weights=e, minlength=n)
    O = np.bincount(iv, weights=e * y, minlength=n)
    P = np.bincount(iv, weights=e * mu_c, minlength=n)
    seen = E > 0
    Z = np.where(seen, E / (E + k), 0.0)
    return np.where(seen, 1 + Z * ((O + 1.0) / (P + 1.0) - 1), 1.0), seen


def run(df_freq, df_sev, pred_dirs, out_dir="v2_clipping", seeds=range(41, 91), models=MODELS, grouping=GROUPING,
        r_halvings=R_HALVINGS):
    """Long table, one row per (seed, model, k, low, high); low = high = nan marks global rescaling alone.
    Resumable per seed."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILE)
    done = set(pd.read_csv(path, usecols=["seed"]).seed) if os.path.exists(path) else set()
    power = pp.power()
    df, num_cols, _ = pp.prepare(df_freq, df_sev)
    df = cs.add_bands(df, num_cols)
    combos = [(K_PAPER, lo, hi) for lo in LOW_GRID for hi in HIGH_GRID]
    combos += [(k, lo, hi) for k in K_GRID if k != K_PAPER for lo, hi in ((LOW_PAPER, HIGH_PAPER), (0.0, np.inf))]
    t0 = time.time()
    for seed in [int(s) for s in seeds]:
        if seed in done:
            continue
        tr, va, te = pp.split(df, seed)
        yv, ev = va[TARGET].to_numpy(float), va[EXPOSURE].to_numpy(float)
        yt, et = te[TARGET].to_numpy(float), te[EXPOSURE].to_numpy(float)
        obs_train = float(np.sum(tr[EXPOSURE].to_numpy(float) * tr[TARGET].to_numpy(float)))
        obs_test = float(np.sum(et * yt))
        (iv, it), n = cs._cell_codes([va, te], cs.GROUPINGS[grouping])
        rng = np.random.default_rng(1000 * seed + 11)
        halves = [rng.integers(0, 2, len(va)).astype(bool) for _ in range(r_halvings)]
        rows = []
        for m in models:
            d = pp.load_predictions(pred_dirs, m, seed)
            for s, frame in (("valid", va), ("test", te)):
                if d["idhash"][s] != pp.id_hash(frame["IDpol"].to_numpy(np.int64)):
                    raise RuntimeError(f"{m} seed {seed}: stored {s} policies differ from the regenerated split")
            c = obs_train / d["train_sum_e_mu"]
            mv, mt = c * d["valid"], c * d["test"]
            base = {"seed": seed, "model": m, "obs_test": obs_test}
            rows.append(base | {"k": np.nan, "low": np.nan, "high": np.nan,
                                "test_dev": _deviance(yt, mt, et, power), "test_gini": rf.gini_exposure(yt, mt, et),
                                "test_B": float(np.sum(et * mt)) / obs_test - 1})
            # cross-fit raw factors once per halving (k = 2000); the bounds are applied afterwards
            cv_parts = []
            for h in halves:
                for fit, ev_ in ((h, ~h), (~h, h)):
                    f, _ = _raw_factor(iv[fit], ev[fit], yv[fit], mv[fit], n, K_PAPER)
                    d0 = _deviance(yv[ev_], mv[ev_], ev[ev_], power)
                    cv_parts.append((f, ev_, d0))
            raw = {k: _raw_factor(iv, ev, yv, mv, n, k) for k in K_GRID}
            for k, lo, hi in combos:
                f0, seen = raw[k]
                f = np.clip(f0, lo, hi)
                mu = mt * f[it]
                row = base | {"k": k, "low": lo, "high": hi, "test_dev": _deviance(yt, mu, et, power),
                              "test_gini": rf.gini_exposure(yt, mu, et), "test_B": float(np.sum(et * mu)) / obs_test - 1,
                              "share_at_low": float(np.mean(f0[seen] <= lo)), "share_at_high": float(np.mean(f0[seen] >= hi))}
                if k == K_PAPER:
                    row["cv_gain"] = float(np.mean([_deviance(yv[e_], mv[e_] * np.clip(fc, lo, hi)[iv[e_]], ev[e_], power) - d0
                                                    for fc, e_, d0 in cv_parts]))
                rows.append(row)
        rf._append(rows, path)
        print(f"seed {seed}: done | {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.read_csv(path)


# ------------------------------------------------------------------------------------------------------------
def _with_changes(res):
    A = res[res.k.isna()].set_index(["seed", "model"])[["test_dev", "test_gini"]]
    x = res[res.k.notna()].join(A, on=["seed", "model"], rsuffix="_A")
    x["dd"] = x.test_dev - x.test_dev_A
    x["dg"] = 1000 * (x.test_gini - x.test_gini_A)
    return x


def _pick(x, k=K_PAPER, lo=LOW_PAPER, hi=HIGH_PAPER):
    return x[(x.k == k) & np.isclose(x.low, lo) & np.isclose(x.high, hi)]


def _summ(q):
    return pd.Series({"deviance change, median": q.dd.median(), "deviance change, mean": q.dd.mean(),
                      "splits worse than rescaling (%)": 100 * (q.dd > 0).mean(),
                      "Gini change (x1000), mean": q.dg.mean(), "mean |B| (%)": 100 * q.test_B.abs().mean(),
                      "pooled bias (%)": 100 * ((1 + q.test_B) * q.obs_test).sum() / q.obs_test.sum() - 100})


def summary(res, out_dir="v2_clipping"):
    """Clipped vs unclipped by model (k = 2000); the bound grid averaged over models; bounds chosen on validation."""
    x = _with_changes(res)
    clip, free = _pick(x), _pick(x, lo=0.0, hi=np.inf)
    rows = []
    for m in MODELS:
        for lab, q in (("clipped [0.95, 1.15]", clip), ("unclipped", free)):
            rows.append(pd.concat([pd.Series({"model": m, "calibrator": lab}), _summ(q[q.model == m])]))
    by_model = pd.DataFrame(rows)
    a = clip.set_index(["seed", "model"]).dd
    b = free.set_index(["seed", "model"]).dd.reindex(a.index)
    overall = pd.DataFrame({"clipped [0.95, 1.15]": _summ(clip), "unclipped": _summ(free)}).T
    overall["clipped better than unclipped (% of model-splits)"] = [100 * np.mean(a < b), np.nan]
    g = x[x.k == K_PAPER]
    grid = g.groupby(["low", "high"]).apply(_summ, include_groups=False).reset_index()
    cv = g.groupby(["low", "high"]).cv_gain.mean().rename("validation cross-fit change, mean").reset_index()
    grid = grid.merge(cv, on=["low", "high"])
    best_cv = g.loc[g.groupby(["seed", "model"]).cv_gain.idxmin()]
    chosen = (best_cv.groupby(["low", "high"]).size() / len(best_cv) * 100).rename("chosen on validation (%)")
    grid = grid.merge(chosen.reset_index(), on=["low", "high"], how="left").fillna({"chosen on validation (%)": 0})
    sel = best_cv[["seed", "model", "dd", "dg", "test_B", "obs_test"]]
    overall.loc["bounds chosen on validation (per split and model)"] = _summ(sel)
    print("clipped vs unclipped, by model (k = 2000), changes relative to global rescaling")
    cs._display(by_model.round(3))
    cs._display(overall.round(3))
    print("bound grid (k = 2000), mean over models and splits")
    cs._display(grid.round(3))
    os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(os.path.join(out_dir, "clipping_tables.xlsx")) as w:
        by_model.round(4).to_excel(w, sheet_name="clip vs no clip by model", index=False)
        overall.round(4).to_excel(w, sheet_name="overall")
        grid.round(4).to_excel(w, sheet_name="bound grid k=2000", index=False)
    return {"by model": by_model, "overall": overall, "grid": grid}


# ------------------------------------------------------------------------------------------------------------
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
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300)


def _location_groups(paper_dir):
    import paper2_paper_figures as pf
    sd = pf.load_policy_pass(paper_dir)["seeds"].set_index("seed")
    return sd.apply(pf._location_group, axis=1), pf.GROUP_ORDER, pf.GROUP_TICK


def tradeoff_figure(res, out_dir="v2_clipping", paper_dir="v2_paper"):
    """Clipped [0.95, 1.15] versus unclipped cells (driver age x Bonus-Malus), relative to global rescaling.
    Row 1, per model: (a) change in test deviance (median, IQR), (b) splits with higher deviance than rescaling,
    (c) change in Gini, (d) pooled bias. Row 2: (e) median change in test deviance against k (median over models,
    band = range over models); (f) unclipped minus clipped test deviance per split, by claim location."""
    import matplotlib.lines as mlines
    plt = _style()
    x = _with_changes(res)
    clip, free = _pick(x), _pick(x, lo=0.0, hi=np.inf)
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.38, wspace=0.12)
    Y = np.arange(len(MODELS))[::-1]
    series = [(clip, C_CLIP, "o", 0.16), (free, C_FREE, "D", -0.16)]

    def rowpanel(ax, title, fn, ref=0.0, iqr=False, xlim=None):
        for q, col, mk, off in series:
            for y, m in zip(Y, MODELS):
                v = q[q.model == m]
                if iqr:
                    q1, q2, q3 = np.percentile(fn(v), [25, 50, 75])
                    ax.plot([q1, q3], [y + off] * 2, color=col, lw=1.4, zorder=2)
                    ax.scatter([q2], [y + off], color=col, marker=mk, s=22, edgecolor="white", lw=0.6, zorder=3)
                else:
                    val = fn(v)
                    ax.plot([ref, val], [y + off] * 2, color=col, lw=1.0, alpha=0.5, zorder=2)
                    ax.scatter([val], [y + off], color=col, marker=mk, s=22, edgecolor="white", lw=0.6, zorder=3)
        ax.axvline(ref, color=INK, lw=0.7, ls="--" if ref else "-")
        ax.set_title(title)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        ax.grid(axis="y", visible=False)
        for yb in Y[:-1]:
            ax.axhline(yb - 0.5, color="#ebe9e4", lw=0.6)
        if xlim:
            ax.set_xlim(*xlim)

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    rowpanel(axes[0], "(a) Δ test deviance", lambda v: v.dd.to_numpy(), iqr=True)
    rowpanel(axes[1], "(b) Splits worse (%)", lambda v: 100 * np.mean(v.dd > 0), xlim=(0, 60))
    rowpanel(axes[2], "(c) Δ Gini (×1000)", lambda v: v.dg.mean())
    rowpanel(axes[3], "(d) Pooled bias (%)",
             lambda v: 100 * ((1 + v.test_B) * v.obs_test).sum() / v.obs_test.sum() - 100)
    axes[0].set_yticks(Y)
    axes[0].set_yticklabels([MODEL_LABEL.get(m, m) for m in MODELS])
    for a in axes[1:]:
        a.set_yticks(Y)
        a.set_yticklabels([])
    # (e) k sensitivity
    g2 = gs[1, :].subgridspec(1, 2, wspace=0.32)
    ae = fig.add_subplot(g2[0, 0])
    for (lo, hi), col, mk, lab in (((LOW_PAPER, HIGH_PAPER), C_CLIP, "o", "clipped"), ((0.0, np.inf), C_FREE, "D", "unclipped")):
        med = []
        for k in K_GRID:
            q = _pick(x, k=k, lo=lo, hi=hi)
            med.append(q.groupby("model").dd.median().reindex(MODELS).to_numpy())
        med = np.array(med)
        xs = np.arange(len(K_GRID))
        ae.fill_between(xs, med.min(1), med.max(1), color=col, alpha=0.15, lw=0)
        ae.plot(xs, np.median(med, 1), color=col, marker=mk, ms=3.5, lw=1.5)
    ae.axhline(0, color=INK, lw=0.7)
    ae.axvline(K_GRID.index(K_PAPER), color="#8a8985", lw=0.7, ls=":")
    ae.set_xticks(np.arange(len(K_GRID)))
    ae.set_xticklabels(K_GRID)
    ae.set_xlabel("Credibility constant k (dotted: k = 2000)")
    ae.set_ylabel("Median Δ test deviance")
    ae.set_title("(e) Sensitivity to k")
    # (f) where unclipped wins and loses
    af = fig.add_subplot(g2[0, 1])
    grp, order, tick = _location_groups(paper_dir)
    a = clip.set_index(["seed", "model"]).test_dev
    b = free.set_index(["seed", "model"]).test_dev.reindex(a.index)
    diff = (b - a).groupby(level="seed").mean()            # mean over models, per split
    rng = np.random.default_rng(3)
    for i, g in enumerate(order):
        v = diff.reindex(grp.index[grp == g]).dropna().to_numpy()
        af.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=10, color=np.where(v < 0, C_FREE, C_CLIP), alpha=0.85,
                   edgecolor="white", lw=0.3, zorder=3)
        af.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=1.4, zorder=4)
    af.axhline(0, color=INK, lw=0.7)
    af.set_xticks(range(len(order)))
    af.set_xticklabels(["Largest\nin test", "2nd/3rd\nin test", "Some in\nvalidation", "All in\ntraining"], fontsize=7)
    af.tick_params(axis="x", length=0)
    af.grid(axis="x", visible=False)
    af.set_ylabel("Δ test deviance (below 0: unclipped better)")
    af.set_title("(f) Unclipped minus clipped, per split")
    handles = [mlines.Line2D([], [], color=C_CLIP, marker="o", ls="", ms=5, label="Clipped to [0.95, 1.15]"),
               mlines.Line2D([], [], color=C_FREE, marker="D", ls="", ms=5, label="Unclipped"),
               mlines.Line2D([], [], color=INK2, lw=1.4, label="(a) interquartile range over the 50 splits")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.91), ncol=3, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.12)
    _save(fig, out_dir, "clipping_tradeoff")
    plt.show()


def bounds_figure(res, out_dir="v2_clipping"):
    """Bound grid at k = 2000, mean over the eight models and 50 splits: (a) median change in test deviance,
    (b) share of model-splits worse than rescaling, (c) pooled bias, (d) share of model-splits in which each pair of
    bounds is chosen by the validation cross-fit. Boxed cell: [0.95, 1.15]."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
    import matplotlib.patches as mpatches
    plt = _style()
    x = _with_changes(res)
    g = x[x.k == K_PAPER]
    best = g.loc[g.groupby(["seed", "model"]).cv_gain.idxmin()]
    lo_lab = ["none" if v == 0 else f"{v:.2f}" for v in LOW_GRID]
    hi_lab = ["none" if np.isinf(v) else f"{v:.2f}" for v in HIGH_GRID]

    def mat(fn):
        M = np.full((len(LOW_GRID), len(HIGH_GRID)), np.nan)
        for i, lo in enumerate(LOW_GRID):
            for j, hi in enumerate(HIGH_GRID):
                q = g[np.isclose(g.low, lo) & np.isclose(g.high, hi)]
                M[i, j] = fn(q)
        return M

    div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f4f2ec", "#f2a07f", "#b03a1a"])
    seq = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
    panels = [("(a) Median Δ test deviance", mat(lambda q: q.dd.median()), div, TwoSlopeNorm(0, -0.2, 0.2), "{:+.3f}"),
              ("(b) Model-splits worse (%)", mat(lambda q: 100 * (q.dd > 0).mean()), seq, Normalize(0, 50), "{:.0f}"),
              ("(c) Pooled bias (%)", mat(lambda q: 100 * ((1 + q.test_B) * q.obs_test).sum() / q.obs_test.sum() - 100),
               div, TwoSlopeNorm(0, -5, 5), "{:+.1f}"),
              ("(d) Chosen on validation (%)", mat(lambda q: 100 * len(best.merge(q[["seed", "model", "low", "high"]]))
                                                    / len(best)), seq, Normalize(0, 40), "{:.0f}")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw={"hspace": 0.45, "wspace": 0.25})
    i0, j0 = LOW_GRID.index(LOW_PAPER), HIGH_GRID.index(HIGH_PAPER)
    for ax, (title, M, cmap, norm, fmt) in zip(axes.flat, panels):
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                rgba = im.cmap(im.norm(M[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if lum < 0.5 else INK)
        ax.add_patch(mpatches.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
        ax.set_xticks(range(len(HIGH_GRID)))
        ax.set_xticklabels(hi_lab)
        ax.set_yticks(range(len(LOW_GRID)))
        ax.set_yticklabels(lo_lab)
        ax.set_xlabel("Upper bound")
        ax.set_ylabel("Lower bound")
        ax.tick_params(length=0)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title)
    _save(fig, out_dir, "clipping_bounds")
    plt.show()
