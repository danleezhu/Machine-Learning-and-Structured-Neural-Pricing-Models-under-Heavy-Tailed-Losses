

import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import paper2_calibration_study as cs
import paper2_paper_figures as pf
import paper2_predictions as pp

MODELS = pf.MODELS
MODEL_LABEL = pf.MODEL_LABEL
VAR_LABEL = pf.VAR_LABEL
TARGET, EXPOSURE = "PurePremium", "Exposure"
NUM_FEATURES = ["DrivAge", "BonusMalus", "VehAge", "VehPower", "Density"]
CAT_FEATURES = ["Region", "Area", "VehBrand", "VehGas"]
# fixed colours per model (same entity -> same colour in every figure of this module)
MODEL_COLOR = {"GLM": "#5f5e5a", "Tweedie GLM": "#8a8985", "EBM": "#1baf7a", "GBM": "#6250d6",
               "LocalGLMnet": "#d6336c", "Neural GA2M": "#eb6834", "Neural GA2M+": "#2a78d6", "Tweedie-CANN": "#b58a00"}
MODEL_STYLE = {"GLM": "--", "Tweedie GLM": ":", "EBM": "-", "GBM": "-", "LocalGLMnet": "-", "Neural GA2M": "-",
               "Neural GA2M+": "-", "Tweedie-CANN": "-."}

# The three calibration methods (same names as paper2_paper_figures.METHOD_NAMES)
M_RAW = "Uncalibrated"
M_A = "Global rescaling"
M_NEW = "Global rescaling + credibility cells (Driver age × Bonus–Malus)"
METHODS = [M_RAW, M_A, M_NEW]


def _tables_dir(out_dir):
    d = os.path.join(out_dir, "tables")
    os.makedirs(d, exist_ok=True)
    return d


def _write(df, out_dir, name, index=False):
    d = _tables_dir(out_dir)
    df.to_csv(os.path.join(d, f"{name}.csv"), index=index)
    try:
        df.to_excel(os.path.join(d, f"{name}.xlsx"), index=index)
    except ImportError:                                  # openpyxl missing: csv only
        pass


def _display(x):
    try:
        from IPython.display import display
        display(x)
    except ImportError:
        print(x)


# =============================================================================================================
# Signed balance on training / validation / test
# =============================================================================================================
def _cell_factors(iv, ev, yv, mv, n, k, c):
    """Credibility cell factors after global rescaling, fitted on the validation set:
    f = clip(1 + Z((O+1)/(c P+1) - 1), 0.95, 1.15), Z = E/(E+k); cells without validation policies get 1."""
    E = np.bincount(iv, weights=ev, minlength=n)
    O = np.bincount(iv, weights=ev * yv, minlength=n)
    P = np.bincount(iv, weights=ev * mv, minlength=n)
    seen = np.bincount(iv, minlength=n) > 0
    Z = np.where(seen, E / (E + k), 0.0)
    f = np.clip(1 + Z * ((O + 1.0) / (c * P + 1.0) - 1), 0.95, 1.15)
    return np.where(seen, f, 1.0)


def signed_balance_table(df_freq, df_sev, pred_dirs, out_dir="v2_paper", seeds=range(41, 91), models=MODELS, k=2000):
    """
    Mean signed balance B = P/O - 1 (%) on the training, validation and test sets, per model and method, and the
    pooled test bias, recomputed from the stored predictions. Training values: uncalibrated = stored training total
    (train_sum_e_mu); global rescaling = 0 by construction; the proposed calibrator has no training value because
    per-policy training predictions were not stored (its cell factors come from the validation set).
    """
    df, num_cols, _ = pp.prepare(df_freq, df_sev)
    df = cs.add_bands(df, num_cols)
    g_age = cs.GROUPINGS[cs.PROPOSED_GROUPING]
    rows, t0 = [], time.time()
    for seed in [int(s) for s in seeds]:
        tr, va, te = pp.split(df, seed)
        yv, ev = va[TARGET].to_numpy(float), va[EXPOSURE].to_numpy(float)
        yt, et = te[TARGET].to_numpy(float), te[EXPOSURE].to_numpy(float)
        O_tr = float(np.sum(tr[EXPOSURE].to_numpy(float) * tr[TARGET].to_numpy(float)))
        O_v, O_t = float(np.sum(ev * yv)), float(np.sum(et * yt))
        (iva, ita), na = cs._cell_codes([va, te], g_age)
        for m in models:
            try:
                d = pp.load_predictions(pred_dirs, m, seed)
            except FileNotFoundError:
                continue
            for s, frame in (("valid", va), ("test", te)):
                if d["idhash"][s] != pp.id_hash(frame["IDpol"].to_numpy(np.int64)):
                    raise RuntimeError(f"{m} seed {seed}: stored {s} policies differ from the regenerated split")
            mv, mt = d["valid"], d["test"]
            c = O_tr / d["train_sum_e_mu"]
            B = lambda e, mu, O: float(np.sum(e * mu)) / O - 1
            fn = _cell_factors(iva, ev, yv, mv, na, k, c)
            preds = {M_RAW: (mv, mt, d["train_sum_e_mu"] / O_tr - 1),
                     M_A: (c * mv, c * mt, 0.0),
                     M_NEW: (c * fn[iva] * mv, c * fn[ita] * mt, np.nan)}
            for meth, (pv, pt, btr) in preds.items():
                rows.append(dict(seed=seed, model=m, method=meth, train_B=btr, valid_B=B(ev, pv, O_v),
                                 test_B=B(et, pt, O_t), pred_test=float(np.sum(et * pt)), obs_test=O_t))
        if seed % 10 == 0:
            print(f"seed {seed} | {(time.time() - t0) / 60:.1f} min", flush=True)
    r = pd.DataFrame(rows)
    g = r.groupby(["model", "method"], sort=False)
    t = pd.DataFrame({"Train B (%)": 100 * g.train_B.mean(), "Valid B (%)": 100 * g.valid_B.mean(),
                      "Test B (%)": 100 * g.test_B.mean(),
                      "Pooled test bias (%)": 100 * (g.pred_test.sum() / g.obs_test.sum() - 1)}).reset_index()
    t["model"] = pd.Categorical(t.model, [m for m in models if m in set(t.model)], ordered=True)
    t["method"] = pd.Categorical(t.method, METHODS, ordered=True)
    t = t.sort_values(["model", "method"]).reset_index(drop=True)
    t["model"] = t.model.astype(str).map(MODEL_LABEL)
    t["method"] = t.method.astype(str)
    _write(r, out_dir, "A3_signed_balance_by_seed")
    _write(t, out_dir, "A3_signed_balance")
    _display(t.round(2))
    return t


# =============================================================================================================
# Stress analysis with the model-free label
# =============================================================================================================
def model_free_stress(run_dir, out_dir="v2_paper", imb=0.40, top1=0.78,
                      paper_seeds=(41, 49, 51, 55, 56, 57, 60, 61, 65, 70, 71, 72, 73, 75, 78, 81, 83, 85, 86, 89, 90)):
    """Regular vs stressed splits, stressed = loss-imbalance score >= imb or test top-1% loss share >= top1 (no model
    enters the label). paper_seeds: the fixed stressed-seed list, for comparison only. Per group and model: mean |B| and pooled bias
    before and after global rescaling, deviance after rescaling, Gini, and the deviance / Gini difference to the GLM
    with the share of splits in which the model is better."""
    s = pd.read_csv(os.path.join(run_dir, "seeds.csv"))
    s["stressed"] = (s.loss_imbalance_score >= imb) | (s.test_top1_share >= top1)
    lab = s.set_index("seed").stressed
    n_st, n_paper = int(lab.sum()), len(paper_seeds)
    both = int(lab.reindex(list(paper_seeds)).fillna(False).sum())
    print(f"model-free label: {n_st} stressed of {len(lab)} splits; fixed list (paper_seeds): {n_paper}; "
          f"in both: {both}")
    perf = pd.read_csv(os.path.join(run_dir, "perf.csv"))
    perf = perf[perf.version.isin(["raw", "train_level"])].copy()
    perf["group"] = np.where(perf.seed.map(lab), "Stressed", "Regular")
    A = perf[perf.version == "train_level"]
    glm = A[A.method == "GLM"].set_index("seed")
    out = []
    for grp in ("Regular", "Stressed"):
        for m in [x for x in MODELS if x in set(perf.method)]:
            raw = perf[(perf.version == "raw") & (perf.method == m) & (perf.group == grp)]
            a = A[(A.method == m) & (A.group == grp)].set_index("seed")
            dd = a.tweedie_deviance - glm.tweedie_deviance.reindex(a.index)
            dg = a.gini - glm.gini.reindex(a.index)
            out.append({"Group": grp, "Splits": len(a), "Model": MODEL_LABEL[m],
                        "|B| raw (%)": 100 * raw.abs_balance_error_pct.mean(),
                        "Pooled bias raw (%)": 100 * (raw.predicted_total.sum() / raw.observed_total.sum() - 1),
                        "|B| rescaled (%)": 100 * a.abs_balance_error_pct.mean(),
                        "Pooled bias rescaled (%)": 100 * (a.predicted_total.sum() / a.observed_total.sum() - 1),
                        "Deviance rescaled": a.tweedie_deviance.mean(), "Gini": a.gini.mean(),
                        "Deviance - GLM": dd.mean() if m != "GLM" else np.nan,
                        "Deviance better than GLM": (dd < 0).mean() if m != "GLM" else np.nan,
                        "Gini - GLM": dg.mean() if m != "GLM" else np.nan,
                        "Gini better than GLM": (dg > 0).mean() if m != "GLM" else np.nan})
    t = pd.DataFrame(out)
    _write(t, out_dir, "S1_model_free_stress")
    _write(s[["seed", "loss_imbalance_score", "test_top1_share", "stressed"]], out_dir, "S1_model_free_labels")
    _display(t.round(3))
    return t


# =============================================================================================================
# Relativities of the categorical levels
# =============================================================================================================
def _centred(pdc, weights=None):
    """Centre the partial dependence within (seed, method, feature): by the mean over levels / grid points, or by the
    exposure-weighted mean over levels when `weights` (feature -> {level: exposure}) is given."""
    d = pdc.copy()
    if weights is None:
        d["c"] = d.pd_log_mu - d.groupby(["seed", "method", "feature"]).pd_log_mu.transform("mean")
        return d
    w = np.array([weights.get(f, {}).get(str(x), np.nan) for f, x in zip(d.feature, d.x.astype(str))])
    d["w"] = np.where(np.isnan(w), 1.0, w)
    d["wy"] = d.w * d.pd_log_mu
    g = d.groupby(["seed", "method", "feature"])
    d["c"] = d.pd_log_mu - g.wy.transform("sum") / g.w.transform("sum")
    return d.drop(columns=["w", "wy"])


def relativity_figure(run_dir, out_dir="v2_paper", df_freq=None, models=("Tweedie GLM", "LocalGLMnet"),
                      features=CAT_FEATURES):
    """
    Relativity of every level = exp(centred partial dependence of log mu). Centring: exposure-weighted mean over the
    levels when df_freq is given (portfolio average relativity = 1), else the plain mean. Median over the splits and
    10-90% range; levels ordered by the median relativity of the first model.
    For LocalGLMnet the partial dependence of a dummy-coded level equals its global coefficient plus the average local
    deviation, i.e. the relativity a rating plan uses.
    """
    plt = pf._plt()
    pdc = pd.read_csv(os.path.join(run_dir, "pd_curves.csv"))
    pdc = pdc[(pdc.kind == "cat") & pdc.feature.isin(features) & pdc.method.isin(models)]
    if pdc.empty:
        raise ValueError(f"{run_dir}/pd_curves.csv has no categorical curves for {models}")
    weights = None
    if df_freq is not None:
        weights = {f: df_freq.groupby(df_freq[f].astype(str))[EXPOSURE].sum().to_dict() for f in features
                   if f in df_freq}
    d = _centred(pdc, weights)
    d["rel"] = np.exp(d.c)
    q = (d.groupby(["feature", "x", "method"]).rel.quantile([0.1, 0.5, 0.9]).unstack()
         .rename(columns={0.1: "p10", 0.5: "median", 0.9: "p90"}).reset_index())
    q["x"] = q.x.astype(str)
    _write(q, out_dir, "A1_relativities")
    widths = {f: max(2, q[q.feature == f].x.nunique()) for f in features}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw={"width_ratios": [widths["Region"], widths["VehBrand"]]
                                                                      if "Region" in widths and "VehBrand" in widths
                                                                      else [1, 1]})
    order_axes = {"Region": axes[0, 0], "VehBrand": axes[0, 1], "Area": axes[1, 0], "VehGas": axes[1, 1]}
    off = np.linspace(-0.18, 0.18, len(models)) if len(models) > 1 else [0.0]
    for f in features:
        ax = order_axes.get(f)
        if ax is None:
            continue
        g = q[q.feature == f]
        ref = g[g.method == models[0]].sort_values("median")
        levels = list(ref.x) + [x for x in g.x.unique() if x not in set(ref.x)]
        pos = {x: i for i, x in enumerate(levels)}
        for j, m in enumerate(models):
            gm = g[g.method == m]
            xs = np.array([pos[x] for x in gm.x]) + off[j]
            ax.errorbar(xs, gm["median"], yerr=[gm["median"] - gm.p10, gm.p90 - gm["median"]], fmt="o", ms=3.2,
                        lw=0.9, capsize=0, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        ax.axhline(1, color=pf.INK2, lw=0.6)
        ax.set_yscale("log")                         # relativities are multiplicative
        lo, hi = g.p10.min(), g.p90.max()
        cand = (0.9, 0.95, 1, 1.05, 1.1, 1.15) if hi / lo < 1.4 else (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8)
        ticks = [t for t in cand if lo * 0.97 <= t <= hi * 1.03] or [1]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, rotation=60 if len(levels) > 8 else 0, ha="right" if len(levels) > 8 else "center",
                           fontsize=6.5 if len(levels) > 12 else 7.5)
        ax.set_title(VAR_LABEL.get(f, f), loc="center")
        ax.set_ylabel("Relativity")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(models), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pf._save(fig, out_dir, "A1_categorical_relativities",
             "Relativities of the levels of the categorical covariates, exp of the partial dependence of the log pure "
             "premium centred at the exposure-weighted portfolio average (log scale). Points: median over the 50 splits; bars: "
             "10-90% range. Levels are ordered by the median relativity of the Tweedie GLM.")
    plt.show()
    return q


# =============================================================================================================
# Partial dependence and shape stability
# =============================================================================================================
def shape_stability(pdc):
    """SD across splits of the centred partial dependence, averaged over grid points / levels, per model and feature
    (same definition as R10 of paper2_analysis_full)."""
    d = _centred(pdc)
    sd = d.groupby(["method", "feature", "x"]).c.std().groupby(["method", "feature"]).mean()
    w = sd.unstack("feature")
    w["Mean (all covariates)"] = w.mean(1)
    num = [f for f in NUM_FEATURES if f in w.columns]
    w["Mean (numerical)"] = w[num].mean(1)
    w = w.reindex([m for m in MODELS if m in w.index])
    w.index = [MODEL_LABEL[m] for m in w.index]
    return w.rename(columns=VAR_LABEL)


def pd_figure(run_dir, out_dir="v2_paper", features=NUM_FEATURES, models=MODELS, band_models=("GLM", "Neural GA2M+")):
    """
    One panel per numerical covariate: median over the splits of the centred partial dependence of log mu for every
    model; 10-90% band for `band_models` only (the complete grid with all bands is R10_shape_stability of
    paper2_analysis_full). Also writes the shape-stability table (A2_shape_stability).
    """
    plt = pf._plt()
    pdc = pd.read_csv(os.path.join(run_dir, "pd_curves.csv"))
    tab = shape_stability(pdc)
    _write(tab.reset_index().rename(columns={"index": "Model"}), out_dir, "A2_shape_stability")
    _display(tab.round(3))
    d = _centred(pdc[pdc.kind == "num"])
    feats = [f for f in features if f in set(d.feature)]
    models = [m for m in models if m in set(d.method)]
    ncol = 3
    nrow = int(np.ceil((len(feats) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.35 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for i, f in enumerate(feats):
        ax = axes[i]
        for m in models:
            g = d[(d.feature == f) & (d.method == m)].pivot_table(index="x_num", columns="seed", values="c")
            if g.empty:
                continue
            if m in band_models:
                ax.fill_between(g.index, g.quantile(0.1, axis=1), g.quantile(0.9, axis=1), color=MODEL_COLOR[m],
                                alpha=0.15, lw=0)
            ax.plot(g.index, g.median(axis=1), color=MODEL_COLOR[m], ls=MODEL_STYLE[m], lw=1.2, label=MODEL_LABEL[m])
        ax.set_title("Density (log scale)" if f == "Density" else VAR_LABEL.get(f, f), loc="center")
        if i % ncol == 0:
            ax.set_ylabel("Centred log pure premium")
    for ax in axes[len(feats):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[len(feats)].legend(h, l, loc="center", ncol=1, title="Median over 50 splits")
    fig.tight_layout()
    pf._save(fig, out_dir, "F10_partial_dependence",
             "Partial dependence of the log pure premium on the numerical covariates, centred within each split. "
             "Lines: median over the 50 splits; shaded: 10-90% range for the GLM and Neural GA2M+.")
    plt.show()
    return tab


# =============================================================================================================
# Checks of descriptive numbers quoted in the paper
# =============================================================================================================
def _pair_key(p):
    a, b = (p.split(" x ") if " x " in p else p.split("|"))
    return " x ".join(sorted([a.strip(), b.strip()]))


def check_numbers(df_freq, df_sev, main_dir, fix_dir, age_max=21, bm_min=100):
    """Prints (i) the share of exposure and losses of drivers aged <= age_max at Bonus-Malus >= bm_min and how many of
    the largest claims they hold; (ii) the agreement of the pair sets with the leave-one-out consensus and its
    Spearman correlation with the share of training losses of the largest training claim (seeds.csv)."""
    df, _, _ = pp.prepare(df_freq, df_sev)
    seg = (df.DrivAge <= age_max) & (df.BonusMalus >= bm_min)
    L, e = df.ClaimAmount, df[EXPOSURE]
    print(f"(i) segment DrivAge <= {age_max} & BonusMalus >= {bm_min}: exposure share {e[seg].sum() / e.sum():.2%}, "
          f"loss share {L[seg].sum() / L.sum():.2%}, pure premium {L[seg].sum() / e[seg].sum():,.0f} vs portfolio "
          f"{L.sum() / e.sum():,.0f}")
    for n in (3, 10, 50, 150):
        top = L.nlargest(n).index
        print(f"    of the {n:>3} largest policy losses: {int(seg.loc[top].sum())} in the segment "
              f"({L.loc[top][seg.loc[top]].sum() / L.loc[top].sum():.0%} of their amount)")
    top3 = L.nlargest(3)
    print(f"    three largest: {', '.join(f'{v:,.0f}' for v in top3)}; share of all losses "
          f"{top3.iloc[0] / L.sum():.2%} (largest), {top3.sum() / L.sum():.2%} (three)")
    raw = json.load(open(os.path.join(main_dir, "ebm_pairs.json")))
    A = {int(s): {" x ".join(sorted(p)) for p in v} for s, v in raw.items()}
    sel = pd.read_csv(os.path.join(fix_dir, "ga2mplus_selection.csv"))
    B = {int(r.seed): {_pair_key(p) for p in json.loads(r.chosen_pairs)} for r in sel.itertuples()}
    sp = pd.read_csv(os.path.join(main_dir, "seeds.csv")).set_index("seed")
    jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else 1.0
    for name, S in (("Neural GA2M (EBM pairs)", A), ("Neural GA2M+", B)):
        seeds = sorted(S)
        loo = []
        for s in seeds:
            o = [t for t in seeds if t != s]
            cnt = pd.Series([p for t in o for p in S[t]]).value_counts()
            loo.append(jac(S[s], set(cnt[cnt >= 0.5 * len(o)].index)))
        r = stats.spearmanr(loo, sp.loc[seeds, "train_max_loss_share"])
        print(f"(ii) {name}: consensus agreement {np.mean(loo):.3f}; Spearman with largest training-claim share "
              f"{r.statistic:+.3f} (p = {r.pvalue:.2g})")
