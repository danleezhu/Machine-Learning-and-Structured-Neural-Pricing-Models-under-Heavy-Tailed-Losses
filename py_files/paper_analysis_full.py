
import json
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import paper2_datasets as dsm

MODEL_ORDER = ["GLM", "Tweedie GLM", "EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Neural GA2M+", "Tweedie-CANN"]
VERSIONS = ["raw", "train_level"]
VERSION_LABELS = {"raw": "Raw", "train_level": "Train level (A)"}
SHORT = {"GLM": "GLM", "Tweedie GLM": "Tweedie GLM", "EBM": "EBM", "GBM": "GBM", "LocalGLMnet": "LocalGLMnet",
         "Neural GA2M": "Neural GA2M", "Neural GA2M+": "Neural GA2M+", "Tweedie-CANN": "CANN"}
COLORS = {"GLM": "#4C4C4C", "Tweedie GLM": "#1F77B4", "EBM": "#2CA02C", "GBM": "#9467BD", "LocalGLMnet": "#D62728",
          "Neural GA2M": "#FF7F0E", "Neural GA2M+": "#8C564B", "Tweedie-CANN": "#17BECF"}
GROUP_LABELS = {"all": "All", "good_seed": "Regular", "bad_seed": "Stressed"}
PAPER_STRESSED_SEEDS = [41, 49, 51, 55, 56, 57, 60, 61, 65, 70, 71, 72, 73, 75, 78, 81, 83, 85, 86, 89, 90]
TEST_FRAC, TRAIN_FRAC = 0.20, 0.60
NUM_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]


# =============================================================================================================
# Helpers
# =============================================================================================================
def _disp(x):
    try:
        from IPython.display import display
        display(x)
    except ImportError:
        print(x)


def _style():
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "pdf.fonttype": 42})


def _models(df):
    present = set(df.method.astype(str))
    return [m for m in MODEL_ORDER if m in present] + sorted(present - set(MODEL_ORDER))


def _holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return out
    q = p[ok]
    order = np.argsort(q)
    m = len(q)
    adj = np.maximum.accumulate((m - np.arange(m)) * q[order])
    tmp = np.empty(m)
    tmp[order] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def paired_stats(x, higher_is_better=False, n_boot=10000, seed=0):
    """Paired differences over seeds: naive and Nadeau-Bengio corrected t intervals, bootstrap, Wilcoxon."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    J = len(x)
    if J < 3:
        return {"n_seeds": J}
    mu, sd = x.mean(), x.std(ddof=1)
    tq = stats.t.ppf(0.975, J - 1)
    se_n, se_c = sd / np.sqrt(J), sd * np.sqrt(1 / J + TEST_FRAC / TRAIN_FRAC)
    boots = np.random.default_rng(seed).choice(x, (n_boot, J)).mean(1)
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_w = stats.wilcoxon(x).pvalue if np.any(x != 0) else np.nan
    except ValueError:
        p_w = np.nan
    better = (x > 0) if higher_is_better else (x < 0)
    return {"n_seeds": J, "mean_diff": mu, "sd_diff": sd, "dz": mu / sd if sd > 0 else np.nan,
            "share_A_better": better.mean(), "naive_lo": mu - tq * se_n, "naive_hi": mu + tq * se_n,
            "naive_p": 2 * stats.t.sf(abs(mu / se_n), J - 1) if sd > 0 else np.nan,
            "NB_lo": mu - tq * se_c, "NB_hi": mu + tq * se_c,
            "NB_p": 2 * stats.t.sf(abs(mu / se_c), J - 1) if sd > 0 else np.nan,
            "boot_lo": np.percentile(boots, 2.5), "boot_hi": np.percentile(boots, 97.5), "wilcoxon_p": p_w}


def _ms(s, scale=1.0, digits=2, signed=False):
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{f.format(scale * s.mean())} ({f.format(scale * s.std()).lstrip('+')})"


class _Saver:
    def __init__(self, out_dir):
        self.t = os.path.join(out_dir, "analysis", "tables") if out_dir else None
        self.f = os.path.join(out_dir, "analysis", "figures") if out_dir else None
        for d in (self.t, self.f):
            if d:
                os.makedirs(d, exist_ok=True)

    def table(self, df, name, latex=True):
        if self.t is None or df is None or len(df) == 0:
            return
        df.to_csv(os.path.join(self.t, f"{name}.csv"), index=False)
        if latex:
            try:
                with open(os.path.join(self.t, f"{name}.tex"), "w") as fh:
                    fh.write(df.to_latex(index=False, float_format="%.3f", na_rep="--", escape=True))
            except Exception:
                pass

    def fig(self, fig, name):
        if self.f is not None:
            fig.savefig(os.path.join(self.f, f"{name}.pdf"), bbox_inches="tight")


# =============================================================================================================
# Loading
# =============================================================================================================
def load_results(out_dir, group="paper", complete_only=True):
    j = lambda f: os.path.join(out_dir, f)
    perf = pd.read_csv(j("perf.csv"))
    perf = perf[perf.version.isin(VERSIONS)]
    cal = pd.read_csv(j("calibration.csv")) if os.path.exists(j("calibration.csv")) else None
    seeds = pd.read_csv(j("seeds.csv"))
    blends = pd.read_csv(j("blends.csv")) if os.path.exists(j("blends.csv")) else None
    pdc = pd.read_csv(j("pd_curves.csv")) if os.path.exists(j("pd_curves.csv")) else None
    hist = [json.loads(l) for l in open(j("nn_history.jsonl"))] if os.path.exists(j("nn_history.jsonl")) else []
    if complete_only and os.path.exists(j("seeds_complete.csv")):
        keep = set(pd.read_csv(j("seeds_complete.csv")).seed)
        perf = perf[perf.seed.isin(keep)]
        cal = cal[cal.seed.isin(keep)] if cal is not None else None
        blends = blends[blends.seed.isin(keep)] if blends is not None else None
        pdc = pdc[pdc.seed.isin(keep)] if pdc is not None else None
    seeds = seeds.drop_duplicates("seed")
    dataset = dsm.dataset_of(out_dir)
    if group == "paper" and dsm.spec(dataset)["stressed_seeds"] is None:
        print(f"{dataset}: no stressed-seed list; group = 'relative' (40% of seeds ranked highest on the split statistics)")
        group = "relative"
    if group == "paper":
        seeds["group"] = np.where(seeds.seed.isin(dsm.spec(dataset)["stressed_seeds"]), "bad_seed", "good_seed")
    else:
        sub = seeds[seeds.seed.isin(perf.seed.unique())]          # quantiles of the 'relative' rule: run's seeds
        seeds["group"] = pd.Series(dsm.stress_labels(sub, group), index=sub.index).reindex(seeds.index).fillna(
            "good_seed")
    perf = perf.merge(seeds[["seed", "group"]], on="seed", how="left")
    perf["method"] = perf.method.astype(str)
    n = perf.groupby("method").seed.nunique()
    print(f"loaded {perf.seed.nunique()} complete seeds; seeds per model: {n.to_dict()}; "
          f"group = {group}: {int((seeds.group == 'bad_seed').sum())} stressed / "
          f"{int((seeds.group == 'good_seed').sum())} regular")
    return {"perf": perf, "cal": cal, "seeds": seeds, "blends": blends, "pd": pdc, "hist": hist, "group": group,
            "dataset": dataset}


# =============================================================================================================
# R1 main table
# =============================================================================================================
def r1_main_table(res, versions=("raw", "train_level")):
    p = res["perf"]
    rows = []
    for v in versions:
        for m in _models(p):
            g = p[(p.method == m) & (p.version == v)]
            if g.empty:
                continue
            rows.append({"Model": SHORT.get(m, m), "Version": VERSION_LABELS[v], "Seeds": g.seed.nunique(),
                         "Abs. B (%)": _ms(g.abs_balance_error_pct, 100),
                         "Signed B (%)": _ms(g.balance_error_pct, 100, signed=True),
                         "Pooled bias (%)": f"{100 * (g.predicted_total.sum() / g.observed_total.sum() - 1):+.2f}",
                         "Deviance": _ms(g.tweedie_deviance, digits=3), "Gini": _ms(g.gini, digits=4),
                         "WMAE": _ms(g.weighted_mae, digits=2),
                         "Minutes": f"{g.fit_seconds.mean() / 60:.1f}"})
    return pd.DataFrame(rows)


# =============================================================================================================
# R2 paired tests
# =============================================================================================================
def default_pairs(models):
    pairs = [(m, "GLM") for m in models if m != "GLM"]
    pairs += [(a, b) for a, b in [("Neural GA2M", "EBM"), ("Neural GA2M", "GBM"), ("LocalGLMnet", "Tweedie GLM"),
                                  ("Tweedie-CANN", "Tweedie GLM"), ("EBM", "GBM"), ("Neural GA2M+", "Neural GA2M"),
                                  ("Neural GA2M+", "EBM")] if a in models and b in models]
    return pairs


def r2_paired_tests(res, metrics=("tweedie_deviance", "gini", "abs_balance_error_pct"), versions=("raw",),
                    pairs=None, groups=("all", "good_seed", "bad_seed")):
    p = res["perf"]
    pairs = pairs or default_pairs(_models(p))
    rows = []
    for v in versions:
        for metric in metrics:
            for g in groups:
                d = p[(p.version == v) & ((p.group == g) if g != "all" else True)]
                W = d.pivot_table(index="seed", columns="method", values=metric)
                for a, b in pairs:
                    if a in W and b in W:
                        s = paired_stats((W[a] - W[b]).dropna(), higher_is_better=(metric == "gini"))
                        rows.append({"version": v, "metric": metric, "group": g, "A": a, "B": b, **s})
    out = pd.DataFrame(rows)
    if len(out) and "NB_p" in out:
        out["NB_p_holm"] = out.groupby(["version", "metric", "group"]).NB_p.transform(lambda s: _holm(s.values))
    return out


def format_paired(t, scale=1.0, digits=3):
    f = lambda v: f"{scale * v:+.{digits}f}" if np.isfinite(v) else "-"
    t = t.reindex(columns=list(dict.fromkeys(list(t.columns) + ["mean_diff", "NB_lo", "NB_hi", "share_A_better",
                                                                 "NB_p_holm", "wilcoxon_p"])))
    o = pd.DataFrame({"group": t.group.map(GROUP_LABELS), "metric": t.metric, "version": t.version,
                      "pair": t.A.map(lambda m: SHORT.get(m, m)) + " - " + t.B.map(lambda m: SHORT.get(m, m)),
                      "n": t.n_seeds, "mean diff": t.mean_diff.map(f),
                      "NB 95% CI": "[" + t.NB_lo.map(f) + ", " + t.NB_hi.map(f) + "]",
                      "A better": t.share_A_better.map(lambda x: f"{100 * x:.0f}%" if np.isfinite(x) else "-"),
                      "p NB (Holm)": t.NB_p_holm.map(lambda x: f"{x:.3f}"),
                      "p Wilcoxon": t.wilcoxon_p.map(lambda x: f"{x:.3f}")})
    return o


def plot_r2(res, saver=None):
    p = res["perf"][res["perf"].version == "raw"]
    models = [m for m in _models(p) if m != "GLM"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, lab in ((axes[0], "tweedie_deviance", "Deviance: model - GLM (lower is better)"),
                            (axes[1], "gini", "Gini: model - GLM (higher is better)")):
        W = p.pivot_table(index="seed", columns="method", values=metric)
        data = [(W[m] - W["GLM"]).dropna().to_numpy() for m in models]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, m in zip(bp["boxes"], models):
            patch.set_facecolor(COLORS.get(m, "#999999"))
            patch.set_alpha(0.35)
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(d)), d, s=8,
                       color=COLORS.get(models[i - 1], "#555555"), alpha=0.7)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels([SHORT.get(m, m) for m in models], rotation=30, ha="right")
        ax.set_title(lab)
    plt.tight_layout()
    if saver:
        saver.fig(fig, "R2_paired_vs_glm")
    plt.show()
    return fig


# =============================================================================================================
# R3 noise floor and level diagnostics
# =============================================================================================================
def r3_noise_floor(res, grid=np.linspace(0.55, 1.35, 801)):
    s = res["seeds"]
    s = s[s.seed.isin(res["perf"].seed.unique())]
    ppf = s.pp_full.iloc[0]
    ratio = s.E_test / s.O_test
    b_int = s.pp_train * ratio - 1
    b_full = ppf * ratio - 1
    curve = pd.DataFrame({"level": grid, "mean_abs_B": [np.mean(np.abs(c * ppf * ratio - 1)) for c in grid]})
    best = curve.loc[curve.mean_abs_B.idxmin()]
    summary = pd.DataFrame([{
        "seeds": len(s), "test pure premium CV (%)": 100 * s.pp_test.std() / s.pp_test.mean(),
        "intercept-only mean |B| (%)": 100 * b_int.abs().mean(), "intercept-only signed B (%)": 100 * b_int.mean(),
        "full-mean constant mean |B| (%)": 100 * b_full.abs().mean(),
        "best constant level (x full mean)": best.level, "best constant mean |B| (%)": 100 * best.mean_abs_B,
        "corr(O_train, O_test)": np.corrcoef(s.O_train, s.O_test)[0, 1]}])
    p = res["perf"].merge(s[["seed", "E_test", "pp_train", "pp_full"]], on="seed")
    p["level_full"] = p.predicted_total / p.E_test / p.pp_full
    ref = p[p.method == "GLM"][["seed", "version", "predicted_total"]].rename(columns={"predicted_total": "ref"})
    p = p.merge(ref, on=["seed", "version"], how="left")
    p["level_vs_glm"] = p.predicted_total / p.ref - 1
    lv = []
    for (m, v), g in p.groupby(["method", "version"]):
        if v not in VERSIONS:
            continue
        lv.append({"method": m, "version": v, "mean |B| (%)": 100 * g.abs_balance_error_pct.mean(),
                   "pooled bias (%)": 100 * (g.predicted_total.sum() / g.observed_total.sum() - 1),
                   "level / full mean": g.level_full.mean(),
                   "mean level vs GLM (%)": 100 * g.level_vs_glm.mean(),
                   "SD level vs GLM (%)": 100 * g.level_vs_glm.std()})
    lv = pd.DataFrame(lv)
    lv["_o"] = lv.method.map({m: i for i, m in enumerate(MODEL_ORDER)})
    lv["_v"] = lv.version.map({v: i for i, v in enumerate(VERSIONS)})
    lv = lv.sort_values(["_v", "_o"]).drop(columns=["_o", "_v"]).reset_index(drop=True)
    return summary, curve, lv


def plot_r3(curve, summary, levels, saver=None):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(curve.level, 100 * curve.mean_abs_B, color="#555555", lw=1.6, label="Constant predictor (no covariates)")
    ax.axvline(1.0, color="#999999", ls="--", lw=0.9)
    for v, mk, fc in (("raw", "o", None), ("train_level", "s", "none")):
        d = levels[levels.version == v]
        for _, r in d.iterrows():
            c = COLORS.get(r.method, "#333333")
            ax.scatter(r["level / full mean"], r["mean |B| (%)"], marker=mk, s=45, color=c,
                       facecolors=c if fc is None else fc, zorder=3,
                       label=f"{SHORT.get(r.method, r.method)}" if v == "raw" else None)
    ax.scatter([], [], marker="o", color="#333333", label="raw (filled)")
    ax.scatter([], [], marker="s", facecolors="none", color="#333333", label="train level (open)")
    b = summary.iloc[0]
    ax.scatter(b["best constant level (x full mean)"], b["best constant mean |B| (%)"], marker="x", color="black",
               s=60, zorder=4, label="best constant")
    ax.set_xlabel("Average predicted level / full-portfolio pure premium")
    ax.set_ylabel("Mean absolute test balance error (%)")
    ax.legend(fontsize=7.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if saver:
        saver.fig(fig, "R3_balance_level_curve")
    plt.show()
    return fig


# =============================================================================================================
# R4 calibration versions
# =============================================================================================================
def r4_versions(res, versions=VERSIONS):
    p = res["perf"]
    W = p.pivot_table(index=["method", "seed"], columns="version", values="abs_balance_error_pct")
    rows = []
    for m in _models(p):
        if m not in W.index.get_level_values(0):
            continue
        g = W.loc[m]
        r = {"Model": SHORT.get(m, m)}
        for v in versions:
            if v in g:
                r[f"|B| {VERSION_LABELS[v]} (%)"] = 100 * g[v].mean()
        for v in versions[1:]:
            if v in g:
                r[f"HR {VERSION_LABELS[v]} (%)"] = 100 * (g[v] > g["raw"]).mean()
        rows.append(r)
    absb = pd.DataFrame(rows)
    dev = (p.pivot_table(index="method", columns="version", values="tweedie_deviance", aggfunc="mean")
           .reindex(_models(p))[list(versions)])
    gini = (p.pivot_table(index="method", columns="version", values="gini", aggfunc="mean")
            .reindex(_models(p))[list(versions)])
    return absb, dev, gini


def plot_r4(res, saver=None, versions=VERSIONS):
    p = res["perf"]
    groups = ["all", "good_seed", "bad_seed"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    x = np.arange(len(versions))
    for ax, g in zip(axes, groups):
        d = p if g == "all" else p[p.group == g]
        for m in _models(d):
            s = d[d.method == m].groupby("version").abs_balance_error_pct.mean().reindex(versions)
            ax.plot(x, 100 * s.to_numpy(), marker="o", lw=1.8, color=COLORS.get(m), label=SHORT.get(m, m))
        ax.set_xticks(x)
        ax.set_xticklabels([VERSION_LABELS[v] for v in versions], rotation=30, ha="right")
        ax.set_title(f"{GROUP_LABELS[g]} seeds (n={d.seed.nunique()})")
    axes[0].set_ylabel("Mean absolute test balance error (%)")
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if saver:
        saver.fig(fig, "R4_calibration_paths")
    plt.show()
    return fig


# =============================================================================================================
# R5 regular vs stressed
# =============================================================================================================
def r5_groups(res, versions=("raw", "train_level")):
    p = res["perf"][res["perf"].version.isin(versions)]
    out = []
    for m in _models(p):
        for g in ("good_seed", "bad_seed"):
            d = p[(p.method == m) & (p.group == g)]
            r = {"Model": SHORT.get(m, m), "Group": GROUP_LABELS[g], "Seeds": d.seed.nunique()}
            for v in versions:
                dv = d[d.version == v]
                r[f"|B| {VERSION_LABELS[v]}"] = 100 * dv.abs_balance_error_pct.mean()
                r[f"signed B {VERSION_LABELS[v]}"] = 100 * dv.balance_error_pct.mean()
            r["Deviance raw"] = d[d.version == "raw"].tweedie_deviance.mean()
            r["Gini raw"] = d[d.version == "raw"].gini.mean()
            out.append(r)
    return pd.DataFrame(out)


def r7_train_valid_test(res, versions=("raw", "train_level")):
    p = res["perf"][res["perf"].version.isin(versions)]
    t = (100 * p.groupby(["method", "version"])[["train_balance_error_pct", "valid_balance_error_pct",
                                                 "balance_error_pct"]].mean()).reset_index()
    t.columns = ["method", "version", "Train signed B (%)", "Valid signed B (%)", "Test signed B (%)"]
    t["fitted on"] = t.version.map({"raw": "-", "train_level": "training (train B in-sample)"})
    t["_o"] = t.method.map({m: i for i, m in enumerate(MODEL_ORDER)})
    t["_v"] = t.version.map({v: i for i, v in enumerate(VERSIONS)})
    return t.sort_values(["_o", "_v"]).drop(columns=["_o", "_v"]).reset_index(drop=True)


# =============================================================================================================
# R9 blends
# =============================================================================================================
def r9_blends(res):
    b = res["blends"]
    if b is None or b.empty:
        return None, None
    summ, tests = [], []
    for (a, bb), g in b.groupby(["A", "B"], sort=False):
        W = g.pivot_table(index="seed", columns="blend", values="tweedie_deviance")
        G = g.pivot_table(index="seed", columns="blend", values="gini")
        best = "A" if W["A"].mean() <= W["B"].mean() else "B"
        r = {"pair": f"{SHORT.get(a, a)} + {SHORT.get(bb, bb)}", "seeds": len(W)}
        for lab in ("A", "B", "50/50", "valid_w"):
            r[f"dev {lab}"] = W[lab].mean()
            r[f"gini {lab}"] = G[lab].mean()
        r["mean weight A (valid)"] = g[g.blend == "valid_w"].weight_A.mean()
        summ.append(r)
        for lab in ("50/50", "valid_w"):
            tests.append({"pair": r["pair"], "blend": lab, "vs": f"best single ({best})", "metric": "deviance",
                          **paired_stats(W[lab] - W[best])})
            tests.append({"pair": r["pair"], "blend": lab, "vs": f"best single ({best})", "metric": "gini",
                          **paired_stats(G[lab] - G[best], higher_is_better=True)})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["NB_p_holm"] = tests.groupby("metric").NB_p.transform(lambda s: _holm(s.values))
    return pd.DataFrame(summ), tests


# =============================================================================================================
# R10 shape stability
# =============================================================================================================
def _centred_pd(pdc):
    d = pdc.copy()
    d["c"] = d.pd_log_mu - d.groupby(["seed", "method", "feature"]).pd_log_mu.transform("mean")
    return d


def r10_shape_stability(res):
    """
    Partial dependence of log mu, centred within each (seed, model, feature). Instability = between-seed SD of the
    centred curve, averaged over grid points / levels; amplitude = median over seeds of max - min of the curve;
    relative instability = instability / amplitude.
    """
    pdc = res["pd"]
    if pdc is None or pdc.empty:
        return None
    d = _centred_pd(pdc)
    sd = d.groupby(["method", "feature", "x"]).c.std().groupby(["method", "feature"]).mean()
    amp = d.groupby(["method", "feature", "seed"]).c.agg(lambda s: s.max() - s.min()).groupby(["method", "feature"]).median()
    t = pd.DataFrame({"instability": sd, "amplitude": amp}).reset_index()
    t["relative"] = t.instability / t.amplitude
    wide = t.pivot(index="method", columns="feature", values="instability").reindex(_models(pdc))
    wide["mean"] = wide.mean(1)
    return t, wide.reset_index()


def plot_r10(res, features=None, saver=None):
    pdc = res["pd"]
    if pdc is None or pdc.empty:
        return None
    d = _centred_pd(pdc[pdc.kind == "num"])
    if features is None:
        features = list(dict.fromkeys(d.feature))
    labels = dsm.spec(res.get("dataset", "freMTPL2"))["labels"]
    models = _models(d)
    feats = [f for f in features if f in set(d.feature)]
    fig, axes = plt.subplots(len(feats), len(models), figsize=(2.3 * len(models), 1.9 * len(feats)), sharey="row")
    axes = np.atleast_2d(axes)
    for i, f in enumerate(feats):
        for j, m in enumerate(models):
            ax = axes[i, j]
            g = d[(d.feature == f) & (d.method == m)].pivot_table(index="x_num", columns="seed", values="c")
            if g.empty:
                continue
            ax.fill_between(g.index, g.quantile(0.1, axis=1), g.quantile(0.9, axis=1), color=COLORS.get(m), alpha=0.25)
            ax.plot(g.index, g.median(axis=1), color=COLORS.get(m), lw=1.5)
            if i == 0:
                ax.set_title(SHORT.get(m, m), fontsize=9)
            if j == 0:
                ax.set_ylabel(labels.get(f, f), fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle("Partial dependence of log pure premium (centred): median and 10-90% band over seeds", y=1.01)
    plt.tight_layout()
    if saver:
        saver.fig(fig, "R10_shape_stability")
    plt.show()
    return fig


# =============================================================================================================
# R11 balance and the allocation of losses; training diagnostics
# =============================================================================================================
def r11_allocation(res, versions=("raw", "train_level")):
    """
    log(1 + B_test) regressed on x_tr = log(pp_test / pp_train) and on x_va = log(pp_test / pp_valid).
    A model whose level follows its training set has slope -1 on x_tr and R^2 close to 1: its test balance is then
    explained by the split, not by the model.
    """
    s = res["seeds"]
    p = res["perf"][res["perf"].version.isin(versions)].merge(s[["seed", "pp_train", "pp_valid", "pp_test"]], on="seed")
    p["yB"] = np.log1p(p.balance_error_pct)
    p["x_tr"] = np.log(p.pp_test / p.pp_train)
    p["x_va"] = np.log(p.pp_test / p.pp_valid)
    rows = []
    for (m, v), g in p.groupby(["method", "version"]):
        r = {"method": m, "version": v, "seeds": len(g)}
        for x in ("x_tr", "x_va"):
            lr = stats.linregress(g[x], g.yB)
            r[f"slope_{x}"], r[f"R2_{x}"] = lr.slope, lr.rvalue ** 2
        r["residual SD of log(1+B) after x_tr (%)"] = 100 * np.std(g.yB - (stats.linregress(g.x_tr, g.yB).intercept
                                                                          + stats.linregress(g.x_tr, g.yB).slope * g.x_tr))
        rows.append(r)
    t = pd.DataFrame(rows)
    t["_o"] = t.method.map({m: i for i, m in enumerate(MODEL_ORDER)})
    t["_v"] = t.version.map({v: i for i, v in enumerate(VERSIONS)})
    return t.sort_values(["_v", "_o"]).drop(columns=["_o", "_v"]).reset_index(drop=True), p


def plot_r11(pdata, saver=None, versions=("raw", "train_level")):
    fig, axes = plt.subplots(1, len(versions), figsize=(5 * len(versions), 4.2), sharey=True)
    for ax, v in zip(np.atleast_1d(axes), versions):
        d = pdata[pdata.version == v]
        x = "x_tr"
        for m in _models(d):
            g = d[d.method == m]
            ax.scatter(g[x], 100 * g.balance_error_pct, s=12, color=COLORS.get(m), alpha=0.75, label=SHORT.get(m, m))
        xx = np.linspace(d[x].min(), d[x].max(), 50)
        ax.plot(xx, 100 * (np.exp(-xx) - 1), color="black", lw=1, ls="--", label="level fixed: B = exp(-x) - 1")
        ax.axhline(0, color="#999999", lw=0.7)
        ax.set_xlabel("log(pp_test / pp_train)")
        ax.set_title(VERSION_LABELS[v])
    np.atleast_1d(axes)[0].set_ylabel("Signed test balance error (%)")
    np.atleast_1d(axes)[-1].legend(fontsize=7.5, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    if saver:
        saver.fig(fig, "R11_balance_vs_allocation")
    plt.show()
    return fig


def r11_training(res):
    p = res["perf"][res["perf"].version == "raw"]
    t = p.groupby("method").agg(minutes=("fit_seconds", lambda s: s.mean() / 60), epochs=("epochs_run", "mean"),
                                best_iteration=("best_iteration", "mean"),
                                best_iteration_sd=("best_iteration", "std")).reindex(_models(p)).reset_index()
    hist_rows = []
    for h in res["hist"]:
        H = pd.DataFrame(h["history"])
        if H.empty:
            continue
        b = int(H.loc[H.criterion.idxmin(), "epoch"])
        hist_rows.append({"method": h["method"], "seed": h["seed"], "best_epoch": b, "epochs": len(H),
                          "valid_gain_after_epoch1": H.valid_loss.iloc[0] - H.valid_loss.min()})
    h = pd.DataFrame(hist_rows)
    hs = (h.groupby("method").agg(best_epoch_median=("best_epoch", "median"),
                                  best_epoch_le3_share=("best_epoch", lambda s: (s <= 3).mean()),
                                  epochs_median=("epochs", "median")).reset_index() if len(h) else None)
    return t, hs


# =============================================================================================================
# Everything
# =============================================================================================================
def run_all(out_dir, group="paper", save=True):
    _style()
    res = load_results(out_dir, group=group)
    sv = _Saver(out_dir if save else None)
    out = {"res": res}

    print("\n" + "=" * 100 + "\nR1  Main performance table\n" + "=" * 100)
    out["R1"] = r1_main_table(res)
    _disp(out["R1"]); sv.table(out["R1"], "R1_main_table")

    print("\n" + "=" * 100 + "\nR2  Paired tests (Nadeau-Bengio corrected; Holm within metric x group)\n" + "=" * 100)
    out["R2"] = r2_paired_tests(res, versions=("raw", "train_level"))
    for metric, scale, dg in (("tweedie_deviance", 1, 3), ("gini", 1, 4), ("abs_balance_error_pct", 100, 2)):
        t = out["R2"][out["R2"].metric == metric]
        if len(t):
            print(f"\n{metric}")
            _disp(format_paired(t, scale, dg))
    sv.table(out["R2"], "R2_paired_tests", latex=False)
    plot_r2(res, sv)

    print("\n" + "=" * 100 + "\nR3  Balance noise floor and level diagnostics\n" + "=" * 100)
    summ, curve, lv = r3_noise_floor(res)
    out["R3"] = {"noise_floor": summ, "levels": lv}
    _disp(summ.round(3)); _disp(lv.round(3))
    sv.table(summ, "R3_noise_floor"); sv.table(lv, "R3_levels")
    plot_r3(curve, summ, lv, sv)

    print("\n" + "=" * 100 + "\nR4  Global rescaling vs raw (mean |B|, harm rate HR = share of seeds worse than raw)\n" + "=" * 100)
    absb, dev, gini = r4_versions(res)
    out["R4"] = {"abs_balance": absb, "deviance": dev, "gini": gini}
    _disp(absb.round(2)); print("mean test deviance by version"); _disp(dev.round(4))
    print("mean Gini by version"); _disp(gini.round(4))
    sv.table(absb, "R4_versions_abs_balance"); sv.table(dev.reset_index(), "R4_versions_deviance")
    plot_r4(res, sv)

    print("\n" + "=" * 100 + "\nR5  Regular vs stressed splits\n" + "=" * 100)
    out["R5"] = r5_groups(res)
    _disp(out["R5"].round(2)); sv.table(out["R5"], "R5_groups")

    print("\n" + "=" * 100 + "\nR7  Signed balance on train / validation / test\n" + "=" * 100)
    out["R7"] = r7_train_valid_test(res)
    _disp(out["R7"].round(2)); sv.table(out["R7"], "R7_train_valid_test")

    print("\n" + "=" * 100 + "\nR9  Blends of two models (weights chosen on validation)\n" + "=" * 100)
    bs, bt = r9_blends(res)
    out["R9"] = {"summary": bs, "tests": bt}
    if bs is not None:
        _disp(bs.round(4))
        _disp(bt[["pair", "blend", "vs", "metric", "n_seeds", "mean_diff", "NB_lo", "NB_hi", "share_A_better",
                  "NB_p_holm"]].round(4))
        sv.table(bs, "R9_blends"); sv.table(bt, "R9_blend_tests", latex=False)

    print("\n" + "=" * 100 + "\nR10 Shape stability across splits (SD of centred partial dependence)\n" + "=" * 100)
    r10 = r10_shape_stability(res)
    if r10 is not None:
        out["R10"] = {"long": r10[0], "wide": r10[1]}
        _disp(r10[1].round(4)); sv.table(r10[1], "R10_shape_instability"); sv.table(r10[0], "R10_long", latex=False)
        plot_r10(res, saver=sv)

    print("\n" + "=" * 100 + "\nR11 Test balance and the allocation of losses; training diagnostics\n" + "=" * 100)
    t11, pdata = r11_allocation(res)
    tr, hs = r11_training(res)
    out["R11"] = {"allocation": t11, "training": tr, "nn_history": hs}
    _disp(t11.round(3)); _disp(tr.round(2))
    if hs is not None:
        _disp(hs.round(2))
    sv.table(t11, "R11_allocation"); sv.table(tr, "R11_training")
    plot_r11(pdata, sv)
    if save:
        print(f"\ntables and figures written to {os.path.join(out_dir, 'analysis')}")
    return out
