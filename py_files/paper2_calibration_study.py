"""
paper2_calibration_study.py — credibility cells for ANY grouping and ANY credibility constant k, computed from the
stored raw predictions (paper2_predictions.py; GA2M+ predictions from ga2m_plus.py). Nothing is fitted.

Design (fixed before looking at the test sets)
  Level        set on the TRAINING set: c = O_train / P_train (global rescaling, "A"). No factor is fitted to the
               validation level.
  Cells        the cell ratios O_l / (c P_l) are taken from the VALIDATION set; the TEST set only evaluates.
  Groupings    GROUPINGS below: categorical variables alone and in pairs; numerical variables in pre-specified bands
               (BANDS; Density: portfolio quintiles of log Density, no target information) alone, in pairs and crossed
               with Region. 20 groupings.
  k grid       K_GRID = 0, 100, 250, 500, 1000, 2000, 5000, 10000, inf (exposure units, Z_l = E_l / (E_l + k)), plus
               "BS" = Buhlmann-Straub estimate (volume units, Z_l = V_l / (V_l + k_hat)).
  Modes
     raw           model output
     A             global rescaling on the training set
     cell_only_A   A followed by unclipped credibility cell factors f_l = 1 + Z_l (O_l / (c P_l) - 1), unseen cells 1:
                   what the cells add beyond the training-set level
     (the proposed calibrator, the clipped version of cell_only_A, is computed in paper2_age_calibration)
  Test metrics balance B, |B|, predicted / observed totals (pooled bias), Tweedie deviance, Gini, and the balance
               over the cells of the SAME grouping: misallocation sum_l |P_l - O_l| / sum_l O_l (primary; money-
               weighted) and the exposure-weighted mean |P_l / O_l - 1| over cells with O_l > 0 (reported only).
  Heterogeneity per (seed, model, grouping) on the validation set: Buhlmann-Straub sigma^2, tau^2, k_hat, mean Z, and
               the permutation p-value of tau^2 = 0 (n_perm permutations).

Consistency check: raw and A reproduce the main run's versions raw and train_level exactly: consistency_check().

    import paper2_calibration_study as cs
    cs.run_study(df_freq, df_sev, pred_dirs=["v2_predictions", "v2_ga2mplus_c995_meanrank/predictions"],
                 out_dir="v2_calibration")
    cs.consistency_check("v2_calibration", "v2_full")
    cs.report("v2_calibration")
"""

import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import paper2_adjust_clean_plus as p2
import paper2_models as pm
import paper2_predictions as pp
import paper2_run_full as rf

K_GRID = [0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, np.inf]
MODES = ["raw", "A", "cell_only_A"]
PROPOSED_GROUPING = "DrivAge x BonusMalus"
MODELS = ["GLM", "Tweedie GLM", "EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Neural GA2M+", "Tweedie-CANN"]
TEST_FRAC, TRAIN_FRAC = 0.20, 0.60

BANDS = {   # left-closed bands; pre-specified (actuarial conventions), not tuned on any target
    "DrivAge": [18, 21, 26, 31, 41, 51, 61, 71, np.inf],
    "BonusMalus": [50, 51, 61, 81, 101, np.inf],
    "VehAge": [0, 1, 3, 6, 11, 16, np.inf],
    "VehPower": [4, 5, 6, 7, 8, 9, 11, np.inf],
    "Density": "quintiles",                     # of log(Density) on the whole portfolio
}


def _b(v):
    return f"{v}_band"


GROUPINGS = {
    # categorical
    "Region": ["Region"], "Area": ["Area"], "VehGas": ["VehGas"], "VehBrand": ["VehBrand"],
    "Region x Area": ["Region", "Area"], "Region x VehGas": ["Region", "VehGas"], "Area x VehGas": ["Area", "VehGas"],
    "Region x VehBrand": ["Region", "VehBrand"], "VehBrand x VehGas": ["VehBrand", "VehGas"],
    # numerical bands
    "DrivAge band": [_b("DrivAge")], "BonusMalus band": [_b("BonusMalus")], "VehAge band": [_b("VehAge")],
    "VehPower band": [_b("VehPower")], "Density band": [_b("Density")],
    "DrivAge x BonusMalus": [_b("DrivAge"), _b("BonusMalus")], "DrivAge x VehPower": [_b("DrivAge"), _b("VehPower")],
    "DrivAge x VehAge": [_b("DrivAge"), _b("VehAge")], "BonusMalus x VehAge": [_b("BonusMalus"), _b("VehAge")],
    # mixed
    "DrivAge band x Region": [_b("DrivAge"), "Region"], "BonusMalus band x Region": [_b("BonusMalus"), "Region"],
}
GROUP_TYPE = {g: ("categorical" if not any(c.endswith("_band") for c in cols) else
                  ("numerical bands" if all(c.endswith("_band") for c in cols) else "mixed"))
              for g, cols in GROUPINGS.items()}


def add_bands(df, num_cols):
    """Band columns of the numerical variables (portfolio-level edges; no target information)."""
    out = df.copy()
    for v in num_cols:
        spec = BANDS.get(v)
        x = out[v].astype(float)
        if spec is None:
            continue
        if isinstance(spec, str):
            edges = np.unique(np.quantile(x, [0.2, 0.4, 0.6, 0.8]))
            out[_b(v)] = np.searchsorted(edges, x, side="right").astype(str)
        else:
            out[_b(v)] = pd.cut(x, spec, right=False).astype(str)
    return out


# =============================================================================================================
# Core computation for one (seed, model, grouping)
# =============================================================================================================
def _tweedie_dev(y, mu, e, power):
    return float(p2.tweedie_deviance(y, mu, e, power=power))


def _segment_metrics(ids, y, mu, e, n):
    """seg_misalloc = sum_l |P_l - O_l| / sum_l O_l: share of the observed loss that is mis-allocated across the cells
    of the grouping (money-weighted, robust to tiny cells; includes the overall imbalance). seg_absB = exposure-
    weighted mean of |P_l / O_l - 1| over cells with O_l > 0 (as paper2_run_full.segment_abs_balance; explodes for
    fine groupings with small cells, so it is reported but not used for conclusions)."""
    o = np.bincount(ids, weights=e * y, minlength=n)
    p = np.bincount(ids, weights=e * mu, minlength=n)
    ex = np.bincount(ids, weights=e, minlength=n)
    keep = o > 0
    return (float(np.sum(np.abs(p - o)) / o.sum()),
            float(np.sum(ex[keep] * np.abs(p[keep] / o[keep] - 1)) / ex[keep].sum()))


def _bs(ids, y, mu, e, n):
    """Buhlmann-Straub on the validation set (q = 1, vectorised on ids)."""
    w = e * mu
    x = y / mu
    n_i = np.bincount(ids, minlength=n).astype(float)
    w_i = np.bincount(ids, weights=w, minlength=n)
    xb_i = np.bincount(ids, weights=w * x, minlength=n) / np.maximum(w_i, 1e-300)
    ss = np.bincount(ids, weights=w * (x - xb_i[ids]) ** 2, minlength=n)
    dof = np.sum(np.maximum(n_i - 1, 0))
    sigma2 = ss.sum() / dof if dof > 0 else np.nan
    W = w_i.sum()
    xbar = np.sum(w_i * xb_i) / W
    I = int(np.sum(n_i > 0))
    denom = W - np.sum(w_i ** 2) / W
    tau2 = (np.sum(w_i * (xb_i - xbar) ** 2) - (I - 1) * sigma2) / denom if denom > 0 else np.nan
    k = sigma2 / tau2 if (np.isfinite(tau2) and tau2 > 0) else np.inf
    return {"n_cells": I, "sigma2": sigma2, "tau2": tau2, "k_hat": k}


def _perm_p(ids, y, mu, e, n, n_perm, seed=0):
    if not n_perm:
        return np.nan
    rng = np.random.default_rng(seed)
    w, x = e * mu, y / mu
    xbar = np.sum(w * x) / np.sum(w)

    def stat(i):
        w_i = np.bincount(i, weights=w, minlength=n)
        xb = np.bincount(i, weights=w * x, minlength=n) / np.maximum(w_i, 1e-300)
        return np.sum(w_i * (xb - xbar) ** 2)

    t0 = stat(ids)
    t = np.array([stat(rng.permutation(ids)) for _ in range(n_perm)])
    return float((1 + np.sum(t >= t0)) / (n_perm + 1))


def evaluate_grouping(ids_fit, y_fit, mu_fit, e_fit, ids_test, y_test, mu_test, e_test, n, c_train, power,
                      n_perm=99):
    """
    raw, A and cell_only_A for one grouping. ids_*: cell ids in [0, n) shared by the validation (fit) and the test set.
    mu_*: RAW predictions; c_train: training-level factor of A. Returns (rows, heterogeneity dict).
    """
    E = np.bincount(ids_fit, weights=e_fit, minlength=n)
    O = np.bincount(ids_fit, weights=e_fit * y_fit, minlength=n)
    P = np.bincount(ids_fit, weights=e_fit * mu_fit, minlength=n)
    seen = np.bincount(ids_fit, minlength=n) > 0
    Xb = np.where(P > 0, O / np.maximum(P, 1e-300), 1.0)
    P_tot = P.sum()
    het = _bs(ids_fit, y_fit, mu_fit, e_fit, n)
    het["p_heterogeneity"] = _perm_p(ids_fit, y_fit, mu_fit, e_fit, n, n_perm)
    Z_bs = P / (P + het["k_hat"]) if np.isfinite(het["k_hat"]) else np.zeros(n)
    Z_bs = np.where(seen, Z_bs, 0.0)
    het["mean_Z_bs"] = float(np.sum(P * Z_bs) / P_tot)
    het["n_cells_test_unseen"] = int(np.sum(~seen[np.unique(ids_test)]))
    obs_test = float(np.sum(e_test * y_test))
    rows = []

    def add(mode, k, f, base=1.0, Z=None):
        fac = np.where(seen, f, 1.0)
        mu = np.clip(np.nan_to_num(base * mu_test * fac[ids_test], nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6, 1e6)
        prd = float(np.sum(e_test * mu))
        rows.append({"mode": mode, "k": k, "test_B": prd / obs_test - 1, "pred_total": prd, "obs_total": obs_test,
                     "deviance": _tweedie_dev(y_test, mu, e_test, power), "gini": rf.gini_exposure(y_test, mu, e_test),
                     **dict(zip(("seg_misalloc", "seg_absB"), _segment_metrics(ids_test, y_test, mu, e_test, n))),
                     "mean_Z": float(np.sum(E * Z) / E.sum()) if Z is not None else np.nan,
                     "factor_min": float(np.min(fac[seen] * base)), "factor_max": float(np.max(fac[seen] * base))})

    ones = np.ones(n)
    add("raw", "-", ones)
    add("A", "-", ones, base=c_train)
    for k in K_GRID:
        Z = np.where(seen, E / (E + k) if np.isfinite(k) else 0.0, 0.0) if k > 0 else seen.astype(float)
        kk = "inf" if not np.isfinite(k) else f"{k:g}"
        add("cell_only_A", kk, 1 + Z * (Xb / c_train - 1), base=c_train, Z=Z)
    add("cell_only_A", "BS", 1 + Z_bs * (Xb / c_train - 1), base=c_train, Z=Z_bs)
    return rows, het


# =============================================================================================================
# Run
# =============================================================================================================
def _cell_codes(frames, cols):
    """Joint integer codes of the cells over several frames (same cell -> same code)."""
    both = pd.concat([f[cols].astype(str) for f in frames], ignore_index=True)
    codes = both.groupby(cols, sort=True, dropna=False).ngroup().to_numpy(np.int64)
    out, start = [], 0
    for f in frames:
        out.append(codes[start:start + len(f)])
        start += len(f)
    return out, int(codes.max()) + 1


def run_study(df_freq, df_sev, pred_dirs=("v2_predictions", "v2_ga2mplus_c995_meanrank/predictions"), out_dir="v2_calibration",
              seeds=range(41, 91), models=None, groupings=None, n_perm=99, df_prepared=None):
    """Resumable per (seed, model). Writes out_dir/results.csv and out_dir/heterogeneity.csv."""
    power = pp.power()
    models = MODELS if models is None else models
    groupings = GROUPINGS if groupings is None else {g: GROUPINGS[g] for g in groupings}
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"pred_dirs": list(pred_dirs), "k_grid": [str(k) for k in K_GRID], "groupings": groupings,
               "bands": {k: (v if isinstance(v, str) else [str(x) for x in v]) for k, v in BANDS.items()},
               "fit_set": "valid", "n_perm": n_perm, "tweedie_power": power},
              open(os.path.join(out_dir, "study_settings.json"), "w"), indent=1)
    R, H = os.path.join(out_dir, "results.csv"), os.path.join(out_dir, "heterogeneity.csv")
    done = set()
    if os.path.exists(H):
        d = pd.read_csv(H, usecols=["seed", "model"])
        done = set(zip(d.seed, d.model))
    df, num_cols, cat_cols = pp.prepare(df_freq, df_sev) if df_prepared is None else df_prepared
    df = add_bands(df, num_cols)
    t_run = time.time()
    for seed in [int(s) for s in seeds]:
        todo = [m for m in models if (seed, m) not in done and pp.available(pred_dirs, m, [seed])]
        if not todo:
            continue
        t0 = time.time()
        tr, va, te = pp.split(df, seed)
        frames = {"train": tr, "valid": va, "test": te}
        codes = {g: _cell_codes([va, te], cols) for g, cols in groupings.items()}
        y = {s: f[pm.TARGET].to_numpy(float) for s, f in frames.items()}
        e = {s: f[pm.EXPOSURE].to_numpy(float) for s, f in frames.items()}
        for m in todo:
            d = pp.load_predictions(pred_dirs, m, seed)
            for s in pp.SETS:
                if d["idhash"][s] != pp.id_hash(frames[s]["IDpol"].to_numpy(np.int64)):
                    raise RuntimeError(f"{m} seed {seed}: stored {s} policies differ from the regenerated split")
            c = float(np.sum(e["train"] * y["train"]) / d["train_sum_e_mu"])
            rows, hets = [], []
            for g, ((ids_v, ids_t), n) in codes.items():
                r, h = evaluate_grouping(ids_v, y["valid"], d["valid"], e["valid"], ids_t, y["test"], d["test"],
                                         e["test"], n, c, power, n_perm=n_perm)
                rows += [dict(seed=seed, model=m, grouping=g, **x) for x in r]
                hets.append(dict(seed=seed, model=m, grouping=g, train_factor=c, **h))
            rf._append(rows, R)
            rf._append(hets, H)                    # heterogeneity last: marks (seed, model) as done
        print(f"seed {seed}: {', '.join(todo)} | {(time.time() - t0) / 60:.1f} min")
    print(f"study finished in {(time.time() - t_run) / 60:.1f} min")
    return load(out_dir)


def load(out_dir):
    r = pd.read_csv(os.path.join(out_dir, "results.csv"), dtype={"k": str})
    h = pd.read_csv(os.path.join(out_dir, "heterogeneity.csv"))
    return r, h


# =============================================================================================================
# Checks and summaries
# =============================================================================================================
MAIN_MAP = {"raw": ("raw", "-"), "train_level": ("A", "-")}


def consistency_check(out_dir, main_dir, tol=1e-5, tol_gini=1e-4):
    """raw and A must reproduce the main run's versions raw and train_level (deviance, B; Gini up to float32 ties)."""
    r, _ = load(out_dir)
    r = r[r.grouping == PROPOSED_GROUPING]
    perf = pd.read_csv(os.path.join(main_dir, "perf.csv")).rename(columns={"method": "model"})
    rows = []
    for v, (mode, k) in MAIN_MAP.items():
        a = r[(r["mode"] == mode) & (r.k == k)].set_index(["seed", "model"])[["deviance", "gini", "test_B"]]
        b = perf[perf.version == v].set_index(["seed", "model"])[["tweedie_deviance", "gini", "balance_error_pct"]]
        j = a.join(b.rename(columns={"gini": "gini_main"}), how="inner")
        if j.empty:
            continue
        rows.append({"main version": v, "study (mode, k)": f"{mode}, {k}", "pairs": len(j),
                     "max rel diff deviance": ((j.deviance - j.tweedie_deviance).abs() / j.tweedie_deviance.abs()).max(),
                     "max abs diff Gini": (j.gini - j.gini_main).abs().max(),
                     "max abs diff B": (j.test_B - j.balance_error_pct).abs().max()})
    t = pd.DataFrame(rows)
    t["ok"] = (t["max rel diff deviance"] < tol) & (t["max abs diff B"] < tol) & (t["max abs diff Gini"] < tol_gini)
    _display(t)
    print("CONSISTENT with the main run" if t.ok.all() else "NOT consistent - see the rows with ok = False")
    return t


def _display(x):
    try:
        from IPython.display import display
        display(x)
    except ImportError:
        print(x)


def _k_order(k):
    special = {"-": -2.0, "inf": 1e8, "BS": 1e9}
    return special[k] if k in special else float(k)


def _nb_p(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    J = len(x)
    if J < 3 or x.std(ddof=1) == 0:
        return np.nan
    se = x.std(ddof=1) * np.sqrt(1 / J + TEST_FRAC / TRAIN_FRAC)
    return float(2 * stats.t.sf(abs(x.mean() / se), J - 1))


def k_table(r, grouping=PROPOSED_GROUPING, modes=("cell_only_A",)):
    """
    Per model x mode x k (one grouping): mean test |B|, pooled bias, mean segment |B| of the grouping, deviance and
    Gini; differences to correction A (paired over seeds): mean, share of seeds worse than A ('harm'), NB-corrected p.
    """
    g = r[r.grouping == grouping].copy()
    g["absB"] = g.test_B.abs()
    base = g[g["mode"] == "A"].set_index(["seed", "model"])
    rows = []
    for (m, mode, k), d in g[g["mode"].isin(modes)].groupby(["model", "mode", "k"]):
        d = d.set_index(["seed", "model"])
        b = base.loc[d.index]
        dd, ds, dg = d.deviance - b.deviance, d.seg_misalloc - b.seg_misalloc, d.gini - b.gini
        rows.append({"model": m, "mode": mode, "k": k, "seeds": len(d),
                     "|B| (%)": 100 * d.absB.mean(), "pooled bias (%)": 100 * (d.pred_total.sum() / d.obs_total.sum() - 1),
                     "segment misallocation (%)": 100 * d.seg_misalloc.mean(), "deviance": d.deviance.mean(),
                     "Gini": d.gini.mean(),
                     "mean Z": d.mean_Z.mean(),
                     "misallocation - A (pp)": 100 * ds.mean(), "harm segment (%)": 100 * (ds > 0).mean(),
                     "p segment (NB)": _nb_p(ds),
                     "deviance - A": dd.mean(), "harm deviance (%)": 100 * (dd > 0).mean(), "p deviance (NB)": _nb_p(dd),
                     "Gini - A": dg.mean()})
    t = pd.DataFrame(rows)
    t["_k"] = t.k.map(_k_order)
    t["_m"] = t.model.map({m: i for i, m in enumerate(MODELS)})
    return t.sort_values(["mode", "_m", "_k"]).drop(columns=["_k", "_m"]).reset_index(drop=True)


def grouping_table(r, h, k_ref=("cell_only_A", "BS")):
    """
    Per grouping x model: heterogeneity on the validation sets (share of seeds with finite k_hat, share with
    permutation p < 0.05, median k_hat, mean Z) and what the Buhlmann-Straub cell correction does on the test sets
    relative to A (segment |B| and deviance; mean difference, harm rate, NB-corrected p).
    """
    mode, k = k_ref
    base = r[r["mode"] == "A"].set_index(["seed", "model", "grouping"])
    var = r[(r["mode"] == mode) & (r.k == k)].set_index(["seed", "model", "grouping"])
    j = var.join(base[["seg_misalloc", "deviance"]], rsuffix="_A", how="inner").reset_index()
    j["d_seg"], j["d_dev"] = j.seg_misalloc - j.seg_misalloc_A, j.deviance - j.deviance_A
    rows = []
    for (g, m), d in j.groupby(["grouping", "model"]):
        hh = h[(h.grouping == g) & (h.model == m)]
        rows.append({"grouping": g, "type": GROUP_TYPE.get(g, ""), "model": m, "cells": hh.n_cells.median(),
                     "finite k_hat (%)": 100 * np.isfinite(hh.k_hat.astype(float)).mean(),
                     "p < 0.05 (%)": 100 * (hh.p_heterogeneity < 0.05).mean(),
                     "median finite k_hat": hh.k_hat[np.isfinite(hh.k_hat.astype(float))].median(),
                     "mean Z (BS)": hh.mean_Z_bs.mean(),
                     "misallocation A (%)": 100 * d.seg_misalloc_A.mean(),
                     "misallocation BS - A (pp)": 100 * d.d_seg.mean(), "harm segment (%)": 100 * (d.d_seg > 0).mean(),
                     "p segment (NB)": _nb_p(d.d_seg), "deviance BS - A": d.d_dev.mean(), "p deviance (NB)": _nb_p(d.d_dev)})
    t = pd.DataFrame(rows)
    t["_g"] = t.grouping.map({g: i for i, g in enumerate(GROUPINGS)})
    t["_m"] = t.model.map({m: i for i, m in enumerate(MODELS)})
    return t.sort_values(["_g", "_m"]).drop(columns=["_g", "_m"]).reset_index(drop=True)


def plot_k_curves(r, grouping=PROPOSED_GROUPING, mode="cell_only_A", saver=None):
    import matplotlib.pyplot as plt
    t = k_table(r, grouping, modes=(mode,))
    ks = sorted(t.k.unique(), key=_k_order)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, col, lab in zip(axes, ["|B| (%)", "segment misallocation (%)", "deviance - A", "Gini - A"],
                            ["test |B| (%)", f"test misallocation over {grouping} (%)", "deviance minus A",
                             "Gini minus A"]):
        for m in [x for x in MODELS if x in set(t.model)]:
            d = t[t.model == m].set_index("k").reindex(ks)
            ax.plot(range(len(ks)), d[col], marker="o", ms=3, label=m)
        if col in ("deviance - A", "Gini - A"):
            ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, rotation=45)
        ax.set_xlabel("k (exposure units; BS = estimated)")
        ax.set_title(lab, fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{mode} on {grouping} (level from training, cells from validation, evaluated on test, mean over "
                 "seeds)", fontsize=11)
    fig.tight_layout()
    if saver:
        saver(fig, f"k_curves_{mode}_{grouping.replace(' ', '')}")
    return fig


def plot_heterogeneity(h, saver=None):
    import matplotlib.pyplot as plt
    t = h.assign(sig=h.p_heterogeneity < 0.05).pivot_table(index="grouping", columns="model", values="sig",
                                                           aggfunc="mean")
    t = t.reindex([g for g in GROUPINGS if g in t.index])[[m for m in MODELS if m in t.columns]]
    fig, ax = plt.subplots(figsize=(1.1 * t.shape[1] + 3, 0.32 * t.shape[0] + 1.5))
    im = ax.imshow(100 * t.values, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels(t.columns, rotation=40, ha="right")
    ax.set_yticks(range(t.shape[0]))
    ax.set_yticklabels(t.index, fontsize=8)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{100 * t.values[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="% of seeds with heterogeneity p < 0.05 (5% expected under no heterogeneity)")
    ax.set_title("Residual cell heterogeneity on the validation sets")
    fig.tight_layout()
    if saver:
        saver(fig, "heterogeneity_heatmap")
    return fig


def report(out_dir, save=True):
    """All tables and figures -> out_dir/tables, out_dir/figures."""
    r, h = load(out_dir)
    tdir, fdir = os.path.join(out_dir, "tables"), os.path.join(out_dir, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)

    def saver(fig, name):
        if save:
            fig.savefig(os.path.join(fdir, name + ".pdf"), bbox_inches="tight")

    print(f"{r.seed.nunique()} seeds | models: {sorted(r.model.unique())} | groupings: {r.grouping.nunique()}")
    out = {}
    print("\n" + "=" * 100 + f"\nC1  k sensitivity on the proposed grouping ({PROPOSED_GROUPING})\n" + "=" * 100)
    out["k_table_proposed"] = k_table(r)
    _display(out["k_table_proposed"].round(4))
    plot_k_curves(r, PROPOSED_GROUPING, "cell_only_A", saver)
    print("\n" + "=" * 100 + "\nC2  heterogeneity and Buhlmann-Straub cell correction for every grouping\n" + "=" * 100)
    out["grouping_table"] = grouping_table(r, h)
    _display(out["grouping_table"].round(4))
    plot_heterogeneity(h, saver)
    print("\n" + "=" * 100 + "\nC3  k sensitivity (cell_only_A) for every grouping: test misallocation minus A (pp), "
          "mean over models and seeds\n" + "=" * 100)
    g = r[r["mode"].isin(["A", "cell_only_A"])]
    base = g[g["mode"] == "A"].set_index(["seed", "model", "grouping"]).seg_misalloc
    v = g[g["mode"] == "cell_only_A"].set_index(["seed", "model", "grouping"])
    v = v.assign(d=100 * (v.seg_misalloc - base.reindex(v.index))).reset_index()
    out["k_by_grouping"] = v.pivot_table(index="grouping", columns="k", values="d", aggfunc="mean")
    out["k_by_grouping"] = out["k_by_grouping"].reindex([x for x in GROUPINGS if x in out["k_by_grouping"].index])[
        sorted(out["k_by_grouping"].columns, key=_k_order)]
    _display(out["k_by_grouping"].round(2))
    if save:
        for k_, t in out.items():
            t.to_csv(os.path.join(tdir, f"{k_}.csv"), index=k_ == "k_by_grouping")
        print(f"tables -> {tdir}, figures -> {fdir}")
    return out
