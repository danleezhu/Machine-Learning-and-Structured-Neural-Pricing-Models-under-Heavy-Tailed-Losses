"""
paper2_calibrator_selection.py — choose the credibility-cell grouping of the calibrator for each model WITHOUT the test set.

Why. Choosing a grouping per model from the test-set heatmap (paper2_age_calibration.figure1) is selection on the
test set, and the reported test gain of the chosen grouping is then optimistic. This module chooses the grouping inside
each split from the validation set only, by cross-fitting, and evaluates the chosen calibrator once on the test set.

Rule (fixed before looking at any test result)
  candidates  : "none" (global rescaling only) and the six groupings of VARIANT_GROUPINGS
  calibrator  : the proposed calibrator, c * mu * clip(1 + Z ((O+1)/(cP+1) - 1), 0.95, 1.15), Z = E/(E+k), k = 2000
  selection   : for each split and model, the validation set is halved at random R = 5 times (10 fits); cell factors
                are estimated on one half and the change in Tweedie deviance relative to global rescaling is measured
                on the other half. The grouping with the lowest mean change is chosen; if no grouping lowers the
                validation deviance, no cells are used ("none").
  final fit   : the chosen grouping's factors are re-estimated on the whole validation set and applied to the test set.
The halvings are the same for all models and groupings of a split, so candidates are compared on the same data.

Inputs: the stored raw predictions (pp.load_predictions) and the data; no model is refitted.

    import paper2_calibrator_selection as sel
    res = sel.run(df_freq, df_sev, [PRED_DIR, f"{FIX_DIR}/predictions"], out_dir="v2_calibrator_selection")
    tabs = sel.summary(res, out_dir="v2_calibrator_selection")
    sel.figure(res, out_dir="v2_calibrator_selection")
"""
import os
import time

import numpy as np
import pandas as pd

import paper2_calibration_study as cs
import paper2_predictions as pp
import paper2_run_full as rf
from paper2_age_calibration import MODELS, VARIANT_GROUPINGS, COL_LABEL, MODEL_LABEL, TARGET, EXPOSURE, _deviance

K = 2000
LOW, HIGH = 0.95, 1.15
R_HALVINGS = 5
FILE = "calibrator_selection.csv"
NONE = "none"


def _factors(iv_fit, e, y, mu_c, n, k=K):
    """Clipped credibility factors per cell from the policies of iv_fit (mu_c = c * mu, already rescaled)."""
    E = np.bincount(iv_fit, weights=e, minlength=n)
    O = np.bincount(iv_fit, weights=e * y, minlength=n)
    P = np.bincount(iv_fit, weights=e * mu_c, minlength=n)
    seen = E > 0
    Z = np.where(seen, E / (E + k), 0.0)
    f = np.clip(1 + Z * ((O + 1.0) / (P + 1.0) - 1), LOW, HIGH)
    return np.where(seen, f, 1.0)


def run(df_freq, df_sev, pred_dirs, out_dir="v2_calibrator_selection", seeds=range(41, 91), models=MODELS,
        groupings=VARIANT_GROUPINGS, k=K, r_halvings=R_HALVINGS):
    """One row per (seed, model): validation cross-fit gain of every candidate, the chosen grouping, and the test
    deviance, Gini and balance of global rescaling, of every grouping, and of the chosen calibrator.
    Resumable per seed (rows are appended to out_dir/calibrator_selection.csv)."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILE)
    done = set(pd.read_csv(path, usecols=["seed"]).seed) if os.path.exists(path) else set()
    power = pp.power()
    df, num_cols, _ = pp.prepare(df_freq, df_sev)
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
        codes = {g: cs._cell_codes([va, te], cs.GROUPINGS[g]) for g in groupings}
        rng = np.random.default_rng(1000 * seed + 7)
        halves = [rng.integers(0, 2, len(va)).astype(bool) for _ in range(r_halvings)]
        rows = []
        for m in models:
            d = pp.load_predictions(pred_dirs, m, seed)
            for s, frame in (("valid", va), ("test", te)):
                if d["idhash"][s] != pp.id_hash(frame["IDpol"].to_numpy(np.int64)):
                    raise RuntimeError(f"{m} seed {seed}: stored {s} policies differ from the regenerated split")
            c = obs_train / d["train_sum_e_mu"]
            mv, mt = c * d["valid"], c * d["test"]
            row = {"seed": seed, "model": m}
            # validation cross-fit: change in deviance relative to global rescaling, averaged over 2 x R fits
            for g, ((iv, it), n) in codes.items():
                gains = []
                for h in halves:
                    for fit, ev_ in ((h, ~h), (~h, h)):
                        f = _factors(iv[fit], ev[fit], yv[fit], mv[fit], n, k)
                        gains.append(_deviance(yv[ev_], mv[ev_] * f[iv[ev_]], ev[ev_], power)
                                     - _deviance(yv[ev_], mv[ev_], ev[ev_], power))
                row[f"cv gain | {g}"] = float(np.mean(gains))
            cv = {g: row[f"cv gain | {g}"] for g in groupings}
            best = min(cv, key=cv.get)
            row["chosen"] = best if cv[best] < 0 else NONE
            # test: global rescaling, every grouping (full validation fit), chosen
            row["test dev | A"] = _deviance(yt, mt, et, power)
            row["test gini | A"] = rf.gini_exposure(yt, mt, et)
            row["test B | A"] = float(np.sum(et * mt)) / obs_test - 1
            for g, ((iv, it), n) in codes.items():
                f = _factors(iv, ev, yv, mv, n, k)
                mu = mt * f[it]
                row[f"test dev | {g}"] = _deviance(yt, mu, et, power)
                row[f"test gini | {g}"] = rf.gini_exposure(yt, mu, et)
                row[f"test B | {g}"] = float(np.sum(et * mu)) / obs_test - 1
            key = "A" if row["chosen"] == NONE else row["chosen"]
            for q in ("dev", "gini", "B"):
                row[f"test {q} | selected"] = row[f"test {q} | {key}"]
            row["obs_test"] = obs_test
            rows.append(row)
        rf._append(rows, path)
        print(f"seed {seed}: done | {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.read_csv(path)


def _stats(res, key):
    dd = res[f"test dev | {key}"] - res["test dev | A"]
    dg = 1000 * (res[f"test gini | {key}"] - res["test gini | A"])
    pred = (1 + res[f"test B | {key}"]) * res.obs_test
    return pd.Series({"deviance change, median": dd.median(), "deviance change, mean": dd.mean(),
                      "splits worse than rescaling (%)": 100 * (dd > 0).mean(),
                      "Gini change (x1000), mean": dg.mean(),
                      "mean |B| (%)": 100 * res[f"test B | {key}"].abs().mean(),
                      "pooled bias (%)": 100 * (pred.sum() / res.obs_test.sum() - 1)})


def summary(res, out_dir="v2_calibrator_selection", fixed="DrivAge x BonusMalus", groupings=VARIANT_GROUPINGS):
    """(1) how often each grouping is chosen, by model; (2) test results of the validation-selected calibrator against
    the fixed grouping and against the best grouping in hindsight (test-chosen, optimistic, for reference only)."""
    freq = (pd.crosstab(res.model, res.chosen, normalize="index") * 100).reindex(
        index=MODELS, columns=[NONE] + list(groupings)).fillna(0)
    rows = []
    for m in MODELS:
        q = res[res.model == m]
        hind = max(groupings, key=lambda g: -(q[f"test dev | {g}"] - q["test dev | A"]).median())
        for lab, key in (("validation-selected", "selected"), (f"fixed: {fixed}", fixed),
                         (f"best in hindsight (test): {hind}", hind)):
            rows.append(pd.concat([pd.Series({"model": m, "calibrator": lab}), _stats(q, key)]))
    by_model = pd.DataFrame(rows)
    overall = pd.DataFrame({lab: _stats(res, key) for lab, key in (("validation-selected", "selected"),
                                                                    (f"fixed: {fixed}", fixed))}).T
    paired = res["test dev | selected"] - res[f"test dev | {fixed}"]
    overall["selected minus fixed: share of model-splits lower"] = [100 * np.mean(paired < 0), np.nan]
    print("share of splits in which each grouping is chosen on the validation set (%)")
    cs._display(freq.round(0))
    print("test results relative to global rescaling")
    cs._display(by_model.round(3))
    cs._display(overall.round(3))
    os.makedirs(out_dir, exist_ok=True)
    with pd.ExcelWriter(os.path.join(out_dir, "calibrator_selection_tables.xlsx")) as w:
        freq.round(1).to_excel(w, sheet_name="chosen grouping (%)")
        by_model.round(4).to_excel(w, sheet_name="test by model", index=False)
        overall.round(4).to_excel(w, sheet_name="test overall")
    return {"frequency": freq, "by model": by_model, "overall": overall}


def figure(res, out_dir="v2_calibrator_selection", fixed="DrivAge x BonusMalus", groupings=VARIANT_GROUPINGS):
    """(a) share of splits in which each grouping is chosen on the validation set; (b) change in test deviance and
    (c) change in Gini relative to global rescaling, validation-selected vs fixed grouping (median and IQR)."""
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#8a8985", "axes.linewidth": 0.7,
                         "xtick.color": "#5f5e5a", "ytick.color": "#5f5e5a", "axes.grid": True, "grid.color": "#ebe9e4",
                         "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.04, "legend.frameon": False})
    ink = "#1f1e1c"
    cols = [NONE] + list(groupings)
    freq = (pd.crosstab(res.model, res.chosen, normalize="index") * 100).reindex(index=MODELS, columns=cols).fillna(0)
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.62, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    seq = LinearSegmentedColormap.from_list("seq", ["#f7f6f2", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])
    im = ax.imshow(freq.to_numpy(float), cmap=seq, vmin=0, vmax=100, aspect="auto")
    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            v = freq.iat[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.5, color="white" if v > 55 else ink)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["None (rescaling only)"] + [COL_LABEL[g].replace("–\n", "–").replace("\n", " ") for g in groupings],
                       rotation=20, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABEL.get(m, m) for m in MODELS])
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(a) Grouping chosen on the validation set (share of the 50 splits)")
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015, aspect=15)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5)
    series = [("selected", "#1baf7a", "D", "Validation-selected grouping"),
              (fixed, "#2a78d6", "o", f"Fixed: {COL_LABEL[fixed].replace(chr(10), ' ')}")]
    Y = np.arange(len(MODELS))[::-1]
    for p, (title, fn) in enumerate((("(b) Change in test deviance", lambda q, k: q[f"test dev | {k}"] - q["test dev | A"]),
                                     ("(c) Change in Gini (×1000)",
                                      lambda q, k: 1000 * (q[f"test gini | {k}"] - q["test gini | A"])))):
        a = fig.add_subplot(gs[1, p])
        for s, (key, col, mk, _) in enumerate(series):
            for y, m in zip(Y, MODELS):
                v = fn(res[res.model == m], key).to_numpy()
                q1, q2, q3 = np.percentile(v, [25, 50, 75])
                yy = y + (0.15 if s == 0 else -0.15)
                a.plot([q1, q3], [yy, yy], color=col, lw=1.4, zorder=2)
                a.scatter([q2], [yy], color=col, marker=mk, s=22, edgecolor="white", lw=0.6, zorder=3)
        a.axvline(0, color=ink, lw=0.7)
        a.set_ylim(-0.6, len(MODELS) - 0.4)
        a.set_yticks(Y)
        a.set_yticklabels([MODEL_LABEL.get(m, m) for m in MODELS] if p == 0 else [])
        a.grid(axis="y", visible=False)
        a.set_title(title)
    handles = [mlines.Line2D([], [], color=c, marker=mk, ls="", ms=5, label=lab) for _, c, mk, lab in series]
    handles.append(mlines.Line2D([], [], color="#5f5e5a", lw=1.4, label="Interquartile range over the 50 splits"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, columnspacing=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"calibrator_selection.{ext}"), dpi=300)
    plt.show()
