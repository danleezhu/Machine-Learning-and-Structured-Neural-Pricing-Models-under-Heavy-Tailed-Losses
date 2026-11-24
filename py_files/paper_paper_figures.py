"""
paper_paper_figures.py — publication figures and tables.

Part G  Neural GA2M+ : stability of the selected interaction pairs, pricing of the extreme-claim policies, test balance
                       and Gini of all models
Part C  Calibration  : test balance B (signed and |B|), pooled bias, deviance and Gini of three calibration methods

Inputs (nothing is refitted):
  RUN_DIR (v2_full)            ebm_pairs.json, seeds.csv
  FIX_DIR                      ga2mplus_selection.csv, predictions/Neural_GA2M+/
  PRED_DIR (v2_predictions)    raw predictions of the 7 main models
  CAL_DIR (v2_calibration)     results.csv
  AGE_DIR (v2_age_calibration) paper_cells_on_A.csv
Outputs: OUT_DIR/figures/*.pdf|png, OUT_DIR/figures/captions.txt, OUT_DIR/tables/*.xlsx

    import paper_paper_figures as pf
    pol = pf.policy_pass(df_freq, df_sev, [PRED_DIR, f"{FIX_DIR}/predictions"], OUT_DIR)      # a few minutes
    tg = pf.ga2m_report(RUN_DIR, FIX_DIR, CAL_DIR, OUT_DIR)
    tc = pf.calibration_report(CAL_DIR, AGE_DIR, OUT_DIR)

Calibration methods (names used in the figures):
  Uncalibrated                         raw model predictions
  Global rescaling                     one factor c = training losses / training predictions
  Global rescaling + credibility cells (Driver age × Bonus–Malus)      (the proposed calibrator)
                                       global rescaling, then clipped credibility cell factors
                                       f = clip(1 + Z((O+1)/(cP+1) - 1), 0.95, 1.15), Z = E/(E+k), with the cell
                                       totals O, P, E of the validation set. No validation-based global factor.
Where each set is used: training set = model fit and the rescaling factor c; validation set = tuning, early
stopping and the cell totals of the credibility cells; test set = every number in the tables and figures.
"""

import itertools
import json
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

import paper_calibration_study as cs
import paper_predictions as pp

MODELS = ["GLM", "Tweedie GLM", "EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Neural GA2M+", "Tweedie-CANN"]
MODEL_LABEL = {"GLM": "GLM", "Tweedie GLM": "Tweedie GLM", "EBM": "EBM", "GBM": "GBM", "LocalGLMnet": "LocalGLMnet",
               "Neural GA2M": "Neural GA2M", "Neural GA2M+": "Neural GA2M+", "Tweedie-CANN": "CANN"}
VAR_LABEL = {"DrivAge": "Driver age", "BonusMalus": "Bonus–Malus", "VehAge": "Vehicle age", "VehPower": "Vehicle power",
             "VehBrand": "Vehicle brand", "VehGas": "Fuel", "Area": "Area", "Region": "Region", "Density": "Density"}
TICK_LABEL = {"Tweedie GLM": "Tweedie\nGLM", "Neural GA2M": "Neural\nGA2M", "Neural GA2M+": "Neural\nGA2M+",
              "LocalGLMnet": "Local-\nGLMnet", "Tweedie-CANN": "CANN"}
TARGET, EXPOSURE = "PurePremium", "Exposure"
C_GRAY, C_BLUE, C_ORANGE, C_AQUA, C_VIOLET, C_LIGHT = "#8a8985", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#c9c7bf"
INK, INK2 = "#1f1e1c", "#5f5e5a"
GROUP_ORDER = ["largest claim in test", "2nd/3rd largest in test", "none in test, some in validation", "all three in training"]
GROUP_TICK = {GROUP_ORDER[0]: "Largest claim\nin the test set", GROUP_ORDER[1]: "2nd or 3rd largest\nin the test set",
              GROUP_ORDER[2]: "None in the test set,\nsome in validation", GROUP_ORDER[3]: "All three in\nthe training set"}
CAPTIONS = {}


# =============================================================================================================
# Style helpers
# =============================================================================================================
def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8985",
                         "axes.linewidth": 0.7, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "axes.grid": True, "grid.color": "#ebe9e4", "grid.linewidth": 0.5,
                         "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight",
                         "savefig.pad_inches": 0.04, "legend.frameon": False})
    return plt


def _save(fig, out_dir, name, caption):
    d = os.path.join(out_dir, "figures")
    os.makedirs(d, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=300)
    path = os.path.join(d, "captions.txt")
    if os.path.exists(path):                       # keep the captions of figures made in an earlier session
        for block in open(path).read().strip().split("\n\n"):
            lines = block.split("\n", 1)
            if len(lines) == 2:
                CAPTIONS.setdefault(lines[0], lines[1])
    CAPTIONS[name] = caption
    with open(path, "w") as fh:
        for k, v in CAPTIONS.items():
            fh.write(f"{k}\n{v}\n\n")


def _pair_label(p):
    a, b = p.split(" x ")
    return f"{VAR_LABEL.get(a, a)} × {VAR_LABEL.get(b, b)}"


def _model_color(m):
    return {"Neural GA2M+": C_BLUE, "Neural GA2M": C_ORANGE}.get(m, C_LIGHT)


def _box(ax, values, pos, color, width=0.6, alpha=0.30, points=True, seed=0):
    ax.boxplot(values, positions=[pos], widths=width, patch_artist=True, showfliers=False,
               medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=color, lw=0.8),
               capprops=dict(color=color, lw=0.8), boxprops=dict(facecolor=color, edgecolor=color, alpha=alpha))
    if points:
        jit = np.random.default_rng(seed).uniform(-width * 0.3, width * 0.3, len(values))
        ax.scatter(np.full(len(values), pos) + jit, values, s=3.5, color=color, lw=0, zorder=3)


def _nb_p(x):
    """Nadeau-Bengio corrected paired t-test (test fraction 0.2, training fraction 0.6)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    J = len(x)
    if J < 3 or x.std(ddof=1) == 0:
        return np.nan
    se = x.std(ddof=1) * np.sqrt(1 / J + 0.2 / 0.6)
    return float(2 * stats.t.sf(abs(x.mean() / se), J - 1))


def _deviance_terms(y, mu, e, p):
    mu = np.maximum(mu, 1e-10)
    return e * 2 * (np.where(y > 0, y ** (2 - p), 0.0) / ((1 - p) * (2 - p)) - y * mu ** (1 - p) / (1 - p)
                    + mu ** (2 - p) / (2 - p))


# =============================================================================================================
# Policy-level pass over the 50 splits (stored predictions only)
# =============================================================================================================
def policy_pass(df_freq, df_sev, pred_dirs, out_dir="v2_paper", seeds=range(41, 91), models=MODELS, n_extreme=3):
    """
    Per seed and model: test deviance, the part contributed by the n_extreme largest claims of that test set, the
    deviance without them; per seed: where the n_extreme largest claims of the PORTFOLIO fall, training and test totals
    (noise floor); predictions of the portfolio's largest-claim policies when they are in the test set; observed and
    predicted loss of the segment DrivAge <= 21 and BonusMalus >= 100. Writes out_dir/tables/policy_pass_*.csv.
    """
    power = pp.power()
    df, num_cols, cat_cols = pp.prepare(df_freq, df_sev)
    top = df.nlargest(n_extreme, "ClaimAmount")[["IDpol", "ClaimAmount", "Exposure", "DrivAge", "BonusMalus", "VehAge",
                                                 "Region", "VehBrand", "Area"]].reset_index(drop=True)
    top["claim"] = [f"C{i + 1}" for i in range(len(top))]
    dev_rows, seed_rows, ext_rows, seg_rows = [], [], [], []
    t0 = time.time()
    for seed in [int(s) for s in seeds]:
        tr, va, te = pp.split(df, seed)
        ids_tr, ids_va = set(tr.IDpol), set(va.IDpol)
        yt, et = te[TARGET].to_numpy(float), te[EXPOSURE].to_numpy(float)
        amt = te["ClaimAmount"].to_numpy(float)
        ext = np.zeros(len(te), bool)
        ext[np.argsort(-amt, kind="mergesort")[:n_extreme]] = True
        seg = ((te["DrivAge"] <= 21) & (te["BonusMalus"] >= 100)).to_numpy()
        where = {cid: ("train" if cid in ids_tr else "valid" if cid in ids_va else "test") for cid in top.IDpol}
        sr = {"seed": seed, "O_train": float(np.sum(tr[EXPOSURE] * tr[TARGET])), "E_train": float(tr[EXPOSURE].sum()),
              "O_test": float(np.sum(et * yt)), "E_test": float(et.sum())}
        for c, cid in zip(top.claim, top.IDpol):
            sr[f"{c}_in"] = where[cid]
        seed_rows.append(sr)
        pos = {cid: int(np.flatnonzero(te.IDpol.to_numpy() == cid)[0]) for cid in top.IDpol if where[cid] == "test"}
        for m in models:
            d = pp.load_predictions(pred_dirs, m, seed)
            if d["idhash"]["test"] != pp.id_hash(te["IDpol"].to_numpy(np.int64)):
                raise RuntimeError(f"{m} seed {seed}: stored test policies differ from the regenerated split")
            mu = d["test"]
            dt = _deviance_terms(yt, mu, et, power)
            E = et.sum()
            dev_rows.append({"seed": seed, "model": m, "deviance": dt.sum() / E, "extreme_part": dt[ext].sum() / E,
                             "deviance_without_extremes": dt[~ext].sum() / et[~ext].sum()})
            seg_rows.append({"seed": seed, "model": m, "E": et[seg].sum(), "O": np.sum(et * yt * seg),
                             "O_no_extremes": np.sum(et * yt * (seg & ~ext)), "P": np.sum(et * mu * seg),
                             "n_extremes_in_segment": int(np.sum(seg & ext))})
            for c, cid in zip(top.claim, top.IDpol):
                if cid in pos:
                    ext_rows.append({"seed": seed, "model": m, "claim": c, "prediction": float(mu[pos[cid]])})
        print(f"seed {seed}: done | {(time.time() - t0) / 60:.1f} min", flush=True)
    out = {"deviance": pd.DataFrame(dev_rows), "seeds": pd.DataFrame(seed_rows),
           "extreme_predictions": pd.DataFrame(ext_rows), "segment": pd.DataFrame(seg_rows), "top_claims": top}
    d = os.path.join(out_dir, "tables")
    os.makedirs(d, exist_ok=True)
    for k, v in out.items():
        v.to_csv(os.path.join(d, f"policy_pass_{k}.csv"), index=False)
    return out


def load_policy_pass(out_dir="v2_paper"):
    d = os.path.join(out_dir, "tables")
    return {k: pd.read_csv(os.path.join(d, f"policy_pass_{k}.csv"))
            for k in ("deviance", "seeds", "extreme_predictions", "segment", "top_claims")}


def _location_group(s):
    ins = [s[f"C{i}_in"] for i in (1, 2, 3)]
    if ins[0] == "test":
        return GROUP_ORDER[0]
    if "test" in ins:
        return GROUP_ORDER[1]
    if "valid" in ins:
        return GROUP_ORDER[2]
    return GROUP_ORDER[3]


# =============================================================================================================
# Part G — Neural GA2M+
# =============================================================================================================
def _pair_key(a, b):
    a, b = sorted((str(a), str(b)))
    return f"{a} x {b}"


def _pair_sets(run_dir, fix_dir):
    main = {int(s): {_pair_key(*p) for p in v} for s, v in json.load(open(os.path.join(run_dir, "ebm_pairs.json"))).items()}
    sel = pd.read_csv(os.path.join(fix_dir, "ga2mplus_selection.csv")).drop_duplicates("seed", keep="last")
    plus = {int(r.seed): set(json.loads(r.chosen_pairs)) for r in sel.itertuples()}
    common = sorted(set(main) & set(plus))
    return {s: main[s] for s in common}, {s: plus[s] for s in common}


def _consensus_jaccard(sets):
    seeds = sorted(sets)
    out = {}
    for s in seeds:
        others = [o for o in seeds if o != s]
        cnt = pd.Series([p for o in others for p in sets[o]]).value_counts()
        cons = set(cnt[cnt >= 0.5 * len(others)].index)
        u = sets[s] | cons
        out[s] = len(sets[s] & cons) / len(u) if u else 1.0
    return pd.Series(out)


def structure_table(run_dir, fix_dir):
    main, plus = _pair_sets(run_dir, fix_dir)
    splits = pd.read_csv(os.path.join(run_dir, "seeds.csv")).set_index("seed")
    rows, loo = {}, {}
    for name, sets in (("Neural GA2M", main), ("Neural GA2M+", plus)):
        seeds = sorted(sets)
        jac = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in itertools.combinations(seeds, 2)]
        cnt = pd.Series([p for s in seeds for p in sets[s]]).value_counts() / len(seeds)
        l = _consensus_jaccard(sets)
        rho = stats.spearmanr(l.values, splits.reindex(seeds).train_max_loss_share.values)
        loo[name] = l
        rows[name] = {"pairs per model": np.mean([len(sets[s]) for s in seeds]),
                      "mean pairwise Jaccard": np.mean(jac), "minimum pairwise Jaccard": np.min(jac),
                      "distinct pairs over 50 splits": len(cnt), "pairs in every split": int((cnt == 1).sum()),
                      "pairs in >= 90% of splits": int((cnt >= 0.9).sum()),
                      "agreement with consensus (Jaccard)": l.mean(),
                      "Spearman with largest training-claim share": rho.statistic, "p-value": rho.pvalue}
    t = pd.DataFrame(rows)
    d = (loo["Neural GA2M+"] - loo["Neural GA2M"]).dropna()
    t.loc["paired Wilcoxon p (agreement)", "Neural GA2M+"] = stats.wilcoxon(d).pvalue
    freq = pd.DataFrame({"Neural GA2M": pd.Series([p for s in main for p in main[s]]).value_counts() / len(main),
                         "Neural GA2M+": pd.Series([p for s in plus for p in plus[s]]).value_counts() / len(plus)}).fillna(0)
    return t, freq, loo, splits


def deviance_tables(pol):
    dv, sd = pol["deviance"], pol["seeds"].set_index("seed")
    grp = sd.apply(_location_group, axis=1)
    w = dv.pivot(index="seed", columns="model", values="deviance")
    x = dv.pivot(index="seed", columns="model", values="extreme_part")
    wo = dv.pivot(index="seed", columns="model", values="deviance_without_extremes")
    d_tot = w["Neural GA2M+"] - w["Neural GA2M"]
    d_ext = x["Neural GA2M+"] - x["Neural GA2M"]
    dec = pd.DataFrame({"group": grp.reindex(d_tot.index), "total": d_tot, "extreme policies": d_ext,
                        "all other policies": d_tot - d_ext})
    rows = []
    for g in GROUP_ORDER + ["all splits"]:
        q = dec if g == "all splits" else dec[dec.group == g]
        rows.append({"location of the three largest portfolio claims": g, "splits": len(q),
                     "deviance difference (mean)": q.total.mean(),
                     "from the extreme-claim policies": q["extreme policies"].mean(),
                     "from all other policies": q["all other policies"].mean(),
                     "splits with lower deviance (%)": 100 * (q.total < 0).mean()})
    t_dec = pd.DataFrame(rows).set_index("location of the three largest portfolio claims")
    rob = pd.DataFrame({"test deviance, mean": w.mean(), "test deviance, median": w.median(),
                        "without the 3 largest test claims, mean": wo.mean(),
                        "without the 3 largest test claims, median": wo.median()}).reindex(MODELS)
    rob["difference vs GLM, without, mean"] = wo.sub(wo["GLM"], axis=0).mean().reindex(MODELS)
    rob["splits better than GLM, without (%)"] = (100 * wo.lt(wo["GLM"], axis=0)).mean().reindex(MODELS)
    return t_dec, dec, rob


def segment_table(pol):
    sg = pol["segment"].groupby("model")[["E", "O", "O_no_extremes", "P", "n_extremes_in_segment"]].sum().reindex(MODELS)
    t = pd.DataFrame({"predicted / observed loss": sg.P / sg.O,
                      "predicted / observed loss, without the 3 largest test claims": sg.P / sg.O_no_extremes})
    return t, sg


def balance_gini_tables(cal_dir):
    """Test balance and Gini of every model, uncalibrated and after global rescaling (from the calibration study)."""
    r, _ = cs.load(cal_dir)
    g = r[r.grouping == cs.PROPOSED_GROUPING]          # raw and A do not depend on the grouping
    rows = []
    for mode, lab in (("raw", "Uncalibrated"), ("A", "Global rescaling")):
        for m in MODELS:
            q = g[(g["mode"] == mode) & (g.model == m)]
            rows.append({"model": m, "version": lab, "mean |B| (%)": 100 * q.test_B.abs().mean(),
                         "median |B| (%)": 100 * q.test_B.abs().median(),
                         "pooled bias (%)": 100 * (q.pred_total.sum() / q.obs_total.sum() - 1), "Gini": q.gini.mean()})
    t = pd.DataFrame(rows)
    b = g[g["mode"].isin(["raw", "A"])].pivot_table(index=["mode", "seed"], columns="model", values="test_B")
    gi = g[g["mode"] == "raw"].pivot_table(index="seed", columns="model", values="gini")
    return t, b, gi


def fig_g1(t, freq, loo, splits, out_dir):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.32})
    f = freq[(freq.max(axis=1) >= 0.1)].sort_values(["Neural GA2M+", "Neural GA2M"], ascending=[True, True])
    y = np.arange(len(f))
    ax = axes[0]
    for i, (a, b) in enumerate(zip(f["Neural GA2M"], f["Neural GA2M+"])):
        ax.plot([a, b], [i, i], color="#dcdad3", lw=1.6, zorder=1)
    ax.scatter(f["Neural GA2M"], y, color=C_ORANGE, s=24, zorder=3, label="Neural GA2M", edgecolor="white", lw=0.6)
    ax.scatter(f["Neural GA2M+"], y, color=C_BLUE, s=24, zorder=3, label="Neural GA2M+", marker="D",
               edgecolor="white", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([_pair_label(p) for p in f.index])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.7, len(f) - 0.3)
    ax.set_xlabel("Share of the 50 splits in which the pair is selected")
    ax.set_title("(a) Selected interaction pairs", loc="center")
    ax = axes[1]
    s = splits.reindex(loo["Neural GA2M+"].index)
    xs = 100 * s.train_max_loss_share
    for name, c, mk in (("Neural GA2M", C_ORANGE, "o"), ("Neural GA2M+", C_BLUE, "D")):
        r = t.loc["Spearman with largest training-claim share", name]
        ax.scatter(xs, loo[name], color=c, marker=mk, s=20, edgecolor="white", lw=0.5, zorder=3,
                   label=f"{name} (Spearman ρ = {r:.2f})")
    ax.set_xlabel("Largest single claim (% of training-set losses)")
    ax.set_ylabel("Agreement with the consensus pair set (Jaccard)")
    ax.set_title("(b) Dependence on the largest training claim", loc="center")
    ax.set_ylim(0, 1.05)
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.subplots_adjust(top=0.84)
    fig.legend(h2, l2, loc="lower center", bbox_to_anchor=(0.5, 0.91), ncol=2, handletextpad=0.3, columnspacing=2)
    _save(fig, out_dir, "G1_interaction_stability",
          "Interaction pairs selected for the Neural GA2M (EBM pair selection) and the Neural GA2M+ over 50 "
          "train/validation/test splits. (a) Share of splits in which each pair is selected (pairs selected in at "
          "least 10% of the splits by either model). (b) Jaccard agreement of each split's pair set with the consensus "
          "set of the other 49 splits, against the share of the training-set losses due to its largest single claim.")
    plt.show()


def fig_g2(dec, out_dir):
    plt = _plt()
    d = dec.copy()
    d["g"] = d.group.map({g: i for i, g in enumerate(GROUP_ORDER)})
    d = d.sort_values(["g", "total"])
    gap = 1.2
    x = np.arange(len(d)) + gap * d.g.to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    e, o = d["extreme policies"].to_numpy(), d["all other policies"].to_numpy()
    pos_base, neg_base = np.zeros(len(d)), np.zeros(len(d))
    for vals, c, lab in ((e, C_BLUE, "Three largest-claim policies"), (o, C_LIGHT, "All other policies")):
        base = np.where(vals >= 0, pos_base, neg_base)
        ax.bar(x, vals, bottom=base, color=c, width=0.82, label=lab, edgecolor="white", linewidth=0.3)
        pos_base += np.where(vals >= 0, vals, 0)
        neg_base += np.where(vals < 0, vals, 0)
    ax.scatter(x, d.total, color=INK, s=7, zorder=4, label="Total")
    ax.axhline(0, color=INK, lw=0.7)
    centers, labels = [], []
    for i, g in enumerate(GROUP_ORDER):
        xi = x[d.g.to_numpy() == i]
        if len(xi):
            centers.append(xi.mean())
            labels.append(f"{GROUP_TICK[g]}\n(n = {len(xi)})")
            if i > 0:
                ax.axvline(xi.min() - (gap + 1) / 2, color="#dcdad3", lw=0.8)
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(x.min() - 1, x.max() + 1)
    ax.set_ylabel("Test deviance: Neural GA2M+ minus Neural GA2M")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    _save(fig, out_dir, "G2_deviance_decomposition",
          "Difference in test deviance between the Neural GA2M+ and the Neural GA2M for each of the 50 splits "
          "(negative: lower deviance for the Neural GA2M+), split into the contribution of the three policies with the "
          "largest claims in the test set and of all other policies. Splits are grouped by where the three largest "
          "claims of the portfolio fall.")
    plt.show()


def fig_g3(pol, out_dir, models=("GLM", "EBM", "GBM", "Neural GA2M", "Neural GA2M+")):
    plt = _plt()
    ep = pol["extreme_predictions"]
    top = pol["top_claims"].set_index("claim")
    claims = [c for c in ("C1", "C2", "C3") if c in set(ep.claim)]
    names = {"C1": "Largest claim", "C2": "Second-largest claim", "C3": "Third-largest claim"}
    fig, axes = plt.subplots(1, len(claims), figsize=(7.2, 3.1), sharey=True, gridspec_kw={"wspace": 0.12})
    axes = np.atleast_1d(axes)
    col = {"GLM": C_GRAY, "EBM": C_AQUA, "GBM": C_VIOLET, "Neural GA2M": C_ORANGE, "Neural GA2M+": C_BLUE}
    for ax, c in zip(axes, claims):
        q = ep[ep.claim == c].pivot(index="seed", columns="model", values="prediction")
        for j, m in enumerate(models):
            v = q[m].to_numpy()
            ax.scatter(np.full(len(v), j) + np.linspace(-0.2, 0.2, len(v)), v, color=col.get(m, C_LIGHT), s=12,
                       zorder=3, edgecolor="white", lw=0.4)
            ax.plot([j - 0.32, j + 0.32], [np.median(v)] * 2, color=INK, lw=1.2, zorder=4)
        r = top.loc[c]
        ax.set_title(f"{names[c]} ({r.ClaimAmount / 1e6:.2f} M), n = {len(q)}\n"
                     f"driver age {int(r.DrivAge)}, Bonus–Malus {int(r.BonusMalus)}", loc="center", fontsize=7.8)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([TICK_LABEL.get(m, m) for m in models], fontsize=7)
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Predicted annual pure premium (log scale)")
    _save(fig, out_dir, "G3_extreme_policy_predictions",
          "Predicted annual pure premium of the three policies with the largest claims in the portfolio, in the "
          "splits in which the policy belongs to the test set (n splits); one point per split, bar = median.")
    plt.show()


def fig_g4(b, out_dir):
    """Signed test balance per model, uncalibrated and after global rescaling; mean |B| above each box."""
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True, gridspec_kw={"hspace": 0.32})
    for ax, (mode, title) in zip(axes, (("raw", "(a) Uncalibrated (numbers: mean |B|, %)"),
                                        ("A", "(b) After global rescaling (numbers: mean |B|, %)"))):
        q = 100 * b.loc[mode]
        lo, hi = np.nanmin(q.values), np.nanmax(q.values)
        top = hi + 0.22 * (hi - lo)
        for j, m in enumerate(MODELS):
            v = q[m].dropna().to_numpy()
            _box(ax, v, j, _model_color(m) if m in ("Neural GA2M", "Neural GA2M+") else C_GRAY, seed=j)
            ax.text(j, top, f"{np.mean(np.abs(v)):.1f}", ha="center", va="center", fontsize=7, color=INK)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_ylim(lo - 0.05 * (hi - lo), top + 0.08 * (hi - lo))
        ax.set_yticks([t for t in ax.get_yticks() if lo - 0.05 * (hi - lo) <= t <= hi + 0.02 * (hi - lo)])
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_ylabel("Test balance B (%)")
        ax.set_title(title, loc="center", pad=4)
        ax.grid(axis="x", visible=False)
    axes[1].set_xticks(range(len(MODELS)))
    axes[1].set_xticklabels([TICK_LABEL.get(m, m) for m in MODELS])
    _save(fig, out_dir, "G4_test_balance_by_model",
          "Test balance B = predicted / observed loss − 1 over the 50 test sets, by model, (a) without calibration and "
          "(b) after rescaling to the training-set loss level. Numbers: mean |B| (%).")
    plt.show()


def fig_g5(gi, out_dir, refs=("Neural GA2M", "EBM", "GBM")):
    """Gini per model, and paired Gini differences of the Neural GA2M+ against other models."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [2.0, 1.25], "wspace": 0.32})
    ax = axes[0]
    for j, m in enumerate(MODELS):
        _box(ax, gi[m].dropna().to_numpy(), j, _model_color(m) if m in ("Neural GA2M", "Neural GA2M+") else C_GRAY,
             seed=j)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([TICK_LABEL.get(m, m) for m in MODELS], fontsize=7)
    ax.set_ylabel("Gini index (test set)")
    ax.set_title("(a) Gini index by model", loc="center")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    labels = []
    for j, ref in enumerate(refs):
        d = 1000 * (gi["Neural GA2M+"] - gi[ref]).dropna().to_numpy()
        _box(ax, d, j, C_BLUE, width=0.55, seed=10 + j)
        labels.append(f"{TICK_LABEL.get(ref, ref)}\n{100 * np.mean(d > 0):.0f}% higher")
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks(range(len(refs)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Gini difference (×1000)")
    ax.set_title("(b) Neural GA2M+ minus", loc="center")
    ax.grid(axis="x", visible=False)
    _save(fig, out_dir, "G5_gini_by_model",
          "(a) Gini index on the 50 test sets by model. (b) Paired difference of the Gini index between the Neural "
          "GA2M+ and other models (×1000); in brackets the share of splits in which the Neural GA2M+ ranks better.")
    plt.show()


def ga2m_report(run_dir="v2_full", fix_dir="v2_ga2mplus_c995_meanrank", cal_dir="v2_calibration", out_dir="v2_paper"):
    pol = load_policy_pass(out_dir)
    t_struct, freq, loo, splits = structure_table(run_dir, fix_dir)
    t_dec, dec, rob = deviance_tables(pol)
    t_seg, sg = segment_table(pol)
    t_bg, b, gi = balance_gini_tables(cal_dir)
    ep = pol["extreme_predictions"]
    t_ext = ep.groupby(["claim", "model"]).prediction.median().unstack().reindex(columns=MODELS)
    t_ext["splits with the claim in the test set"] = ep.groupby("claim").seed.nunique()
    q = ep.pivot_table(index=["claim", "seed"], columns="model", values="prediction")
    t_ext["Neural GA2M+ above Neural GA2M (splits)"] = (q["Neural GA2M+"] > q["Neural GA2M"]).groupby(level="claim").sum()
    gd = pd.DataFrame({f"vs {ref}": {"mean Gini difference (x1000)": 1000 * (gi["Neural GA2M+"] - gi[ref]).mean(),
                                     "splits with higher Gini (%)": 100 * (gi["Neural GA2M+"] > gi[ref]).mean(),
                                     "Nadeau-Bengio p": _nb_p(gi["Neural GA2M+"] - gi[ref])}
                       for ref in MODELS if ref != "Neural GA2M+"})
    tables = {"G1 structure stability": t_struct,
              "G1 pair frequency": freq.sort_values("Neural GA2M+", ascending=False).rename(index=_pair_label),
              "G2 deviance decomposition": t_dec, "G3 extreme-claim policies": t_ext, "G4 balance and Gini": t_bg,
              "G5 Gini differences": gd, "G6 high-risk segment": t_seg, "G7 deviance without extremes": rob}
    d = os.path.join(out_dir, "tables")
    with pd.ExcelWriter(os.path.join(d, "paper_tables_ga2m.xlsx")) as w:
        for k, v in tables.items():
            v.round(4).to_excel(w, sheet_name=k[:31], index=k != "G4 balance and Gini")
    for k, v in tables.items():
        print("\n" + k)
        cs._display(v.round(3))
    print(f"\nsegment DrivAge <= 21 and BonusMalus >= 100: {100 * sg.E.iloc[0] / pol['seeds'].E_test.sum():.2f}% of test "
          f"exposure, contains {int(sg.n_extremes_in_segment.iloc[0])} of the {3 * len(pol['seeds'])} largest test claims")
    fig_g1(t_struct, freq, loo, splits, out_dir)
    fig_g2(dec, out_dir)
    fig_g3(pol, out_dir)
    fig_g4(b, out_dir)
    fig_g5(gi, out_dir)
    return tables


# =============================================================================================================
# Part C — Calibration
# =============================================================================================================
METHOD_NAMES = {
    "raw": "Uncalibrated",
    "A": "Global rescaling",
    "prop": "Global rescaling + credibility cells ({cells})",
}
CELL_LABEL = {"DrivAge x BonusMalus": "Driver age × Bonus–Malus", "DrivAge band": "Driver age",
              "BonusMalus x VehAge": "Bonus–Malus × Vehicle age", "DrivAge x VehAge": "Driver age × Vehicle age",
              "BonusMalus band": "Bonus–Malus", "DrivAge band x Region": "Driver age × Region"}


def _colors(methods):
    """Fixed colour per method, so a method keeps its colour whatever else is compared."""
    out = []
    for lab, src, mode, g, k in methods:
        if mode == "raw":
            out.append(C_GRAY)
        elif mode == "A":
            out.append(C_BLUE)
        else:
            out.append(C_AQUA)
    return out


def methods_for(proposed_grouping="DrivAge x BonusMalus", k=2000):
    """(label, source, mode, grouping, k) of the three calibration methods compared: uncalibrated, global rescaling on
    the training set, and global rescaling + clipped credibility cells on proposed_grouping (the proposed calibrator).
    Uncalibrated and global rescaling do not depend on the grouping; their rows are read for proposed_grouping."""
    cells = CELL_LABEL.get(proposed_grouping, proposed_grouping)
    return [(METHOD_NAMES["raw"], "study", "raw", proposed_grouping, "-"),
            (METHOD_NAMES["A"], "study", "A", proposed_grouping, "-"),
            (METHOD_NAMES["prop"].format(cells=cells), "age", "paper cells on A", proposed_grouping, str(k))]


def _wrap(label):
    return label.replace(" (", "\n(").replace(" + ", " +\n", 1) if "(" in label else label


def calibration_data(cal_dir, age_dir, methods):
    r, _ = cs.load(cal_dir)
    var = pd.read_csv(os.path.join(age_dir, "paper_cells_on_A.csv"))
    obs = r[(r["mode"] == "A")].set_index(["seed", "model", "grouping"]).obs_total
    out = []
    for lab, src, mode, g, k in methods:
        if src == "study":
            d = r[(r["mode"] == mode) & (r.grouping == g) & (r.k == k)][["seed", "model", "test_B", "deviance", "gini"]].copy()
        else:
            d = var[(var.grouping == g) & (var.k == int(k))][["seed", "model", "test_B", "deviance", "gini"]].copy()
        if d.empty:
            raise ValueError(f"no results for {lab} ({mode}, {g}, k = {k})")
        d["obs_total"] = np.asarray(pd.MultiIndex.from_arrays([d.seed, d.model, [g] * len(d)]).map(obs), float)
        d["method"] = lab
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    raw = d[d.method == methods[0][0]].set_index(["seed", "model"])
    d = d.join(raw[["deviance", "gini"]].rename(columns=lambda c: c + "_raw"), on=["seed", "model"])
    d["pred_total"] = d.obs_total * (1 + d.test_B)
    return d


def noise_floor(pol):
    s = pol["seeds"]
    b = (s.O_train / s.E_train * s.E_test) / s.O_test - 1
    return float(100 * b.abs().mean())


def calibration_tables(d, methods):
    labs = [v[0] for v in methods]
    rows = []
    for (m, v), q in d.groupby(["model", "method"]):
        dd, dg = q.deviance - q.deviance_raw, q.gini - q.gini_raw
        first = v == labs[0]
        rows.append({"model": m, "method": v, "mean |B| (%)": 100 * q.test_B.abs().mean(),
                     "median |B| (%)": 100 * q.test_B.abs().median(),
                     "pooled bias (%)": 100 * (q.pred_total.sum() / q.obs_total.sum() - 1),
                     "test deviance": q.deviance.mean(), "deviance change vs uncalibrated": dd.mean(),
                     "splits with lower deviance (%)": np.nan if first else 100 * (dd < 0).mean(),
                     "Wilcoxon p (deviance)": np.nan if first else float(stats.wilcoxon(dd).pvalue),
                     "Nadeau-Bengio p (deviance)": np.nan if first else _nb_p(dd),
                     "Gini": q.gini.mean(), "Gini change vs uncalibrated (x1000)": 1000 * dg.mean()})
    t = pd.DataFrame(rows)
    t["method"] = pd.Categorical(t.method, labs, ordered=True)
    t["model"] = pd.Categorical(t.model, MODELS, ordered=True)
    t = t.sort_values(["model", "method"]).reset_index(drop=True)
    avg = t.drop(columns="model").groupby("method", observed=True).mean(numeric_only=True)
    return t, avg


def _method_legend(fig, methods, y=1.0, ncol=2):
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.6, label=lab) for (lab, *_), c in zip(methods, _colors(methods))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=1.2,
               columnspacing=2.2)


def fig_c1(d, out_dir, methods, tag=""):
    """Signed test balance per split, by model and method; mean |B| above each box."""
    plt = _plt()
    labs = [v[0] for v in methods]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.2), sharey=True, gridspec_kw={"hspace": 0.30, "wspace": 0.10})
    allb = 100 * d.test_B
    lo, hi = allb.min(), allb.max()
    top = hi + 0.16 * (hi - lo)
    for ax, m in zip(axes.flat, MODELS):
        for j, (lab, c) in enumerate(zip(labs, _colors(methods))):
            b = 100 * d[(d.model == m) & (d.method == lab)].test_B.to_numpy()
            _box(ax, b, j, c, width=0.62, seed=j)
            ax.text(j, top, f"{np.mean(np.abs(b)):.1f}", ha="center", va="center", fontsize=6.8, color=INK)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_ylim(lo - 0.04 * (hi - lo), top + 0.07 * (hi - lo))
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(labs) - 0.4)
        ax.set_title(MODEL_LABEL[m], loc="center", pad=3)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test balance B (%)")
    _method_legend(fig, methods, y=0.93)
    ks = sorted({m[4] for m in methods[2:]}, key=float)
    _save(fig, out_dir, f"C1_test_balance_by_model{tag}",
          "Test balance B = predicted / observed loss − 1 over the 50 test sets, by model and calibration method. "
          f"Numbers above the boxes: mean |B| (%). Credibility constant k = {', '.join(ks)}.")
    plt.show()


def fig_c2(avg, floor, out_dir, methods, tag=""):
    """Averages over the eight models: mean |B|, pooled bias, deviance and Gini relative to the uncalibrated models."""
    import matplotlib.lines as mlines
    plt = _plt()
    labs = [v[0] for v in methods]
    panels = [("mean |B| (%)", "(a) Mean |B| (%)"), ("pooled bias (%)", "(b) Pooled bias (%)"),
              ("deviance change vs uncalibrated", "(c) Change in test deviance"),
              ("Gini change vs uncalibrated (x1000)", "(d) Change in Gini (×1000)")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4 + 0.7 * max(0, len(labs) - 4)), sharey=True, gridspec_kw={"hspace": 0.45, "wspace": 0.12})
    y = np.arange(len(labs))[::-1]
    for ax, (col, title) in zip(axes.flat, panels):
        v = avg.reindex(labs)[col].to_numpy()
        ax.barh(y, v, color=_colors(methods), height=0.6)
        ax.axvline(0, color=INK, lw=0.7)
        if col == "mean |B| (%)":
            ax.axvline(floor, color=INK, lw=0.9, ls="--")
        lo, hi = min(0.0, np.nanmin(v)), max(0.0, np.nanmax(v), floor if col == "mean |B| (%)" else 0)
        span = hi - lo if hi > lo else 1.0
        for yi, vi in zip(y, v):
            if col == "mean |B| (%)":      # inside the bar, clear of the noise-floor line
                ax.text(vi - 0.02 * span, yi, f"{vi:.2f}", va="center", ha="right", fontsize=7, color="white")
            else:
                ax.text(vi + (0.02 if vi >= 0 else -0.02) * span, yi, f"{vi:.2f}", va="center",
                        ha="left" if vi >= 0 else "right", fontsize=7, color=INK)
        ax.set_xlim(lo - (0.22 * span if lo < 0 else 0), hi + 0.22 * span)
        ax.set_title(title, loc="center")
        ax.grid(axis="y", visible=False)
    for ax in axes[:, 0]:
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap(l) for l in labs], fontsize=7.2)
    nf = mlines.Line2D([], [], color=INK, lw=0.9, ls="--", label=f"Noise floor of mean |B| ({floor:.1f}%)")
    fig.legend(handles=[nf], loc="lower center", bbox_to_anchor=(0.5, 0.96))
    _save(fig, out_dir, f"C2_calibration_summary{tag}",
          "Calibration methods, averaged over the eight models and 50 test sets: (a) mean absolute test balance |B|; "
          "(b) pooled bias, total predicted over total observed loss − 1; (c) change in test deviance and (d) change "
          "in the Gini index relative to the uncalibrated predictions. Dashed line: mean |B| of an intercept-only "
          "model at the training-set loss level (noise floor of the test balance).")
    plt.show()


def intercept_only_B(pol):
    """Test balance B of an intercept-only premium at the training-set loss level, per split (the noise floor)."""
    sd = pol["seeds"].set_index("seed")
    return sd.O_train / sd.E_train * sd.E_test / sd.O_test - 1


def location_tables(d, pol, methods):
    """Mean signed test balance B (%) and mean |B| (%) by claim-location group, for an intercept-only premium and every
    calibration method (mean over the eight models and the splits of the group)."""
    grp = pol["seeds"].set_index("seed").apply(_location_group, axis=1)
    b0 = intercept_only_B(pol)
    signed, absb = {}, {}
    rows = [("Intercept-only premium (training level)", b0.rename("test_B").reset_index().assign(model="-"))]
    rows += [(lab, d[d.method == lab][["seed", "model", "test_B"]]) for lab, *_ in methods]
    for lab, q in rows:
        g = grp.reindex(q.seed).to_numpy()
        signed[lab] = {gg: 100 * q.test_B[g == gg].mean() for gg in GROUP_ORDER} | {"all splits": 100 * q.test_B.mean()}
        absb[lab] = {gg: 100 * q.test_B[g == gg].abs().mean() for gg in GROUP_ORDER} | {"all splits": 100 * q.test_B.abs().mean()}
    n = grp.value_counts().reindex(GROUP_ORDER)
    cols = {gg: f"{gg} (n={int(n[gg])})" for gg in GROUP_ORDER}
    return pd.DataFrame(signed).T.rename(columns=cols), pd.DataFrame(absb).T.rename(columns=cols)


def fig_c3(d, pol, out_dir, methods, tag=""):
    """Signed test balance per split by where the three largest claims fall: intercept-only premium and the three
    calibration methods (one point per split, B averaged over the eight models)."""
    import matplotlib.patches as mpatches
    plt = _plt()
    grp = pol["seeds"].set_index("seed").apply(_location_group, axis=1)
    series = [("Intercept-only premium", C_LIGHT, 100 * intercept_only_B(pol))]
    for (lab, *_), c in zip(methods, _colors(methods)):
        series.append((lab, c, 100 * d[d.method == lab].groupby("seed").test_B.mean()))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.8 / len(series)
    for gi, gg in enumerate(GROUP_ORDER):
        seeds = grp.index[grp == gg]
        for j, (lab, c, b) in enumerate(series):
            x = gi + (j - (len(series) - 1) / 2) * w
            _box(ax, b.reindex(seeds).to_numpy(), x, c, width=w * 0.82, seed=gi * 10 + j)
        if gi:
            ax.axvline(gi - 0.5, color="#ebe9e4", lw=0.8)
    ax.axhline(0, color=INK, lw=0.7)
    n = grp.value_counts().reindex(GROUP_ORDER).astype(int)
    ax.set_xticks(range(len(GROUP_ORDER)))
    ax.set_xticklabels([f"{GROUP_TICK[g]}\n(n = {k})" for g, k in zip(GROUP_ORDER, n)])
    ax.set_xlim(-0.5, len(GROUP_ORDER) - 0.5)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Test balance B (%)")
    handles = [mpatches.Patch(facecolor=c, edgecolor=c, alpha=0.6, label=_wrap(lab).replace("\n", " "))
               for lab, c, _ in series]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, handlelength=1.2,
              columnspacing=2.0)
    _save(fig, out_dir, f"C3_test_balance_by_claim_location{tag}",
          "Test balance B = predicted / observed loss - 1 for each split, by where the three largest claims of the "
          "portfolio fall (n = number of splits): an intercept-only premium at the training-set loss level and the "
          "three calibration methods (B averaged over the eight models).")
    plt.show()


def _iqr_point(ax, v, y, color, marker, ms=24):
    q1, q2, q3 = np.nanpercentile(v, [25, 50, 75])
    ax.plot([q1, q3], [y, y], color=color, lw=1.4, solid_capstyle="butt", zorder=2)
    ax.scatter([q2], [y], color=color, marker=marker, s=ms, edgecolor="white", lw=0.6, zorder=3)


def fig_c4(t, floor, out_dir, methods, tag="", d=None):
    """Per model, over the 50 test sets: (a) pooled bias; (b) change in test deviance and (d) change in Gini relative
    to the uncalibrated predictions, median and interquartile range over the splits; (c) share of splits with lower
    test deviance than the uncalibrated predictions."""
    import matplotlib.lines as mlines
    plt = _plt()
    labs, cols = [m[0] for m in methods], _colors(methods)
    marks = ["o", "s", "D"]
    dd = d.assign(dd=d.deviance - d.deviance_raw, dg=1000 * (d.gini - d.gini_raw))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.5), sharey=True, gridspec_kw={"wspace": 0.10})
    Y = np.arange(len(MODELS))[::-1]
    off = {0: 0.24, 1: 0.0, 2: -0.24}
    for i, lab in enumerate(labs):                                              # (a) pooled bias
        q = t[t.method == lab].set_index("model").reindex(MODELS)
        axes[0].scatter(q["pooled bias (%)"], Y + off[i], color=cols[i], marker=marks[i], s=24, edgecolor="white",
                        lw=0.6, zorder=3)
    for i, lab in enumerate(labs[1:], start=1):
        for y, m in zip(Y, MODELS):
            q = dd[(dd.method == lab) & (dd.model == m)]
            _iqr_point(axes[1], q.dd.to_numpy(), y + off[i] + 0.12, cols[i], marks[i])     # (b) deviance
            axes[2].barh(y + off[i] + 0.12, 100 * np.mean(q.dd < 0), height=0.22, color=cols[i], alpha=0.85)  # (c)
            if i == len(labs) - 1:
                _iqr_point(axes[3], q.dg.to_numpy(), y, cols[i], marks[i])                  # (d) Gini
    for ax, ref, title in zip(axes, (0, 0, 50, 0), ("(a) Pooled bias (%)", "(b) Δ test deviance",
                                                     "(c) Splits improved (%)", "(d) Δ Gini (×1000)")):
        ax.axvline(ref, color=INK, lw=0.7, ls="--" if ref == 50 else "-")
        ax.set_title(title, loc="center")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(MODELS) - 0.4)
        for yb in Y[:-1]:
            ax.axhline(yb - 0.5, color="#ebe9e4", lw=0.6)
    axes[2].set_xlim(0, 100)
    axes[0].set_yticks(Y)
    axes[0].set_yticklabels([MODEL_LABEL[m] for m in MODELS])
    handles = [mlines.Line2D([], [], color=c, marker=mk, ls="", ms=5, label=_wrap(l).replace("\n", " "))
               for l, c, mk in zip(labs, cols, marks)]
    handles.append(mlines.Line2D([], [], color=INK2, lw=1.4, label="Interquartile range over the 50 splits"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.93), ncol=2, handletextpad=0.4,
               columnspacing=1.6)
    _save(fig, out_dir, f"C4_calibration_by_model{tag}",
          "Calibration methods by model over the 50 test sets: (a) pooled bias, total predicted over total observed "
          "loss - 1; (b) change in test deviance and (d) change in the Gini index relative to the uncalibrated "
          "predictions, median (marker) and interquartile range (line) over the splits; (c) share of splits with "
          "lower test deviance than the uncalibrated predictions. Global rescaling leaves the Gini index unchanged.")
    plt.show()


def fig_c5(d, out_dir, methods, tag=""):
    """Per model: per-split change in test deviance relative to the uncalibrated predictions; share of splits with
    lower deviance above each box."""
    plt = _plt()
    labs, cols = [m[0] for m in methods][1:], _colors(methods)[1:]
    dd = d.assign(dd=d.deviance - d.deviance_raw)
    lo, hi = np.nanpercentile(dd[dd.method.isin(labs)].dd, [0.5, 99.5])
    top = hi + 0.18 * (hi - lo)
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.0), sharey=True, gridspec_kw={"hspace": 0.30, "wspace": 0.10})
    for ax, m in zip(axes.flat, MODELS):
        for j, (lab, c) in enumerate(zip(labs, cols)):
            v = dd[(dd.model == m) & (dd.method == lab)].dd.to_numpy()
            _box(ax, np.clip(v, lo, hi), j, c, width=0.62, seed=j)
            ax.text(j, top, f"{100 * np.mean(v < 0):.0f}", ha="center", va="center", fontsize=6.8, color=INK)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_ylim(lo - 0.05 * (hi - lo), top + 0.08 * (hi - lo))
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(labs) - 0.4)
        ax.set_title(MODEL_LABEL[m], loc="center", pad=3)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Change in test deviance")
    _method_legend(fig, methods[1:], y=0.93)
    _save(fig, out_dir, f"C5_deviance_change_by_model{tag}",
          "Change in test deviance relative to the uncalibrated predictions for each of the 50 splits, by model and "
          "calibration method (negative: lower deviance). Numbers above the boxes: share of splits (%) with lower "
          "deviance. Values beyond the 0.5% and 99.5% quantiles are drawn at these limits.")
    plt.show()


def calibration_report(cal_dir="v2_calibration", age_dir="v2_age_calibration", out_dir="v2_paper",
                       proposed_grouping="DrivAge x BonusMalus", k=2000, tag=""):
    """Tables and figures C1-C5 for uncalibrated, global rescaling and the proposed calibrator.
    proposed_grouping: "DrivAge x BonusMalus" (default) or "DrivAge band"; k in 100, 500, 1000, 2000, 5000, 10000;
    tag: suffix of the figure / table file names."""
    methods = methods_for(proposed_grouping, k)
    pol = load_policy_pass(out_dir)
    d = calibration_data(cal_dir, age_dir, methods)
    floor = noise_floor(pol)
    t, avg = calibration_tables(d, methods)
    loc_signed, loc_abs = location_tables(d, pol, methods)
    print(f"noise floor (intercept-only model at the training level): mean |B| = {floor:.2f}%")
    cs._display(avg.round(3))
    print("mean signed test balance B (%) by claim location")
    cs._display(loc_signed.round(2))
    print("mean |B| (%) by claim location")
    cs._display(loc_abs.round(2))
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    with pd.ExcelWriter(os.path.join(out_dir, "tables", f"paper_tables_calibration{tag}.xlsx")) as w:
        pd.DataFrame({"noise floor mean |B| (%)": [floor]}).to_excel(w, sheet_name="noise floor", index=False)
        avg.round(4).to_excel(w, sheet_name="average over models")
        t.round(4).to_excel(w, sheet_name="by model", index=False)
        loc_signed.round(3).to_excel(w, sheet_name="signed B by claim location")
        loc_abs.round(3).to_excel(w, sheet_name="abs B by claim location")
    fig_c1(d, out_dir, methods, tag)
    fig_c2(avg, floor, out_dir, methods, tag)
    fig_c3(d, pol, out_dir, methods, tag)
    fig_c4(t, floor, out_dir, methods, tag, d=d)
    fig_c5(d, out_dir, methods, tag)
    return {"by model": t, "average": avg, "signed B by claim location": loc_signed,
            "abs B by claim location": loc_abs, "noise floor": floor, "data": d}
