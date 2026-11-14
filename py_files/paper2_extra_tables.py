"""
paper2_extra_tables.py — paired-comparison tables with bootstrap confidence intervals.

Table "vsref": every model except Neural GA2M+ minus the reference model GBM, after global rescaling.
    For each model and metric: mean difference, 95% bootstrap CI over the 50 splits, share of splits in which the
    model is better than GBM, Wilcoxon signed-rank p (Holm over the six models within a metric).

Table "groups": Neural GA2M+ minus Neural GA2M by claim-location group (where the three largest claims fall).
    Test deviance before rescaling, split into the contribution of the
    policies with the three largest claims and of all other policies (columns "Contribution of"); Gini index (unchanged by rescaling).
    For each group: number of splits, mean difference, 95% bootstrap CI over the splits of the group,
    share of splits in which Neural GA2M+ is better, Wilcoxon signed-rank p (exact; Holm over the four groups).
    Below the table: permutation test of the largest-claim-in-test group against the other 40 splits.

Inputs
    --perf   perf.csv of the merged run with Neural GA2M+ (v2_full_plus995/perf.csv)
    --dev    policy_pass_deviance.csv (paper2_paper_figures.policy_pass; seed x model: deviance, extreme_part,
             deviance_without_extremes; before rescaling)
    --seeds  policy_pass_seeds.csv (seed, C1_in, C2_in, C3_in: set holding the 1st, 2nd and 3rd largest claim)

Usage
    python src/paper2_extra_tables.py --perf v2_full_plus995/perf.csv --dev v2_paper/tables/policy_pass_deviance.csv \
        --seeds v2_paper/tables/policy_pass_seeds.csv --out v2_paper/tables/paired
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

B_BOOT = 10_000
SEED = 2026
GROUPS = {1: "Largest claim in test", 2: "2nd or 3rd largest in test", 3: "None in test, some in validation",
          4: "All three in training"}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj, run, m = np.empty(len(p)), 0.0, len(p)
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def boot_ci(d, rng):
    d = np.asarray(d, float)
    means = rng.choice(d, size=(B_BOOT, len(d)), replace=True).mean(axis=1)
    return np.quantile(means, 0.025), np.quantile(means, 0.975)


def group_of(r):
    if r.C1_in == "test":
        return 1
    if "test" in (r.C2_in, r.C3_in):
        return 2
    if "valid" in (r.C1_in, r.C2_in, r.C3_in):
        return 3
    return 4


# ----------------------------------------------------------------------------------------------- table "vsref"
REF = "GBM"
OTHERS = ["EBM", "Neural GA2M", "LocalGLMnet", "CANN", "Tweedie GLM", "GLM"]


def table_vsref(perf, rng):
    p = perf[perf.version == "train_level"]
    rows = []
    for col, lab, sign in (("tweedie_deviance", "Test deviance", -1), ("gini", "Gini index", +1)):
        w = p.pivot(index="seed", columns="method", values=col)
        block = []
        for m in OTHERS:
            d = (w[m] - w[REF]).to_numpy()
            lo, hi = boot_ci(d, rng)
            block.append({"metric": lab, "model": m, "mean": d.mean(), "boot_lo": lo, "boot_hi": hi,
                          "better": float(np.mean(sign * d > 0)), "wilcoxon_p": stats.wilcoxon(d).pvalue})
        for r, q in zip(block, holm([r["wilcoxon_p"] for r in block])):
            r["wilcoxon_p_holm"] = q
        rows += block
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- table "groups"
def table_groups(perf, dev, seeds, rng):
    g = seeds.set_index("seed").apply(group_of, axis=1)
    comp = {}
    for col, lab in (("deviance", "total"), ("extreme_part", "extreme"), ("deviance_without_extremes", "other")):
        w = dev.pivot(index="seed", columns="model", values=col)
        comp[lab] = w["Neural GA2M+"] - w["Neural GA2M"]
    wg = perf[perf.version == "train_level"].pivot(index="seed", columns="method", values="gini")
    comp["gini"] = wg["Neural GA2M+"] - wg["Neural GA2M"]
    wr = perf[perf.version == "train_level"].pivot(index="seed", columns="method", values="tweedie_deviance")
    comp["total_rescaled"] = wr["Neural GA2M+"] - wr["Neural GA2M"]
    sign = {"total": -1, "extreme": -1, "other": -1, "gini": +1, "total_rescaled": -1}

    rows = []
    for key in list(GROUPS) + ["all"]:
        idx = g.index if key == "all" else g.index[g == key]
        r = {"group": key, "label": "All splits" if key == "all" else GROUPS[key], "n": len(idx)}
        for c, d in comp.items():
            x = d.loc[idx].to_numpy()
            lo, hi = boot_ci(x, rng)
            r.update({f"{c}_mean": x.mean(), f"{c}_lo": lo, f"{c}_hi": hi,
                      f"{c}_better": float(np.mean(sign[c] * x > 0)), f"{c}_p": stats.wilcoxon(x).pvalue})
        rows.append(r)
    t = pd.DataFrame(rows)
    grp = t.group != "all"
    for c in comp:
        t.loc[grp, f"{c}_p_holm"] = holm(t.loc[grp, f"{c}_p"])
        t.loc[~grp, f"{c}_p_holm"] = t.loc[~grp, f"{c}_p"]

    extra = {}
    for c in ("total", "gini", "total_rescaled"):
        d = comp[c]
        a, b = d[g == 1].to_numpy(), d[g != 1].to_numpy()
        obs = sign[c] * (a.mean() - b.mean())
        pool = np.concatenate([a, b])
        perm_rng = np.random.default_rng(SEED)
        cnt = 0
        for _ in range(B_BOOT):
            perm_rng.shuffle(pool)
            cnt += sign[c] * (pool[:len(a)].mean() - pool[len(a):].mean()) >= obs
        extra[f"perm_group1_{c}"] = (cnt + 1) / (B_BOOT + 1)
    return t, pd.Series(extra)


# ---------------------------------------------------------------------------------------------------- LaTeX
def num(x, dec):
    s = f"{x:+.{dec}f}"
    return "0." + "0" * dec if float(s) == 0 else s.replace("-", "$-$")


def pval(p):
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(np.log10(p)))
    c = round(p / 10 ** e)
    if c == 10:
        c, e = 1, e + 1
    return f"${c}\\times10^{{{e}}}$"


def latex_vsref(t):
    L = [r"\begin{table}[H]",
         r"\caption{Paired differences from GBM after global rescaling, 50 splits (model minus GBM; a positive deviance or a negative Gini difference means worse than GBM). Bootstrap "
         r"CI: percentile interval of the mean difference over 10,000 resamples of the splits. Better: share of "
         r"splits in which the model has a lower deviance or a higher Gini index than GBM. $p$: Wilcoxon signed-rank "
         r"test, Holm-adjusted over the six models.\label{tab:vsgbm}}",
         r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrcrr rcrr}", r"\toprule",
         r" & \multicolumn{4}{c}{Test deviance} & \multicolumn{4}{c}{Gini index} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Model & Mean & 95\% CI, bootstrap & Better & $p$ & Mean & 95\% CI, bootstrap & Better & $p$ \\",
         r"\midrule"]
    d = t[t.metric == "Test deviance"].set_index("model")
    g = t[t.metric == "Gini index"].set_index("model")
    for m in OTHERS:
        a, b = d.loc[m], g.loc[m]
        L.append(f"{m} & {num(a['mean'], 3)} & [{num(a.boot_lo, 3)}, {num(a.boot_hi, 3)}] & {100 * a.better:.0f}\\% & "
                 f"{pval(a.wilcoxon_p_holm)} & {num(b['mean'], 4)} & [{num(b.boot_lo, 4)}, {num(b.boot_hi, 4)}] & "
                 f"{100 * b.better:.0f}\\% & {pval(b.wilcoxon_p_holm)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L)


def latex_groups(t, extra):
    L = [r"\begin{table}[H]",
         r"\caption{Neural GA2M+ minus Neural GA2M by claim-location group (where the three largest claims of the portfolio fall). Test "
         r"deviance before rescaling, split into the contribution of the policies with the three largest claims "
         r"and of all other policies; Gini index. Mean over the splits of the group; bootstrap CI: percentile "
         r"interval over 10,000 resamples of these splits. Better: share of splits in which Neural GA2M+ has the "
         r"lower deviance or the higher Gini index. $p$: Wilcoxon signed-rank test (exact), Holm-adjusted over "
         r"the four groups (unadjusted for all splits).\label{tab:groups_ga2m}}",
         r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrrcrrrrrrr}", r"\toprule",
         r" & & \multicolumn{4}{c}{Test deviance} & \multicolumn{2}{c}{Contribution of} & \multicolumn{3}{c}{Gini index} \\",
         r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}",
         r"Group & Splits & Mean & 95\% CI, bootstrap & Better & $p$ & Largest claims & Other policies & Mean & Better & $p$ \\",
         r"\midrule"]
    for _, r in t.iterrows():
        if r.group == "all":
            L.append(r"\midrule")
        L.append(f"{r.label} & {r.n} & {num(r.total_mean, 3)} & [{num(r.total_lo, 3)}, {num(r.total_hi, 3)}] & "
                 f"{100 * r.total_better:.0f}\\% & {pval(r.total_p_holm)} & {num(r.extreme_mean, 3)} & "
                 f"{num(r.other_mean, 3)} & {num(r.gini_mean, 4)} & {100 * r.gini_better:.0f}\\% & "
                 f"{pval(r.gini_p_holm)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
          r"\noindent{\footnotesize Permutation test of the first group against the other 40 splits: "
          f"$p$ = {pval(extra.perm_group1_total)} for the deviance and {pval(extra.perm_group1_gini)} for the Gini index." + r"}", r"\end{table}"]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    perf = pd.read_csv(a.perf)
    perf["method"] = perf.method.replace({"Tweedie-CANN": "CANN"})
    dev, seeds = pd.read_csv(a.dev), pd.read_csv(a.seeds)

    t1 = table_vsref(perf, rng)
    t1.to_csv(os.path.join(a.out, "vsgbm.csv"), index=False)
    open(os.path.join(a.out, "table_vsgbm.tex"), "w").write(latex_vsref(t1) + "\n")

    t2, extra = table_groups(perf, dev, seeds, rng)
    t2.to_csv(os.path.join(a.out, "ga2m_groups.csv"), index=False)
    extra.to_csv(os.path.join(a.out, "ga2m_groups_tests.csv"))
    open(os.path.join(a.out, "table_groups.tex"), "w").write(latex_groups(t2, extra) + "\n")

    pd.set_option("display.width", 250)
    print(t1.round(4).to_string(index=False))
    print(t2.round(4).T.to_string()); print(extra.round(5))


if __name__ == "__main__":
    main()
