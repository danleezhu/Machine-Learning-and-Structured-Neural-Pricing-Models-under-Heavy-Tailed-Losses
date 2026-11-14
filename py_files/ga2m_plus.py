"""
ga2m_plus.py — Neural GA2M+: the Neural GA2M architecture with interaction pairs screened on a capped target over
half-samples of the training set.

Screening (training set only; the validation and test sets are not used)
  1. cap = q-quantile of the positive policy claim amounts of the training set (q = 0.995);
     y_q = min(ClaimAmount, cap) / Exposure (capped pure premium, exposure weights).
  2. Main-effects EBM on y_q (Tweedie(p) objective, learning_rate / max_rounds of the selected EBM configuration,
     no interactions); init score = log of its fitted means.
  3. Complementary-pairs subsampling (Shah & Samworth 2013): n_splits = 10 random halvings of the training set,
     stratified by claim indicator x claim-size class, both halves used -> 20 subsamples. On each subsample the FAST
     strength of every pair is measured on the residuals of the main-effects fit; the ranking on the full training set
     is recorded as well.
  4. Selection rule (fixed_cap_sets):
       "meanrank" : the k_top = 10 pairs with the best mean rank over the 20 half-samples (the rule of the paper);
       "pi"       : stability selection, pairs with selection frequency pi >= pi_threshold.
  The Neural GA2M (selected configuration, all n_bag members, same model seed as the Neural GA2M of the main run) is
  then fitted on the selected pairs and evaluated exactly as in paper2_run_full.

Paper pipeline (fixed cap 0.995)
    import ga2m_plus as gp
    gp.screen_fixed_cap(df_freq, df_sev, range(41, 91), "v2_ga2mplus_screen995", CONFIGS, cap=0.995)
    rr = gp.rule_report("v2_full", "v2_ga2mplus_screen995", 0.995)          # training data only
    gp.run_fixed_cap(df_freq, df_sev, range(41, 91), "v2_ga2mplus_screen995", "v2_ga2mplus_c995_meanrank",
                     CONFIGS, cap=0.995, rule="meanrank")
    gp.structure_stability("v2_full", "v2_ga2mplus_c995_meanrank")
    gp.merge_results("v2_full", "v2_ga2mplus_c995_meanrank", "v2_full_plus995")   # then paper2_analysis_full.run_all

Variant with the cap chosen on the validation set (run_ga2m_plus): screening at caps 0.995 and 0.999, member 0 of the
ensemble fitted for every distinct pair set, the cap with the lower validation deviance kept, remaining members fitted
for that set only. run_screening_ablation / ablation_report repeat the screening on the uncapped target to separate the
effect of the cap from that of the half-sample ranking.

Files (out_dir): perf.csv, calibration.csv, pd_curves.csv, nn_history.jsonl (same formats as the main run),
ga2mplus_selection.csv (one row per seed), ga2mplus_frequencies.csv (seed x cap x pair: pi, mean half-sample rank,
full-sample strength and rank), seeds_complete.csv, predictions/Neural_GA2M+/seed<s>.npz. Settings are stored in
ga2mplus_settings.json; a resumed run with different settings is refused.
"""

import itertools
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from scipy import stats

import paper2_adjust_clean_plus as p2
import paper2_datasets as dsm
import paper2_models as pm
import paper2_run_full as rf

METHOD = "Neural GA2M+"
DEFAULT_SETTINGS = dict(caps=[0.995, 0.999], k_top=10, pi_threshold=0.6, n_splits=10)
STRESS_COLS = ("train_top1_share", "train_max_loss_share")


# =============================================================================================================
# Data-set hooks (same as the main run)
# =============================================================================================================
def _power():
    return dsm.power()


def _dataset():
    return dsm.active()


def _prepare(df_freq, df_sev):
    return dsm.prepare(df_freq, df_sev)


def _split(df, seed):
    return dsm.split(df, seed)


def _check_dir(out_dir, power):
    dsm.check_run_dir(out_dir, power)


def pair_key(a, b):
    a, b = sorted((str(a), str(b)))
    return f"{a} x {b}"


def _display(x):
    try:
        from IPython.display import display
        display(x)
    except ImportError:
        print(x)


# =============================================================================================================
# Screening
# =============================================================================================================
def _subsample_strata(train_df):
    amt = train_df["ClaimAmount"].to_numpy(float)
    pos = amt[amt > 0]
    bins = np.quantile(pos, [0.50, 0.75, 0.90, 0.97, 0.99]) if len(pos) else np.array([])
    cls = np.where(amt > 0, np.searchsorted(bins, amt) + 1, 0)
    while True:                       # merge classes with < 4 policies into the next lower claim-size class
        counts = pd.Series(cls).value_counts()
        small = sorted(c for c in counts.index if counts[c] < 4 and c > 0)
        if not small:
            return cls
        c = small[-1]
        cls[cls == c] = max(k for k in counts.index if k < c)


def _parse(res, names):
    items = res.items() if isinstance(res, dict) else res
    out = []
    for item in items:
        a, b = item[0][0], item[0][1]
        if isinstance(a, (int, np.integer)):
            a, b = names[int(a)], names[int(b)]
        out.append((pair_key(a, b), float(item[1])))
    return out


def _fast_strengths(X, y, init, e, objective, names, n_pairs):
    """FAST strength of every pair on the residuals of the main-effects fit (one call of measure_interactions)."""
    from interpret.utils import measure_interactions
    res = measure_interactions(X, y, interactions=None, init_score=init, sample_weight=e, objective=objective)
    st = dict(_parse(res, names))
    if len(st) != n_pairs:
        raise RuntimeError(f"measure_interactions returned {len(st)} pairs, expected {n_pairs}")
    return st


def screen_pairs(train_df, num_cols, cat_cols, power, ebm_cfg, cap_q, k_top, pi_threshold, n_splits, seed,
                 n_jobs=1):
    """
    cap_q = None: uncapped target (ablation). Returns (selected pairs as [a, b] lists, frequency table, cap value,
    minutes). The frequency table has one row per pair: pi, mean_rank (half samples), full_strength / full_rank
    (FAST on the full training set).
    """
    from interpret.glassbox import ExplainableBoostingRegressor
    from sklearn.model_selection import train_test_split
    t0 = time.time()
    names = list(num_cols) + list(cat_cols)
    X = train_df[names]
    e = train_df["Exposure"].to_numpy(float)
    amt = train_df["ClaimAmount"].to_numpy(float)
    cap = float(np.quantile(amt[amt > 0], cap_q)) if cap_q is not None else np.inf
    y = np.minimum(amt, cap) / e
    objective = f"tweedie_deviance:variance_power={power}"
    kw = dict(objective=objective, interactions=0, random_state=pm.EBM_CONSTANTS["random_state"])
    for k in ("learning_rate", "max_rounds"):
        if ebm_cfg.get(k) is not None:
            kw[k] = ebm_cfg[k]
    mains = ExplainableBoostingRegressor(**kw).fit(X, y, sample_weight=e)
    init = np.log(np.maximum(np.asarray(mains.predict(X), float), 1e-10))
    all_pairs = [pair_key(a, b) for a, b in itertools.combinations(names, 2)]
    n_pairs = len(all_pairs)

    strata = _subsample_strata(train_df)
    idx = np.arange(len(train_df))
    halves = []
    for s in range(int(n_splits)):
        h1, h2 = train_test_split(idx, train_size=0.5, stratify=strata, random_state=1000 * int(seed) + s)
        halves += [h1, h2]
    jobs = [(X.iloc[h], y[h], init[h], e[h]) for h in halves] + [(X, y, init, e)]   # last = full training set
    if n_jobs == 1:
        results = [_fast_strengths(a, b, c, d, objective, names, n_pairs) for a, b, c, d in jobs]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_fast_strengths)(a, b, c, d, objective, names, n_pairs)
                                          for a, b, c, d in jobs)
    full = results.pop()
    tops, ranks = [], []
    for st in results:
        order = sorted(st, key=st.get, reverse=True)
        tops.append(set(order[:k_top]))
        ranks.append({p: r + 1 for r, p in enumerate(order)})
    full_order = sorted(full, key=full.get, reverse=True)
    full_rank = {p: r + 1 for r, p in enumerate(full_order)}
    freq = pd.DataFrame({"pair": all_pairs,
                         "pi": [np.mean([p in t for t in tops]) for p in all_pairs],
                         "mean_rank": [np.mean([r[p] for r in ranks]) for p in all_pairs],
                         "full_strength": [full[p] for p in all_pairs],
                         "full_rank": [full_rank[p] for p in all_pairs]})
    freq = freq.sort_values(["pi", "mean_rank"], ascending=[False, True]).reset_index(drop=True)
    sel = freq[freq.pi >= pi_threshold]
    if sel.empty:
        sel = freq[freq.pi == freq.pi.max()]
    pairs = [p.split(" x ") for p in sel.pair]
    return pairs, freq, cap, (time.time() - t0) / 60


# =============================================================================================================
# Neural GA2M members (same construction as paper2_models.NeuralGA2MWrapper.fit, member by member)
# =============================================================================================================
def _fit_members(model, train_df, valid_df, bags):
    """
    Trains members `bags` of a NeuralGA2MWrapper exactly as NeuralGA2MWrapper.fit does (same preprocessing, same
    tensors, same initial bias, member seed = seed + 100000 b) and appends them to model.models_. Members can be added
    in several calls; the wrapper's predict() averages all members present.
    """
    import torch
    if not hasattr(model, "_member_infos"):
        model.prep_.fit(train_df)
        model.models_, model._member_infos, model._members = [], [], []
    tr_t, va_t = pm.make_tensors(model.prep_, train_df), pm.make_tensors(model.prep_, valid_df)
    init_mean = float(np.average(train_df[pm.TARGET], weights=train_df[pm.EXPOSURE]))
    for b in bags:
        if b in model._members:
            continue
        s = model.seed + 100000 * int(b)
        torch.manual_seed(s)
        m = pm.NeuralGA2MModule(model.numerical_cols, model.categorical_cols, model.prep_.category_sizes(), model.pairs,
                                init_mean, tuple(model.arch["hidden"]))
        m, info = pm.train_network(m, tr_t, va_t, model.tweedie_power, model.train_cfg, s, tag=f"{METHOD} #{b}",
                                   verbose=model.verbose)
        model.models_.append(m)
        model._member_infos.append(info)
        model._members.append(int(b))
    infos = model._member_infos
    model.train_info_ = dict(infos[0])
    model.train_info_.update({"member_best_epochs": [i["best_epoch"] for i in infos],
                              "member_epochs_run": [i["epochs_run"] for i in infos],
                              "share_grad_norm_over_5": float(np.mean([i["share_grad_norm_over_5"] for i in infos]))})
    model.best_iteration_ = float(np.mean(model.train_info_["member_best_epochs"]))
    return model


def _valid_devs(model, y_tr, e_tr, train_df, y_va, e_va, valid_df, power):
    p_tr, p_va = pm.predict(model, train_df), pm.predict(model, valid_df)
    c = float(np.sum(e_tr * y_tr) / np.sum(e_tr * p_tr))
    return (float(p2.tweedie_deviance(y_va, c * p_va, e_va, power=power)),
            float(p2.tweedie_deviance(y_va, p_va, e_va, power=power)))


# =============================================================================================================
# Run
# =============================================================================================================
def _check_settings(out_dir, settings):
    path = os.path.join(out_dir, "ga2mplus_settings.json")
    canon = json.dumps(settings, sort_keys=True)
    if os.path.exists(path):
        if json.dumps(json.load(open(path)), sort_keys=True) != canon:
            raise RuntimeError(f"{out_dir} was started with different settings ({path}); use a new out_dir")
    else:
        json.dump(settings, open(path, "w"), indent=1, sort_keys=True)


FREQ_COLUMNS = ["pair", "pi", "mean_rank", "full_strength", "full_rank", "seed", "cap_level", "cap_value"]
SEL_COLUMNS = ["seed", "cap_levels", "cap_values", "sets", "set_sizes", "valid_dev_level", "valid_dev_raw",
               "chosen_cap_level", "chosen_pairs", "chosen_size", "final_valid_dev_level", "final_valid_dev_raw",
               "screen_minutes", "select_minutes", "final_minutes"]


def run_ga2m_plus(df_freq, df_sev, seeds, out_dir, configs_path, caps=None, k_top=None, pi_threshold=None,
                  n_splits=None, power=None, n_perm=199, df_prepared=None, compute_pd=True, n_jobs=4,
                  save_predictions=True):
    """GA2M+ with the cap chosen on the validation set (see the module docstring).
    n_jobs: parallel FAST calls in the screening, as threads (interpret's native code runs outside the GIL); results do
    not depend on it; 1 = sequential.
    save_predictions: raw validation and test predictions (float32), the training sum of exposure x prediction and
    hashes of the split are written to out_dir/predictions/Neural_GA2M+/seed<s>.npz (same layout as
    paper2_predictions.py), so the calibration studies need no refit."""
    power = _power() if power is None else power
    configs = rf.load_configs(configs_path)
    settings = dict(DEFAULT_SETTINGS)
    for k, v in (("caps", caps), ("k_top", k_top), ("pi_threshold", pi_threshold), ("n_splits", n_splits)):
        if v is not None:
            settings[k] = list(v) if k == "caps" else v
    settings["caps"] = sorted(float(c) for c in settings["caps"])
    settings_full = {**settings, "ga2m_config": configs["Neural GA2M"], "ebm_config": configs["EBM"],
                     "tweedie_power": power, "dataset": _dataset(), "cap_selection": "member 0 of the ensemble"}
    os.makedirs(out_dir, exist_ok=True)
    _check_dir(out_dir, power)
    _check_settings(out_dir, settings_full)
    P = {k: os.path.join(out_dir, f) for k, f in (("perf", "perf.csv"), ("cal", "calibration.csv"),
                                                  ("pd", "pd_curves.csv"), ("hist", "nn_history.jsonl"),
                                                  ("sel", "ga2mplus_selection.csv"),
                                                  ("freq", "ga2mplus_frequencies.csv"),
                                                  ("done", "seeds_complete.csv"))}
    done = set(pd.read_csv(P["done"]).seed) if os.path.exists(P["done"]) else set()
    df, num_cols, cat_cols = _prepare(df_freq, df_sev) if df_prepared is None else df_prepared
    grid, levels = rf.make_pd_grid(df, num_cols, cat_cols)
    n_bag = int(configs["Neural GA2M"]["arch"]["n_bag"])
    print("settings:", json.dumps(settings), "| data set:", _dataset(), "| p =", power)
    print("GA2M configuration:", json.dumps(configs["Neural GA2M"], sort_keys=True))

    for seed in [int(s) for s in seeds]:
        if seed in done:
            continue
        t_seed = time.time()
        print(f"\n{'=' * 100}\nseed {seed}\n{'=' * 100}")
        p2.set_all_seeds(seed)
        train_df, valid_df, test_df = _split(df, seed)
        y_tr, e_tr = train_df[pm.TARGET].to_numpy(float), train_df[pm.EXPOSURE].to_numpy(float)
        y_va, e_va = valid_df[pm.TARGET].to_numpy(float), valid_df[pm.EXPOSURE].to_numpy(float)
        y_te, e_te = test_df[pm.TARGET].to_numpy(float), test_df[pm.EXPOSURE].to_numpy(float)
        sets = (("train", train_df), ("valid", valid_df), ("test", test_df))

        # 1. screening on the training set, one set per cap
        cand, freq_rows, cap_values, t_screen = {}, [], [], 0.0
        for q in settings["caps"]:
            pairs, freq, cap, mins = screen_pairs(train_df, num_cols, cat_cols, power, configs["EBM"], q,
                                                  settings["k_top"], settings["pi_threshold"], settings["n_splits"],
                                                  seed, n_jobs=n_jobs)
            cand[q] = pairs
            cap_values.append(cap)
            t_screen += mins
            freq_rows.append(freq.assign(seed=seed, cap_level=q, cap_value=cap))
            print(f"  cap {q}: {cap:,.0f} | {len(pairs)} pairs selected | {mins:.1f} min | "
                  f"{', '.join(' x '.join(p) for p in pairs)}")

        # 2. member 0 per distinct set, cap chosen on validation
        key_of = {q: tuple(sorted(pair_key(*p) for p in cand[q])) for q in settings["caps"]}
        fitted, t0 = {}, time.time()
        for q in settings["caps"]:
            key = key_of[q]
            if key not in fitted:
                model = pm.build_model("Neural GA2M", num_cols, cat_cols, seed, power, configs, ebm_pairs=cand[q])
                _fit_members(model, train_df, valid_df, [0])
                dl, dr = _valid_devs(model, y_tr, e_tr, train_df, y_va, e_va, valid_df, power)
                fitted[key] = {"model": model, "caps": [], "dev_level": dl, "dev_raw": dr}
            fitted[key]["caps"].append(q)
        t_select = (time.time() - t0) / 60
        best_key = min(fitted, key=lambda k: (round(fitted[k]["dev_level"], 12), -max(fitted[k]["caps"])))
        best = fitted[best_key]
        q_star = max(best["caps"])
        print("  validation deviance of member 0 (level-corrected): " +
              ", ".join(f"caps {f['caps']}: {f['dev_level']:.4f}" for f in fitted.values()) + f" -> cap {q_star}")

        # 3. remaining members for the chosen set
        t0 = time.time()
        model = _fit_members(best["model"], train_df, valid_df, range(1, n_bag))
        t_final = (time.time() - t0) / 60
        fin_level, fin_raw = _valid_devs(model, y_tr, e_tr, train_df, y_va, e_va, valid_df, power)
        fit_secs = 60 * t_final + 60 * t_select / max(len(fitted), 1)   # chosen set: member 0 + other members

        # 4. evaluation exactly as in the main run
        raw = {s: pm.predict(model, d) for s, d in sets}
        if save_predictions:
            save_prediction_file(out_dir, METHOD, seed, raw, sets)
        preds, cal_info = rf.calibrated_versions(raw, sets, y_tr, e_tr, valid_df, y_va, n_perm, seed)
        info = getattr(model, "train_info_", {}) or {}
        rows = []
        for version in rf.VERSIONS:
            p = preds[version]
            rows.append({"seed": seed, "method": METHOD, "version": version,
                         **rf.test_metrics(test_df, y_te, p["test"], e_te, power),
                         "valid_balance_error_pct": rf.balance(y_va, p["valid"], e_va),
                         "train_balance_error_pct": rf.balance(y_tr, p["train"], e_tr),
                         "train_observed_total": float(np.sum(e_tr * y_tr)),
                         "train_predicted_total": float(np.sum(e_tr * p["train"])),
                         "fit_seconds": fit_secs, "best_iteration": getattr(model, "best_iteration_", np.nan),
                         "epochs_run": info.get("epochs_run", np.nan), "stopped_by": info.get("stopped_by", "")})
        anchors = rf._anchors(train_df, cat_cols)
        pdc = rf.pd_curves(pm, model, train_df, grid, levels, anchors, seed=seed).assign(seed=seed, method=METHOD) \
            if compute_pd else None
        sel_row = {"seed": seed, "cap_levels": json.dumps(settings["caps"]), "cap_values": json.dumps(cap_values),
                   "sets": json.dumps({str(q): list(key_of[q]) for q in settings["caps"]}),
                   "set_sizes": json.dumps({str(q): len(cand[q]) for q in settings["caps"]}),
                   "valid_dev_level": json.dumps({str(q): fitted[key_of[q]]["dev_level"] for q in settings["caps"]}),
                   "valid_dev_raw": json.dumps({str(q): fitted[key_of[q]]["dev_raw"] for q in settings["caps"]}),
                   "chosen_cap_level": q_star, "chosen_pairs": json.dumps(list(best_key)),
                   "chosen_size": len(best_key), "final_valid_dev_level": fin_level, "final_valid_dev_raw": fin_raw,
                   "screen_minutes": t_screen, "select_minutes": t_select, "final_minutes": t_final}
        rf._append([sel_row], P["sel"], SEL_COLUMNS)
        rf._append(pd.concat(freq_rows, ignore_index=True), P["freq"])
        rf._append([{"seed": seed, "method": METHOD, **cal_info,
                     "share_grad_norm_over_5": info.get("share_grad_norm_over_5", np.nan)}], P["cal"], rf.CAL_COLUMNS)
        rf._append(pdc, P["pd"])
        if info.get("history"):
            with open(P["hist"], "a") as fh:
                fh.write(json.dumps({"seed": seed, "method": METHOD, "history": info["history"]}) + "\n")
        rf._append(rows, P["perf"])
        rf._append([{"seed": seed, "minutes": (time.time() - t_seed) / 60}], P["done"])
        r0, rl = rows[0], rows[rf.VERSIONS.index("train_level")]
        print(f"  {METHOD} | dev {r0['tweedie_deviance']:.4f} -> {rl['tweedie_deviance']:.4f} (train level) | "
              f"Gini {r0['gini']:.4f} | seed done in {(time.time() - t_seed) / 60:.1f} min "
              f"(screening {t_screen:.1f}, selection {t_select:.1f}, final members {t_final:.1f})")
        del fitted, model, raw, preds, train_df, valid_df, test_df
    return pd.read_csv(P["perf"]) if os.path.exists(P["perf"]) else None


def prediction_path(out_dir, method, seed):
    return os.path.join(out_dir, "predictions", method.replace(" ", "_"), f"seed{int(seed)}.npz")


def save_prediction_file(out_dir, method, seed, raw, sets):
    """Same format as paper2_predictions.save_file: valid / test predictions (float32), the training sum of
    exposure x prediction, SHA-1 of the IDpol of every set (no policy ids, no training predictions)."""
    import paper2_predictions as pp
    pp.save_file(prediction_path(out_dir, method, seed), raw, sets)


# =============================================================================================================
# Fixed-cap GA2M+: one cap for every seed, no cap choice on the validation set
# =============================================================================================================
def _evaluate_store(model, seed, sets, y_tr, e_tr, y_va, e_va, y_te, e_te, valid_df, test_df, train_df, cat_cols,
                    grid, levels, power, n_perm, P, out_dir, fit_secs, compute_pd, save_predictions):
    """Same evaluation and files as run_ga2m_plus (raw and train_level, metrics, partial dependence, diagnostics)."""
    raw = {s: pm.predict(model, d) for s, d in sets}
    if save_predictions:
        save_prediction_file(out_dir, METHOD, seed, raw, sets)
    preds, cal_info = rf.calibrated_versions(raw, sets, y_tr, e_tr, valid_df, y_va, n_perm, seed)
    info = getattr(model, "train_info_", {}) or {}
    rows = []
    for version in rf.VERSIONS:
        p = preds[version]
        rows.append({"seed": seed, "method": METHOD, "version": version,
                     **rf.test_metrics(test_df, y_te, p["test"], e_te, power),
                     "valid_balance_error_pct": rf.balance(y_va, p["valid"], e_va),
                     "train_balance_error_pct": rf.balance(y_tr, p["train"], e_tr),
                     "train_observed_total": float(np.sum(e_tr * y_tr)),
                     "train_predicted_total": float(np.sum(e_tr * p["train"])),
                     "fit_seconds": fit_secs, "best_iteration": getattr(model, "best_iteration_", np.nan),
                     "epochs_run": info.get("epochs_run", np.nan), "stopped_by": info.get("stopped_by", "")})
    if compute_pd:
        anchors = rf._anchors(train_df, cat_cols)
        rf._append(rf.pd_curves(pm, model, train_df, grid, levels, anchors, seed=seed).assign(seed=seed, method=METHOD),
                   P["pd"])
    rf._append([{"seed": seed, "method": METHOD, **cal_info,
                 "share_grad_norm_over_5": info.get("share_grad_norm_over_5", np.nan)}], P["cal"], rf.CAL_COLUMNS)
    if info.get("history"):
        with open(P["hist"], "a") as fh:
            fh.write(json.dumps({"seed": seed, "method": METHOD, "history": info["history"]}) + "\n")
    return rows


def _load_freq(d):
    """ga2mplus_frequencies.csv of a run_ga2m_plus or screen_fixed_cap directory, restricted to COMPLETE seeds
    (seeds_complete.csv / screen_complete.csv) and without the duplicate rows an interruption between two file
    appends can leave. Row order inside a seed is the screening order (pi descending, mean rank ascending)."""
    fr = pd.read_csv(os.path.join(d, "ga2mplus_frequencies.csv"))
    for f in ("screen_complete.csv", "seeds_complete.csv"):
        if os.path.exists(os.path.join(d, f)):
            fr = fr[fr.seed.isin(set(pd.read_csv(os.path.join(d, f)).seed))]
            break
    else:
        fr = fr.iloc[:0]
    fr = fr.assign(cap_level=fr.cap_level.astype(float))
    return fr.drop_duplicates(["seed", "cap_level", "pair"], keep="last")


def screen_fixed_cap(df_freq, df_sev, seeds, screen_dir, configs_path, cap=0.995, plus_dir=None, k_top=None,
                     pi_threshold=None, n_splits=None, power=None, df_prepared=None, n_jobs=4):
    """
    Screening of GA2M+ (training set only, no network) at ONE fixed cap for every seed -> screen_dir/
    ga2mplus_frequencies.csv (same columns as run_ga2m_plus). Optional plus_dir: a run_ga2m_plus directory with this
    cap among its caps and the same screening settings; its stored rows are copied instead of screening again (the
    screening is deterministic: EBM random_state fixed, halvings random_state = 1000 seed + s). Resumable; a different
    cap or setting needs a new screen_dir.
    """
    power = _power() if power is None else power
    configs = rf.load_configs(configs_path)
    s_ = dict(DEFAULT_SETTINGS)
    for k, v in (("k_top", k_top), ("pi_threshold", pi_threshold), ("n_splits", n_splits)):
        if v is not None:
            s_[k] = v
    settings = {"fixed_cap": float(cap), "k_top": s_["k_top"], "pi_threshold": s_["pi_threshold"],
                "n_splits": s_["n_splits"], "ebm_config": configs["EBM"], "tweedie_power": power, "dataset": _dataset()}
    os.makedirs(screen_dir, exist_ok=True)
    _check_dir(screen_dir, power)
    _check_settings(screen_dir, settings)
    path, done_path = os.path.join(screen_dir, "ga2mplus_frequencies.csv"), os.path.join(screen_dir, "screen_complete.csv")
    done = set(pd.read_csv(done_path).seed) if os.path.exists(done_path) else set()
    src = None
    if plus_dir is not None and os.path.exists(os.path.join(plus_dir, "ga2mplus_frequencies.csv")):
        ps = json.load(open(os.path.join(plus_dir, "ga2mplus_settings.json")))
        if "dataset" not in ps and os.path.exists(os.path.join(plus_dir, "dataset.json")):
            ps["dataset"] = json.load(open(os.path.join(plus_dir, "dataset.json"))).get("dataset")
        same = {k: ps.get(k) == settings[k] for k in ("k_top", "pi_threshold", "n_splits", "tweedie_power", "dataset")}
        same["ebm_config"] = json.dumps(ps.get("ebm_config"), sort_keys=True) == json.dumps(configs["EBM"], sort_keys=True)
        same["cap"] = any(np.isclose(float(c), float(cap)) for c in ps.get("caps", []))
        if not all(same.values()):
            raise RuntimeError(f"{plus_dir} was screened with other settings ({[k for k, v in same.items() if not v]});"
                               " use plus_dir=None to screen everything anew")
        src = _load_freq(plus_dir)
        src = src[np.isclose(src.cap_level, float(cap))]
    df = None
    n_copy = n_new = 0
    for seed in [int(s) for s in seeds]:
        if seed in done:
            continue
        if src is not None and seed in set(src.seed):
            rf._append(src[src.seed == seed].reindex(columns=FREQ_COLUMNS), path, FREQ_COLUMNS)
            rf._append([{"seed": seed, "minutes": 0.0, "source": os.path.relpath(plus_dir)}], done_path)
            n_copy += 1
            continue
        if df is None:
            df, num_cols, cat_cols = _prepare(df_freq, df_sev) if df_prepared is None else df_prepared
        train_df, _, _ = _split(df, seed)
        pairs, freq, cap_value, mins = screen_pairs(train_df, num_cols, cat_cols, power, configs["EBM"], float(cap),
                                                    s_["k_top"], s_["pi_threshold"], s_["n_splits"], seed,
                                                    n_jobs=n_jobs)
        rf._append(freq.assign(seed=seed, cap_level=float(cap), cap_value=cap_value), path, FREQ_COLUMNS)
        rf._append([{"seed": seed, "minutes": mins, "source": "screened"}], done_path)
        n_new += 1
        print(f"seed {seed}: cap {cap}: {cap_value:,.0f} | {len(pairs)} pairs (pi >= {s_['pi_threshold']}) | "
              f"{mins:.1f} min | {', '.join(' x '.join(p) for p in pairs)}")
    n_all = len(set(pd.read_csv(done_path).seed)) if os.path.exists(done_path) else 0
    print(f"screening at cap {cap}: {n_copy} seeds copied from {plus_dir}, {n_new} screened, {n_all} complete in "
          f"{screen_dir}")
    return _load_freq(screen_dir) if n_all else None


def fixed_cap_sets(screen_dir, cap=0.995, rule="pi", k_top=None, pi_threshold=None):
    """
    Pair list per seed from a stored screening (screen_fixed_cap or run_ga2m_plus directory), for ONE cap:
      rule "pi"       : pi >= pi_threshold (fallback: largest pi), in the screening order -> exactly the list
                        run_ga2m_plus passes to the network for that cap (the order of the pairs fixes the order of
                        the pair sub-networks and hence their initialisation)
      rule "meanrank" : the k_top pairs with the best mean rank over the 2 n_splits half samples, ordered by mean rank
    No model is fitted; only the training sets entered the screening.
    """
    k_top = DEFAULT_SETTINGS["k_top"] if k_top is None else k_top
    thr = DEFAULT_SETTINGS["pi_threshold"] if pi_threshold is None else pi_threshold
    fr = _load_freq(screen_dir)
    fr = fr[np.isclose(fr.cap_level, float(cap))]
    if fr.empty:
        raise ValueError(f"cap {cap} not screened in {screen_dir}")
    out = {}
    for seed, g in fr.groupby("seed", sort=True):
        if rule == "pi":
            sel = g[g.pi >= thr]
            sel = sel if len(sel) else g[g.pi == g.pi.max()]
        elif rule == "meanrank":
            sel = g.sort_values(["mean_rank", "pi", "pair"], ascending=[True, False, True], kind="mergesort").head(k_top)
        else:
            raise ValueError("rule must be 'pi' or 'meanrank'")
        out[int(seed)] = list(sel.pair)
    return out


def rule_report(main_dir, screen_dir, cap=0.995, ablation_dir=None, k_top=None, pi_threshold=None,
                stress_cols=STRESS_COLS):
    """
    Training-only choice between the two GA2M+ rules at a fixed cap, BEFORE any network is fitted: structure
    stability over the seeds (set size, Jaccard, pairs always / >= 90%, agreement with the leave-one-out consensus,
    its correlation with the training-tail statistics) of EBM pairs, single FAST top-k, stability selection (pi) and
    half-sample mean-rank top-k; raw-target rows if ablation_dir is given. Recommended rule = the one of pi / meanrank
    with the larger mean agreement with the leave-one-out consensus (ties -> pi).
    """
    k_top = DEFAULT_SETTINGS["k_top"] if k_top is None else k_top
    fr = _load_freq(screen_dir)
    fr = fr[np.isclose(fr.cap_level, float(cap))]
    main = _sets_from_main(main_dir)
    rules = {"EBM pairs (main GA2M)": main}
    if "full_rank" in fr and fr.full_rank.notna().all():
        rules[f"cap {cap}, single FAST top-{k_top}"] = {int(s): set(d.nsmallest(k_top, "full_rank").pair)
                                                        for s, d in fr.groupby("seed")}
    rules.update({f"cap {cap}, stability selection (rule pi)": {s: set(v) for s, v in
                                                           fixed_cap_sets(screen_dir, cap, "pi", k_top, pi_threshold).items()},
             f"cap {cap}, half-sample mean-rank top-{k_top} (rule meanrank)":
                 {s: set(v) for s, v in fixed_cap_sets(screen_dir, cap, "meanrank", k_top, pi_threshold).items()}})
    common = sorted(set(main) & set(fr.seed.astype(int)))
    if ablation_dir is not None and os.path.exists(os.path.join(ablation_dir, "ablation_raw_frequencies.csv")):
        ab = pd.read_csv(os.path.join(ablation_dir, "ablation_raw_frequencies.csv")).drop_duplicates(["seed", "pair"], keep="last")
        thr = DEFAULT_SETTINGS["pi_threshold"] if pi_threshold is None else pi_threshold
        if "full_rank" in ab and ab.full_rank.notna().all():
            rules[f"raw, single FAST top-{k_top}"] = {int(s): set(d.nsmallest(k_top, "full_rank").pair) for s, d in ab.groupby("seed")}
        rules["raw, stability selection"] = {int(s): (set(d[d.pi >= thr].pair) or set(d[d.pi == d.pi.max()].pair))
                                             for s, d in ab.groupby("seed")}
    seeds_csv = os.path.join(main_dir, "seeds.csv")
    splits = pd.read_csv(seeds_csv) if os.path.exists(seeds_csv) else None
    res, loos = {}, {}
    for name, sets in rules.items():
        sets = {s: sets[s] for s in common if s in sets}
        if len(sets) >= 3:
            res[name], loos[name] = _set_stats(sets, splits, stress_cols)
    table = pd.DataFrame(res).T
    _display(table.round(3))
    k_pi = f"cap {cap}, stability selection (rule pi)"
    k_mr = f"cap {cap}, half-sample mean-rank top-{k_top} (rule meanrank)"
    d = (loos[k_mr] - loos[k_pi]).dropna()
    try:
        p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
    except ValueError:
        p_w = np.nan
    rec = "meanrank" if d.mean() > 0 else "pi"
    print(f"{len(common)} seeds | agreement with the leave-one-out consensus, meanrank - pi: mean {d.mean():+.3f}, "
          f"meanrank better in {(d > 0).mean():.0%} of seeds, Wilcoxon p = {p_w:.3g} -> recommended rule: {rec}")
    return {"table": table, "recommended": rec, "agreement_diff": d, "seeds": common}


def _purge_partial(P, seed):
    """Removes the rows of an unfinished seed (an interruption between two file appends) before it is redone."""
    for key in ("perf", "cal", "pd", "sel"):
        f = P.get(key)
        if f and os.path.exists(f):
            d = pd.read_csv(f)
            if (d.seed == seed).any():
                d[d.seed != seed].to_csv(f, index=False)
    f = P.get("hist")
    if f and os.path.exists(f):
        lines = open(f).readlines()
        keep = [ln for ln in lines if json.loads(ln).get("seed") != seed]
        if len(keep) != len(lines):
            open(f, "w").writelines(keep)


def run_fixed_cap(df_freq, df_sev, seeds, screen_dir, out_dir, configs_path, cap=0.995, rule="pi", plus_dir=None,
                  k_top=None, pi_threshold=None, power=None, n_perm=199, df_prepared=None, compute_pd=True,
                  save_predictions=True, reuse=True):
    """
    GA2M+ with a FIXED cap (no choice of the cap on the validation set). Pair lists: fixed_cap_sets(screen_dir, ...)
    (nothing is screened here; run screen_fixed_cap first). The Neural GA2M (selected configuration, all n_bag
    members, same model seed as the main run) is fitted on these pairs and evaluated exactly as in run_ga2m_plus.
    reuse: for a seed complete in plus_dir whose stored model used exactly this pair list (rule "pi" and chosen cap =
    cap) the stored results and predictions are copied instead of refitting the identical model.
    Method name "Neural GA2M+"; out_dir has the same files as a run_ga2m_plus directory, so merge_results /
    structure_stability / compare_runs / the analysis use it unchanged. Resumable; seeds not yet screened are skipped.
    """
    power = _power() if power is None else power
    configs = rf.load_configs(configs_path)
    sets_all = fixed_cap_sets(screen_dir, cap, rule, k_top, pi_threshold)
    settings = {"fixed_cap": float(cap), "rule": rule, "k_top": DEFAULT_SETTINGS["k_top"] if k_top is None else k_top,
                "pi_threshold": DEFAULT_SETTINGS["pi_threshold"] if pi_threshold is None else pi_threshold,
                "screening_dir": os.path.relpath(screen_dir), "ga2m_config": configs["Neural GA2M"],
                "tweedie_power": power, "dataset": _dataset()}
    sset = json.load(open(os.path.join(screen_dir, "ga2mplus_settings.json")))
    if "fixed_cap" in sset and not np.isclose(float(sset["fixed_cap"]), float(cap)):
        raise RuntimeError(f"{screen_dir} was screened at cap {sset['fixed_cap']}, not {cap}")
    os.makedirs(out_dir, exist_ok=True)
    _check_dir(out_dir, power)
    _check_settings(out_dir, settings)
    P = {k: os.path.join(out_dir, f) for k, f in (("perf", "perf.csv"), ("cal", "calibration.csv"),
                                                  ("pd", "pd_curves.csv"), ("hist", "nn_history.jsonl"),
                                                  ("sel", "ga2mplus_selection.csv"), ("done", "seeds_complete.csv"))}
    done = set(pd.read_csv(P["done"]).seed) if os.path.exists(P["done"]) else set()
    fr = _load_freq(screen_dir)
    cap_value_of = fr[np.isclose(fr.cap_level, float(cap))].groupby("seed").cap_value.first().to_dict()
    src_sel, src_done = None, set()
    if reuse and plus_dir is not None and os.path.exists(os.path.join(plus_dir, "seeds_complete.csv")):
        ps = json.load(open(os.path.join(plus_dir, "ga2mplus_settings.json")))
        if json.dumps(ps.get("ga2m_config"), sort_keys=True) != json.dumps(configs["Neural GA2M"], sort_keys=True):
            raise RuntimeError(f"{plus_dir} used another GA2M configuration; use plus_dir=None")
        src_done = set(pd.read_csv(os.path.join(plus_dir, "seeds_complete.csv")).seed)
        src_sel = pd.read_csv(os.path.join(plus_dir, "ga2mplus_selection.csv")).drop_duplicates("seed", keep="last") \
            .set_index("seed")
    df = None
    grid = levels = None
    n_bag = int(configs["Neural GA2M"]["arch"]["n_bag"])
    n_reused = n_fit = n_skip = 0
    print(f"fixed cap {cap}, rule {rule} | screening from {screen_dir} | {len(sets_all)} seeds screened")
    for seed in [int(s) for s in seeds]:
        if seed in done:
            continue
        if seed not in sets_all:
            n_skip += 1
            continue
        t0 = time.time()
        _purge_partial(P, seed)
        pairs = sets_all[seed]
        cap_value = float(cap_value_of.get(seed, np.nan))
        sel_row = {"seed": seed, "cap_levels": json.dumps([float(cap)]), "cap_values": json.dumps([cap_value]),
                   "sets": json.dumps({str(float(cap)): pairs}), "set_sizes": json.dumps({str(float(cap)): len(pairs)}),
                   "valid_dev_level": json.dumps({}), "valid_dev_raw": json.dumps({}),
                   "chosen_cap_level": float(cap), "chosen_pairs": json.dumps(pairs), "chosen_size": len(pairs),
                   "screen_minutes": 0.0, "select_minutes": 0.0}
        same = (src_sel is not None and seed in src_done and seed in src_sel.index and rule == "pi"
                and np.isclose(float(src_sel.loc[seed, "chosen_cap_level"]), float(cap))
                and sorted(json.loads(src_sel.loc[seed, "chosen_pairs"])) == sorted(pairs))
        if same:
            j = os.path.join
            for key, f in (("perf", "perf.csv"), ("cal", "calibration.csv"), ("pd", "pd_curves.csv")):
                if os.path.exists(j(plus_dir, f)):
                    d = pd.read_csv(j(plus_dir, f))
                    d = d[d.seed == seed]
                    if key == "perf":       # keep only the versions of rf.VERSIONS
                        d = d[d.version.isin(rf.VERSIONS)].drop_duplicates(["seed", "method", "version"], keep="last")
                    elif key == "cal":
                        d = d.drop_duplicates(["seed", "method"], keep="last").reindex(columns=rf.CAL_COLUMNS)
                    rf._append(d, P[key], rf.CAL_COLUMNS if key == "cal" else None)
            hp = j(plus_dir, "nn_history.jsonl")
            if os.path.exists(hp):
                last = None
                with open(hp) as fh:
                    for line in fh:
                        if json.loads(line).get("seed") == seed:
                            last = line
                if last is not None:
                    with open(P["hist"], "a") as out:
                        out.write(last)
            src_pred = prediction_path(plus_dir, METHOD, seed)
            if os.path.exists(src_pred):
                os.makedirs(os.path.dirname(prediction_path(out_dir, METHOD, seed)), exist_ok=True)
                shutil.copy(src_pred, prediction_path(out_dir, METHOD, seed))
            sel_row.update(final_valid_dev_level=src_sel.loc[seed].get("final_valid_dev_level", np.nan),
                           final_valid_dev_raw=src_sel.loc[seed].get("final_valid_dev_raw", np.nan), final_minutes=0.0)
            rf._append([sel_row], P["sel"], SEL_COLUMNS)
            rf._append([{"seed": seed, "minutes": 0.0}], P["done"])
            n_reused += 1
            print(f"seed {seed}: same pairs as the stored GA2M+ model (cap {cap}) - results copied")
            continue
        if df is None:
            df, num_cols, cat_cols = _prepare(df_freq, df_sev) if df_prepared is None else df_prepared
            grid, levels = rf.make_pd_grid(df, num_cols, cat_cols)
        p2.set_all_seeds(seed)
        train_df, valid_df, test_df = _split(df, seed)
        y_tr, e_tr = train_df[pm.TARGET].to_numpy(float), train_df[pm.EXPOSURE].to_numpy(float)
        y_va, e_va = valid_df[pm.TARGET].to_numpy(float), valid_df[pm.EXPOSURE].to_numpy(float)
        y_te, e_te = test_df[pm.TARGET].to_numpy(float), test_df[pm.EXPOSURE].to_numpy(float)
        sets = (("train", train_df), ("valid", valid_df), ("test", test_df))
        model = pm.build_model("Neural GA2M", num_cols, cat_cols, seed, power, configs,
                               ebm_pairs=[p.split(" x ") for p in pairs])
        _fit_members(model, train_df, valid_df, range(n_bag))
        fit_min = (time.time() - t0) / 60
        fin_level, fin_raw = _valid_devs(model, y_tr, e_tr, train_df, y_va, e_va, valid_df, power)
        rows = _evaluate_store(model, seed, sets, y_tr, e_tr, y_va, e_va, y_te, e_te, valid_df, test_df, train_df,
                               cat_cols, grid, levels, power, n_perm, P, out_dir, 60 * fit_min, compute_pd,
                               save_predictions)
        sel_row.update(final_valid_dev_level=fin_level, final_valid_dev_raw=fin_raw, final_minutes=fit_min)
        rf._append([sel_row], P["sel"], SEL_COLUMNS)
        rf._append(rows, P["perf"])
        rf._append([{"seed": seed, "minutes": (time.time() - t0) / 60}], P["done"])
        n_fit += 1
        r0, rl = rows[0], rows[rf.VERSIONS.index("train_level")]
        print(f"seed {seed}: {len(pairs)} pairs | dev {r0['tweedie_deviance']:.4f} -> {rl['tweedie_deviance']:.4f} "
              f"(train level) | Gini {r0['gini']:.4f} | {(time.time() - t0) / 60:.1f} min")
        del model, train_df, valid_df, test_df
    print(f"done: {n_reused} seeds copied, {n_fit} seeds fitted" + (f", {n_skip} not screened yet" if n_skip else ""))
    return pd.read_csv(P["perf"]) if os.path.exists(P["perf"]) else None


def compare_runs(main_dir, runs, version="train_level", ref="Neural GA2M"):
    """
    Paired comparison of GA2M+ variants (runs = {label: directory}) with a main-run model on the common seeds:
    test deviance (given version) and Gini; mean, median, share of seeds better, Wilcoxon and Nadeau-Bengio p.
    """
    perf = pd.read_csv(os.path.join(main_dir, "perf.csv"))
    base = perf[(perf.method == ref)]
    rows = []
    for label, d in runs.items():
        p = pd.read_csv(os.path.join(d, "perf.csv"))
        p = p[p.method == METHOD]
        for metric, ver, better in (("tweedie_deviance", version, -1), ("gini", "raw", 1)):
            a = p[p.version == ver].set_index("seed")[metric]
            b = base[base.version == ver].set_index("seed")[metric]
            x = (a - b).dropna()
            J = len(x)
            if J < 3:
                continue
            se = x.std(ddof=1) * np.sqrt(1 / J + 0.2 / 0.6)
            rows.append({"run": label, "metric": f"{metric} ({ver})", "seeds": J, f"mean diff vs {ref}": x.mean(),
                         "median diff": x.median(), "share better": float(np.mean(better * x > 0)),
                         "Wilcoxon p": stats.wilcoxon(x).pvalue if np.any(x != 0) else np.nan,
                         "NB p": 2 * stats.t.sf(abs(x.mean() / se), J - 1) if se > 0 else np.nan})
    t = pd.DataFrame(rows)
    _display(t.round(4))
    return t


# =============================================================================================================
# Ablation: screening on the uncapped target (no network)
# =============================================================================================================
def run_screening_ablation(df_freq, df_sev, seeds, out_dir, configs_path, k_top=None, pi_threshold=None,
                           n_splits=None, power=None, df_prepared=None, n_jobs=4):
    """Same screening as GA2M+ with cap = None (raw pure premium): stability-selection set and full-sample FAST
    ranking per seed -> out_dir/ablation_raw_frequencies.csv. Resumable."""
    power = _power() if power is None else power
    configs = rf.load_configs(configs_path)
    s_ = dict(DEFAULT_SETTINGS)
    for k, v in (("k_top", k_top), ("pi_threshold", pi_threshold), ("n_splits", n_splits)):
        if v is not None:
            s_[k] = v
    os.makedirs(out_dir, exist_ok=True)
    _check_dir(out_dir, power)
    _check_settings(out_dir, {"k_top": s_["k_top"], "pi_threshold": s_["pi_threshold"], "n_splits": s_["n_splits"],
                              "ebm_config": configs["EBM"], "tweedie_power": power, "dataset": _dataset()})
    path = os.path.join(out_dir, "ablation_raw_frequencies.csv")
    done = set(pd.read_csv(path).seed) if os.path.exists(path) else set()
    df, num_cols, cat_cols = _prepare(df_freq, df_sev) if df_prepared is None else df_prepared
    for seed in [int(s) for s in seeds]:
        if seed in done:
            continue
        train_df, _, _ = _split(df, seed)
        pairs, freq, cap, mins = screen_pairs(train_df, num_cols, cat_cols, power, configs["EBM"], None, s_["k_top"],
                                              s_["pi_threshold"], s_["n_splits"], seed, n_jobs=n_jobs)
        rf._append(freq.assign(seed=seed, cap_level="raw", cap_value=np.inf), path)
        print(f"seed {seed}: raw target | {len(pairs)} pairs (pi >= {s_['pi_threshold']}) | {mins:.1f} min | "
              f"{', '.join(' x '.join(p) for p in pairs)}")
    return pd.read_csv(path) if os.path.exists(path) else None


# =============================================================================================================
# Summaries, merge, structure stability
# =============================================================================================================
def merge_results(main_dir, plus_dir, out_dir):
    """Writes a directory that paper2_analysis_full.run_all can read: main-run files + GA2M+ rows. Only seeds that
    are complete in BOTH runs are kept (seeds_complete.csv = intersection)."""
    os.makedirs(out_dir, exist_ok=True)
    j = os.path.join
    keep = set(pd.read_csv(j(main_dir, "seeds_complete.csv")).seed) & set(pd.read_csv(j(plus_dir, "seeds_complete.csv")).seed)
    for f in ("perf.csv", "calibration.csv", "pd_curves.csv"):
        a = pd.read_csv(j(main_dir, f))
        b = pd.read_csv(j(plus_dir, f)) if os.path.exists(j(plus_dir, f)) else a.iloc[:0]
        if list(a.columns) != list(b.columns):
            raise RuntimeError(f"{f}: column layouts differ between the two runs")
        d = pd.concat([a, b], ignore_index=True)
        d[d.seed.isin(keep)].to_csv(j(out_dir, f), index=False)
    with open(j(out_dir, "nn_history.jsonl"), "w") as out:
        for d in (main_dir, plus_dir):
            if os.path.exists(j(d, "nn_history.jsonl")):
                out.write(open(j(d, "nn_history.jsonl")).read())
    for f in ("seeds.csv", "blends.csv"):
        if os.path.exists(j(main_dir, f)):
            d = pd.read_csv(j(main_dir, f))
            d[d.seed.isin(keep)].to_csv(j(out_dir, f), index=False)
    for f in ("ebm_pairs.json", "configs_used.json", "dataset.json"):
        if os.path.exists(j(main_dir, f)):
            shutil.copy(j(main_dir, f), j(out_dir, f))
    pd.DataFrame({"seed": sorted(keep)}).to_csv(j(out_dir, "seeds_complete.csv"), index=False)
    print(f"merged into {out_dir}: {len(keep)} seeds complete in both runs")


def selection_summary(plus_dir):
    """Chosen cap, set sizes, and per cap the share of seeds whose set contains each pair (with its mean pi)."""
    sel = pd.read_csv(os.path.join(plus_dir, "ga2mplus_selection.csv"))
    fr = pd.read_csv(os.path.join(plus_dir, "ga2mplus_frequencies.csv"))
    n = sel.seed.nunique()
    print(f"{n} seeds | chosen cap: " + ", ".join(f"{q}: {c}" for q, c in sel.chosen_cap_level.value_counts()
                                                    .sort_index().items()))
    sizes = pd.DataFrame([json.loads(x) for x in sel.set_sizes])
    print("set size per cap (mean, min, max): " + "; ".join(f"{c}: {sizes[c].mean():.1f}, {sizes[c].min()}, "
                                                           f"{sizes[c].max()}" for c in sizes))
    print(f"chosen set size: mean {sel.chosen_size.mean():.1f}, min {sel.chosen_size.min()}, "
          f"max {sel.chosen_size.max()}")
    dv = pd.DataFrame([json.loads(x) for x in sel.valid_dev_level])
    if dv.shape[1] == 2:
        d = dv.iloc[:, 0] - dv.iloc[:, 1]
        same = sum(json.loads(r)[dv.columns[0]] == json.loads(r)[dv.columns[1]] for r in sel.sets)
        print(f"member-0 validation deviance, cap {dv.columns[0]} - cap {dv.columns[1]}: mean {d.mean():+.4f}, "
              f"SD {d.std():.4f}; identical sets in {same} of {n} seeds")
    if "screen_minutes" in sel:
        print("minutes per seed: screening {:.1f}, selection {:.1f}, final members {:.1f}".format(
            sel.screen_minutes.mean(), sel.get("select_minutes", pd.Series([np.nan])).mean(),
            sel.get("final_minutes", pd.Series([np.nan])).mean()))
    sets = {(r.seed, q): set(v) for r in sel.itertuples() for q, v in json.loads(r.sets).items()}
    caps = sorted({q for _, q in sets}, key=float)
    tab = {}
    for q in caps:
        tab[f"in set, cap {q}"] = pd.Series([p for (s, qq), v in sets.items() if qq == q for p in v]).value_counts() / n
        tab[f"mean pi, cap {q}"] = fr[fr.cap_level.astype(str) == str(q)].groupby("pair").pi.mean()
    tab["in chosen set"] = pd.Series([p for x in sel.chosen_pairs for p in json.loads(x)]).value_counts() / n
    out = pd.DataFrame(tab).fillna(0).sort_values("in chosen set", ascending=False)
    _display(out[(out.filter(like="in ").max(axis=1) > 0)].round(2))
    return out


def _sets_from_main(main_dir):
    raw = json.load(open(os.path.join(main_dir, "ebm_pairs.json")))
    return {int(s): {pair_key(*p) for p in v} for s, v in raw.items()}


def _sets_from_plus(plus_dir):
    d = pd.read_csv(os.path.join(plus_dir, "ga2mplus_selection.csv"))
    return {int(r.seed): set(json.loads(r.chosen_pairs)) for r in d.itertuples()}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def _set_stats(sets, splits, stress_cols=STRESS_COLS):
    seeds = sorted(sets)
    jac = [_jaccard(sets[a], sets[b]) for a, b in itertools.combinations(seeds, 2)]
    loo = []
    for s in seeds:
        others = [o for o in seeds if o != s]
        counts = pd.Series([p for o in others for p in sets[o]]).value_counts()
        cons = set(counts[counts >= 0.5 * len(others)].index)
        loo.append(_jaccard(sets[s], cons))
    loo = pd.Series(loo, index=seeds)
    allp = set().union(*sets.values())
    out = {"seeds": len(seeds), "mean set size": np.mean([len(sets[s]) for s in seeds]),
           "mean pairwise Jaccard": np.mean(jac), "min pairwise Jaccard": np.min(jac),
           "pairs ever selected": len(allp),
           "pairs always selected": len(set.intersection(*sets.values())),
           "pairs in >= 90% of seeds": int((pd.Series([p for s in seeds for p in sets[s]]).value_counts()
                                           >= 0.9 * len(seeds)).sum()),
           "mean Jaccard to leave-one-out consensus": loo.mean()}
    if splits is not None:
        for c in stress_cols:
            if c in splits:
                r = stats.spearmanr(loo.values, splits.set_index("seed").reindex(seeds)[c].values)
                out[f"Spearman(consensus Jaccard, {c})"] = r.statistic
                out[f"p ({c})"] = r.pvalue
    return out, loo


def structure_stability(main_dir, plus_dir, stress_cols=STRESS_COLS):
    """
    Pair sets of the main-run GA2M (= EBM pairs) and of GA2M+ across seeds: size, pairwise Jaccard, pairs ever /
    always selected, agreement with the leave-one-out consensus (pairs selected in >= 50% of the other seeds) and its
    Spearman correlation with training-tail statistics; Wilcoxon test of the per-seed consensus agreement.
    """
    a, b = _sets_from_main(main_dir), _sets_from_plus(plus_dir)
    common = sorted(set(a) & set(b))
    a, b = {s: a[s] for s in common}, {s: b[s] for s in common}
    seeds_csv = os.path.join(main_dir, "seeds.csv")
    splits = pd.read_csv(seeds_csv) if os.path.exists(seeds_csv) else None
    sa, la = _set_stats(a, splits, stress_cols)
    sb, lb = _set_stats(b, splits, stress_cols)
    table = pd.DataFrame({"Neural GA2M (EBM pairs)": sa, "Neural GA2M+": sb})
    d = (lb - la).dropna()
    try:
        p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
    except ValueError:
        p_w = np.nan
    freq = pd.DataFrame({"GA2M": pd.Series([p for s in common for p in a[s]]).value_counts() / len(common),
                         "GA2M+": pd.Series([p for s in common for p in b[s]]).value_counts() / len(common)}).fillna(0)
    freq = freq.sort_values(["GA2M+", "GA2M"], ascending=False)
    _display(table.round(3))
    print(f"per-seed agreement with the consensus, GA2M+ - GA2M: mean {d.mean():+.3f}, "
          f"better in {(d > 0).mean():.0%} of {len(d)} seeds, Wilcoxon p = {p_w:.3g}")
    _display(freq.round(2))
    return {"table": table, "frequency": freq, "agreement_diff": d}


def ablation_report(main_dir, plus_dir, ablation_dir=None, k_top=None, pi_threshold=None, stress_cols=STRESS_COLS):
    """
    Structure stability of every selection rule on the same seeds:
      EBM pairs (main-run GA2M) | raw single FAST top-k | raw stability | cap q single top-k | cap q stability |
      GA2M+ (chosen) | raw single FAST, size-matched to GA2M+ (top-|S| of the raw ranking, per seed).
    Raw rules need run_screening_ablation(); they are skipped if ablation_dir is None.
    Capping effect = (cap q, single) vs (raw, single); stability-selection effect = (stability) vs (single) at the
    same target; the size-matched row checks that GA2M+'s gain is not just a smaller set.
    """
    k_top = DEFAULT_SETTINGS["k_top"] if k_top is None else k_top
    pi_thr = DEFAULT_SETTINGS["pi_threshold"] if pi_threshold is None else pi_threshold
    fr = pd.read_csv(os.path.join(plus_dir, "ga2mplus_frequencies.csv"))
    fr["cap_level"] = fr.cap_level.astype(str)
    if ablation_dir is not None and os.path.exists(os.path.join(ablation_dir, "ablation_raw_frequencies.csv")):
        ab = pd.read_csv(os.path.join(ablation_dir, "ablation_raw_frequencies.csv"))
        ab["cap_level"] = "raw"
        fr = pd.concat([fr, ab], ignore_index=True)
    plus = _sets_from_plus(plus_dir)
    main = _sets_from_main(main_dir)
    common = sorted(set(plus) & set(main) & set(fr.seed))
    if ablation_dir is not None and "raw" in set(fr.cap_level):
        common = sorted(set(common) & set(fr[fr.cap_level == "raw"].seed))

    def _stab(level):
        g = fr[fr.cap_level == level]
        out = {}
        for s, d in g.groupby("seed"):
            sel = set(d[d.pi >= pi_thr].pair)
            out[int(s)] = sel if sel else set(d[d.pi == d.pi.max()].pair)
        return out

    def _single(level, size=None):
        g = fr[fr.cap_level == level]
        return {int(s): set(d.nsmallest(size(int(s)) if size else k_top, "full_rank").pair) for s, d in g.groupby("seed")
                if size is None or int(s) in plus}

    rules = {"EBM pairs (main GA2M)": main}
    levels = sorted([c for c in fr.cap_level.unique() if c != "raw"], key=float)
    if "raw" in set(fr.cap_level):
        rules[f"raw, single FAST top-{k_top}"] = _single("raw")
        rules["raw, stability selection"] = _stab("raw")
    def _meanrank(level):
        g = fr[fr.cap_level == level]
        return {int(s): set(d.nsmallest(k_top, "mean_rank").pair) for s, d in g.groupby("seed")}

    for q in levels:
        rules[f"cap {q}, single FAST top-{k_top}"] = _single(q)
        rules[f"cap {q}, stability selection"] = _stab(q)
        rules[f"cap {q}, half-sample mean-rank top-{k_top}"] = _meanrank(q)
    rules["GA2M+ (chosen cap)"] = plus
    if "raw" in set(fr.cap_level):
        rules["raw single FAST, size-matched to GA2M+"] = _single("raw", size=lambda s: len(plus[s]))
    seeds_csv = os.path.join(main_dir, "seeds.csv")
    splits = pd.read_csv(seeds_csv) if os.path.exists(seeds_csv) else None
    res, loos = {}, {}
    for name, sets in rules.items():
        sets = {s: sets[s] for s in common if s in sets}
        res[name], loos[name] = _set_stats(sets, splits, stress_cols)
    table = pd.DataFrame(res).T
    _display(table.round(3))
    if "raw single FAST, size-matched to GA2M+" in loos:
        d = (loos["GA2M+ (chosen cap)"] - loos["raw single FAST, size-matched to GA2M+"]).dropna()
        try:
            p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
        except ValueError:
            p_w = np.nan
        print(f"GA2M+ vs size-matched raw ranking, consensus agreement: mean diff {d.mean():+.3f}, "
              f"better in {(d > 0).mean():.0%} of {len(d)} seeds, Wilcoxon p = {p_w:.3g}")
    return table


# =============================================================================================================
# Smoke test
# =============================================================================================================
def smoke_test(df_freq, df_sev, configs_path, n_sample=60000, seed=49, out_dir="ga2mplus_smoke", n_jobs=2):
    """Every code path on a subsample: 2 halvings, GA2M with 2 epochs, both caps, ablation; then a resume."""
    import copy
    checks = {}
    cfg = json.load(open(configs_path))
    small = copy.deepcopy(cfg)
    small["Neural GA2M"]["train"]["max_epochs"] = 2
    small["Neural GA2M"]["arch"]["n_bag"] = min(2, int(small["Neural GA2M"]["arch"]["n_bag"]))
    small_path = os.path.join(os.path.dirname(os.path.abspath(configs_path)), "_ga2mplus_smoke_configs.json")
    json.dump(small, open(small_path, "w"))
    df, num_cols, cat_cols = _prepare(df_freq, df_sev)
    df = df.sample(n=min(n_sample, len(df)), random_state=0).reset_index(drop=True)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    t0 = time.time()
    try:
        perf = run_ga2m_plus(None, None, [seed], out_dir, small_path, n_splits=2, n_perm=19,
                             df_prepared=(df, num_cols, cat_cols), n_jobs=n_jobs)
        checks["minutes"] = round((time.time() - t0) / 60, 1)
        checks[f"rows = {len(rf.VERSIONS)} versions"] = len(perf) == len(rf.VERSIONS)
        checks["all finite"] = bool(np.isfinite(perf[["tweedie_deviance", "gini", "balance_error_pct"]]).all().all())
        sel = pd.read_csv(os.path.join(out_dir, "ga2mplus_selection.csv"))
        fr = pd.read_csv(os.path.join(out_dir, "ga2mplus_frequencies.csv"))
        n_pairs = len(num_cols + cat_cols) * (len(num_cols + cat_cols) - 1) // 2
        checks[f"frequencies: {n_pairs} pairs x 2 caps"] = len(fr) == 2 * n_pairs
        checks["sum of pi per cap = k_top"] = bool(np.allclose(fr.groupby("cap_level").pi.sum(),
                                                               DEFAULT_SETTINGS["k_top"]))
        checks["full-sample ranks 1..n per cap"] = bool(all(sorted(g.full_rank) == list(range(1, n_pairs + 1))
                                                            for _, g in fr.groupby("cap_level")))
        checks["chosen cap in caps"] = float(sel.chosen_cap_level.iloc[0]) in DEFAULT_SETTINGS["caps"]
        n_rows = len(pd.read_csv(os.path.join(out_dir, "perf.csv")))
        run_ga2m_plus(None, None, [seed], out_dir, small_path, n_splits=2, n_perm=19,
                      df_prepared=(df, num_cols, cat_cols), n_jobs=n_jobs)
        checks["resume does not refit"] = len(pd.read_csv(os.path.join(out_dir, "perf.csv"))) == n_rows
        z = np.load(prediction_path(out_dir, METHOD, seed))
        checks["predictions saved (valid/test + training sum + split hashes)"] = (
            len(z["valid_pred"]) + len(z["test_pred"]) + int(z["train_n"]) == len(df)
            and np.isfinite(float(z["train_sum_e_mu"])) and len(str(z["test_idhash"])) == 40)
        try:
            run_ga2m_plus(None, None, [seed], out_dir, small_path, n_splits=3, df_prepared=(df, num_cols, cat_cols))
            checks["changed settings refused"] = False
        except RuntimeError:
            checks["changed settings refused"] = True
        ab = run_screening_ablation(None, None, [seed], os.path.join(out_dir, "ablation"), small_path, n_splits=2,
                                    df_prepared=(df, num_cols, cat_cols), n_jobs=n_jobs)
        checks["ablation (raw target) written"] = ab is not None and len(ab) == n_pairs
    finally:
        os.remove(small_path)
    for k, v in checks.items():
        print(f"{k}: {v}")
    ok = all(v for k, v in checks.items() if k != "minutes")
    print("\nGA2M+ SMOKE TEST PASSED" if ok else "\nGA2M+ SMOKE TEST FAILED - see the values above")
    return checks
