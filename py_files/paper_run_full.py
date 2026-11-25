

import gc
import json
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd

import paper_adjust_clean_plus as p2
import paper_datasets as dsm


def check_dependencies():
    """Fails at import time (not hours into a run) if a helper module lacks a required function."""
    problems = [f"paper_adjust_clean_plus.py has no {f}" for f in ("tweedie_deviance", "set_all_seeds")
                if not hasattr(p2, f)]
    if "seg_cols" not in dsm.spec():
        problems.append("paper_datasets.spec() has no 'seg_cols'")
    if problems:
        raise ImportError("; ".join(problems) + ". Replace the file(s) and restart the kernel.")


check_dependencies()


TARGET, EXPOSURE = "PurePremium", "Exposure"
SEG_COLS = ["Region", "Area", "VehGas", "VehBrand"]


def seg_cols():
    return list(dsm.spec()["seg_cols"])


MODEL_ORDER = ["GLM", "Tweedie GLM", "EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Tweedie-CANN"]
CONFIGURED = ["EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Tweedie-CANN"]
VERSIONS = ["raw", "train_level"]
BLEND_PAIRS = [("Neural GA2M", "EBM"), ("Neural GA2M", "GBM"), ("EBM", "GBM"),
               ("LocalGLMnet", "Tweedie GLM"), ("Tweedie-CANN", "Tweedie GLM"), ("Neural GA2M", "LocalGLMnet")]
PD_N_SAMPLE, PD_N_GRID = 2000, 25


# =============================================================================================================
# Metrics
# =============================================================================================================
def gini_exposure(y, pred, e):
    order = np.argsort(pred, kind="mergesort")
    ce = np.concatenate([[0.0], np.cumsum(e[order])])
    cl = np.concatenate([[0.0], np.cumsum((e * y)[order])])
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(1.0 - 2.0 * trap(cl / cl[-1], ce / ce[-1]))


def balance(y, pred, e):
    return float(np.sum(e * pred) / np.sum(e * y) - 1.0)


def segment_abs_balance(df, y, pred, e, col):
    """Exposure-weighted mean of |P_l / O_l - 1| over the levels l of col with positive observed loss."""
    g = pd.DataFrame({"l": df[col].astype(str).to_numpy(), "o": e * y, "p": e * pred, "e": e}).groupby("l").sum()
    g = g[g.o > 0]
    return float(np.sum(g.e * np.abs(g.p / g.o - 1)) / g.e.sum())


def test_metrics(df, y, pred, e, power):
    obs, prd = float(np.sum(e * y)), float(np.sum(e * pred))
    out = {"observed_total": obs, "predicted_total": prd, "balance_error_pct": prd / obs - 1.0,
           "abs_balance_error_pct": abs(prd / obs - 1.0),
           "tweedie_deviance": float(p2.tweedie_deviance(y, pred, e, power=power)),
           "gini": gini_exposure(y, pred, e),
           "weighted_mae": float(np.average(np.abs(y - pred), weights=e)),
           "weighted_rmse": float(np.sqrt(np.average((y - pred) ** 2, weights=e)))}
    for c in seg_cols():
        out[f"seg_absB_{c}"] = segment_abs_balance(df, y, pred, e, c)
    return out


def clean(pred):
    return np.clip(np.nan_to_num(np.asarray(pred, float), nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6, 1e6)


# =============================================================================================================
# Poisson quasi-likelihood GLMs (numpy IRLS)
# =============================================================================================================
def poisson_irls(X, y, e, offset, beta0, max_iter=50, tol=1e-9):
    """Score equations sum_i e_i (y_i - mu_i) x_i = 0, mu = exp(offset + X beta); step halving on the deviance."""
    from scipy.special import xlogy

    def dev(m):
        return 2.0 * np.sum(e * (xlogy(y, y) - xlogy(y, m) - (y - m)))

    beta = np.asarray(beta0, float).copy()
    eta = np.clip(offset + X @ beta, -30, 30)
    d = dev(np.exp(eta))
    converged, it = False, 0
    for it in range(1, max_iter + 1):
        m = np.exp(eta)
        w = e * m
        A = X.T @ (X * w[:, None])
        b = X.T @ (w * (eta - offset) + e * (y - m))
        step = np.linalg.solve(A, b) - beta
        for _ in range(30):
            cand = beta + step
            eta_c = np.clip(offset + X @ cand, -30, 30)
            d_c = dev(np.exp(eta_c))
            if d_c <= d * (1 + 1e-12):
                break
            step /= 2
        beta, eta, d = cand, eta_c, d_c
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    return beta, converged, it


def calibration_slope(y, mu, e):
    """Diagnostic only: Poisson quasi-likelihood GLM log E[y] = a + b log mu on the given data; returns b (1 = calibrated
    spread, < 1 predictions too spread out, > 1 too compressed)."""
    lm = np.log(np.clip(np.asarray(mu, float), 1e-10, None))
    m0, s0 = np.average(lm, weights=e), np.sqrt(np.average((lm - np.average(lm, weights=e)) ** 2, weights=e))
    X = np.column_stack([np.ones(len(lm)), (lm - m0) / s0])
    b0 = np.array([m0 + np.log(np.sum(e * y) / np.sum(e * mu)), s0])
    beta, conv, _ = poisson_irls(X, np.asarray(y, float), np.asarray(e, float), np.zeros(len(lm)), b0)
    return float(beta[1] / s0) if conv else np.nan


# =============================================================================================================
# Partial dependence of log mu (shape stability across splits)
# =============================================================================================================
def make_pd_grid(df, num_cols, cat_cols, n_grid=PD_N_GRID):
    """Grid from the covariates of the full portfolio (no target information); identical for every seed.
    Density is on the log scale (the model input)."""
    grid = {}
    for c in num_cols:
        x = df[c].astype(float)
        v = np.unique(np.quantile(x, np.linspace(0.01, 0.99, n_grid)))
        if np.allclose(x, np.round(x)):
            v = np.unique(np.round(v))
        grid[c] = v
    levels = {c: sorted(df[c].astype(str).unique()) for c in cat_cols}
    return grid, levels


def _anchors(train_df, cat_cols):
    """One training row per level of every factor, appended to every prediction frame so that models that build dummies
    with pandas.get_dummies(drop_first=True) see all levels (a frame with one level would be coded as the reference)."""
    rows = [train_df[train_df[c].astype(str) == l].iloc[[0]] for c in cat_cols
            for l in sorted(train_df[c].astype(str).unique())]
    return pd.concat(rows, ignore_index=True)


def _predict_framed(pm, model, frame, anchors):
    full = pd.concat([frame, anchors], ignore_index=True)
    return pm.predict(model, full)[:len(frame)]


def pd_curves(pm, model, train_df, grid, levels, anchors, n=PD_N_SAMPLE, seed=0):
    samp = train_df.sample(n=min(n, len(train_df)), random_state=int(seed)).reset_index(drop=True)
    out = []
    for f, g in grid.items():
        big = pd.concat([samp.assign(**{f: v}) for v in g], ignore_index=True)
        lv = np.log(_predict_framed(pm, model, big, anchors)).reshape(len(g), len(samp)).mean(1)
        out += [{"feature": f, "kind": "num", "x": str(v), "x_num": float(v), "pd_log_mu": float(p)}
                for v, p in zip(g, lv)]
    for f, levs in levels.items():
        big = pd.concat([samp.assign(**{f: l}) for l in levs], ignore_index=True)
        lv = np.log(_predict_framed(pm, model, big, anchors)).reshape(len(levs), len(samp)).mean(1)
        out += [{"feature": f, "kind": "cat", "x": str(l), "x_num": np.nan, "pd_log_mu": float(p)}
                for l, p in zip(levs, lv)]
    return pd.DataFrame(out)


# =============================================================================================================
# Split statistics, blends
# =============================================================================================================
def split_statistics(seed, train_df, valid_df, test_df, full_df):
    pos = full_df.loc[full_df["ClaimAmount"] > 0, "ClaimAmount"]
    large = float(pos.quantile(0.999))
    out = {"seed": int(seed), "large_claim_threshold": large}
    for pre, d in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        loss, e = d["ClaimAmount"].to_numpy(float), d["Exposure"].to_numpy(float)
        tot = loss.sum()
        out.update({f"E_{pre}": e.sum(), f"O_{pre}": tot, f"pp_{pre}": tot / e.sum(),
                    f"{pre}_top1_share": loss[loss >= np.quantile(loss, 0.99)].sum() / tot,
                    f"{pre}_top01_share": loss[loss >= np.quantile(loss, 0.999)].sum() / tot,
                    f"{pre}_max_loss_share": loss.max() / tot, f"{pre}_n_large": int(np.sum(loss >= large)),
                    f"{pre}_claims": float(d["ClaimNb"].sum())})
    out["pp_full"] = full_df["ClaimAmount"].sum() / full_df["Exposure"].sum()
    out["loss_imbalance_score"] = abs(out["pp_valid"] / out["pp_train"] - 1) + abs(out["pp_test"] / out["pp_train"] - 1)
    return out


def blends_for_seed(seed, tmp_dir, valid_df, test_df, power, pairs=BLEND_PAIRS, grid=np.linspace(0, 1, 11)):
    yv, ev = valid_df[TARGET].to_numpy(float), valid_df[EXPOSURE].to_numpy(float)
    yt, et = test_df[TARGET].to_numpy(float), test_df[EXPOSURE].to_numpy(float)
    rows = []
    for a, b in pairs:
        fa, fb = (os.path.join(tmp_dir, f"{m.replace(' ', '_')}.npz") for m in (a, b))
        if not (os.path.exists(fa) and os.path.exists(fb)):
            warnings.warn(f"seed {seed}: predictions of {a} or {b} missing; blend skipped")
            continue
        A, B = np.load(fa), np.load(fb)
        va, vb = A["valid_raw"].astype(float), B["valid_raw"].astype(float)
        ta, tb = A["test_raw"].astype(float), B["test_raw"].astype(float)
        vdev = [p2.tweedie_deviance(yv, w * va + (1 - w) * vb, ev, power=power) for w in grid]
        w_star = float(grid[int(np.argmin(vdev))])
        for label, w in (("A", 1.0), ("B", 0.0), ("50/50", 0.5), ("valid_w", w_star)):
            pred = w * ta + (1 - w) * tb
            rows.append({"seed": int(seed), "A": a, "B": b, "blend": label, "weight_A": w,
                         "tweedie_deviance": float(p2.tweedie_deviance(yt, pred, et, power=power)),
                         "gini": gini_exposure(yt, pred, et), "balance_error_pct": balance(yt, pred, et),
                         "abs_balance_error_pct": abs(balance(yt, pred, et))})
    return pd.DataFrame(rows)


# =============================================================================================================
# Calibrated versions
# =============================================================================================================
def calibrated_versions(raw, sets, y_tr, e_tr, valid_df, y_va, n_perm=None, seed=None):
    """raw: {"train","valid","test"} raw predictions; sets: (("train", df), ("valid", df), ("test", df)).
    Versions: raw and train_level (global rescaling on the training set). n_perm and seed are accepted for the callers'
    signature and not used."""
    preds = {"raw": raw}
    c = float(np.sum(e_tr * y_tr) / np.sum(e_tr * raw["train"]))
    preds["train_level"] = {s: clean(c * raw[s]) for s, _ in sets}
    info = {"train_factor": c,
            "valid_calibration_slope": calibration_slope(y_va, raw["valid"], valid_df[EXPOSURE].to_numpy(float))}
    return preds, info


# =============================================================================================================
# Runner
# =============================================================================================================
CAL_COLUMNS = ["seed", "method", "train_factor", "valid_calibration_slope", "share_grad_norm_over_5", "ebm_max_rounds",
               "ebm_best_iteration_max", "ebm_best_iteration_median", "ebm_share_bags_at_cap"]


def _append(rows, path, columns=None):
    """Appends rows; the column layout is fixed by the first write (or by `columns`) and may never change."""
    if rows is None or not len(rows):
        return
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if os.path.exists(path):
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if columns is not None and header != list(columns):
            raise RuntimeError(f"{path} has a different column layout; use a new out_dir")
        columns = header
    if columns is not None:
        unknown = set(df.columns) - set(columns)
        if unknown:
            raise KeyError(f"unexpected columns {sorted(unknown)} for {path}")
        df = df.reindex(columns=columns)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def load_configs(configs_path):
    if configs_path is None or not os.path.exists(configs_path):
        raise FileNotFoundError("configs_path is required (tuning/selected_configs.json from paper_tuning.py)")
    cfg = json.load(open(configs_path))
    missing = [m for m in CONFIGURED if m not in cfg]
    if missing:
        raise KeyError(f"{configs_path}: configuration missing for {missing}")
    return {m: cfg[m] for m in CONFIGURED}


def _check_configs(out_dir, configs):
    path = os.path.join(out_dir, "configs_used.json")
    canon = json.dumps(configs, sort_keys=True)
    if os.path.exists(path):
        if json.dumps(json.load(open(path)), sort_keys=True) != canon:
            raise RuntimeError(f"{out_dir} was started with a different configuration ({path}); use a new out_dir")
    else:
        json.dump(configs, open(path, "w"), indent=1, sort_keys=True)


def run_full(df_freq, df_sev, seeds=range(41, 91), out_dir="v2_full", configs_path=None, model_names=None,
             power=None, n_perm=199, df_prepared=None, compute_pd=True, verbose=False):
    import paper_models as pm
    power = dsm.power() if power is None else power
    configs = load_configs(configs_path)
    names = MODEL_ORDER if model_names is None else [m for m in MODEL_ORDER if m in model_names]
    os.makedirs(out_dir, exist_ok=True)
    meta_p = json.load(open(configs_path)).get("_meta", {}).get("tweedie_power")
    if meta_p is not None and abs(float(meta_p) - power) > 1e-12:
        raise RuntimeError(f"{configs_path} was tuned with p = {meta_p}, this run uses p = {power}")
    dsm.check_run_dir(out_dir, power)
    _check_configs(out_dir, configs)
    P = {k: os.path.join(out_dir, f) for k, f in (("perf", "perf.csv"), ("cal", "calibration.csv"),
                                                  ("seeds", "seeds.csv"), ("blends", "blends.csv"),
                                                  ("pd", "pd_curves.csv"), ("pairs", "ebm_pairs.json"),
                                                  ("hist", "nn_history.jsonl"), ("done", "seeds_complete.csv"))}
    done_models = set()
    if os.path.exists(P["perf"]):
        d = pd.read_csv(P["perf"], usecols=["seed", "method"])
        done_models = set(zip(d.seed.astype(int), d.method))
    complete = set(pd.read_csv(P["done"]).seed) if os.path.exists(P["done"]) else set()
    seeds_logged = set(pd.read_csv(P["seeds"]).seed) if os.path.exists(P["seeds"]) else set()
    pairs_cache = json.load(open(P["pairs"])) if os.path.exists(P["pairs"]) else {}

    df, num_cols, cat_cols = dsm.prepare(df_freq, df_sev) if df_prepared is None else df_prepared
    grid, levels = make_pd_grid(df, num_cols, cat_cols)
    print("configurations:")
    for m in CONFIGURED:
        print(f"  {m:13s} {json.dumps(configs[m], sort_keys=True)}")
    t_run = time.time()

    for seed in [int(s) for s in seeds]:
        if seed in complete:
            continue
        todo = [m for m in names if (seed, m) not in done_models]
        t_seed = time.time()
        print(f"\n{'=' * 100}\nseed {seed}: {', '.join(todo) if todo else 'all models done; blends only'}\n{'=' * 100}")
        p2.set_all_seeds(seed)
        train_df, valid_df, test_df = dsm.split(df, seed)
        if seed not in seeds_logged:
            _append([split_statistics(seed, train_df, valid_df, test_df, df)], P["seeds"])
        y_tr, e_tr = train_df[TARGET].to_numpy(float), train_df[EXPOSURE].to_numpy(float)
        y_va, e_va = valid_df[TARGET].to_numpy(float), valid_df[EXPOSURE].to_numpy(float)
        y_te, e_te = test_df[TARGET].to_numpy(float), test_df[EXPOSURE].to_numpy(float)
        sets = (("train", train_df), ("valid", valid_df), ("test", test_df))
        tmp = os.path.join(out_dir, "_tmp", f"seed{seed}")
        os.makedirs(tmp, exist_ok=True)
        anchors = _anchors(train_df, cat_cols)
        tweedie_glm = None

        for name in todo:
            ebm_pairs = None
            if name == "Neural GA2M":
                if str(seed) not in pairs_cache:
                    print("  fitting EBM (selected configuration) to obtain its interaction pairs")
                    e_ = pm.fit_model("EBM", pm.build_model("EBM", num_cols, cat_cols, seed, power, configs),
                                      train_df, valid_df)
                    pairs_cache[str(seed)] = [list(p) for p in e_.interaction_pairs_]
                    json.dump(pairs_cache, open(P["pairs"], "w"), indent=1)
                    del e_
                ebm_pairs = pairs_cache[str(seed)]
            glm_for_cann = tweedie_glm if name == "Tweedie-CANN" else None
            model = pm.build_model(name, num_cols, cat_cols, seed, power, configs, ebm_pairs=ebm_pairs,
                                   tweedie_glm=glm_for_cann, verbose=verbose)
            t0 = time.time()
            pm.fit_model(name, model, train_df, valid_df)
            secs = time.time() - t0
            if name == "Tweedie GLM":
                tweedie_glm = model                          # the same fitted GLM is CANN's offset
            if name == "EBM":
                pairs_cache[str(seed)] = [list(p) for p in model.interaction_pairs_]
                json.dump(pairs_cache, open(P["pairs"], "w"), indent=1)

            raw = {s: pm.predict(model, d) for s, d in sets}
            np.savez_compressed(os.path.join(tmp, f"{name.replace(' ', '_')}.npz"),
                                valid_raw=raw["valid"].astype(np.float32), test_raw=raw["test"].astype(np.float32))
            preds, cal_info = calibrated_versions(raw, sets, y_tr, e_tr, valid_df, y_va, n_perm, seed)

            info = getattr(model, "train_info_", {}) or {}
            rows = []
            for version in VERSIONS:
                p = preds[version]
                rows.append({"seed": seed, "method": name, "version": version,
                             **test_metrics(test_df, y_te, p["test"], e_te, power),
                             "valid_balance_error_pct": balance(y_va, p["valid"], e_va),
                             "train_balance_error_pct": balance(y_tr, p["train"], e_tr),
                             "train_observed_total": float(np.sum(e_tr * y_tr)),
                             "train_predicted_total": float(np.sum(e_tr * p["train"])),
                             "fit_seconds": secs, "best_iteration": getattr(model, "best_iteration_", np.nan),
                             "epochs_run": info.get("epochs_run", np.nan), "stopped_by": info.get("stopped_by", "")})
            pdc = pd_curves(pm, model, train_df, grid, levels, anchors, seed=seed).assign(seed=seed, method=name) \
                if compute_pd else None
            ebm_info = getattr(model, "ebm_rounds_", None) or {}
            _append([{"seed": seed, "method": name, **cal_info,
                      "share_grad_norm_over_5": info.get("share_grad_norm_over_5", np.nan),
                      **{f"ebm_{k}": v for k, v in ebm_info.items()}}], P["cal"], CAL_COLUMNS)
            _append(pdc, P["pd"])
            if info.get("history"):
                with open(P["hist"], "a") as fh:
                    fh.write(json.dumps({"seed": seed, "method": name, "history": info["history"]}) + "\n")
            _append(rows, P["perf"])                         # perf last: it marks the model as done

            r0, rl = rows[0], rows[VERSIONS.index("train_level")]
            print(f"  {name:13s} | {secs / 60:5.1f} min | dev {r0['tweedie_deviance']:.4f} -> {rl['tweedie_deviance']:.4f}"
                  f" (train level) | Gini {r0['gini']:.4f} | B raw {100 * r0['balance_error_pct']:+6.1f}% train-level "
                  f"{100 * rl['balance_error_pct']:+6.1f}% | train B raw {100 * r0['train_balance_error_pct']:+6.1f}%")
            if name != "Tweedie GLM":
                del model
            del raw, preds, rows, pdc
            gc.collect()

        _append(blends_for_seed(seed, tmp, valid_df, test_df, power), P["blends"])
        _append([{"seed": seed, "minutes": (time.time() - t_seed) / 60}], P["done"])
        shutil.rmtree(tmp, ignore_errors=True)
        del train_df, valid_df, test_df, anchors, tweedie_glm
        gc.collect()
        print(f"seed {seed} complete in {(time.time() - t_seed) / 60:.1f} min")
    shutil.rmtree(os.path.join(out_dir, "_tmp"), ignore_errors=True)
    print(f"\nrun finished in {(time.time() - t_run) / 3600:.1f} h")
    return pd.read_csv(P["perf"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="50-seed run of all models")
    ap.add_argument("--freq", required=True)
    ap.add_argument("--sev", required=True)
    ap.add_argument("--configs", required=True, help="tuning/selected_configs.json")
    ap.add_argument("--out", default="v2_full")
    ap.add_argument("--seeds", default="41-90", help="e.g. 41-90 or 49,50,57")
    a = ap.parse_args()
    if "-" in a.seeds:
        lo, hi = map(int, a.seeds.split("-"))
        seeds = range(lo, hi + 1)
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
    run_full(pd.read_csv(a.freq), pd.read_csv(a.sev), seeds=seeds, out_dir=a.out, configs_path=a.configs)
