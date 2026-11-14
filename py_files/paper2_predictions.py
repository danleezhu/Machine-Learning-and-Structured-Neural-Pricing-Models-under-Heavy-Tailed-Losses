"""
paper2_predictions.py — refits the models of the 50-seed run and KEEPS their raw predictions, so that any calibration
study (other cells, numerical bands, other credibility constants k, other clip bands) can be computed afterwards
without fitting anything again (paper2_calibration_study.py).

Reproduction of the main run
    Same data preparation, same split (seed), same configurations (tuning/selected_configs.json, checked against
    v2_full/configs_used.json), same model seeds, the Neural GA2M uses the EBM pairs stored by the main run
    (v2_full/ebm_pairs.json, in the stored order) and Tweedie-CANN the Tweedie GLM of the same split. Every fit is
    checked against the main run: raw test deviance, Gini and balance must coincide with v2_full/perf.csv
    (verification.csv; "identical" < 1e-6 relative, "close" < 1e-3, otherwise "DIFFERENT").

Files (out_dir)
    <Model_name>/seed<s>.npz      valid_pred, test_pred (float32, lossless for the study: relative rounding 6e-8),
                                  train_sum_e_mu (float64: the only training quantity the study needs, for the
                                  training-level factor c), <set>_idhash (SHA-1 of the IDpol of every set: the split
                                  is regenerated and checked, so no policy ids are stored), train_pred only if
                                  keep_train=True
    verification.csv              one row per (seed, model)
    Size: about 1 MB per model and seed (keep_train=True: about 2.4 MB); 7 models x 50 seeds about 0.35 GB.
    float32 values are almost incompressible (zlib saves about 10%, byte-shuffling another 10%); the real savings are
    not storing policy ids and training predictions. A lossy float16 / log-quantised store would break the exact
    reproduction of the main run (relative errors of about 1e-3) and is not used.

    import paper2_predictions as pp
    pp.archive_predictions(df_freq, df_sev, seeds=range(41, 91), out_dir="v2_predictions", main_dir="v2_full",
                           configs_path="tuning/selected_configs.json")
    d = pp.load_predictions(["v2_predictions", "v2_ga2mplus_c995_meanrank/predictions"], "EBM", 49)
"""

import gc
import json
import os
import time

import numpy as np
import pandas as pd

import paper2_adjust_clean_plus as p2
import paper2_datasets as dsm
import paper2_models as pm
import paper2_run_full as rf

MODELS = ["GLM", "Tweedie GLM", "EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Tweedie-CANN"]
SETS = ("train", "valid", "test")


# ---- data-set hooks (same as the main run) -----------------------------------------------------------------
def power():
    return dsm.power()


def prepare(df_freq, df_sev):
    return dsm.prepare(df_freq, df_sev)


def split(df, seed):
    return dsm.split(df, seed)


def folder(model):
    return model.replace(" ", "_")


def model_path(out_dir, model, seed):
    return os.path.join(out_dir, folder(model), f"seed{int(seed)}.npz")


def id_hash(ids):
    import hashlib
    return hashlib.sha1(np.ascontiguousarray(np.asarray(ids, np.int64)).tobytes()).hexdigest()


def save_file(path, raw, sets, keep_train=False):
    """raw: {"train","valid","test"} -> raw predictions (float64); sets: (("train", df), ("valid", df), ("test", df))."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = dict(sets)
    tr = frames["train"]
    arrays = {"valid_pred": np.asarray(raw["valid"], np.float32), "test_pred": np.asarray(raw["test"], np.float32),
              "train_sum_e_mu": np.float64(np.sum(tr[pm.EXPOSURE].to_numpy(float) * np.asarray(raw["train"], float)))}
    if keep_train:
        arrays["train_pred"] = np.asarray(raw["train"], np.float32)
    for name, d in sets:
        arrays[f"{name}_idhash"] = np.array(id_hash(d["IDpol"].to_numpy(np.int64)))
        arrays[f"{name}_n"] = np.int64(len(d))
    np.savez_compressed(path, **arrays)


def load_predictions(pred_dirs, model, seed):
    """
    Raw predictions of `model` for `seed` from the first directory that has them. pred_dirs: a directory or a list
    (e.g. ["v2_predictions", "v2_ga2mplus_c995_meanrank/predictions"]). Returns {"valid": array, "test": array,
    "train": array (only if stored), "train_sum_e_mu": float, "idhash": {set: sha1}, "n": {set: n}} (float64
    predictions).
    """
    for d in ([pred_dirs] if isinstance(pred_dirs, str) else pred_dirs):
        path = model_path(d, model, seed)
        if not os.path.exists(path):
            continue
        z = np.load(path)
        out = {s: z[f"{s}_pred"].astype(float) for s in SETS if f"{s}_pred" in z.files}
        out["train_sum_e_mu"] = float(z["train_sum_e_mu"])
        out["idhash"] = {s: str(z[f"{s}_idhash"]) for s in SETS}
        out["n"] = {s: int(z[f"{s}_n"]) for s in SETS}
        return out
    raise FileNotFoundError(f"no predictions of {model} for seed {seed} in {pred_dirs}")


def available(pred_dirs, model, seeds):
    dirs = [pred_dirs] if isinstance(pred_dirs, str) else pred_dirs
    return [int(s) for s in seeds if any(os.path.exists(model_path(d, model, s)) for d in dirs)]


# ---- run ----------------------------------------------------------------------------------------------------
def _status(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return "no reference"
    r = abs(a - b) / max(abs(b), 1e-12)
    return "identical" if r < 1e-6 else ("close" if r < 1e-3 else "DIFFERENT")


def archive_predictions(df_freq, df_sev, seeds=range(41, 91), out_dir="v2_predictions", main_dir="v2_full",
                        configs_path="tuning/selected_configs.json", models=None, df_prepared=None, verbose=False,
                        keep_train=False):
    """Resumable: a (model, seed) whose file exists is skipped. models: subset of MODELS (default all).
    keep_train=True also stores the training predictions (2.4x the space; only needed for calibrations FITTED on the
    training set, which the study does not use)."""
    p = power()
    configs = rf.load_configs(configs_path)
    used = os.path.join(main_dir, "configs_used.json")
    if os.path.exists(used) and json.dumps(json.load(open(used)), sort_keys=True) != json.dumps(configs, sort_keys=True):
        raise RuntimeError(f"{configs_path} differs from {used}: the refits would not reproduce the main run")
    models = MODELS if models is None else [m for m in MODELS if m in models]
    pairs_main = json.load(open(os.path.join(main_dir, "ebm_pairs.json")))
    perf = pd.read_csv(os.path.join(main_dir, "perf.csv"))
    perf = perf[perf.version == "raw"].set_index(["seed", "method"])
    os.makedirs(out_dir, exist_ok=True)
    dsm.check_run_dir(out_dir, p)
    ver_path = os.path.join(out_dir, "verification.csv")
    df, num_cols, cat_cols = prepare(df_freq, df_sev) if df_prepared is None else df_prepared
    t_run = time.time()
    for seed in [int(s) for s in seeds]:
        todo = [m for m in models if not os.path.exists(model_path(out_dir, m, seed))]
        if not todo:
            continue
        t_seed = time.time()
        print(f"\n{'=' * 90}\nseed {seed}: {', '.join(todo)}\n{'=' * 90}")
        p2.set_all_seeds(seed)
        train_df, valid_df, test_df = split(df, seed)
        sets = (("train", train_df), ("valid", valid_df), ("test", test_df))
        y_te, e_te = test_df[pm.TARGET].to_numpy(float), test_df[pm.EXPOSURE].to_numpy(float)
        tweedie_glm = None
        rows = []
        for name in todo:
            if name == "Tweedie-CANN" and tweedie_glm is None:          # CANN's offset: the Tweedie GLM of this split
                tweedie_glm = pm.fit_model("Tweedie GLM", pm.build_model("Tweedie GLM", num_cols, cat_cols, seed, p,
                                                                         configs), train_df, valid_df)
            ebm_pairs = pairs_main.get(str(seed)) if name == "Neural GA2M" else None
            if name == "Neural GA2M" and ebm_pairs is None:
                raise RuntimeError(f"no EBM pairs of seed {seed} in {main_dir}/ebm_pairs.json")
            model = pm.build_model(name, num_cols, cat_cols, seed, p, configs, ebm_pairs=ebm_pairs,
                                   tweedie_glm=tweedie_glm if name == "Tweedie-CANN" else None, verbose=verbose)
            t0 = time.time()
            pm.fit_model(name, model, train_df, valid_df)
            secs = time.time() - t0
            if name == "Tweedie GLM":
                tweedie_glm = model
            raw = {s: pm.predict(model, d) for s, d in sets}
            save_file(model_path(out_dir, name, seed), raw, sets, keep_train=keep_train)
            dev = float(p2.tweedie_deviance(y_te, raw["test"], e_te, power=p))
            gini = rf.gini_exposure(y_te, raw["test"], e_te)
            bal = rf.balance(y_te, raw["test"], e_te)
            ref = perf.loc[(seed, name)] if (seed, name) in perf.index else None
            r = {"seed": seed, "model": name, "minutes": secs / 60, "test_deviance": dev, "gini": gini, "balance": bal,
                 "main_deviance": ref.tweedie_deviance if ref is not None else np.nan,
                 "main_gini": ref.gini if ref is not None else np.nan,
                 "main_balance": ref.balance_error_pct if ref is not None else np.nan}
            r["deviance_status"] = _status(dev, r["main_deviance"])
            r["gini_status"] = _status(gini, r["main_gini"])
            if name == "EBM":
                same = [list(x) for x in model.interaction_pairs_] == pairs_main.get(str(seed))
                r["ebm_pairs_as_main"] = bool(same)
            rows.append(r)
            print(f"  {name:13s} | {secs / 60:5.1f} min | test dev {dev:.4f} (main {r['main_deviance']:.4f}: "
                  f"{r['deviance_status']}) | Gini {gini:.4f} ({r['gini_status']})")
            if name != "Tweedie GLM":
                del model
            del raw
            gc.collect()
        rf._append(rows, ver_path)
        del train_df, valid_df, test_df, tweedie_glm
        gc.collect()
        print(f"seed {seed} done in {(time.time() - t_seed) / 60:.1f} min")
    print(f"\nfinished in {(time.time() - t_run) / 3600:.2f} h")
    return verification_summary(out_dir)


def verification_summary(out_dir):
    path = os.path.join(out_dir, "verification.csv")
    if not os.path.exists(path):
        return None
    v = pd.read_csv(path)
    v["rel_diff"] = (v.test_deviance - v.main_deviance).abs() / v.main_deviance.abs()
    t = v.groupby("model").agg(seeds=("seed", "nunique"),
                               share_identical=("deviance_status", lambda s: (s == "identical").mean()),
                               share_identical_or_close=("deviance_status",
                                                         lambda s: s.isin(["identical", "close"]).mean()),
                               max_rel_deviance_diff=("rel_diff", "max"),
                               minutes_per_fit=("minutes", "mean"))
    t = t.reindex([m for m in MODELS if m in t.index])
    bad = v[v.deviance_status == "DIFFERENT"]
    try:
        from IPython.display import display
    except ImportError:
        display = print
    display(t)
    if len(bad):
        print(f"WARNING: {len(bad)} fits differ from the main run (relative deviance difference > 1e-3):")
        display(bad[["seed", "model", "test_deviance", "main_deviance"]])
    else:
        print("all refits reproduce the main run (relative deviance difference < 1e-3)")
    return t


def smoke_test(df_freq, df_sev, configs_path, n_sample=60000, seed=49, out_dir="predictions_smoke"):
    """Every code path on a subsample (networks with 2 epochs); no comparison with the main run."""
    import copy
    import shutil
    cfg = json.load(open(configs_path))
    small = copy.deepcopy(cfg)
    for m in ("LocalGLMnet", "Neural GA2M", "Tweedie-CANN"):
        small[m]["train"]["max_epochs"] = 2
    small["Neural GA2M"]["arch"]["n_bag"] = 1
    tmp_main = os.path.join(out_dir + "_main")
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(tmp_main, ignore_errors=True)
    os.makedirs(tmp_main)
    small_path = os.path.join(tmp_main, "configs.json")
    json.dump(small, open(small_path, "w"))
    df, num_cols, cat_cols = prepare(df_freq, df_sev)
    df = df.sample(n=min(n_sample, len(df)), random_state=0).reset_index(drop=True)
    tr, va, te = split(df, seed)
    e = pm.fit_model("EBM", pm.build_model("EBM", num_cols, cat_cols, seed, power(), rf.load_configs(small_path)),
                     tr, va)
    json.dump({str(seed): [list(x) for x in e.interaction_pairs_]}, open(os.path.join(tmp_main, "ebm_pairs.json"), "w"))
    pd.DataFrame(columns=["seed", "method", "version", "tweedie_deviance", "gini", "balance_error_pct"]).to_csv(
        os.path.join(tmp_main, "perf.csv"), index=False)
    t0 = time.time()
    archive_predictions(None, None, [seed], out_dir, tmp_main, small_path, df_prepared=(df, num_cols, cat_cols))
    checks = {"minutes": round((time.time() - t0) / 60, 1)}
    sizes = {"train": len(tr), "valid": len(va), "test": len(te)}
    for m in MODELS:
        d = load_predictions(out_dir, m, seed)
        checks[f"{m}: files, lengths, id hashes"] = (
            all(len(d[s]) == sizes[s] for s in ("valid", "test")) and d["n"] == sizes
            and all(np.isfinite(d[s]).all() for s in ("valid", "test")) and np.isfinite(d["train_sum_e_mu"])
            and d["idhash"]["test"] == id_hash(te["IDpol"].to_numpy(np.int64)))
    n_before = len(pd.read_csv(os.path.join(out_dir, "verification.csv")))
    archive_predictions(None, None, [seed], out_dir, tmp_main, small_path, df_prepared=(df, num_cols, cat_cols))
    checks["resume does not refit"] = len(pd.read_csv(os.path.join(out_dir, "verification.csv"))) == n_before
    for k, v in checks.items():
        print(f"{k}: {v}")
    ok = all(v for k, v in checks.items() if k != "minutes")
    print("\nPREDICTIONS SMOKE TEST PASSED" if ok else "\nPREDICTIONS SMOKE TEST FAILED - see the values above")
    return checks, (df, num_cols, cat_cols)
