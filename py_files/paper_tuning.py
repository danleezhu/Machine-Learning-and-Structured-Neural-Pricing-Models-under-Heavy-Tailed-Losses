"""
paper_tuning.py — validation-only hyper-parameter selection before the 50-seed run.

The TEST set is never used: for every seed the split is made, the test part is discarded immediately.

Selection criterion   valid_dev_level = Tweedie deviance (p = TWEEDIE_POWER) on the VALIDATION set of the predictions
                      multiplied by the training-set level factor c = sum_train e y / sum_train e mu (the 'train_level'
                      correction of the main run; c uses training data only). This selects on ranking and shape, not on
                      the level noise of the validation set. Raw validation deviance, Gini and balances are recorded too.
Decision rule         one-standard-error rule on paired (per-seed) differences: the best configuration b has the lowest
                      mean; every configuration whose mean exceeds b's by at most one SE of the paired difference is
                      'equivalent'; the FIRST equivalent configuration in the listed order is chosen. Every grid lists
                      the reference setting first, so a change needs evidence beyond one SE.
Seeds                 SEEDS_5 (stages G, 1, 1b, 2, 2c) and SEEDS_10 (stage E and the confirmation stage 3), evenly
                      spaced over the loss-imbalance score within the regular and stressed groups. They are among the 50
                      evaluation seeds; only their validation sets are used here.

Stages (each resumable; every fit is appended to out_dir/results.csv and reused whenever the same configuration and seed
are needed again, e.g. by a later stage):
    E   EBM            reference (learning_rate 0.03, max_rounds 500) vs installed interpret defaults; SEEDS_10, final.
                       Also stores the EBM pairs of the selected configuration (needed by Neural GA2M).
    G   GBM            num_leaves {15, 31, 63} x min_child_samples {200, 500, 1000} x categorical {ordinal, native}
    1   optimisation   LocalGLMnet, Tweedie-CANN: lr {3e-4, 1e-3} x criterion {raw, ma, level}
                       Neural GA2M: lr {3e-4, 1e-3} x criterion {raw, ma, level} x scheduler patience {8, off}
    1b  clipping       stage-1 winner with gradient clipping 5 vs off
    2   capacity       LocalGLMnet eta_bound {1, 0.5, 2} x lambda_local {1e-4, 1e-3}; CANN delta_bound {1, 0.5, 2};
                       Neural GA2M hidden {(32,16), (64,32)}
    2c  nagging        Neural GA2M n_bag {1, 5}: 5 is used only if better by more than one SE
    3   confirmation   GBM, LocalGLMnet, CANN, Neural GA2M: reference vs stage winner on SEEDS_10 -> final

Patience is tied to the stopping criterion: LocalGLMnet and CANN use 8 epochs for 'raw' and 'level' and 15 for 'ma'
(a moving average needs a longer window); Neural GA2M uses 15 for every criterion (slow, flat validation curves).

    import paper_tuning as tu
    tu.smoke_test(df_freq, df_sev)                    # every code path on a subsample; must PASS
    tu.run_tuning(df_freq, df_sev, out_dir="tuning")  # all stages; resumable; writes tuning/selected_configs.json
"""

import copy
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import paper_adjust_clean_plus as p2
import paper_models as pm

SEEDS_5 = [49, 50, 57, 62, 67]
SEEDS_10 = [41, 47, 49, 50, 57, 62, 67, 74, 78, 85]
METRIC = "valid_dev_level"

EBM_REF = {"learning_rate": 0.03, "max_rounds": 500}
EBM_LIBRARY_DEFAULT = {"learning_rate": None, "max_rounds": None}    # installed interpret defaults
GBM_REF = {"num_leaves": 15, "min_child_samples": 500, "categorical": "ordinal"}
NN_REF = {
    "LocalGLMnet": {"arch": {"hidden_dims": [128, 64], "dropout": 0.05, "eta_bound": 1.0, "lambda_local": 1e-4},
                    "train": {"lr": 3e-4, "batch_size": 4096, "criterion": "raw", "patience": 8,
                              "scheduler_patience": None, "grad_clip": 5.0}},
    "Tweedie-CANN": {"arch": {"hidden_dims": [64, 32], "dropout": 0.05, "delta_bound": 1.0, "lambda_delta": 1e-3},
                     "train": {"lr": 3e-4, "batch_size": 4096, "criterion": "raw", "patience": 8,
                               "scheduler_patience": None, "grad_clip": 5.0}},
    "Neural GA2M": {"arch": {"hidden": [32, 16], "n_bag": 1},
                    "train": {"lr": 1e-3, "batch_size": 8192, "criterion": "ma", "patience": 15,
                              "scheduler_patience": 8, "grad_clip": 5.0}},
}


def patience_for(model, criterion):
    if model == "Neural GA2M":
        return 15
    return 15 if criterion == "ma" else 8


# =============================================================================================================
# Configurations, identifiers, storage
# =============================================================================================================
def config_id(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def describe(model, cfg):
    if model == "EBM":
        return f"lr={cfg['learning_rate']} rounds={cfg['max_rounds']}"
    if model == "GBM":
        return f"leaves={cfg['num_leaves']} min_child={cfg['min_child_samples']} cat={cfg['categorical']}"
    t, a = cfg["train"], cfg["arch"]
    arch = " ".join(f"{k}={v}" for k, v in a.items())
    return (f"lr={t['lr']:g} crit={t['criterion']} pat={t['patience']} sched={t['scheduler_patience']} "
            f"clip={t['grad_clip']} | {arch}")


RESULT_COLUMNS = ["model", "config_id", "seed", "stage_first_used", "config", "valid_dev_level", "valid_dev_raw",
                  "valid_gini", "valid_balance_raw", "valid_balance_level", "train_balance_raw", "train_level_factor",
                  "minutes", "best_iteration", "epochs_run", "stopped_by", "share_grad_norm_over_5",
                  "ebm_max_rounds", "ebm_best_iteration_max", "ebm_best_iteration_median", "ebm_share_bags_at_cap"]


def append_rows(rows, path, columns):
    """Appends rows with a FIXED column set (CSV appends must never change the column layout)."""
    df = pd.DataFrame(rows)
    unknown = set(df.columns) - set(columns)
    if unknown:
        raise KeyError(f"unexpected columns {sorted(unknown)} for {path}")
    df = df.reindex(columns=columns)
    if os.path.exists(path):
        header = pd.read_csv(path, nrows=0).columns.tolist()
        if header != list(columns):
            raise RuntimeError(f"{path} has a different column layout; use a new output directory")
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


class Store:
    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.results = os.path.join(out_dir, "results.csv")
        self.selections = os.path.join(out_dir, "selections.json")
        self.histories = os.path.join(out_dir, "histories.jsonl")

    def load(self):
        return pd.read_csv(self.results) if os.path.exists(self.results) else pd.DataFrame(
            columns=["model", "config_id", "seed"])

    def done(self):
        r = self.load()
        return set(zip(r.model, r.config_id, r.seed.astype(int)))

    def append(self, row, history=None):
        append_rows([row], self.results, RESULT_COLUMNS)
        if history is not None:
            with open(self.histories, "a") as fh:
                fh.write(json.dumps({"model": row["model"], "config_id": row["config_id"], "seed": row["seed"],
                                     "history": history}) + "\n")

    def get_sel(self):
        return json.load(open(self.selections)) if os.path.exists(self.selections) else {}

    def set_sel(self, key, value):
        s = self.get_sel()
        s[key] = value
        json.dump(s, open(self.selections, "w"), indent=1)

    def pairs_path(self, ebm_cid):
        return os.path.join(self.dir, f"ebm_pairs_{ebm_cid}.json")


# =============================================================================================================
# One fit, validation only
# =============================================================================================================
def _split(df, seed):
    tr, va, te = p2.make_train_valid_test_stratified(df, random_state=int(seed))
    del te                                                           # the test set is never used in tuning
    return tr, va


def evaluate(name, cfg, seed, train_df, valid_df, num_cols, cat_cols, power, pairs=None, glm=None):
    torch.manual_seed(pm.model_seed(name, seed) if name in pm.NN_MODELS else int(seed))
    np.random.seed(int(seed))
    configs = {name: cfg}
    model = pm.build_model(name, num_cols, cat_cols, seed, power, configs, ebm_pairs=pairs, tweedie_glm=glm)
    t0 = time.time()
    pm.fit_model(name, model, train_df, valid_df)
    minutes = (time.time() - t0) / 60
    y_tr, e_tr = train_df[pm.TARGET].to_numpy(float), train_df[pm.EXPOSURE].to_numpy(float)
    y_va, e_va = valid_df[pm.TARGET].to_numpy(float), valid_df[pm.EXPOSURE].to_numpy(float)
    p_tr, p_va = pm.predict(model, train_df), pm.predict(model, valid_df)
    c = float(np.sum(e_tr * y_tr) / np.sum(e_tr * p_tr))
    info = getattr(model, "train_info_", {}) or {}
    row = {"model": name, "config_id": config_id(cfg), "seed": int(seed), "config": json.dumps(cfg, sort_keys=True),
           "valid_dev_level": float(p2.tweedie_deviance(y_va, c * p_va, e_va, power=power)),
           "valid_dev_raw": float(p2.tweedie_deviance(y_va, p_va, e_va, power=power)),
           "valid_gini": pm_gini(y_va, p_va, e_va),
           "valid_balance_raw": float(np.sum(e_va * p_va) / np.sum(e_va * y_va) - 1),
           "valid_balance_level": float(np.sum(e_va * c * p_va) / np.sum(e_va * y_va) - 1),
           "train_balance_raw": float(np.sum(e_tr * p_tr) / np.sum(e_tr * y_tr) - 1), "train_level_factor": c,
           "minutes": minutes, "best_iteration": getattr(model, "best_iteration_", np.nan),
           "epochs_run": info.get("epochs_run", np.nan), "stopped_by": info.get("stopped_by", ""),
           "share_grad_norm_over_5": info.get("share_grad_norm_over_5", np.nan)}
    if name == "EBM" and getattr(model, "ebm_rounds_", None):
        row.update({f"ebm_{k}": v for k, v in model.ebm_rounds_.items()})
    return row, model, info.get("history")


def pm_gini(y, pred, e):
    order = np.argsort(pred, kind="mergesort")
    ce = np.concatenate([[0.0], np.cumsum(e[order])])
    cl = np.concatenate([[0.0], np.cumsum((e * y)[order])])
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(1.0 - 2.0 * trap(cl / cl[-1], ce / ce[-1]))


def run_grid(store, data, name, configs, seeds, stage, power, pairs_by_seed=None, verbose=True):
    """Fits every (config, seed) not yet in results.csv. Seeds outer loop (one split / GLM per seed)."""
    df, num_cols, cat_cols = data
    done = store.done()
    todo = [(s, c) for s in seeds for c in configs if (name, config_id(c), int(s)) not in done]
    if not todo:
        return
    print(f"\n[stage {stage}] {name}: {len(todo)} fits to run ({len(configs)} configurations x {len(seeds)} seeds, "
          f"{len(configs) * len(seeds) - len(todo)} reused)")
    t_start, n_done = time.time(), 0
    for seed in seeds:
        cfgs = [c for c in configs if (name, config_id(c), int(seed)) not in done]
        if not cfgs:
            continue
        tr, va = _split(df, seed)
        glm = None
        if name == "Tweedie-CANN":
            glm = p2.TweedieGLM(num_cols, cat_cols, tweedie_power=power).fit(tr)
        pairs = None
        if name == "Neural GA2M":
            if pairs_by_seed is None or str(seed) not in pairs_by_seed:
                raise RuntimeError(f"EBM pairs for seed {seed} missing: run stage E first")
            pairs = pairs_by_seed[str(seed)]
        for cfg in cfgs:
            row, model, hist = evaluate(name, cfg, seed, tr, va, num_cols, cat_cols, power, pairs, glm)
            row["stage_first_used"] = stage
            store.append(row, hist)
            if name == "EBM":
                path = store.pairs_path(row["config_id"])
                pb = json.load(open(path)) if os.path.exists(path) else {}
                pb[str(seed)] = [list(p) for p in model.interaction_pairs_]
                json.dump(pb, open(path, "w"), indent=1)
            n_done += 1
            eta = (time.time() - t_start) / n_done * (len(todo) - n_done) / 60
            if verbose:
                extra = (f" | epochs {row['epochs_run']:.0f} (best {row['best_iteration']:.0f}, {row['stopped_by']})"
                         if name in pm.NN_MODELS else f" | best iteration {row['best_iteration']}")
                print(f"  seed {seed} | {describe(name, cfg)} | dev(level) {row['valid_dev_level']:.4f} | "
                      f"dev(raw) {row['valid_dev_raw']:.4f} | train B {100 * row['train_balance_raw']:+.1f}%{extra} | "
                      f"{row['minutes']:.1f} min | about {eta:.0f} min left in this grid")
            del model
        del tr, va, glm


# =============================================================================================================
# Selection
# =============================================================================================================
def one_se(store, name, configs, seeds, metric=METRIC):
    r = store.load()
    ids = [config_id(c) for c in configs]
    d = r[(r.model == name) & r.config_id.isin(ids) & r.seed.isin(seeds)]
    W = d.pivot_table(index="seed", columns="config_id", values=metric)
    missing = [(i, s) for i in ids for s in seeds if i not in W.columns or s not in W.index or pd.isna(W.loc[s, i])]
    if missing:
        raise RuntimeError(f"{name}: results missing for {len(missing)} (config, seed) pairs; run the grid first")
    W = W.loc[seeds, ids]
    best = W.mean().idxmin()
    chosen = None
    rows = []
    for i, c in zip(ids, configs):
        diff = W[i] - W[best]
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        dref = W[i] - W[ids[0]]
        g = d[d.config_id == i]
        rows.append({"config_id": i, "description": describe(name, c), "mean": W[i].mean(),
                     "diff_vs_best": diff.mean(), "se_vs_best": se,
                     "diff_vs_reference": dref.mean(), "se_vs_reference": dref.std(ddof=1) / np.sqrt(len(dref)),
                     "seeds_better_than_reference": int((dref < 0).sum()),
                     "valid_dev_raw": g.valid_dev_raw.mean(), "valid_gini": g.valid_gini.mean(),
                     "train_balance_%": 100 * g.train_balance_raw.mean(), "minutes": g.minutes.mean(),
                     "epochs": g.epochs_run.mean(), "best_iteration": g.best_iteration.mean(),
                     "stopped_by_max": int((g.stopped_by == "max_epochs").sum()),
                     "grad_norm_over_5": g.share_grad_norm_over_5.mean()})
        if chosen is None and diff.mean() <= se + 1e-12:
            chosen = i
    table = pd.DataFrame(rows)
    table["chosen"] = table.config_id == chosen
    table["best"] = table.config_id == best
    return configs[ids.index(chosen)], table


def _show(title, table):
    print(f"\n{title}")
    cols = ["description", "mean", "diff_vs_reference", "se_vs_reference", "seeds_better_than_reference",
            "diff_vs_best", "se_vs_best", "valid_dev_raw", "valid_gini", "train_balance_%", "epochs", "best_iteration",
            "stopped_by_max", "grad_norm_over_5", "minutes", "chosen", "best"]
    t = table[[c for c in cols if c in table.columns]]
    try:
        from IPython.display import display
        with pd.option_context("display.max_colwidth", 90, "display.width", 250):
            display(t.round(4))
    except ImportError:
        print(t.round(4).to_string())
    if "stopped_by_max" in t and t.stopped_by_max.sum() > 0:
        print("WARNING: some fits reached max_epochs before early stopping; consider a larger max_epochs")


# =============================================================================================================
# Grids
# =============================================================================================================
def _nn(model, arch=None, **train):
    c = copy.deepcopy(NN_REF[model])
    if arch:
        c["arch"].update(arch)
    c["train"].update(train)
    if "criterion" in train:
        c["train"]["patience"] = patience_for(model, train["criterion"])
    return c


def grid_ebm():
    return [EBM_REF, EBM_LIBRARY_DEFAULT]


def grid_gbm():
    g = [GBM_REF]
    for cat in ("ordinal", "native"):
        for leaves in (15, 31, 63):
            for mcs in (200, 500, 1000):
                c = {"num_leaves": leaves, "min_child_samples": mcs, "categorical": cat}
                if c != GBM_REF:
                    g.append(c)
    return g


def grid_stage1(model):
    ref = NN_REF[model]
    g = [ref]
    scheds = (8, None) if model == "Neural GA2M" else (None,)
    for lr in (3e-4, 1e-3):
        for crit in ("raw", "ma", "level"):
            for sp in scheds:
                c = _nn(model, lr=lr, criterion=crit, scheduler_patience=sp)
                if config_id(c) != config_id(ref):
                    g.append(c)
    return g


def grid_clip(winner):
    other = copy.deepcopy(winner)
    other["train"]["grad_clip"] = None if winner["train"]["grad_clip"] is not None else 5.0
    return [winner, other]


def grid_capacity(model, winner):
    g = [winner]
    if model == "LocalGLMnet":
        opts = [{"eta_bound": b, "lambda_local": l} for b in (1.0, 0.5, 2.0) for l in (1e-4, 1e-3)]
    elif model == "Tweedie-CANN":
        opts = [{"delta_bound": b} for b in (1.0, 0.5, 2.0)]
    else:
        opts = [{"hidden": [32, 16]}, {"hidden": [64, 32]}]
    for o in opts:
        c = copy.deepcopy(winner)
        c["arch"].update(o)
        if config_id(c) not in [config_id(x) for x in g]:
            g.append(c)
    return g


def grid_nagging(winner):
    c5 = copy.deepcopy(winner)
    c5["arch"]["n_bag"] = 5
    return [winner, c5]


# =============================================================================================================
# Stages
# =============================================================================================================
def _ebm_pairs(store):
    sel = store.get_sel()
    if "EBM" not in sel:
        raise RuntimeError("run stage E (EBM) first")
    return json.load(open(store.pairs_path(config_id(sel["EBM"]))))


def stage_E(store, data, power):
    print("installed interpret defaults:", pm.ebm_library_defaults())
    run_grid(store, data, "EBM", grid_ebm(), SEEDS_10, "E", power)
    chosen, t = one_se(store, "EBM", grid_ebm(), SEEDS_10)
    _show("EBM (SEEDS_10)", t)
    r = store.load()
    cols = [c for c in r.columns if c.startswith("ebm_")]
    if cols:
        print("EBM boosting rounds per configuration (share_bags_at_cap > 0 means the cap was binding):")
        print(r[r.model == "EBM"].groupby("config_id")[cols].mean().round(3).to_string())
    store.set_sel("EBM", chosen)
    return chosen


def stage_G(store, data, power):
    run_grid(store, data, "GBM", grid_gbm(), SEEDS_5, "G", power)
    chosen, t = one_se(store, "GBM", grid_gbm(), SEEDS_5)
    _show("GBM (SEEDS_5)", t)
    store.set_sel("GBM/stage_G", chosen)
    return chosen


def stage_nn(store, data, power, model, stage):
    sel = store.get_sel()
    pairs = _ebm_pairs(store) if model == "Neural GA2M" else None
    if stage == "1":
        grid = grid_stage1(model)
    elif stage == "1b":
        grid = grid_clip(sel[f"{model}/stage_1"])
    elif stage == "2":
        grid = grid_capacity(model, sel[f"{model}/stage_1b"])
    elif stage == "2c":
        grid = grid_nagging(sel[f"{model}/stage_2"])
    else:
        raise ValueError(stage)
    run_grid(store, data, model, grid, SEEDS_5, stage, power, pairs)
    chosen, t = one_se(store, model, grid, SEEDS_5)
    _show(f"{model}, stage {stage} (SEEDS_5)", t)
    store.set_sel(f"{model}/stage_{stage}", chosen)
    return chosen


def stage_3(store, data, power, models):
    sel = store.get_sel()
    finals = {}
    for model in models:
        if model == "GBM":
            winner, ref = sel["GBM/stage_G"], GBM_REF
        else:
            last = "2c" if model == "Neural GA2M" else "2"
            winner, ref = sel[f"{model}/stage_{last}"], NN_REF[model]
        grid = [ref] if config_id(winner) == config_id(ref) else [ref, winner]
        pairs = _ebm_pairs(store) if model == "Neural GA2M" else None
        run_grid(store, data, model, grid, SEEDS_10, "3", power, pairs)
        chosen, t = one_se(store, model, grid, SEEDS_10)
        _show(f"{model}, confirmation (SEEDS_10)", t)
        store.set_sel(model, chosen)
        finals[model] = chosen
    return finals


def write_selected_configs(store):
    sel = store.get_sel()
    need = ["EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Tweedie-CANN"]
    missing = [m for m in need if m not in sel]
    if missing:
        raise RuntimeError(f"final selections missing for {missing}")
    out = {m: sel[m] for m in need}
    out["_meta"] = {"data": "log(Density) for all models", "selection_metric": METRIC,
                    "rule": "one-SE on paired differences, reference first", "seeds_5": SEEDS_5, "seeds_10": SEEDS_10,
                    "tweedie_power": p2.TWEEDIE_POWER, "train_constants": pm.TRAIN_CONSTANTS,
                    "gbm_constants": pm.GBM_CONSTANTS, "ebm_constants": pm.EBM_CONSTANTS,
                    "ebm_library_defaults": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                                             for k, v in pm.ebm_library_defaults().items()},
                    "written": time.strftime("%Y-%m-%d %H:%M")}
    path = os.path.join(store.dir, "selected_configs.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nfinal configurations written to {path}")
    for m in need:
        print(f"  {m:13s} {describe(m, out[m])}")
    return out


ALL_STAGES = ("E", "G", "1", "1b", "2", "2c", "3")


def run_tuning(df_freq, df_sev, out_dir="tuning", stages=ALL_STAGES, nn_models=tuple(pm.NN_MODELS), power=None):
    """Runs the requested stages in order (each resumable). Stage 3 also writes selected_configs.json."""
    power = p2.TWEEDIE_POWER if power is None else power
    store = Store(out_dir)
    data = pm.prepare_data(df_freq, df_sev)
    for st in stages:
        t0 = time.time()
        print(f"\n{'#' * 100}\n# stage {st}\n{'#' * 100}")
        if st == "E":
            stage_E(store, data, power)
        elif st == "G":
            stage_G(store, data, power)
        elif st in ("1", "1b", "2"):
            for m in nn_models:
                stage_nn(store, data, power, m, st)
        elif st == "2c":
            if "Neural GA2M" in nn_models:
                stage_nn(store, data, power, "Neural GA2M", "2c")
        elif st == "3":
            stage_3(store, data, power, ["GBM"] + [m for m in pm.NN_MODELS if m in nn_models])
            if set(nn_models) == set(pm.NN_MODELS):
                write_selected_configs(store)
        else:
            raise ValueError(st)
        print(f"stage {st} finished in {(time.time() - t0) / 3600:.2f} h")
    return store.get_sel()


# =============================================================================================================
# Smoke test: every code path, small subsample, few epochs
# =============================================================================================================
def smoke_test(df_freq, df_sev, n_sample=60000, seed=49, out_dir="tuning_smoke"):
    import shutil
    import paper_run_full as rf
    checks = {}
    df_raw, _, _ = p2.prepare_fremtpl2(df_freq, df_sev)
    df, num_cols, cat_cols = pm.prepare_data(df_freq, df_sev)
    checks["log Density applied"] = bool(np.allclose(np.exp(df.Density), df_raw.Density))
    a = p2.make_train_valid_test_stratified(df_raw, random_state=seed)
    b = p2.make_train_valid_test_stratified(df, random_state=seed)
    checks["split unchanged by log Density"] = all(set(x.IDpol) == set(y.IDpol) for x, y in zip(a, b))
    del a, b, df_raw
    df = df.sample(n=min(n_sample, len(df)), random_state=0).reset_index(drop=True)
    tr, va = _split(df, seed)
    power = p2.TWEEDIE_POWER
    short = {"max_epochs": 2}

    glm = p2.TweedieGLM(num_cols, cat_cols, tweedie_power=power).fit(tr)
    lg = pm.LocalGLMNetWrapper(num_cols, cat_cols, power, NN_REF["LocalGLMnet"]["arch"],
                               {**NN_REF["LocalGLMnet"]["train"], "max_epochs": 0}, seed=1).fit(tr, va)
    with torch.no_grad():
        t0_ = pm._xonly(lg.prep_, tr)
        _, _, _, z0 = lg.model_.components(t0_[0], t0_[1])
        glm_part = (z0 @ lg.model_.beta_global).double().numpy()
    checks["LocalGLMnet global part = Tweedie GLM (max rel diff)"] = float(
        np.max(np.abs(np.exp(glm_part) / glm.predict(tr) - 1)))
    checks["LocalGLMnet start = Tweedie GLM (max rel diff)"] = float(np.max(np.abs(lg.predict(tr) / glm.predict(tr) - 1)))
    for crit in ("raw", "ma", "level"):
        for clip in (5.0, None):
            cfg = _nn("LocalGLMnet", criterion=crit, grad_clip=clip)
            cfg["train"].update(short)
            m = pm.LocalGLMNetWrapper(num_cols, cat_cols, power, cfg["arch"], cfg["train"], seed=1).fit(tr, va)
            checks[f"LocalGLMnet {crit} clip={clip} finite"] = bool(np.isfinite(m.predict(va)).all())
    with torch.no_grad():
        t = pm._xonly(m.prep_, va)
        eta, beta, d, z = m.model_.components(t[0], t[1])
        checks["LocalGLMnet |eta - eta_global| <= bound"] = float((eta - z @ m.model_.beta_global).abs().max())

    cann0 = pm.TweedieCANNWrapper(num_cols, cat_cols, power, NN_REF["Tweedie-CANN"]["arch"],
                                  {**NN_REF["Tweedie-CANN"]["train"], "max_epochs": 0}, seed=3, glm=glm).fit(tr, va)
    checks["CANN start ~ Tweedie GLM (max |delta|)"] = float(np.max(np.abs(cann0.delta(va))))
    cfg = _nn("Tweedie-CANN", criterion="level", grad_clip=None)
    cfg["arch"]["delta_bound"] = 0.5
    cfg["train"].update(short)
    cann = pm.TweedieCANNWrapper(num_cols, cat_cols, power, cfg["arch"], cfg["train"], seed=3, glm=glm).fit(tr, va)
    checks["CANN |delta| <= 0.5"] = float(np.max(np.abs(cann.delta(va))))
    checks["CANN prediction = GLM x exp(delta)"] = float(np.max(np.abs(cann.predict(va) /
                                                                     (glm.predict(va) * np.exp(cann.delta(va))) - 1)))

    print("installed interpret defaults:", pm.ebm_library_defaults())
    ebm = pm.EBMWrapperV2(num_cols, cat_cols, power, EBM_REF).fit(tr)
    checks["EBM pairs"] = len(ebm.interaction_pairs_)
    checks["EBM rounds info available"] = ebm.ebm_rounds_ is not None
    for sp in (8, None):
        cfg = _nn("Neural GA2M", criterion="level", scheduler_patience=sp)
        cfg["arch"].update({"n_bag": 2, "hidden": [64, 32]})
        cfg["train"].update(short)
        ga = pm.NeuralGA2MWrapper(num_cols, cat_cols, power, ebm.interaction_pairs_, cfg["arch"], cfg["train"],
                                  seed=2).fit(tr, va)
        checks[f"GA2M n_bag=2 sched={sp} finite"] = bool(np.isfinite(ga.predict(va)).all())
    cfg = _nn("Neural GA2M")
    cfg["train"].update(short)
    ga1 = pm.NeuralGA2MWrapper(num_cols, cat_cols, power, ebm.interaction_pairs_, cfg["arch"], cfg["train"],
                               seed=2).fit(tr, va)
    tc = ga1.term_contributions(va)
    checks["GA2M terms reproduce log prediction (n_bag=1)"] = float(np.max(np.abs(np.exp(tc.sum(1)) / ga1.predict(va) - 1)))
    for cat in ("ordinal", "native"):
        g = pm.GBMWrapperV2(num_cols, cat_cols, power, {**GBM_REF, "categorical": cat}).fit(tr, va)
        checks[f"GBM {cat} best iteration"] = int(g.best_iteration_)

    # tuning machinery on 2 seeds, tiny grids
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    store = Store(out_dir)
    data = (df, num_cols, cat_cols)
    g2 = [GBM_REF, {**GBM_REF, "categorical": "native"}]
    run_grid(store, data, "GBM", g2, [seed, 50], "smoke", power, verbose=False)
    run_grid(store, data, "GBM", g2, [seed, 50], "smoke", power, verbose=False)        # must reuse, not refit
    checks["results reused (2 configs x 2 seeds = 4 rows)"] = len(store.load()) == 4
    chosen, _ = one_se(store, "GBM", g2, [seed, 50])
    checks["one-SE returns a configuration"] = chosen in g2

    # main runner on the subsample with smoke configurations
    smoke_cfg = {"EBM": EBM_REF, "GBM": GBM_REF}
    for mname in pm.NN_MODELS:
        c = copy.deepcopy(NN_REF[mname])
        c["train"]["max_epochs"] = 2
        smoke_cfg[mname] = c
    cfg_path = os.path.join(out_dir, "smoke_configs.json")
    json.dump(smoke_cfg, open(cfg_path, "w"))
    run_dir = os.path.join(out_dir, "run")
    perf = rf.run_full(None, None, seeds=[seed], out_dir=run_dir, configs_path=cfg_path, n_perm=19,
                       df_prepared=(df, num_cols, cat_cols))
    checks[f"runner rows (7 models x {len(rf.VERSIONS)} versions)"] = len(perf)
    checks["runner all finite"] = bool(np.isfinite(perf[["tweedie_deviance", "gini", "balance_error_pct"]]).all().all())
    p = perf.set_index(["method", "version"])
    checks["train_level exact on train (max |B|)"] = float(p.xs("train_level", level="version").train_balance_error_pct.abs().max())

    for k, v in checks.items():
        print(f"{k}: {v}")
    ok = (checks["log Density applied"] and checks["split unchanged by log Density"]
          and checks["LocalGLMnet global part = Tweedie GLM (max rel diff)"] < 1e-4
          and checks["LocalGLMnet start = Tweedie GLM (max rel diff)"] < 1e-4
          and all(v for k, v in checks.items() if k.endswith("finite"))
          and checks["LocalGLMnet |eta - eta_global| <= bound"] <= 1.0 + 1e-5
          and checks["CANN start ~ Tweedie GLM (max |delta|)"] < 0.05
          and checks["CANN |delta| <= 0.5"] <= 0.5 + 1e-6
          and checks["CANN prediction = GLM x exp(delta)"] < 1e-4
          and checks["GA2M terms reproduce log prediction (n_bag=1)"] < 1e-4
          and checks["results reused (2 configs x 2 seeds = 4 rows)"] and checks["one-SE returns a configuration"]
          and checks[f"runner rows (7 models x {len(rf.VERSIONS)} versions)"] == 7 * len(rf.VERSIONS)
          and checks["train_level exact on train (max |B|)"] < 1e-8)
    print("\nSMOKE TEST PASSED" if ok else "\nSMOKE TEST FAILED - see the values above")
    return checks


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Validation-only tuning before the 50-seed run")
    ap.add_argument("--freq", required=True)
    ap.add_argument("--sev", required=True)
    ap.add_argument("--out", default="tuning")
    ap.add_argument("--stages", default=",".join(ALL_STAGES), help="e.g. E,G or 1,1b,2,2c,3")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    f, s = pd.read_csv(a.freq), pd.read_csv(a.sev)
    if a.smoke:
        smoke_test(f, s)
    else:
        run_tuning(f, s, out_dir=a.out, stages=tuple(x.strip() for x in a.stages.split(",")))
