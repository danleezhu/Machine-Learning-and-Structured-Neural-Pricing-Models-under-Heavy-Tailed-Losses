"""
paper2_datasets.py — data-set settings used by every module of the pipeline: feature lists, split strata, balance
segments, stressed-seed labels and the Tweedie power.

    import paper2_datasets as dsm
    df, num_cols, cat_cols = dsm.prepare(df_freq, df_sev)
    train, valid, test = dsm.split(df, seed=41)

freMTPL2  French MTPL, 678k policies (freMTPL2freq + freMTPL2sev). Preparation:
          paper2_models.prepare_data (log Density for the models); split:
          paper2_adjust_clean_plus.make_train_valid_test_stratified (60 / 20 / 20, strata: claim indicator x
          claim-size class x Region).

Every output directory records the data set and the Tweedie power in dataset.json; a directory written with another
data set or power is refused (check_run_dir).
"""

import numpy as np

SPECS = {
    "freMTPL2": dict(
        num_cols=["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"],
        cat_cols=["VehBrand", "VehGas", "Area", "Region"],
        log_cols=["Density"],
        strata_col="Region",
        seg_cols=["Region", "Area", "VehGas", "VehBrand"],
        stressed_seeds=[41, 49, 51, 55, 56, 57, 60, 61, 65, 70, 71, 72, 73, 75, 78, 81, 83, 85, 86, 89, 90],
        labels={"Density": "log Density"},
        tweedie_power=1.6,
    ),
}

_ACTIVE = "freMTPL2"
_POWER = {}                                     # data set -> Tweedie power set with set_power()


def use(name):
    """Selects the data set for every module of the pipeline."""
    global _ACTIVE
    if name not in SPECS:
        raise KeyError(f"unknown data set {name!r}; registered: {list(SPECS)}")
    _ACTIVE = name
    print(f"data set: {name}")


def active():
    return _ACTIVE


def set_power(p, source=""):
    """Tweedie power of the active data set for every model, objective and deviance of the pipeline."""
    p = float(p)
    if not 1.0 < p < 2.0:
        raise ValueError("the Tweedie power must lie in (1, 2) (compound Poisson-gamma)")
    _POWER[_ACTIVE] = p
    print(f"{_ACTIVE}: Tweedie power p = {p:g} {source}")


def power():
    """The Tweedie power of the active data set (set_power() value, else the registered value)."""
    p = _POWER.get(_ACTIVE, spec()["tweedie_power"])
    if p is None:
        raise RuntimeError(f"{_ACTIVE}: no Tweedie power set; call set_power() first")
    return float(p)


def spec(name=None):
    return SPECS[name or _ACTIVE]


# =============================================================================================================
# Preparation and split
# =============================================================================================================
def prepare(df_freq=None, df_sev=None):
    """Model-ready data of the active data set: (df, num_cols, cat_cols)."""
    if _ACTIVE == "freMTPL2":
        import paper2_models as pm
        return pm.prepare_data(df_freq, df_sev)
    raise KeyError(_ACTIVE)


def prepare_for_interactions(df_freq=None, df_sev=None):
    """Data used by the interaction diagnostics (raw Density; EBM bins on quantiles, so the FAST strengths do not
    depend on the log)."""
    if _ACTIVE == "freMTPL2":
        import paper2_adjust_clean_plus as p2
        return p2.prepare_fremtpl2(df_freq, df_sev)
    return prepare(df_freq, df_sev)


def split(df, seed):
    """Train / valid / test (60 / 20 / 20) of the active data set for one seed."""
    if _ACTIVE == "freMTPL2":
        import paper2_adjust_clean_plus as p2
        return p2.make_train_valid_test_stratified(df, random_state=int(seed))
    raise KeyError(_ACTIVE)


# =============================================================================================================
# Stressed / regular seeds
# =============================================================================================================
def stress_labels(seeds_df, group):
    """
    "paper"         : the fixed list spec()["stressed_seeds"] (21 seeds).
    "model_free"    : loss-imbalance score >= 0.40 or test top-1% loss share >= 0.78.
    "relative"      : the same two statistics, data-relative: score = max(percentile rank of the loss-imbalance
                      score, percentile rank of the test top-1% share) over the seeds of the run; the 40% of seeds with
                      the highest score are stressed.
    Returns an array of "bad_seed" / "good_seed".
    """
    s = seeds_df
    if group == "paper":
        st = spec()["stressed_seeds"]
        if st is None:
            raise ValueError(f"no stressed-seed list for {_ACTIVE}; use group='relative'")
        return np.where(s.seed.isin(st), "bad_seed", "good_seed")
    if group == "model_free":
        bad = (s.loss_imbalance_score >= 0.40) | (s.test_top1_share >= 0.78)
    elif group == "relative":
        score = np.maximum(s.loss_imbalance_score.rank(pct=True), s.test_top1_share.rank(pct=True))
        bad = score.rank(ascending=False, method="first") <= int(round(0.40 * len(s)))
    else:
        raise ValueError("group must be 'paper', 'model_free' or 'relative'")
    return np.where(bad, "bad_seed", "good_seed")


# =============================================================================================================
# Output directories
# =============================================================================================================
def check_run_dir(out_dir, p=None):
    """Every output directory records its data set and Tweedie power; a directory of another data set or p is
    refused. p: the power actually used (default power())."""
    import json
    import os
    p = power() if p is None else float(p)
    path = os.path.join(out_dir, "dataset.json")
    if os.path.exists(path):
        rec = json.load(open(path))
        found = rec.get("dataset")
        if found != _ACTIVE:
            raise RuntimeError(f"{out_dir} holds results for {found!r}, the active data set is {_ACTIVE!r}; "
                               f"call paper2_datasets.use({found!r}) or use another directory")
        if "tweedie_power" in rec and abs(rec["tweedie_power"] - p) > 1e-12:
            raise RuntimeError(f"{out_dir} was run with p = {rec['tweedie_power']:g}, the current p is {p:g}; "
                               f"use another directory")
        if "tweedie_power" not in rec:
            rec["tweedie_power"] = p
            json.dump(rec, open(path, "w"))
    else:
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"dataset": _ACTIVE, "tweedie_power": p}, open(path, "w"))


def dataset_of(out_dir, default="freMTPL2"):
    import json
    import os
    path = os.path.join(out_dir, "dataset.json")
    return json.load(open(path)).get("dataset", default) if os.path.exists(path) else default
