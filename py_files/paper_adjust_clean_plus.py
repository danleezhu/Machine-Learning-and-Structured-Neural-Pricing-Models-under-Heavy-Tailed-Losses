"""
paper_adjust_clean_plus.py — shared helpers of the pipeline: data preparation, Tweedie deviance, stratified
train / validation / test split, the two GLM benchmarks, tabular preprocessing for the networks, seeding.

TWEEDIE_POWER = 1.6 is used to train every Tweedie model and to evaluate the Tweedie deviance (profile-likelihood
estimates on the training sets: 1.54-1.58).
"""

import random

import numpy as np
import pandas as pd
import matplotlib as mpl
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_tweedie_deviance

import torch
import torch.nn as nn

TWEEDIE_POWER = 1.6


def prepare_fremtpl2(df_freq, df_sev=None):
    """
    freMTPL2:
    df_freq: policy-level frequency data
    df_sev: claim-level severity data, optional

    Target:
        PurePremium = ClaimAmount / Exposure
    """

    df = df_freq.copy()

    if df_sev is not None:
        sev_agg = (
            df_sev.groupby("IDpol")["ClaimAmount"]
            .sum()
            .reset_index()
        )
        df = df.merge(sev_agg, on="IDpol", how="left")
    else:
        if "ClaimAmount" not in df.columns:
            raise ValueError("Need ClaimAmount column or df_sev.")

    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)

    df = df[df["Exposure"] > 0].copy()

    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

    categorical_cols = ["VehBrand", "VehGas", "Area", "Region"]
    numerical_cols = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]

    return df, numerical_cols, categorical_cols


def tweedie_deviance(y_true, y_pred, exposure=None, power=TWEEDIE_POWER):
    """
    y_true and y_pred are PurePremium scale.
    """
    y_pred = np.maximum(y_pred, 1e-10)

    if exposure is None:
        return mean_tweedie_deviance(
            y_true, y_pred, power=power
        )

    return mean_tweedie_deviance(
        y_true, y_pred, power=power, sample_weight=exposure
    )


def make_train_valid_test_stratified(
    df,
    target_col="PurePremium",
    test_size=0.2,
    valid_size=0.2,
    random_state=42
):
    df = df.copy()

    df["has_claim"] = (df["ClaimAmount"] > 0).astype(int)

    positive = df.loc[df["ClaimAmount"] > 0, "ClaimAmount"]

    df["loss_bin"] = 0
    if len(positive) > 0:
        bins = positive.quantile([0.50, 0.75, 0.90, 0.97, 0.99]).values
        df.loc[df["ClaimAmount"] > 0, "loss_bin"] = np.searchsorted(
            bins,
            df.loc[df["ClaimAmount"] > 0, "ClaimAmount"].values
        ) + 1

    df["strata"] = (
        df["has_claim"].astype(str)
        + "_"
        + df["loss_bin"].astype(str)
        + "_"
        + df["Region"].astype(str)
    )

    # rare strata fallback
    counts = df["strata"].value_counts()
    rare = counts[counts < 5].index
    df.loc[df["strata"].isin(rare), "strata"] = "rare"

    train_valid, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["strata"]
    )
    valid_size_adjust = valid_size/(1-test_size)

    train, valid = train_test_split(
        train_valid,
        test_size=valid_size_adjust,
        random_state=random_state,
        stratify=train_valid["strata"]
    )

    drop_cols = ["has_claim", "loss_bin", "strata"]

    return (
        train.drop(columns=drop_cols).reset_index(drop=True),
        valid.drop(columns=drop_cols).reset_index(drop=True),
        test.drop(columns=drop_cols).reset_index(drop=True)
    )


class PoissonGammaGLM:
    """
    Classical actuarial GLM:
        frequency ~ Poisson(log link) with offset log(Exposure)
        severity ~ Gamma(log link), trained on positive claims
        pure premium = freq * sev
    """

    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.freq_columns_ = None
        self.sev_columns_ = None
        self.freq_result_ = None
        self.sev_result_ = None

    def _build_X(self, df):
        X_num = df[self.numerical_cols].copy()
        X_cat = pd.get_dummies(df[self.categorical_cols], drop_first=True, dtype=float)
        X = pd.concat([X_num, X_cat], axis=1)
        X = sm.add_constant(X, has_constant="add")
        return X

    def fit(self, df):
        X = self._build_X(df)
        self.freq_columns_ = X.columns.tolist()

        # Frequency model
        freq_model = sm.GLM(
            df["ClaimNb"].values,
            X,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=np.log(df["Exposure"].values)
        )
        self.freq_result_ = freq_model.fit()

        # Severity model: on positive claims
        sev_df = df[df["ClaimNb"] > 0].copy()
        sev_df["AvgClaim"] = sev_df["ClaimAmount"] / sev_df["ClaimNb"]

        X_sev = self._build_X(sev_df)
        self.sev_columns_ = X_sev.columns.tolist()

        sev_model = sm.GLM(
            sev_df["AvgClaim"].values,
            X_sev,
            family=sm.families.Gamma(link=sm.families.links.Log()),
            # average of ClaimNb claims: Var = phi * mu^2 / ClaimNb  -> variance weights.
            # (freq_weights gave identical estimates but treated each policy as ClaimNb replicates.)
            var_weights=sev_df["ClaimNb"].values
        )
        self.sev_result_ = sev_model.fit()
        return self

    def predict(self, df):
        X = self._build_X(df)
        for c in self.freq_columns_:
            if c not in X.columns:
                X[c] = 0.0
        X_freq = X[self.freq_columns_]

        pred_freq_total = self.freq_result_.predict(
            X_freq,
            offset=np.log(df["Exposure"].values)
        )
        pred_freq_per_exposure = pred_freq_total / df["Exposure"].values

        X2 = self._build_X(df)
        for c in self.sev_columns_:
            if c not in X2.columns:
                X2[c] = 0.0
        X_sev = X2[self.sev_columns_]

        pred_sev = self.sev_result_.predict(X_sev)

        pure_premium = pred_freq_per_exposure * pred_sev
        return np.maximum(pure_premium, 1e-8)


class TweedieGLM:
    """
    Direct Tweedie GLM on pure premium with exposure weights.
    """

    def __init__(self, numerical_cols, categorical_cols, tweedie_power=TWEEDIE_POWER):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.tweedie_power = tweedie_power
        self.columns_ = None
        self.result_ = None

    def _build_X(self, df):
        X_num = df[self.numerical_cols].copy()
        X_cat = pd.get_dummies(df[self.categorical_cols], drop_first=True, dtype=float)
        X = pd.concat([X_num, X_cat], axis=1)
        X = sm.add_constant(X, has_constant="add")
        return X

    def fit(self, df, target_col="PurePremium", exposure_col="Exposure"):
        X = self._build_X(df)
        self.columns_ = X.columns.tolist()

        model = sm.GLM(
            df[target_col].values,
            X,
            family=sm.families.Tweedie(
                var_power=self.tweedie_power,
                link=sm.families.links.Log()
            ),
            # Y = L / e with Var(Y) = phi * mu^p / e -> exposure is a variance weight, not a
            # frequency (replication) weight. Point estimates are identical to freq_weights.
            var_weights=df[exposure_col].values
        )
        self.result_ = model.fit()
        return self

    def predict(self, df):
        X = self._build_X(df)
        for c in self.columns_:
            if c not in X.columns:
                X[c] = 0.0
        X = X[self.columns_]
        pred = self.result_.predict(X)
        return np.maximum(pred, 1e-8)


class TabularPreprocessor:
    def __init__(self, numerical_cols, categorical_cols):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.num_means_ = None
        self.num_stds_ = None
        self.cat_maps_ = {}

    def fit(self, df):
        X_num = df[self.numerical_cols].astype(float)
        self.num_means_ = X_num.mean().values
        self.num_stds_ = X_num.std().replace(0, 1).values

        for c in self.categorical_cols:
            vals = sorted(df[c].astype(str).unique())
            self.cat_maps_[c] = {v: i for i, v in enumerate(vals)}
        return self

    def transform(self, df):
        X_num = df[self.numerical_cols].astype(float).values
        X_num = (X_num - self.num_means_) / self.num_stds_

        X_cat = np.zeros((len(df), len(self.categorical_cols)), dtype=int)
        for j, c in enumerate(self.categorical_cols):
            mapping = self.cat_maps_[c]
            X_cat[:, j] = df[c].astype(str).map(mapping).fillna(0).astype(int).values

        return X_num, X_cat

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def category_sizes(self):
        return [len(self.cat_maps_[c]) for c in self.categorical_cols]


class EmbeddingBlock(nn.Module):
    def __init__(self, category_sizes, emb_dims=None):
        super().__init__()
        if emb_dims is None:
            emb_dims = [min(16, (k + 1) // 2) for k in category_sizes]
        self.emb_layers = nn.ModuleList([
            nn.Embedding(k, d) for k, d in zip(category_sizes, emb_dims)
        ])
        self.out_dim = sum(emb_dims)

    def forward(self, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        return torch.cat(embs, dim=1) if len(embs) > 0 else None


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "figure.facecolor": "white",
        "axes.facecolor": "white"
    })


# matplotlib defaults used for every figure (the figure modules update further keys)
set_paper_style()
