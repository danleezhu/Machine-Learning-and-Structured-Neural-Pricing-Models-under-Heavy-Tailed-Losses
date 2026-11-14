"""
paper2_models.py — model definitions and fitting (used by paper2_tuning.py, paper2_run_full.py and ga2m_plus.py).

Every tunable setting is passed EXPLICITLY through a configuration dictionary; a missing key raises an error instead
of falling back to a hidden default. Constants that are not tuned are listed below with their source.

Data
    prepare_data(): paper2_adjust_clean_plus.prepare_fremtpl2 + Density replaced by log(Density) for ALL models.
    The split (make_train_valid_test_stratified) depends only on ClaimAmount and Region, so it is unchanged.

Models
    GLM, Tweedie GLM          paper2_adjust_clean_plus (no hyper-parameters)
    EBM                       EBMWrapperV2: Tweedie(p) objective, interactions = 10, random_state = 42;
                              learning_rate / max_rounds from the configuration (None = installed interpret default);
                              every other argument = interpret default (outer bagging, internal validation split)
    GBM                       GBMWrapperV2: LightGBM Tweedie(p); num_leaves, min_child_samples and the categorical
                              treatment ('ordinal' = OrdinalEncoder codes used as numbers;
                              'native' = LightGBM categorical splits) from the configuration; the rest GBM_CONSTANTS
    LocalGLMnet               dummy-coded linear part, GLM-initialised global coefficients, smooth premium bound
    Neural GA2M               EBM's pairs (or the GA2M+ pairs), neural shape functions; optional nagging (average of
                              n_bag networks)
    Tweedie-CANN              log mu = log mu_GLM (frozen Tweedie GLM, offset) + delta(x), delta = B tanh(g(x) / B)

Shared network trainer (train_network): AdamW, weight decay only on weight matrices / embeddings, early stopping with
    criterion 'raw'   : validation Tweedie loss of the epoch
              'ma'    : moving average of the last ma_window validation losses
              'level' : validation Tweedie loss after multiplying the predictions by the training-set level factor
                        c = sum_train e y / sum_train e mu (the correction applied later as 'train_level'), so that the
                        stopping decision is not driven by the level noise of the validation set
    optional ReduceLROnPlateau (scheduler_patience None = off), optional gradient-norm clipping (None = off). The share
    of batches whose gradient norm exceeds 5 is recorded in every case (diagnostic of the clipping).
"""

import gc
import inspect
import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import statsmodels.api as sm

import paper2_adjust_clean_plus as p2

TARGET, EXPOSURE = "PurePremium", "Exposure"
NN_MODELS = ["LocalGLMnet", "Neural GA2M", "Tweedie-CANN"]
MODEL_ORDER = ["GLM", "Tweedie GLM", "EBM", "GBM", "LocalGLMnet", "Neural GA2M", "Tweedie-CANN"]
NN_SEED_OFFSET = {"LocalGLMnet": 11, "Neural GA2M": 22, "Tweedie-CANN": 33}      # model seed = 1000 * seed + offset

# ---- constants that are not tuned ---------------------------------------------------------------------------
TRAIN_CONSTANTS = dict(max_epochs=150,          # upper bound only; runs that reach it are flagged (stopped_by)
                       ma_window=5,             # window of the 'ma' criterion
                       min_rel_delta=1e-5,      # relative improvement needed to count as an improvement
                       sched_factor=0.5, min_lr=1e-5,   # ReduceLROnPlateau: halve, floor 1e-5
                       weight_decay=1e-5,
                       eval_batch=65536)        # memory only
TRAIN_REQUIRED = ("lr", "batch_size", "criterion", "patience", "scheduler_patience", "grad_clip")
CLIP_REFERENCE = 5.0                            # gradient-norm threshold; used for the diagnostic share
GBM_CONSTANTS = dict(learning_rate=0.01, n_estimators=5000, subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                     reg_alpha=0.5, reg_lambda=2.0, random_state=42, stopping_rounds=200)
EBM_CONSTANTS = dict(interactions=10, random_state=42)
ARCH_REQUIRED = {"LocalGLMnet": ("hidden_dims", "dropout", "eta_bound", "lambda_local"),
                 "Neural GA2M": ("hidden", "n_bag"),
                 "Tweedie-CANN": ("hidden_dims", "dropout", "delta_bound", "lambda_delta")}


def _require(d, keys, what):
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"{what}: missing setting(s) {missing}; every tunable setting must be given explicitly")


# =============================================================================================================
# Data
# =============================================================================================================
def prepare_data(df_freq, df_sev, log_density=True):
    df, num_cols, cat_cols = p2.prepare_fremtpl2(df_freq, df_sev)
    if log_density:
        dens = df["Density"].astype(float)
        if (dens <= 0).any():
            raise ValueError("Density must be positive to take logs")
        df["Density"] = np.log(dens)
    return df, num_cols, cat_cols


def clean(pred):
    return np.clip(np.nan_to_num(np.asarray(pred, float), nan=1e-6, posinf=1e6, neginf=1e-6), 1e-6, 1e6)


# =============================================================================================================
# Shared network trainer
# =============================================================================================================
def tweedie_nll_eta(y, eta, e, power):
    """e * [exp((2-p) eta)/(2-p) - y exp((1-p) eta)/(1-p)]: Tweedie negative log-likelihood up to terms free of mu."""
    eta = torch.clamp(eta, -15.0, 15.0)
    return e * (torch.exp((2.0 - power) * eta) / (2.0 - power) - y * torch.exp((1.0 - power) * eta) / (1.0 - power))


def make_tensors(prep, df, offset=None, target_col=TARGET, exposure_col=EXPOSURE):
    """(x_num, x_cat, y, exposure, offset); offset = log mu of a frozen base model (CANN) or 0."""
    Xn, Xc = prep.transform(df)
    off = np.zeros(len(df), np.float32) if offset is None else np.asarray(offset, np.float32)
    return (torch.as_tensor(np.asarray(Xn, np.float32)), torch.as_tensor(np.asarray(Xc, np.int64)),
            torch.as_tensor(df[target_col].to_numpy(np.float32)), torch.as_tensor(df[exposure_col].to_numpy(np.float32)),
            torch.as_tensor(off))


def _param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if (p.ndim < 2 or name.startswith("beta_global")) else decay).append(p)
    return decay, no_decay


def _clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def network_eta(model, tensors, batch=65536):
    """Full linear predictor (network output + offset) on a data set, as float64 numpy."""
    model.eval()
    xn, xc, _, _, off = tensors
    out = [(model(xn[s:s + batch], xc[s:s + batch]) + off[s:s + batch]).double().numpy()
           for s in range(0, len(xn), batch)]
    return np.concatenate(out)


def _weighted_loss(eta, y, e, power):
    """sum e l / sum e for float64 numpy inputs (the validation criterion)."""
    eta = np.clip(eta, -15.0, 15.0)
    l = e * (np.exp((2.0 - power) * eta) / (2.0 - power) - y * np.exp((1.0 - power) * eta) / (1.0 - power))
    return float(l.sum() / e.sum())


def train_network(model, train_t, valid_t, power, cfg, seed, tag="", verbose=False):
    _require(cfg, TRAIN_REQUIRED, f"{tag} training configuration")
    cfg = {**TRAIN_CONSTANTS, **cfg}
    if cfg["criterion"] not in ("raw", "ma", "level"):
        raise ValueError("criterion must be 'raw', 'ma' or 'level'")
    torch.manual_seed(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    decay, no_decay = _param_groups(model)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"])
    sched = None
    if cfg["scheduler_patience"] is not None:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg["sched_factor"],
                                                           patience=int(cfg["scheduler_patience"]), min_lr=cfg["min_lr"])
    clip = math.inf if cfg["grad_clip"] is None else float(cfg["grad_clip"])
    xn, xc, y, e, off = train_t
    y_tr, e_tr = y.double().numpy(), e.double().numpy()
    y_va, e_va = valid_t[2].double().numpy(), valid_t[3].double().numpy()
    n, bs = len(y), int(cfg["batch_size"])
    best_state, best_crit, best_epoch, wait = _clone_state(model), math.inf, 0, 0
    history, valid_hist = [], []
    n_skipped = n_batches = n_big = 0
    stopped_by = "max_epochs"

    for epoch in range(int(cfg["max_epochs"])):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tr_num = tr_den = 0.0
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            eta, pen = model(xn[idx], xc[idx], with_penalty=True)
            eta = eta + off[idx]
            l_obs = tweedie_nll_eta(y[idx], eta, e[idx], power)
            loss = l_obs.mean() + pen
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            n_batches += 1
            n_big += norm > CLIP_REFERENCE
            opt.step()
            tr_num += l_obs.detach().double().sum().item()
            tr_den += e[idx].double().sum().item()

        eta_va = network_eta(model, valid_t, cfg["eval_batch"])
        va = _weighted_loss(eta_va, y_va, e_va, power)
        valid_hist.append(va)
        rec = {"epoch": epoch + 1, "train_loss": tr_num / max(tr_den, 1e-12), "valid_loss": va,
               "lr": opt.param_groups[0]["lr"]}
        if cfg["criterion"] == "level":
            eta_tr = network_eta(model, train_t, cfg["eval_batch"])
            log_c = math.log(np.sum(e_tr * y_tr) / np.sum(e_tr * np.exp(np.clip(eta_tr, -30, 30))))
            crit = _weighted_loss(eta_va + log_c, y_va, e_va, power)
            rec.update({"valid_loss_level": crit, "train_level_factor": math.exp(log_c)})
        elif cfg["criterion"] == "ma":
            crit = float(np.mean(valid_hist[-int(cfg["ma_window"]):]))
        else:
            crit = va
        rec["criterion"] = crit
        history.append(rec)
        if verbose:
            print(f"[{tag}] epoch {epoch + 1}: train {rec['train_loss']:.5f} valid {va:.5f} crit {crit:.5f} "
                  f"lr {rec['lr']:.1e}")

        threshold = best_crit - cfg["min_rel_delta"] * abs(best_crit) if math.isfinite(best_crit) else math.inf
        if math.isfinite(crit) and crit < threshold:
            best_crit, best_epoch, wait = crit, epoch + 1, 0
            best_state = _clone_state(model)
        else:
            wait += 1
        if sched is not None and math.isfinite(crit):
            sched.step(crit)
        if wait >= int(cfg["patience"]):
            stopped_by = "patience"
            break

    model.load_state_dict(best_state)
    if n_skipped:
        warnings.warn(f"[{tag}] {n_skipped} batches skipped because of a non-finite loss")
    return model, {"history": history, "best_epoch": best_epoch, "epochs_run": len(history),
                   "best_criterion": best_crit, "stopped_by": stopped_by, "n_skipped_batches": n_skipped,
                   "share_grad_norm_over_5": n_big / max(n_batches, 1)}


def _to_mu(eta):
    return np.clip(np.exp(np.clip(np.asarray(eta, float), -30, 30)), 1e-6, 1e6)


# =============================================================================================================
# LocalGLMnet
# =============================================================================================================
class LocalGLMNetModule(nn.Module):
    """
    z(x) = (1, standardized numerical variables, dummies; reference = largest training exposure)
    eta(x) = (beta_global + s(x) d(x))^T z(x),  d = head(body(x_num, embeddings)),  r = d^T z,
    s = B tanh(r / B) / r  =>  eta(x) - beta_global^T z(x) = B tanh(r / B) in (-B, B).
    Penalty (training loss only): lambda * mean((s d)^2).
    """

    def __init__(self, n_num, category_sizes, ref_levels, beta_init, hidden_dims, dropout, eta_bound, lambda_local):
        super().__init__()
        self.n_num = int(n_num)
        self.category_sizes = [int(k) for k in category_sizes]
        self.eta_bound, self.lambda_local = eta_bound, float(lambda_local)
        self.emb = p2.EmbeddingBlock(self.category_sizes)
        layers, prev = [], self.n_num + self.emb.out_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        for c, (k, r) in enumerate(zip(self.category_sizes, ref_levels)):
            self.register_buffer(f"keep_{c}", torch.tensor([l for l in range(k) if l != int(r)], dtype=torch.long))
        self.n_coef = 1 + self.n_num + sum(k - 1 for k in self.category_sizes)
        beta_init = torch.as_tensor(np.asarray(beta_init, dtype=np.float32))
        if beta_init.numel() != self.n_coef:
            raise ValueError(f"beta_init has {beta_init.numel()} entries, expected {self.n_coef}")
        self.beta_global = nn.Parameter(beta_init.clone())
        self.head = nn.Linear(prev, self.n_coef)
        # zero output layer: d(x) = 0 at the start, so training starts EXACTLY at the Tweedie GLM (beta_global).
        # Its own gradient (dL/dd * h) is non-zero, so it moves at the first step and the body learns from then on.
        # (A small random init, std 1e-3, moved the starting premium by up to ~6% for policies with extreme z.)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def design(self, xn, xc):
        cols = [torch.ones(len(xn), 1, dtype=xn.dtype, device=xn.device), xn]
        for c, k in enumerate(self.category_sizes):
            cols.append(F.one_hot(xc[:, c], num_classes=k).to(xn.dtype)[:, getattr(self, f"keep_{c}")])
        return torch.cat(cols, dim=1)

    def components(self, xn, xc):
        z = self.design(xn, xc)
        d = self.head(self.body(torch.cat([xn, self.emb(xc)], dim=1)))
        if self.eta_bound is not None:
            B = float(self.eta_bound)
            r = (d * z).sum(1)
            small = r.abs() < 1e-6
            r_safe = torch.where(small, torch.ones_like(r), r)
            s = torch.where(small, torch.ones_like(r), B * torch.tanh(r_safe / B) / r_safe)
            d = s.unsqueeze(1) * d
        beta = self.beta_global.unsqueeze(0) + d
        return (beta * z).sum(1), beta, d, z

    def forward(self, xn, xc, with_penalty=False):
        eta, beta, d, z = self.components(xn, xc)
        return (eta, self.lambda_local * torch.mean(d ** 2)) if with_penalty else eta


class LocalGLMNetWrapper:
    def __init__(self, numerical_cols, categorical_cols, tweedie_power, arch, train, seed, verbose=False):
        _require(arch, ARCH_REQUIRED["LocalGLMnet"], "LocalGLMnet architecture")
        _require(train, TRAIN_REQUIRED, "LocalGLMnet training")
        self.numerical_cols, self.categorical_cols = list(numerical_cols), list(categorical_cols)
        self.tweedie_power, self.arch, self.train_cfg = tweedie_power, dict(arch), dict(train)
        self.seed, self.verbose = int(seed), verbose
        self.prep_ = p2.TabularPreprocessor(self.numerical_cols, self.categorical_cols)

    def _numpy_design(self, Xn, Xc):
        cols = [np.ones((len(Xn), 1)), np.asarray(Xn, float)]
        for c, k in enumerate(self.category_sizes_):
            pos = -np.ones(k, dtype=int)
            pos[self.keep_[c]] = np.arange(len(self.keep_[c]))
            col = pos[Xc[:, c]]
            D = np.zeros((len(Xc), len(self.keep_[c])))
            rows = np.flatnonzero(col >= 0)
            D[rows, col[rows]] = 1.0
            cols.append(D)
        return np.hstack(cols)

    def fit(self, train_df, valid_df, target_col=TARGET, exposure_col=EXPOSURE):
        Xn, Xc = self.prep_.fit_transform(train_df)
        self.category_sizes_ = self.prep_.category_sizes()
        e_tr, y_tr = train_df[exposure_col].to_numpy(float), train_df[target_col].to_numpy(float)
        self.ref_levels_ = [int(np.argmax(np.bincount(Xc[:, c], weights=e_tr, minlength=k)))
                            for c, k in enumerate(self.category_sizes_)]
        self.keep_ = [np.array([l for l in range(k) if l != r]) for k, r in zip(self.category_sizes_, self.ref_levels_)]
        names = ["intercept"] + self.numerical_cols
        for c, col in enumerate(self.categorical_cols):
            inv = {v: k for k, v in self.prep_.cat_maps_[col].items()}
            names += [f"{col}={inv[l]}" for l in self.keep_[c]]
        self.coef_names_ = names
        X = self._numpy_design(Xn, Xc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GLM(y_tr, X, family=sm.families.Tweedie(var_power=self.tweedie_power, link=sm.families.links.Log()),
                         var_weights=e_tr).fit()
        if not bool(getattr(res, "converged", True)):
            raise RuntimeError("LocalGLMnet: the GLM used for the initialisation did not converge")
        beta0 = np.asarray(res.params, float)
        self.glm_init_coef_ = pd.Series(beta0, index=self.coef_names_)
        del X
        gc.collect()
        torch.manual_seed(self.seed)
        a = self.arch
        self.model_ = LocalGLMNetModule(len(self.numerical_cols), self.category_sizes_, self.ref_levels_, beta0,
                                        tuple(a["hidden_dims"]), a["dropout"], a["eta_bound"], a["lambda_local"])
        tr_t, va_t = make_tensors(self.prep_, train_df), make_tensors(self.prep_, valid_df)
        self.model_, self.train_info_ = train_network(self.model_, tr_t, va_t, self.tweedie_power, self.train_cfg,
                                                      self.seed, tag="LocalGLMnet", verbose=self.verbose)
        self.best_iteration_ = self.train_info_["best_epoch"]
        return self

    def predict(self, df):
        return _to_mu(network_eta(self.model_, _xonly(self.prep_, df)))

    @torch.no_grad()
    def predict_local_coefficients(self, df, batch=65536):
        self.model_.eval()
        Xn, Xc = self.prep_.transform(df)
        xn, xc = torch.as_tensor(np.asarray(Xn, np.float32)), torch.as_tensor(np.asarray(Xc, np.int64))
        out = [self.model_.components(xn[s:s + batch], xc[s:s + batch])[1].numpy() for s in range(0, len(xn), batch)]
        return pd.DataFrame(np.vstack(out), columns=self.coef_names_, index=df.index)


def _xonly(prep, df, offset=None):
    """Tensors for prediction (targets not needed): y and exposure are dummies."""
    Xn, Xc = prep.transform(df)
    n = len(df)
    off = np.zeros(n, np.float32) if offset is None else np.asarray(offset, np.float32)
    return (torch.as_tensor(np.asarray(Xn, np.float32)), torch.as_tensor(np.asarray(Xc, np.int64)),
            torch.zeros(n), torch.ones(n), torch.as_tensor(off))


# =============================================================================================================
# Neural GA2M
# =============================================================================================================
def _mlp(in_dim, hidden, out_std=1e-3):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    last = nn.Linear(prev, 1)
    nn.init.normal_(last.weight, mean=0.0, std=out_std)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*layers, last)


class NeuralGA2MModule(nn.Module):
    """eta = b + sum_j f_j(x_j) + sum_k g_k(level_k) + sum_{(u,v) in EBM pairs} h_uv(x_u, x_v)."""

    def __init__(self, num_names, cat_names, category_sizes, pairs, init_mean, hidden):
        super().__init__()
        self.num_names, self.cat_names = list(num_names), list(cat_names)
        self.category_sizes = [int(k) for k in category_sizes]
        self.num_main = nn.ModuleList([_mlp(1, hidden) for _ in self.num_names])
        self.cat_main = nn.ModuleList([nn.Embedding(k, 1) for k in self.category_sizes])
        for emb in self.cat_main:
            nn.init.zeros_(emb.weight)
        self.pair_spec, self.pair_embs, self.pair_nets = [], nn.ModuleList(), nn.ModuleList()
        for a, b in pairs:
            spec, embs, in_dim = [], nn.ModuleList(), 0
            for v in (a, b):
                if v in self.num_names:
                    spec.append(("num", self.num_names.index(v)))
                    in_dim += 1
                elif v in self.cat_names:
                    c = self.cat_names.index(v)
                    k = self.category_sizes[c]
                    d = min(8, max(2, (k + 1) // 2))
                    emb = nn.Embedding(k, d)
                    nn.init.normal_(emb.weight, mean=0.0, std=0.1)
                    embs.append(emb)
                    spec.append(("cat", c))
                    in_dim += d
                else:
                    raise ValueError(f"pair member {v!r} is not a model feature")
            self.pair_spec.append(spec)
            self.pair_embs.append(embs)
            self.pair_nets.append(_mlp(in_dim, hidden))
        self.bias = nn.Parameter(torch.tensor(float(np.log(max(init_mean, 1e-6)))))
        self.term_names = self.num_names + self.cat_names + [f"{a} x {b}" for a, b in pairs]

    def terms(self, xn, xc):
        out = [net(xn[:, j:j + 1]).squeeze(1) for j, net in enumerate(self.num_main)]
        out += [emb(xc[:, c]).squeeze(1) for c, emb in enumerate(self.cat_main)]
        for spec, embs, net in zip(self.pair_spec, self.pair_embs, self.pair_nets):
            parts, ei = [], 0
            for kind, i in spec:
                if kind == "num":
                    parts.append(xn[:, i:i + 1])
                else:
                    parts.append(embs[ei](xc[:, i]))
                    ei += 1
            out.append(net(torch.cat(parts, dim=1)).squeeze(1))
        return torch.stack(out, dim=1)

    def forward(self, xn, xc, with_penalty=False):
        eta = self.bias + self.terms(xn, xc).sum(1)
        return (eta, eta.new_zeros(())) if with_penalty else eta


class NeuralGA2MWrapper:
    """
    n_bag networks with different initialisations (member b uses seed + 100000 b); the prediction is the arithmetic
    mean of their means (nagging predictor, Richman & Wuthrich). n_bag = 1 is a single network.
    """

    def __init__(self, numerical_cols, categorical_cols, tweedie_power, pairs, arch, train, seed, verbose=False):
        _require(arch, ARCH_REQUIRED["Neural GA2M"], "Neural GA2M architecture")
        _require(train, TRAIN_REQUIRED, "Neural GA2M training")
        if pairs is None:
            raise ValueError("Neural GA2M needs the EBM interaction pairs of the same split")
        self.numerical_cols, self.categorical_cols = list(numerical_cols), list(categorical_cols)
        self.tweedie_power, self.pairs = tweedie_power, [tuple(p) for p in pairs]
        self.arch, self.train_cfg, self.seed, self.verbose = dict(arch), dict(train), int(seed), verbose
        self.prep_ = p2.TabularPreprocessor(self.numerical_cols, self.categorical_cols)

    def fit(self, train_df, valid_df, target_col=TARGET, exposure_col=EXPOSURE):
        self.prep_.fit(train_df)
        tr_t, va_t = make_tensors(self.prep_, train_df), make_tensors(self.prep_, valid_df)
        init_mean = float(np.average(train_df[target_col], weights=train_df[exposure_col]))
        self.models_, infos = [], []
        for b in range(int(self.arch["n_bag"])):
            s = self.seed + 100000 * b
            torch.manual_seed(s)
            m = NeuralGA2MModule(self.numerical_cols, self.categorical_cols, self.prep_.category_sizes(), self.pairs,
                                 init_mean, tuple(self.arch["hidden"]))
            m, info = train_network(m, tr_t, va_t, self.tweedie_power, self.train_cfg, s, tag=f"Neural GA2M #{b}",
                                    verbose=self.verbose)
            self.models_.append(m)
            infos.append(info)
        self.train_info_ = dict(infos[0])
        self.train_info_.update({"member_best_epochs": [i["best_epoch"] for i in infos],
                                 "member_epochs_run": [i["epochs_run"] for i in infos],
                                 "share_grad_norm_over_5": float(np.mean([i["share_grad_norm_over_5"] for i in infos]))})
        self.best_iteration_ = float(np.mean(self.train_info_["member_best_epochs"]))
        return self

    def predict(self, df):
        t = _xonly(self.prep_, df)
        return np.mean([_to_mu(network_eta(m, t)) for m in self.models_], axis=0)

    @torch.no_grad()
    def term_contributions(self, df, batch=65536):
        """Centred additive contributions on the log scale (exact decomposition of log mu for n_bag = 1; for
        n_bag > 1 the member average, an approximation of the log of the averaged prediction)."""
        Xn, Xc = self.prep_.transform(df)
        xn, xc = torch.as_tensor(np.asarray(Xn, np.float32)), torch.as_tensor(np.asarray(Xc, np.int64))
        T = np.zeros((len(df), len(self.models_[0].term_names)))
        icpt = 0.0
        for m in self.models_:
            m.eval()
            Tm = np.vstack([m.terms(xn[s:s + batch], xc[s:s + batch]).double().numpy() for s in range(0, len(xn), batch)])
            mean_m = Tm.mean(0)
            T += (Tm - mean_m) / len(self.models_)
            icpt += (float(m.bias.detach()) + mean_m.sum()) / len(self.models_)
        out = pd.DataFrame(T, columns=self.models_[0].term_names, index=df.index)
        out.insert(0, "intercept", icpt)
        return out


# =============================================================================================================
# Tweedie-CANN (GLM offset + bounded neural correction)
# =============================================================================================================
class CANNModule(nn.Module):
    """delta(x) = B tanh(g(x) / B), g = MLP(x_num, embeddings); the frozen GLM enters as an offset in the trainer."""

    def __init__(self, n_num, category_sizes, hidden_dims, dropout, delta_bound, lambda_delta):
        super().__init__()
        self.emb = p2.EmbeddingBlock([int(k) for k in category_sizes])
        layers, prev = [], int(n_num) + self.emb.out_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.body = nn.Sequential(*layers)
        self.out = nn.Linear(prev, 1)
        nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)        # start (almost) at the GLM
        nn.init.zeros_(self.out.bias)
        self.delta_bound, self.lambda_delta = delta_bound, float(lambda_delta)

    def forward(self, xn, xc, with_penalty=False):
        g = self.out(self.body(torch.cat([xn, self.emb(xc)], dim=1))).squeeze(1)
        delta = g if self.delta_bound is None else float(self.delta_bound) * torch.tanh(g / float(self.delta_bound))
        return (delta, self.lambda_delta * torch.mean(delta ** 2)) if with_penalty else delta


class TweedieCANNWrapper:
    """log mu = log mu_GLM + delta(x). The Tweedie GLM is fitted on the training set (or a fitted one is passed)."""

    def __init__(self, numerical_cols, categorical_cols, tweedie_power, arch, train, seed, glm=None, verbose=False):
        _require(arch, ARCH_REQUIRED["Tweedie-CANN"], "Tweedie-CANN architecture")
        _require(train, TRAIN_REQUIRED, "Tweedie-CANN training")
        self.numerical_cols, self.categorical_cols = list(numerical_cols), list(categorical_cols)
        self.tweedie_power, self.arch, self.train_cfg = tweedie_power, dict(arch), dict(train)
        self.seed, self.glm_, self.verbose = int(seed), glm, verbose
        self.prep_ = p2.TabularPreprocessor(self.numerical_cols, self.categorical_cols)

    def _log_glm(self, df):
        return np.log(clean(self.glm_.predict(df)))

    def fit(self, train_df, valid_df, target_col=TARGET, exposure_col=EXPOSURE):
        if self.glm_ is None:
            self.glm_ = p2.TweedieGLM(self.numerical_cols, self.categorical_cols, tweedie_power=self.tweedie_power).fit(
                train_df, target_col=target_col, exposure_col=exposure_col)
        self.prep_.fit(train_df)
        tr_t = make_tensors(self.prep_, train_df, self._log_glm(train_df))
        va_t = make_tensors(self.prep_, valid_df, self._log_glm(valid_df))
        torch.manual_seed(self.seed)
        a = self.arch
        self.model_ = CANNModule(len(self.numerical_cols), self.prep_.category_sizes(), tuple(a["hidden_dims"]),
                                 a["dropout"], a["delta_bound"], a["lambda_delta"])
        self.model_, self.train_info_ = train_network(self.model_, tr_t, va_t, self.tweedie_power, self.train_cfg,
                                                      self.seed, tag="Tweedie-CANN", verbose=self.verbose)
        self.best_iteration_ = self.train_info_["best_epoch"]
        return self

    def predict(self, df):
        return _to_mu(network_eta(self.model_, _xonly(self.prep_, df, self._log_glm(df))))

    @torch.no_grad()
    def delta(self, df):
        self.model_.eval()
        t = _xonly(self.prep_, df)
        return np.concatenate([self.model_(t[0][s:s + 65536], t[1][s:s + 65536]).double().numpy()
                               for s in range(0, len(df), 65536)])


# =============================================================================================================
# EBM and GBM
# =============================================================================================================
def ebm_library_defaults():
    """learning_rate and max_rounds defaults of the INSTALLED interpret version (read, not assumed)."""
    from interpret.glassbox import ExplainableBoostingRegressor
    sig = inspect.signature(ExplainableBoostingRegressor.__init__).parameters
    return {k: sig[k].default for k in ("learning_rate", "max_rounds", "outer_bags", "validation_size",
                                         "early_stopping_rounds") if k in sig}


def extract_ebm_pairs(ebm, feature_names):
    groups = getattr(ebm, "term_features_", None)
    if groups is None:
        groups = getattr(ebm, "feature_groups_", None)
    if groups is None:
        raise AttributeError("cannot find the EBM terms (term_features_ / feature_groups_)")
    names = getattr(ebm, "feature_names_in_", None)
    names = list(feature_names if names is None else names)
    return [(str(names[g[0]]), str(names[g[1]])) for g in groups if len(g) == 2]


class EBMWrapperV2:
    """
    params: {"learning_rate": float or None, "max_rounds": int or None}; None = interpret default (read from the
    installed version). interactions = 10 and random_state = 42; everything else interpret default.
    """

    def __init__(self, numerical_cols, categorical_cols, tweedie_power, params):
        _require(params, ("learning_rate", "max_rounds"), "EBM configuration")
        from interpret.glassbox import ExplainableBoostingRegressor
        self.numerical_cols, self.categorical_cols = list(numerical_cols), list(categorical_cols)
        kw = dict(objective=f"tweedie_deviance:variance_power={tweedie_power}", **EBM_CONSTANTS)
        for k in ("learning_rate", "max_rounds"):
            if params[k] is not None:
                kw[k] = params[k]
        self.kwargs_ = kw
        self.model_ = ExplainableBoostingRegressor(**kw)

    def fit(self, df, target_col=TARGET, exposure_col=EXPOSURE):
        X = df[self.numerical_cols + self.categorical_cols]
        y, w = df[target_col].to_numpy(float), df[exposure_col].to_numpy(float)
        self.model_.fit(X, y, sample_weight=w)
        pred = self.model_.predict(X)
        level = np.average(pred, weights=w) / np.average(y, weights=w)
        if np.any(pred <= 0) or not (0.5 < level < 2.0):
            raise RuntimeError(f"EBM sanity check failed (min pred {pred.min():.3g}, level ratio {level:.3f})")
        self.interaction_pairs_ = extract_ebm_pairs(self.model_, self.numerical_cols + self.categorical_cols)
        bi = getattr(self.model_, "best_iteration_", None)
        cap = self.model_.get_params().get("max_rounds")
        self.ebm_rounds_ = None
        if bi is not None:
            # interpret counts boosting STEPS (one step per term; max_rounds is in ROUNDS of n_terms steps):
            # best_iteration_ has shape (n_stages, n_outer_bags); stage 0 = main effects, stage 1 = pairs.
            # Converted to rounds so that it is comparable with max_rounds.
            bi = np.atleast_2d(np.asarray(bi, float))
            n_terms = [len(self.numerical_cols) + len(self.categorical_cols), max(len(self.interaction_pairs_), 1)]
            rounds = np.vstack([bi[s] / n_terms[min(s, 1)] for s in range(bi.shape[0])])
            self.ebm_rounds_ = {"max_rounds": cap, "best_iteration_max": float(np.nanmax(rounds)),
                                "best_iteration_median": float(np.nanmedian(rounds[0])),   # main-effect stage
                                "share_bags_at_cap": float(np.mean(rounds >= 0.98 * cap)) if cap else np.nan}
            self.best_iteration_ = float(np.nanmedian(rounds[0]))
        else:
            self.best_iteration_ = np.nan
        return self

    def predict(self, df):
        return np.maximum(self.model_.predict(df[self.numerical_cols + self.categorical_cols]), 1e-8)


class GBMWrapperV2:
    """
    LightGBM Tweedie(p). params: {"num_leaves", "min_child_samples", "categorical": "ordinal" | "native"}.
    Other settings: GBM_CONSTANTS. Categorical variables are integer-coded by OrdinalEncoder in both cases
    (unknown levels -> -1); 'native' passes them to LightGBM as categorical features (negative codes = missing).
    """

    def __init__(self, numerical_cols, categorical_cols, tweedie_power, params):
        _require(params, ("num_leaves", "min_child_samples", "categorical"), "GBM configuration")
        if params["categorical"] not in ("ordinal", "native"):
            raise ValueError("categorical must be 'ordinal' or 'native'")
        import lightgbm as lgb
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OrdinalEncoder
        self.numerical_cols, self.categorical_cols = list(numerical_cols), list(categorical_cols)
        self.params = dict(params)
        c = GBM_CONSTANTS
        self.preprocessor_ = ColumnTransformer([("num", "passthrough", self.numerical_cols),
                                                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                                       unknown_value=-1), self.categorical_cols)])
        self.model_ = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=tweedie_power,
                                        learning_rate=c["learning_rate"], n_estimators=c["n_estimators"],
                                        num_leaves=int(params["num_leaves"]),
                                        min_child_samples=int(params["min_child_samples"]),
                                        subsample=c["subsample"], subsample_freq=c["subsample_freq"],
                                        colsample_bytree=c["colsample_bytree"], reg_alpha=c["reg_alpha"],
                                        reg_lambda=c["reg_lambda"], random_state=c["random_state"],
                                        force_row_wise=True, verbose=-1)

    def fit(self, train_df, valid_df, target_col=TARGET, exposure_col=EXPOSURE):
        import lightgbm as lgb
        cols = self.numerical_cols + self.categorical_cols
        X_tr = self.preprocessor_.fit_transform(train_df[cols])
        X_va = self.preprocessor_.transform(valid_df[cols])
        cat_idx = list(range(len(self.numerical_cols), len(cols))) if self.params["categorical"] == "native" else "auto"
        self.model_.fit(X_tr, train_df[target_col].to_numpy(float), sample_weight=train_df[exposure_col].to_numpy(float),
                        eval_set=[(X_va, valid_df[target_col].to_numpy(float))],
                        eval_sample_weight=[valid_df[exposure_col].to_numpy(float)], eval_metric="tweedie",
                        categorical_feature=cat_idx,
                        callbacks=[lgb.early_stopping(GBM_CONSTANTS["stopping_rounds"], first_metric_only=True,
                                                      verbose=False)])
        self.best_iteration_ = self.model_.best_iteration_
        return self

    def predict(self, df):
        Xp = self.preprocessor_.transform(df[self.numerical_cols + self.categorical_cols])
        return np.maximum(self.model_.predict(Xp, num_iteration=self.best_iteration_), 1e-8)


# =============================================================================================================
# Factory
# =============================================================================================================
def model_seed(name, seed):
    return 1000 * int(seed) + NN_SEED_OFFSET[name]


def build_model(name, num_cols, cat_cols, seed, power, configs, ebm_pairs=None, tweedie_glm=None, verbose=False):
    """configs: {"EBM": {...}, "GBM": {...}, "LocalGLMnet": {"arch": {...}, "train": {...}}, ...}."""
    if name == "GLM":
        return p2.PoissonGammaGLM(num_cols, cat_cols)
    if name == "Tweedie GLM":
        return p2.TweedieGLM(num_cols, cat_cols, tweedie_power=power)
    if name not in configs:
        raise KeyError(f"no configuration for {name}")
    c = configs[name]
    if name == "EBM":
        return EBMWrapperV2(num_cols, cat_cols, power, c)
    if name == "GBM":
        return GBMWrapperV2(num_cols, cat_cols, power, c)
    if name == "LocalGLMnet":
        return LocalGLMNetWrapper(num_cols, cat_cols, power, c["arch"], c["train"], model_seed(name, seed), verbose)
    if name == "Neural GA2M":
        return NeuralGA2MWrapper(num_cols, cat_cols, power, ebm_pairs, c["arch"], c["train"], model_seed(name, seed),
                                 verbose)
    if name == "Tweedie-CANN":
        return TweedieCANNWrapper(num_cols, cat_cols, power, c["arch"], c["train"], model_seed(name, seed),
                                  glm=tweedie_glm, verbose=verbose)
    raise ValueError(name)


def fit_model(name, model, train_df, valid_df):
    """Fits a model with the right signature; GLM / Tweedie GLM / EBM use the training set only."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        if name == "GLM":
            model.fit(train_df)
        elif name in ("Tweedie GLM", "EBM"):
            model.fit(train_df, target_col=TARGET, exposure_col=EXPOSURE)
        else:
            model.fit(train_df, valid_df, target_col=TARGET, exposure_col=EXPOSURE)
    return model


def predict(model, df):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return clean(model.predict(df))
