import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


"""
Phase-wise feature analysis vs. val_accs.

Schema
------
    arch, val_accs,
    phase{p}_triangle
    phase{p}_diamond
    phase{p}_line
    phase{p}_only_merge_at_output
    phase{p}_input_to_output_edge     -> 5 binary motif flags (0/1)
    phase{p}_output_merge_count
    phase{p}_input_fork_count
    phase{p}_missing_nodes_count      -> 3 integer counts
  for p in {0, 1, 2}

Missingness convention (IMPORTANT)
----------------------------------
    NaN  = the phase does not exist for that architecture
    0    = motif absent / count is zero, in a phase that DOES exist

Absent phases are excluded from per-phase statistics; `n_phases` is kept as
its own predictor so phase count is never silently confounded with phase
content.

Usage
-----
    df  = pd.read_csv("your_data.csv")
    res = run(df, min_support=5)
    df2 = res["df_with_recs"]
"""

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 0. Schema
# ----------------------------------------------------------------------
N_PHASES = 3
TARGET = "val_accs"
ID_COLS = ["arch"]

BIN_FEATS = [
    "triangle",
    "diamond",
    "line",
    "only_merge_at_output",
    "input_to_output_edge",
]
CNT_FEATS = [
    "output_merge_count",
    "input_fork_count",
    "missing_nodes_count",
]
ALL_FEATS = BIN_FEATS + CNT_FEATS

MAX_LEVELS = 8          # above this many distinct values -> quantile bins

SHORT = {
    "triangle": "tri",
    "diamond": "diam",
    "line": "line",
    "only_merge_at_output": "merge@out",
    "input_to_output_edge": "in\u2192out",
    "output_merge_count": "n_merge",
    "input_fork_count": "n_fork",
    "missing_nodes_count": "n_missing",
}


def pcol(phase, feat):
    return f"phase{phase}_{feat}"


def phase_cols(phase):
    return [pcol(phase, f) for f in ALL_FEATS]


def check_schema(df):
    """Validate columns, dtypes and the NaN-means-absent convention."""
    expected = [TARGET] + [pcol(p, f) for p in range(N_PHASES) for f in ALL_FEATS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise KeyError(f"missing expected columns: {missing}")

    for p in range(N_PHASES):
        for f in BIN_FEATS:
            vals = set(pd.unique(df[pcol(p, f)].dropna()))
            if not vals <= {0, 1, 0.0, 1.0}:
                warnings.warn(
                    f"{pcol(p, f)} is not 0/1: {sorted(vals)[:8]} "
                    "-- consider moving it to CNT_FEATS"
                )
        # partial missingness: some but not all columns NaN within a phase
        nan_mask = df[phase_cols(p)].isna()
        partial = nan_mask.any(axis=1) & ~nan_mask.all(axis=1)
        if partial.any():
            warnings.warn(
                f"phase {p}: {int(partial.sum())} row(s) have SOME columns NaN. "
                "NaN should mark whole-phase absence; these rows are dropped "
                "from phase-level stats."
            )
    return True


def coerce_dtypes(df):
    """Nullable ints so NaN can coexist with integer flags/counts."""
    df = df.copy()
    for p in range(N_PHASES):
        for f in BIN_FEATS:
            df[pcol(p, f)] = df[pcol(p, f)].astype("Int8")
        for f in CNT_FEATS:
            df[pcol(p, f)] = df[pcol(p, f)].astype("Int16")
    return df


# ----------------------------------------------------------------------
# 1. Phase presence, phase count, wide -> long
# ----------------------------------------------------------------------
def add_phase_count(df):
    """A phase is present iff its columns are non-NaN (all of them)."""
    df = df.copy()
    for p in range(N_PHASES):
        df[f"phase{p}_present"] = df[phase_cols(p)].notna().all(axis=1).astype(int)
    df["n_phases"] = df[[f"phase{p}_present" for p in range(N_PHASES)]].sum(axis=1)
    return df


def to_long(df):
    """One row per (architecture, present phase)."""
    frames = []
    keep = [c for c in ID_COLS if c in df.columns] + [TARGET, "n_phases"]
    for p in range(N_PHASES):
        ren = {pcol(p, f): f for f in ALL_FEATS}
        sub = df.loc[df[f"phase{p}_present"] == 1, list(ren) + keep].copy()
        sub = sub.rename(columns=ren)
        sub["phase"] = p
        sub["sample_id"] = sub.index
        frames.append(sub)

    long = pd.concat(frames, ignore_index=True)
    assert long[ALL_FEATS].notna().all().all(), "NaN inside a present phase"
    long[BIN_FEATS] = long[BIN_FEATS].astype(int)
    long[CNT_FEATS] = long[CNT_FEATS].astype(float)
    return long


def phase_count_effect(df):
    """Is n_phases alone driving val_accs? (confounder check)"""
    out = (df.groupby("n_phases")[TARGET]
             .agg(mean="mean", std="std", n="count", median="median")
             .reset_index())
    groups = [g[TARGET].dropna().values for _, g in df.groupby("n_phases")
              if len(g) > 2]
    if len(groups) >= 2:
        out.attrs["kruskal_p"] = stats.kruskal(*groups).pvalue
    return out


# ----------------------------------------------------------------------
# 2. Marginal effects
# ----------------------------------------------------------------------
def _bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    if pv.size == 0:
        return q
    order = np.argsort(pv)
    ranked = np.empty_like(pv)
    ranked[order] = (pv[order] * len(pv) / (np.arange(len(pv)) + 1)).clip(max=1.0)
    q[ok] = np.minimum.accumulate(ranked[::-1])[::-1]
    return q


def binary_effects(long, min_group=5):
    """On/off contrast of each motif flag on val_accs, per phase."""
    rows = []
    for p, feat in itertools.product(range(N_PHASES), BIN_FEATS):
        d = long[long["phase"] == p]
        on = d.loc[d[feat] == 1, TARGET].dropna()
        off = d.loc[d[feat] == 0, TARGET].dropna()
        rec = dict(phase=p, feature=feat, n_on=len(on), n_off=len(off),
                   mean_on=on.mean() if len(on) else np.nan,
                   mean_off=off.mean() if len(off) else np.nan)
        if len(on) < min_group or len(off) < min_group:
            rows.append({**rec, "delta": np.nan, "cohens_d": np.nan,
                         "p_value": np.nan, "p_mannwhitney": np.nan,
                         "note": "insufficient support"})
            continue
        _, pv_t = stats.ttest_ind(on, off, equal_var=False)           # Welch
        _, pv_u = stats.mannwhitneyu(on, off, alternative="two-sided")
        pooled = np.sqrt(((len(on) - 1) * on.var(ddof=1) +
                          (len(off) - 1) * off.var(ddof=1)) /
                         (len(on) + len(off) - 2))
        rows.append({**rec,
                     "delta": on.mean() - off.mean(),
                     "cohens_d": (on.mean() - off.mean()) / pooled if pooled > 0 else np.nan,
                     "p_value": pv_t, "p_mannwhitney": pv_u, "note": ""})
    out = pd.DataFrame(rows)
    out["q_value"] = _bh_fdr(out["p_value"])
    return out.sort_values(["phase", "delta"], ascending=[True, False])


def count_effects(long, min_group=5):
    """Counts are ordinal: monotone trend + per-level (or per-bin) means."""
    trend, levels = [], []
    for p, feat in itertools.product(range(N_PHASES), CNT_FEATS):
        d = long[long["phase"] == p].dropna(subset=[feat, TARGET])
        if len(d) < 10 or d[feat].nunique() < 2:
            continue
        rho, pv_s = stats.spearmanr(d[feat], d[TARGET])
        r, pv_p = stats.pearsonr(d[feat], d[TARGET])
        trend.append(dict(phase=p, feature=feat, n=len(d),
                          spearman_rho=rho, p_value=pv_s,
                          pearson_r=r, p_pearson=pv_p,
                          n_levels=d[feat].nunique()))

        if d[feat].nunique() <= MAX_LEVELS:
            key, mode = d[feat], "level"
        else:
            key = pd.qcut(d[feat], q=min(5, d[feat].nunique()), duplicates="drop")
            mode = "bin"

        lv = (d.groupby(key, observed=True)[TARGET]
                .agg(mean="mean", std="std", n="count").reset_index())
        lv.columns = ["value", "mean", "std", "n"]
        lv["value"] = lv["value"].astype(str)
        lv.insert(0, "kind", mode)
        lv.insert(0, "feature", feat)
        lv.insert(0, "phase", p)
        levels.append(lv[lv["n"] >= min_group])

    trend = pd.DataFrame(trend)
    if not trend.empty:
        trend["q_value"] = _bh_fdr(trend["p_value"])
    levels = pd.concat(levels, ignore_index=True) if levels else pd.DataFrame()
    return trend, levels


# ----------------------------------------------------------------------
# 3. Combinations and interactions
# ----------------------------------------------------------------------
def _config_label(bits):
    on = [SHORT[f] for f, b in zip(BIN_FEATS, bits) if b == "1"]
    return "+".join(on) if on else "(none)"


def combination_table(long, min_support=5):
    """Exact 5-bit motif configuration -> val_accs, per phase."""
    d = long.copy()
    d["config"] = d[BIN_FEATS].astype(int).astype(str).agg("".join, axis=1)
    g = (d.groupby(["phase", "config"])[TARGET]
           .agg(mean="mean", std="std", n="count", median="median").reset_index())
    g = g[g["n"] >= min_support].copy()
    if g.empty:
        return g
    # Lower confidence bound penalises configs that look good on few rows.
    g["lcb95"] = g["mean"] - 1.96 * g["std"].fillna(0) / np.sqrt(g["n"])
    g["label"] = g["config"].apply(_config_label)
    return g.sort_values(["phase", "lcb95"], ascending=[True, False])


def pairwise_interactions(long, min_support=5):
    """Deviation of the joint (1,1) cell from the additive expectation."""
    rows = []
    for p in range(N_PHASES):
        d = long[long["phase"] == p]
        base = d[TARGET].mean()
        for a, b in itertools.combinations(BIN_FEATS, 2):
            cells = {}
            ok = True
            for va, vb in itertools.product([0, 1], [0, 1]):
                s = d.loc[(d[a] == va) & (d[b] == vb), TARGET].dropna()
                if len(s) < min_support:
                    ok = False
                    break
                cells[(va, vb)] = s.mean()
            if not ok:
                continue
            add_expect = (cells[(1, 0)] + cells[(0, 1)]) - cells[(0, 0)]
            rows.append(dict(phase=p, feat_a=a, feat_b=b,
                             m00=cells[(0, 0)], m10=cells[(1, 0)],
                             m01=cells[(0, 1)], m11=cells[(1, 1)],
                             expected_11=add_expect,
                             interaction=cells[(1, 1)] - add_expect,
                             baseline=base))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out["interaction"].abs().sort_values(ascending=False).index)
    return out


def cross_phase_combinations(df, min_support=5):
    """Does a motif's value depend on WHICH phase it sits in?"""
    d = add_phase_count(df)
    rows = []
    for feat in BIN_FEATS:
        cols = [pcol(p, feat) for p in range(N_PHASES)]
        pat = (d[cols].fillna(0).astype(int).astype(str).agg("".join, axis=1))
        g = (d.assign(pattern=pat).groupby(["pattern"])[TARGET]
               .agg(mean="mean", std="std", n="count").reset_index())
        g = g[g["n"] >= min_support]
        g.insert(0, "feature", feat)
        rows.append(g)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out.sort_values(["feature", "mean"], ascending=[True, False])


# ----------------------------------------------------------------------
# 4. Multivariate model (optional, for ranking)
# ----------------------------------------------------------------------
def fit_model(df):
    """Gradient boosting on the wide frame; NaN handled natively."""
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
    except ImportError:
        warnings.warn("scikit-learn unavailable; skipping model")
        return None, np.nan, pd.DataFrame()

    feats = [pcol(p, f) for p in range(N_PHASES) for f in ALL_FEATS] + ["n_phases"]
    d = add_phase_count(df)
    X = d[feats].astype(float)
    y = d[TARGET].astype(float)
    ok = y.notna()
    X, y = X[ok], y[ok]

    model = HistGradientBoostingRegressor(max_depth=4, random_state=0)
    r2 = float(np.mean(cross_val_score(model, X, y, cv=5, scoring="r2")))
    model.fit(X, y)

    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(model, X, y, n_repeats=10, random_state=0)
        imp = (pd.DataFrame({"feature": feats,
                             "importance": pi.importances_mean,
                             "std": pi.importances_std})
                 .sort_values("importance", ascending=False))
    except Exception:
        imp = pd.DataFrame()
    return model, r2, imp


# ----------------------------------------------------------------------
# 5. Recommended configuration + new columns
# ----------------------------------------------------------------------
def best_config_per_phase(long, combos, min_support=5, strategy="config"):
    """Pick the best motif pattern per phase, then the best count level in it."""
    rec = {}
    for p in range(N_PHASES):
        d = long[long["phase"] == p]

        if strategy == "config" and not combos.empty:
            c = combos[combos["phase"] == p]
            if not c.empty:
                bits = c.loc[c["lcb95"].idxmax(), "config"]
                flags = {f: int(b) for f, b in zip(BIN_FEATS, bits)}
            else:
                strategy_p = "marginal"
                flags = None
            if c.empty:
                flags = None
        else:
            flags = None

        if flags is None:                       # marginal fallback
            flags = {}
            for f in BIN_FEATS:
                on = d.loc[d[f] == 1, TARGET].mean()
                off = d.loc[d[f] == 0, TARGET].mean()
                flags[f] = int(np.nan_to_num(on, nan=-np.inf) >
                               np.nan_to_num(off, nan=-np.inf))

        # counts chosen CONDITIONAL on the selected motif pattern
        mask = np.ones(len(d), dtype=bool)
        for f, v in flags.items():
            mask &= (d[f] == v).to_numpy()
        sub = d[mask] if mask.sum() >= min_support else d

        counts = {}
        for f in CNT_FEATS:
            lv = sub.groupby(f)[TARGET].agg(["mean", "count"])
            lv = lv[lv["count"] >= max(2, min_support // 2)]
            counts[f] = float(lv["mean"].idxmax()) if not lv.empty else float(sub[f].median())

        rec[p] = {**flags, **counts}
    return rec


def add_recommended_columns(df, rec):
    """rec_phase{p}_{feat} columns + Hamming distance to the recommendation."""
    d = add_phase_count(df)
    dist = np.zeros(len(d))
    for p in range(N_PHASES):
        present = d[f"phase{p}_present"] == 1
        for f in ALL_FEATS:
            col = f"rec_{pcol(p, f)}"
            d[col] = np.where(present, rec[p][f], np.nan)
        for f in BIN_FEATS:
            actual = d[pcol(p, f)].astype("float")
            dist += np.where(present & (actual != rec[p][f]), 1.0, 0.0)
    d["rec_distance"] = dist
    return d


def validate_recommendation(df_rec):
    """val_accs should fall monotonically as distance from the recipe grows."""
    return (df_rec.groupby("rec_distance")[TARGET]
                  .agg(mean="mean", std="std", n="count")
                  .reset_index())


# ----------------------------------------------------------------------
# 6. Plots  (per-bar alpha is baked into RGBA -- matplotlib needs scalar alpha)
# ----------------------------------------------------------------------
def _rgba(colors, alphas):
    return [(*mcolors.to_rgb(c), float(a)) for c, a in zip(colors, alphas)]


def plot_binary_effects(eff):
    fig, axes = plt.subplots(1, N_PHASES, figsize=(4.2 * N_PHASES, 3.6),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for p, ax in enumerate(axes):
        d = eff[(eff["phase"] == p)].dropna(subset=["delta"])
        if d.empty:
            ax.set_title(f"phase {p}: no support")
            ax.axis("off")
            continue
        d = d.sort_values("delta")
        lab = d["feature"].map(SHORT).tolist()
        delta = d["delta"].to_numpy(dtype=float)
        sig = (d["q_value"].fillna(1).to_numpy() < 0.05)
        base = np.where(delta > 0, "#2a9d5c", "#c0392b")
        ax.barh(lab, delta, color=_rgba(base, np.where(sig, 1.0, 0.35)))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"phase {p}")
        ax.set_xlabel("Δ val_accs (on − off)")
    fig.suptitle("Motif on/off effect (solid = FDR q < 0.05)", y=1.03)
    fig.tight_layout()
    return fig


def plot_count_levels(levels):
    if levels.empty:
        return None
    fig, axes = plt.subplots(len(CNT_FEATS), N_PHASES,
                             figsize=(4.0 * N_PHASES, 3.0 * len(CNT_FEATS)),
                             squeeze=False)
    for i, feat in enumerate(CNT_FEATS):
        for p in range(N_PHASES):
            ax = axes[i][p]
            d = levels[(levels["phase"] == p) & (levels["feature"] == feat)]
            if d.empty:
                ax.axis("off")
                continue
            err = (d["std"] / np.sqrt(d["n"])).fillna(0).to_numpy()
            ax.errorbar(range(len(d)), d["mean"].to_numpy(), yerr=err,
                        marker="o", capsize=3, color="#2b6cb0")
            ax.set_xticks(range(len(d)))
            ax.set_xticklabels(d["value"], rotation=45, ha="right", fontsize=8)
            ax.set_title(f"phase {p} · {SHORT[feat]}", fontsize=9)
            if p == 0:
                ax.set_ylabel("mean val_accs")
    fig.tight_layout()
    return fig


def plot_top_configs(combos, top_n=8):
    if combos.empty:
        return None
    fig, axes = plt.subplots(1, N_PHASES, figsize=(5.0 * N_PHASES, 4.0))
    axes = np.atleast_1d(axes)
    for p, ax in enumerate(axes):
        d = combos[combos["phase"] == p].nlargest(top_n, "lcb95").iloc[::-1]
        if d.empty:
            ax.set_title(f"phase {p}: no support")
            ax.axis("off")
            continue
        err = (d["std"] / np.sqrt(d["n"])).fillna(0).to_numpy()
        ax.barh(d["label"], d["mean"].to_numpy(), xerr=err,
                color="#4c78a8", alpha=0.85)
        for y, n in enumerate(d["n"]):
            ax.text(ax.get_xlim()[0], y, f" n={n}", va="center", fontsize=7)
        ax.set_title(f"phase {p}: top configs")
        ax.set_xlabel("mean val_accs")
    fig.tight_layout()
    return fig


def plot_phase_count(pc):
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    err = (pc["std"] / np.sqrt(pc["n"])).fillna(0).to_numpy()
    ax.bar(pc["n_phases"].astype(str), pc["mean"], yerr=err,
           color="#6a51a3", alpha=0.85, capsize=4)
    ax.set_xlabel("n_phases")
    ax.set_ylabel("mean val_accs")
    ax.set_title("val_accs by phase count")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# 7. Driver
# ----------------------------------------------------------------------
def run(df, min_support=5, strategy="config", make_plots=True, verbose=True):
    check_schema(df)
    df = coerce_dtypes(df)
    df = add_phase_count(df)
    long = to_long(df)

    pc = phase_count_effect(df)
    eff = binary_effects(long, min_group=min_support)
    trend, levels = count_effects(long, min_group=min_support)
    combos = combination_table(long, min_support=min_support)
    inter = pairwise_interactions(long, min_support=min_support)
    cross = cross_phase_combinations(df, min_support=min_support)
    model, r2, imp = fit_model(df)

    rec = best_config_per_phase(long, combos, min_support=min_support,
                                strategy=strategy)
    df_rec = add_recommended_columns(df, rec)
    check = validate_recommendation(df_rec)

    if verbose:
        print("=== val_accs by n_phases  (confounder check) ===")
        print(pc.to_string(index=False))
        print("kruskal p:", pc.attrs.get("kruskal_p"))

        print("\n=== binary motif effects (by |delta|) ===")
        e = eff.dropna(subset=["delta"])
        print(e.reindex(e["delta"].abs().sort_values(ascending=False).index)
               .to_string(index=False))

        print("\n=== count feature trends ===")
        print(trend.to_string(index=False) if not trend.empty else "(none)")

        print("\n=== count feature per-level means ===")
        print(levels.to_string(index=False) if not levels.empty else "(none)")

        print("\n=== top configs per phase (by lcb95) ===")
        if not combos.empty:
            print(combos.groupby("phase").head(5).to_string(index=False))

        print("\n=== strongest pairwise interactions ===")
        print(inter.head(10).to_string(index=False) if not inter.empty else "(none)")

        print(f"\n=== model CV R^2: {r2:.3f} ===")
        if not imp.empty:
            print(imp.head(10).to_string(index=False))

        print("\n=== recommended configuration ===")
        for p in range(N_PHASES):
            print(f"  phase {p}: {rec[p]}")

        print("\n=== validation: val_accs vs distance from recipe ===")
        print(check.to_string(index=False))

    if make_plots:
        plot_phase_count(pc)
        plot_binary_effects(eff)
        plot_count_levels(levels)
        plot_top_configs(combos)
        plt.show()

    return dict(df=df, long=long, phase_count=pc, binary=eff, trend=trend,
                levels=levels, combos=combos, interactions=inter,
                cross_phase=cross, importance=imp, model=model, r2=r2,
                recipe=rec, df_with_recs=df_rec, validation=check)


#usage
# res = run(result_df, min_support=5)


def analyze_features(
    result_df,
    target_col="val_accs",
    top_n=5,
    top_importances=15,
    save_plots=True,
):
    """
    Run PCA + feature-importance analysis on architecture feature table,
    relating structural features to a chosen target metric (e.g. val_accs
    or flops).

    Parameters
    ----------
    result_df : pd.DataFrame
        Must contain "arch", `target_col`, and structural feature columns
        (binary 0/1 or non-negative counts). NaNs are allowed where a
        phase (e.g. phase2) doesn't exist for that arch.
    target_col : str
        Which column to treat as the target/metric to analyze against
        (e.g. "val_accs" or "flops"). Any other numeric metric columns
        present in result_df besides this one are excluded from the
        feature set automatically (see KNOWN_METRIC_COLS below) so they
        don't leak in as spurious "features".
    top_n : int
        Number of top-`target_col` architectures to annotate on the PCA scatter.
    top_importances : int
        Number of top features to show in the importance bar chart.
    save_plots : bool
        Whether to save figures to disk (pca_scatter.png, feature_importance.png,
        per_feature_scatter.png).

    Returns
    -------
    dict with keys:
        "X"            : feature matrix used for modeling (NaNs -> -1 sentinel)
        "X_pca"        : 2D PCA projection
        "pca"          : fitted PCA object
        "rf"           : fitted RandomForestRegressor
        "importances"  : pd.Series of feature importances, sorted descending
    """

    # Columns that represent *metrics/targets* rather than structural
    # features. If result_df has both val_accs and flops (or others), only
    # target_col is used as y; the rest of these are dropped from the
    # feature matrix so one metric doesn't get used to "predict" another.
    KNOWN_METRIC_COLS = {"val_accs", "flops", "arch"}

    if target_col not in result_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in result_df columns")

    exclude_cols = KNOWN_METRIC_COLS | {target_col}
    feature_cols = [c for c in result_df.columns if c not in exclude_cols]

    # -------------------------------------------------------------
    # 1. Prepare feature matrix
    # -------------------------------------------------------------
    X_raw = result_df[feature_cols].copy()

    # NaNs appear where a phase (e.g. phase2) doesn't exist for that arch.
    # All real feature values are either binary (0/1) or non-negative counts,
    # so filling missing values with 0 would be indistinguishable from a real
    # "False"/"zero count" value. Instead we use -1 as an explicit sentinel
    # meaning "not applicable / phase absent", which no real value can take.
    # We also add an explicit "phase_present" flag so the model can use that
    # information directly and doesn't have to infer it from the sentinel.
    #
    # Phases are detected dynamically from column names (e.g. "phase0_x",
    # "phase3_y", "phase12_z", ...) instead of a hardcoded list, so this
    # works regardless of how many phases exist in a given dataset.
    phase_pattern = re.compile(r"^(phase\d+)_")
    phases = sorted({
        m.group(1) for c in feature_cols
        if (m := phase_pattern.match(c))
    })

    for phase in phases:
        phase_cols = [c for c in feature_cols if c.startswith(f"{phase}_")]
        if phase_cols:
            X_raw[f"{phase}_present"] = X_raw[phase_cols].notna().any(axis=1).astype(int)

    X = X_raw.fillna(-1)
    y = result_df[target_col].values

    # -------------------------------------------------------------
    # 2. PCA (feature reduction) on standardized features
    # -------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_scaled)

    print(f"Explained variance ratio (PC1, PC2): {pca.explained_variance_ratio_}")
    print(f"Total variance captured: {pca.explained_variance_ratio_.sum():.2%}")

    # -------------------------------------------------------------
    # 3. Scatter plot: PC1 vs PC2, colored by target_col
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", s=60, edgecolor="k", alpha=0.85)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(target_col)

    # annotate top-N architectures by target_col
    top_idx = np.argsort(y)[-top_n:]
    for i in top_idx:
        ax.annotate(str(result_df["arch"].iloc[i]), (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Architectures in PCA space, colored by {target_col}")
    plt.tight_layout()
    if save_plots:
        plt.savefig("pca_scatter.png", dpi=150)
    plt.show()

    # -------------------------------------------------------------
    # 4. Feature importance: which structural features drive target_col?
    # -------------------------------------------------------------
    # Note: Random Forest is tree-based, so it naturally handles the -1
    # sentinel as "just another value to split on" without assuming any
    # ordinal relationship to 0/1 - it can still learn "phase absent" as
    # a meaningful split. This avoids the distortion 0-filling would cause.
    rf = RandomForestRegressor(n_estimators=500, random_state=0)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(f"\nTop features driving {target_col}:")
    print(importances.head(top_importances))

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.head(top_importances).sort_values().plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature importance (Random Forest)")
    ax.set_title(f"Top structural features predicting {target_col}")
    plt.tight_layout()
    if save_plots:
        plt.savefig("feature_importance.png", dpi=150)
    plt.show()

    # -------------------------------------------------------------
    # 5. Per-feature scatter vs target_col (grid of subplots)
    # -------------------------------------------------------------
    # Plot from result_df (with real NaNs, not the -1-filled X) so gaps in the
    # scatter honestly show "phase not present" rather than a fake data point.
    n_cols = 4
    n_rows = int(np.ceil(len(feature_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, feature_cols):
        ax.scatter(result_df[col], y, s=15, alpha=0.6)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel(target_col)

    for ax in axes[len(feature_cols):]:
        ax.axis("off")

    plt.tight_layout()
    if save_plots:
        plt.savefig("per_feature_scatter.png", dpi=150)
    plt.show()

    return {
        "X": X,
        "X_pca": X_pca,
        "pca": pca,
        "rf": rf,
        "importances": importances,
    }


def analyze_generational_trends(
    result_df,
    generation_col="generation",
    val_col="val_accs",
    top_frac=0.3,
    top_n_plots=12,
    save_plots=True,
):
    """
    Track how structural features evolve across generations and whether their
    prevalence trend correlates with the trend in val_accs. Also identifies
    which features are simply *maintained* (stable) across generations,
    independent of any relationship to val_accs.

    Parameters
    ----------
    result_df : pd.DataFrame
        Must contain `generation`, `val_accs`, and feature columns (binary
        0/1 or numeric counts; NaN allowed for phase-absent features).
    generation_col : str
        Column identifying generation number (0, 1, 2, ...).
    val_col : str
        Column with validation accuracy.
    top_frac : float
        Fraction of top performers (by val_accs) per generation to use for
        the "elite persistence" view - i.e. which features survive/spread
        among the best architectures over time.
    top_n_plots : int
        How many top-correlated / top-maintained features to show in plots
        and printed tables.
    save_plots : bool
        Whether to save figures to disk.

    Returns
    -------
    dict with keys:
        "gen_means"         : per-generation mean of every feature + val_accs
        "elite_gen_means"   : same, but computed only over top performers per generation
        "overall_corr"      : Series, correlation of each feature's per-gen mean with val_accs per-gen mean
        "elite_corr"        : same, using elite-only per-generation means
        "persistence"       : DataFrame of stability stats per feature (mean_level, cv, slope, min, max)
        "maintained_ranked" : persistence, sorted by cv ascending (excludes always-absent features)
    """
    feature_cols = [c for c in result_df.columns
                    if c not in ("arch", val_col, generation_col)]

    df = result_df.copy()

    # Force feature + val columns to numeric float64. Upstream steps (e.g.
    # concatenating True/False/NaN, or a phase column that is entirely NaN
    # for some rows) can leave a column as object dtype even though every
    # value is numeric-looking. That silently breaks things like
    # np.polyfit (UFuncTypeError: dtype('O') -> float64) and .std()/.corr()
    # in less obvious ways. Cast defensively right up front.
    for c in feature_cols + [val_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------------------------------------------
    # 1. Per-generation means (overall population)
    # -------------------------------------------------------------
    gen_means = df.groupby(generation_col)[feature_cols + [val_col]].mean().sort_index()

    # -------------------------------------------------------------
    # 2. Per-generation means among top performers only ("elites")
    #    -> shows which features persist/spread among the best architectures
    #
    #    NOTE: avoid groupby(...).apply(...) here. In pandas 2.2+, apply()
    #    excludes the grouping column from the group passed into the
    #    function by default (heading toward include_groups=False), which
    #    silently drops `generation_col` and breaks the regroup below with
    #    a KeyError. Rank-based filtering sidesteps this entirely.
    # -------------------------------------------------------------
    rank_desc = df.groupby(generation_col)[val_col].rank(method="first", ascending=False)
    group_sizes = df.groupby(generation_col)[val_col].transform("size")
    k_per_row = np.ceil(group_sizes * top_frac).clip(lower=1)

    elite_mask = rank_desc <= k_per_row
    elite_df = df.loc[elite_mask]

    elite_gen_means = elite_df.groupby(generation_col)[feature_cols + [val_col]].mean().sort_index()

    # -------------------------------------------------------------
    # 3. Correlate each feature's per-generation trend with val_accs trend
    # -------------------------------------------------------------
    def trend_correlation(means_df):
        corr = {}
        for col in feature_cols:
            valid = means_df[[col, val_col]].dropna()
            if len(valid) >= 3 and valid[col].std() > 0:
                corr[col] = valid[col].corr(valid[val_col])
            else:
                corr[col] = np.nan
        return pd.Series(corr).sort_values(key=lambda s: s.abs(), ascending=False)

    overall_corr = trend_correlation(gen_means)
    elite_corr = trend_correlation(elite_gen_means)

    print("Correlation of feature's per-generation mean with val_accs per-generation mean (overall population):")
    print(overall_corr.head(top_n_plots))
    print("\nSame, computed only over top performers per generation (elite persistence):")
    print(elite_corr.head(top_n_plots))

    # -------------------------------------------------------------
    # 3b. Persistence: which features are *maintained* across generations,
    #     regardless of whether they track val_accs at all.
    #
    #     For each feature, computed across its per-generation means:
    #       - mean_level : average prevalence/value across all generations
    #       - cv         : coefficient of variation (std/mean) -> LOW cv means
    #                      the feature's level barely changes gen-to-gen
    #                      (maintained); HIGH cv means it fluctuates a lot
    #       - slope      : linear trend slope (near 0 = flat/maintained,
    #                      far from 0 = steadily growing or disappearing)
    #       - min/max    : range of the per-generation mean across generations
    #
    #     A feature is "maintained" if it has a stable, non-trivial mean_level
    #     (not stuck at 0) AND low cv AND a near-zero slope.
    # -------------------------------------------------------------
    def persistence_stats(means_df):
        stats = {}
        for col in feature_cols:
            vals = means_df[col].dropna()
            if len(vals) < 2:
                continue

            # Defensive cast: groupby/mean on an object-dtype column (which
            # can happen if a source column had NaNs mixed with bool/int
            # upstream) can yield an object-dtype Series here too. np.polyfit
            # and .std() need real float64, not python objects.
            vals = vals.astype(float)

            mean_level = float(vals.mean())
            std_level = float(vals.std())
            cv = (std_level / mean_level) if mean_level != 0 else np.nan
            gens = vals.index.values.astype(float)
            y_vals = vals.values.astype(float)
            slope = np.polyfit(gens, y_vals, 1)[0] if std_level > 0 else 0.0
            stats[col] = {
                "mean_level": mean_level,
                "std_level": std_level,
                "cv": cv,
                "slope": slope,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        return pd.DataFrame(stats).T

    persistence = persistence_stats(gen_means)
    # Rank "most maintained": lowest cv, excluding features that are
    # essentially always-absent (mean_level ~ 0), which trivially have low
    # cv but aren't meaningfully "maintained" - they're just never there.
    maintained_ranked = persistence[persistence["mean_level"] > 1e-6].sort_values("cv")

    print("\nFeature persistence across generations (lower cv = more consistently maintained):")
    print(maintained_ranked[["mean_level", "cv", "slope", "min", "max"]].head(top_n_plots))

    # -------------------------------------------------------------
    # 4. Plot: val_accs trend across generations
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gen_means.index, gen_means[val_col], marker="o", label="population mean")
    ax.plot(elite_gen_means.index, elite_gen_means[val_col], marker="s", label=f"top {int(top_frac*100)}% mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel(val_col)
    ax.set_title("val_accs across generations")
    ax.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("val_accs_by_generation.png", dpi=150)
    plt.show()

    # -------------------------------------------------------------
    # 5. Grid of trend plots for top-correlated features (dual axis:
    #    feature prevalence/mean vs val_accs, both across generations)
    # -------------------------------------------------------------
    top_features = overall_corr.dropna().head(top_n_plots).index.tolist()
    if top_features:
        n_cols = 4
        n_rows = int(np.ceil(len(top_features) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for ax, col in zip(axes, top_features):
            ax2 = ax.twinx()
            ax.plot(gen_means.index, gen_means[col], color="tab:blue", marker="o", label="feature")
            ax2.plot(gen_means.index, gen_means[val_col], color="tab:red", linestyle="--", marker="x", label="val_accs")
            ax.set_title(f"{col}\n(r={overall_corr[col]:.2f})", fontsize=9)
            ax.set_xlabel("Generation")
            ax.tick_params(axis="y", labelcolor="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:red")

        for ax in axes[len(top_features):]:
            ax.axis("off")

        plt.tight_layout()
        if save_plots:
            plt.savefig("feature_trends_vs_val_accs.png", dpi=150)
        plt.show()

    # -------------------------------------------------------------
    # 6. Heatmap: feature prevalence across generations, ordered by
    #    persistence (most-maintained at top) so drift/stability is
    #    visible at a glance across the whole feature set.
    # -------------------------------------------------------------
    heatmap_cols = maintained_ranked.index.tolist()
    if heatmap_cols:
        heat_data = gen_means[heatmap_cols].T  # rows=features, cols=generations

        fig, ax = plt.subplots(figsize=(1.0 * len(gen_means.index) + 3, 0.35 * len(heatmap_cols) + 2))
        im = ax.imshow(heat_data.values, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(heat_data.columns)))
        ax.set_xticklabels(heat_data.columns)
        ax.set_yticks(range(len(heat_data.index)))
        ax.set_yticklabels(heat_data.index, fontsize=8)
        ax.set_xlabel("Generation")
        ax.set_title("Feature prevalence/mean across generations\n(sorted: most-maintained (low cv) at top)")
        fig.colorbar(im, ax=ax, label="mean value")
        plt.tight_layout()
        if save_plots:
            plt.savefig("feature_persistence_heatmap.png", dpi=150)
        plt.show()

    return {
        "gen_means": gen_means,
        "elite_gen_means": elite_gen_means,
        "overall_corr": overall_corr,
        "elite_corr": elite_corr,
        "persistence": persistence,
        "maintained_ranked": maintained_ranked,
    }


# Example usage:
# results = analyze_generational_trends(result_df)
# results["maintained_ranked"].head(10)   # most consistently maintained features
# results["overall_corr"].head(10)        # features most correlated with val_accs trend

# Example usage:
# results = analyze_features(result_df)
# results["importances"].head(10)
def analyze_generational_trendss(
    result_df,
    generation_col="generation",
    val_col="val_accs",
    top_frac=0.3,
    top_n_plots=12,
    save_plots=True,
):
    """
    Track how structural features evolve across generations and whether their
    prevalence trend correlates with the trend in val_accs.

    Parameters
    ----------
    result_df : pd.DataFrame
        Must contain `generation`, `val_accs`, and feature columns (binary
        0/1 or numeric counts; NaN allowed for phase-absent features).
    generation_col : str
        Column identifying generation number (0, 1, 2, ...).
    val_col : str
        Column with validation accuracy.
    top_frac : float
        Fraction of top performers (by val_accs) per generation to use for
        the "elite persistence" view - i.e. which features survive/spread
        among the best architectures over time.
    top_n_plots : int
        How many top-correlated features to show in the trend-grid plot.
    save_plots : bool
        Whether to save figures to disk.

    Returns
    -------
    dict with keys:
        "gen_means"        : per-generation mean of every feature + val_accs
        "elite_gen_means"  : same, but computed only over top performers per generation
        "overall_corr"     : Series, correlation of each feature's per-gen mean with val_accs per-gen mean
        "elite_corr"       : same, using elite-only per-generation means
    """
    feature_cols = [c for c in result_df.columns
                    if c not in ("arch", val_col, generation_col)]

    df = result_df.copy()

    # -------------------------------------------------------------
    # 1. Per-generation means (overall population)
    # -------------------------------------------------------------
    gen_means = df.groupby(generation_col)[feature_cols + [val_col]].mean().sort_index()

    # -------------------------------------------------------------
    # 2. Per-generation means among top performers only ("elites")
    #    -> shows which features persist/spread among the best architectures
    #
    #    NOTE: avoid groupby(...).apply(...) here. In pandas 2.2+, apply()
    #    excludes the grouping column from the group passed into the
    #    function by default (heading toward include_groups=False), which
    #    silently drops `generation_col` and breaks the regroup below with
    #    a KeyError. Rank-based filtering sidesteps this entirely.
    # -------------------------------------------------------------
    rank_desc = df.groupby(generation_col)[val_col].rank(method="first", ascending=False)
    group_sizes = df.groupby(generation_col)[val_col].transform("size")
    k_per_row = np.ceil(group_sizes * top_frac).clip(lower=1)

    elite_mask = rank_desc <= k_per_row
    elite_df = df.loc[elite_mask]

    elite_gen_means = elite_df.groupby(generation_col)[feature_cols + [val_col]].mean().sort_index()

    # -------------------------------------------------------------
    # 3. Correlate each feature's per-generation trend with val_accs trend
    # -------------------------------------------------------------
    def trend_correlation(means_df):
        corr = {}
        for col in feature_cols:
            valid = means_df[[col, val_col]].dropna()
            if len(valid) >= 3 and valid[col].std() > 0:
                corr[col] = valid[col].corr(valid[val_col])
            else:
                corr[col] = np.nan
        return pd.Series(corr).sort_values(key=lambda s: s.abs(), ascending=False)

    overall_corr = trend_correlation(gen_means)
    elite_corr = trend_correlation(elite_gen_means)

    print("Correlation of feature's per-generation mean with val_accs per-generation mean (overall population):")
    print(overall_corr.head(top_n_plots))
    print("\nSame, computed only over top performers per generation (elite persistence):")
    print(elite_corr.head(top_n_plots))

    # -------------------------------------------------------------
    # 4. Plot: val_accs trend across generations
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gen_means.index, gen_means[val_col], marker="o", label="population mean")
    ax.plot(elite_gen_means.index, elite_gen_means[val_col], marker="s", label=f"top {int(top_frac*100)}% mean")
    ax.set_xlabel("Generation")
    ax.set_ylabel(val_col)
    ax.set_title("val_accs across generations")
    ax.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("val_accs_by_generation.png", dpi=150)
    plt.show()

    # -------------------------------------------------------------
    # 5. Grid of trend plots for top-correlated features (dual axis:
    #    feature prevalence/mean vs val_accs, both across generations)
    # -------------------------------------------------------------
    top_features = overall_corr.dropna().head(top_n_plots).index.tolist()
    n_cols = 4
    n_rows = int(np.ceil(len(top_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, top_features):
        ax2 = ax.twinx()
        ax.plot(gen_means.index, gen_means[col], color="tab:blue", marker="o", label="feature")
        ax2.plot(gen_means.index, gen_means[val_col], color="tab:red", marker="x", linestyle="--", label="val_accs")
        ax.set_title(f"{col}\n(r={overall_corr[col]:.2f})", fontsize=9)
        ax.set_xlabel("Generation")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    for ax in axes[len(top_features):]:
        ax.axis("off")

    plt.tight_layout()
    if save_plots:
        plt.savefig("feature_trends_by_generation.png", dpi=150)
    plt.show()

    return {
        "gen_means": gen_means,
        "elite_gen_means": elite_gen_means,
        "overall_corr": overall_corr,
        "elite_corr": elite_corr,
    }

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def analyze_top_model_lineage(
    result_df,
    generation_col="generation",
    val_col="val_accs",
    top_n_numeric_plots=12,
    save_plots=True,
):
    """
    Track the single BEST architecture (by val_accs) in each generation and
    see which structural features are shared/continued across those top-1
    architectures over time.

    Unlike analyze_generational_trends (which averages features over a group,
    e.g. top 30% per generation), this looks at literal individual top-1 rows
    per generation - i.e. "does the winning architecture's feature set carry
    over into the next generation's winning architecture?"

    Binary (0/1) features and numeric (count) features are handled
    differently, since squashing a count like output_merge_count=5 down to
    just "present" (1) throws away the magnitude that actually matters for
    that kind of feature:

    - Binary features -> presence/absence per generation's winner, with
      continuity score (fraction of generations present) and longest
      consecutive-presence streak.
    - Numeric features -> the winner's actual raw value per generation is
      tracked directly (no 0/1 squashing), plus stability stats (mean, cv,
      slope) and a line plot showing the value's trajectory across
      generations so you can see literal continuity of the value itself,
      not just "on/off".

    Parameters
    ----------
    result_df : pd.DataFrame
        Must contain `generation`, `val_accs`, "arch", and feature columns.
    generation_col : str
        Column identifying generation number (0, 1, 2, ...).
    val_col : str
        Column with validation accuracy (or whichever metric defines "best").
    top_n_numeric_plots : int
        How many numeric features to show in the value-continuity line-plot
        grid (ranked by lowest cv = most stable/continuous first).
    save_plots : bool
        Whether to save figures to disk.

    Returns
    -------
    dict with keys:
        "top_per_gen"          : DataFrame, one row per generation = the best
                                  architecture in that generation (arch,
                                  val_accs, all feature columns)
        "binary_cols"          : list of feature columns treated as binary
        "numeric_cols"         : list of feature columns treated as numeric
        "presence_matrix"      : DataFrame (generations x binary features),
                                  1/0/NaN showing whether each binary feature
                                  was present in that generation's winner
        "continuity"           : Series per binary feature - fraction of
                                  generations (among those where applicable)
                                  where the winner had that feature present.
                                  1.0 = present in every generation's winner.
        "streaks"              : dict per binary feature -> longest
                                  consecutive run of generations present
        "numeric_value_matrix" : DataFrame (generations x numeric features) -
                                  the winner's actual raw value per generation
                                  (NOT squashed to 0/1)
        "numeric_trend_stats"  : DataFrame per numeric feature - mean_level,
                                  cv (stability of magnitude across winners),
                                  slope (trend direction), min, max
    """
    feature_cols = [c for c in result_df.columns
                    if c not in ("arch", val_col, generation_col)]

    df = result_df.copy()
    for c in feature_cols + [val_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------------------------------------------
    # 1. Pull literal top-1 architecture per generation
    # -------------------------------------------------------------
    idx = df.groupby(generation_col)[val_col].idxmax()
    top_per_gen = df.loc[idx].set_index(generation_col).sort_index()

    print("Top architecture per generation:")
    print(top_per_gen[["arch", val_col]])

    # -------------------------------------------------------------
    # 2. Classify features as binary (0/1 only, ignoring NaN) vs numeric
    # -------------------------------------------------------------
    binary_cols, numeric_cols = [], []
    for col in feature_cols:
        vals = set(df[col].dropna().unique().tolist())
        if vals and vals.issubset({0.0, 1.0}):
            binary_cols.append(col)
        else:
            numeric_cols.append(col)

    print(f"\nTreating as binary flags ({len(binary_cols)}): {binary_cols}")
    print(f"Treating as numeric counts ({len(numeric_cols)}): {numeric_cols}")

    # -------------------------------------------------------------
    # 3. Binary features: presence matrix + continuity + streaks
    # -------------------------------------------------------------
    presence_matrix = pd.DataFrame(index=top_per_gen.index)
    continuity = pd.Series(dtype=float)
    streaks = {}

    if binary_cols:
        presence_matrix = top_per_gen[binary_cols].apply(
            lambda col: col.where(col.isna(), (col > 0).astype(float))
        )

        continuity = presence_matrix.mean(skipna=True).sort_values(ascending=False)

        print("\nFeature continuity across winning architectures (1.0 = always present in the top model, 0.0 = never):")
        print(continuity)

        def longest_streak(series):
            best = cur = 0
            for v in series:
                if v == 1:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 0
            return best

        streaks = {
            col: longest_streak(presence_matrix[col].fillna(0).values)
            for col in binary_cols
        }
        streaks = dict(sorted(streaks.items(), key=lambda kv: kv[1], reverse=True))

        print("\nLongest consecutive-generation streak of presence in the winning architecture:")
        for k, v in list(streaks.items())[:15]:
            print(f"  {k}: {v} generations")

    # -------------------------------------------------------------
    # 4. Numeric features: track actual raw value per generation (no 0/1
    #    squashing), plus stability stats
    # -------------------------------------------------------------
    numeric_value_matrix = pd.DataFrame(index=top_per_gen.index)
    numeric_trend_stats = pd.DataFrame()

    if numeric_cols:
        numeric_value_matrix = top_per_gen[numeric_cols]

        stats = {}
        for col in numeric_cols:
            vals = numeric_value_matrix[col].dropna().astype(float)
            if len(vals) < 2:
                continue
            mean_level = float(vals.mean())
            std_level = float(vals.std())
            cv = (std_level / mean_level) if mean_level > 1e-6 else np.nan
            gens = vals.index.values.astype(float)
            slope = np.polyfit(gens, vals.values, 1)[0] if std_level > 0 else 0.0
            stats[col] = {
                "mean_level": mean_level,
                "std_level": std_level,
                "cv": cv,
                "slope": slope,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

        numeric_trend_stats = pd.DataFrame(stats).T.sort_values("cv", na_position="last")

        print("\nNumeric feature value trend across winning architectures (sorted by cv = stability of magnitude; "
              "low cv means the winner's value for this feature stayed fairly continuous/consistent across generations):")
        print(numeric_trend_stats)

    # -------------------------------------------------------------
    # 5. Plot: val_accs of the winner across generations
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(top_per_gen.index, top_per_gen[val_col], marker="o", color="tab:green")
    for gen, row in top_per_gen.iterrows():
        ax.annotate(str(row["arch"]), (gen, row[val_col]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Generation")
    ax.set_ylabel(val_col)
    ax.set_title("Best architecture's val_accs per generation")
    plt.tight_layout()
    if save_plots:
        plt.savefig("top_model_val_accs_by_generation.png", dpi=150)
    plt.show()

    # -------------------------------------------------------------
    # 6. Heatmap: BINARY feature presence (1/0/NaN) in the winning
    #    architecture, rows = features (sorted by continuity), columns =
    #    generations
    # -------------------------------------------------------------
    if binary_cols:
        ordered_cols = continuity.index.tolist()
        heat_data = presence_matrix[ordered_cols].T

        fig, ax = plt.subplots(figsize=(max(8, len(top_per_gen) * 0.5), max(6, len(ordered_cols) * 0.3)))
        im = ax.imshow(heat_data.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(len(ordered_cols)))
        ax.set_yticklabels(ordered_cols, fontsize=8)
        ax.set_xticks(range(len(heat_data.columns)))
        ax.set_xticklabels(heat_data.columns, fontsize=8)
        ax.set_xlabel("Generation")
        ax.set_title("BINARY feature presence in each generation's WINNING architecture\n(green=present, red=absent, white=phase not applicable)")
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(["absent", "present"])
        plt.tight_layout()
        if save_plots:
            plt.savefig("top_model_feature_lineage_heatmap.png", dpi=150)
        plt.show()

    # -------------------------------------------------------------
    # 7. Heatmap: NUMERIC feature values (normalized per-feature 0-1 across
    #    its own min/max) in the winning architecture, rows = features
    #    (sorted by cv, most stable/continuous first), columns = generations
    # -------------------------------------------------------------
    if numeric_cols:
        ordered_numeric = numeric_trend_stats.index.tolist()
        raw = numeric_value_matrix[ordered_numeric].T.astype(float)

        # normalize each row (feature) to its own 0-1 range so the color
        # scale is meaningful regardless of the feature's absolute scale
        row_min = raw.min(axis=1)
        row_max = raw.max(axis=1)
        row_range = (row_max - row_min).replace(0, np.nan)
        norm = raw.sub(row_min, axis=0).div(row_range, axis=0)

        fig, ax = plt.subplots(figsize=(max(8, len(top_per_gen) * 0.5), max(6, len(ordered_numeric) * 0.3)))
        im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_yticks(range(len(ordered_numeric)))
        ax.set_yticklabels(
            [f"{c} (cv={numeric_trend_stats.loc[c, 'cv']:.2f})" if pd.notna(numeric_trend_stats.loc[c, "cv"])
             else f"{c} (cv=n/a)" for c in ordered_numeric],
            fontsize=8,
        )
        ax.set_xticks(range(len(norm.columns)))
        ax.set_xticklabels(norm.columns, fontsize=8)
        ax.set_xlabel("Generation")
        ax.set_title("NUMERIC feature values in each generation's WINNING architecture\n"
                     "(color = value normalized to that feature's own min-max range; rows sorted by stability/cv)")
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
        cbar.ax.set_yticklabels(["min", "max"])
        plt.tight_layout()
        if save_plots:
            plt.savefig("top_model_numeric_feature_heatmap.png", dpi=150)
        plt.show()

    # -------------------------------------------------------------
    # 8. Line-plot grid: literal value continuity of each numeric feature
    #    across generations, for the winning architecture. This shows the
    #    actual trajectory (not just on/off), so you can see whether the
    #    value climbs, drops, or holds steady from one generation's winner
    #    to the next.
    # -------------------------------------------------------------
    if numeric_cols:
        plot_cols = numeric_trend_stats.index.tolist()[:top_n_numeric_plots]
        n_cols_grid = 4
        n_rows_grid = int(np.ceil(len(plot_cols) / n_cols_grid))
        fig, axes = plt.subplots(n_rows_grid, n_cols_grid,
                                  figsize=(4.5 * n_cols_grid, 3.5 * n_rows_grid))
        axes = np.atleast_1d(axes).flatten()

        for ax, col in zip(axes, plot_cols):
            vals = numeric_value_matrix[col]
            ax.plot(vals.index, vals.values, marker="o", color="tab:blue")
            cv_val = numeric_trend_stats.loc[col, "cv"]
            cv_label = f"{cv_val:.2f}" if pd.notna(cv_val) else "n/a"
            ax.set_title(f"{col}\n(cv={cv_label})", fontsize=9)
            ax.set_xlabel("Generation")
            ax.set_ylabel("value")

        for ax in axes[len(plot_cols):]:
            ax.axis("off")

        fig.suptitle("Continuity of numeric feature VALUES across generations (winning architecture only)",
                      y=1.02)
        plt.tight_layout()
        if save_plots:
            plt.savefig("top_model_numeric_feature_continuity_lines.png", dpi=150)
        plt.show()

    return {
        "top_per_gen": top_per_gen,
        "binary_cols": binary_cols,
        "numeric_cols": numeric_cols,
        "presence_matrix": presence_matrix,
        "continuity": continuity,
        "streaks": streaks,
        "numeric_value_matrix": numeric_value_matrix,
        "numeric_trend_stats": numeric_trend_stats,
    }


# Example usage:
# results = analyze_top_model_lineage(result_df)
# results["top_per_gen"]           # the actual best arch each generation
# results["continuity"]            # binary features: which ones the winners kept using
# results["streaks"]               # binary features: longest unbroken run of "winning" presence
# results["numeric_trend_stats"]   # numeric features: stability (cv), trend (slope) of value across winners
# results["numeric_value_matrix"]  # numeric features: raw value per generation, for custom plotting