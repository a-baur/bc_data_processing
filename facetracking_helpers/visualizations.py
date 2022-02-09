from importlib import reload
import pandas as pd
import numpy as np
import facetracking_helpers.calculations as calculations
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
reload(calculations)

LAGS_IN_MS = 75  # Lag in ms


def visualize_synchrony(df_subj1: pd.DataFrame, df_subj2: pd.DataFrame, how="windowed", **kwargs):
    assert how in ["windowed", "rolling"]
    if how == "windowed":
        return visualize_synchrony_windowed(df_subj1, df_subj2, **kwargs)
    elif how == "rolling":
        return visualize_synchrony_rolling(df_subj1, df_subj2, **kwargs)


def visualize_synchrony_windowed(df_subj1: pd.DataFrame, df_subj2: pd.DataFrame, window_size=30, **kwargs):
    data = get_heatmap_data(df_subj1, df_subj2, window_size)
    ax = heatmap(data, **kwargs)
    return ax


def visualize_synchrony_rolling(df_subj1: pd.DataFrame, df_subj2: pd.DataFrame, **kwargs):
    # Rolling window time lagged cross correlation
    lag_seconds = LAGS_IN_MS/10
    fps = 10
    window_size = 300  # samples
    t_start = 0
    t_end = t_start + window_size
    step_size = 1
    rss = []
    print(len(df_subj1.index))
    while t_end < len(df_subj1.index):
        d1 = df_subj1.mean(axis=1).iloc[t_start:t_end]
        d2 = df_subj2.mean(axis=1).iloc[t_start:t_end]
        rs = [calculations.crosscorr(d1, d2, lag) for lag in
                range(-int(lag_seconds * fps), int(lag_seconds * fps + 1))]
        rs = np.nan_to_num(rs)
        rss.append(np.abs(rs))
        t_start = t_start + step_size
        t_end = t_end + step_size

    rss = pd.DataFrame(rss)

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(rss, ax=ax, vmin=0.0, vmax=0.6, **kwargs)
    x = 2 * lag_seconds * fps
    ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation',
           xlim=[0, int(x) + 1], xlabel='Time-lag of cross-correlation', ylabel='Epochs')
    ax.set_xticks([0, x / 4, x / 2, 3 * x / 4, x])
    ax.set_xticklabels([-x / 20, -x / 40, 0, x / 40, x / 20])
    return plt


def heatmap(heatmap_data, title_suffix='', **kwargs):
    f, ax = plt.subplots(figsize=(10, 5))
    heatmap_data = np.array(heatmap_data)
    print(heatmap_data.max(axis=1).mean())
    print(np.abs(heatmap_data.argmax(axis=1) - LAGS_IN_MS).mean())
    sns.heatmap(heatmap_data, vmin=0.0, vmax=0.6, ax=ax, **kwargs)
    ax.set(title=f'Windowed Time Lagged Cross Correlation'+title_suffix, xlim=[0, LAGS_IN_MS],
           xlabel='Time-lag of cross-correlation', ylabel='Time intervals')
    x = 2 * (LAGS_IN_MS/10) / 0.1
    ax.set_xticks([0, x / 4, x / 2, 3 * x / 4, x])
    ax.set_xticklabels([-x / 20, -x / 40, 0, x / 40, x / 20])
    ax.axvline(x=0, color="black")
    return ax


def get_heatmap_data(df_subj1: pd.DataFrame, df_subj2: pd.DataFrame, window_size=30):
    s1_groups = df_subj1.resample(f'{window_size}s')
    s2_groups = df_subj2.resample(f'{window_size}s')
    corr_vals = list()
    for (time_idx1, s1), (time_idx2, s2) in zip(s1_groups, s2_groups):
        pair_syncs = pair_synchrony(s1, s2)
        corr_vals.append(pair_syncs)
    heatmap_data = np.apply_along_axis(
        lambda x: calculations._inverse_fisher_z(x), 0, corr_vals
    )
    return heatmap_data


def pair_synchrony(df_subj1, df_subj2):
    if not all([col in df_subj2.columns for col in df_subj1.columns]):
        raise AssertionError("DataFrames must have same columns")
    if not len(df_subj1.index) == len(df_subj2.index):
        raise AssertionError(f"DataFrames must have same length ({len(df_subj1.index)} != {len(df_subj2.index)})")  # Join by timestamp?

    lags = np.arange(-LAGS_IN_MS, LAGS_IN_MS, 1)

    df_fs_corr = pd.DataFrame()
    for col in df_subj1.columns:
        # Compute Cross-Correlation at different time lags
        rs = np.nan_to_num(
            [calculations.crosscorr(df_subj1[col], df_subj2[col], lag) for lag in lags])
        norm_vals = np.abs(calculations.v_fisher_z(rs))
        df_fs_corr = pd.concat([df_fs_corr, pd.Series(norm_vals)], axis=1)

    return df_fs_corr.mean(axis=1)
