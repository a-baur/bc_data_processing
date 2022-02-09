import pandas as pd
import numpy as np
from datetime import datetime
from random import shuffle
import matplotlib.pyplot as plt

rpss = []


##### PREPPERS
def sync_pair(df1, df2):
    """ Synchronize pair of time-dataframes."""
    n1, n2 = df1.name, df2.name
    df_joined = df1.join(df2, how="outer", lsuffix='_1', rsuffix='_2')
    df_joined = df_joined.dropna()

    _df1 = df_joined.filter(regex='_1')
    _df1.columns = _df1.columns.str.replace(r'_1$', '')
    _df1.name = n1

    _df2 = df_joined.filter(regex='_2')
    _df2.columns = _df2.columns.str.replace(r'_2$', '')
    _df2.name = n2

    return _df1, _df2


def get_overlapping_slice(df_subj1, df_subj2):
    """ Cut timeseries down to overlapping time frame """
    n1, n2 = df_subj1.name, df_subj2.name
    start_time = datetime.fromisoformat(str(max(df_subj2.index.min(), df_subj1.index.min()))).time()
    end_time = datetime.fromisoformat(str(min(df_subj2.index.max(), df_subj1.index.max()))).time()
    df_subj1 = df_subj1.between_time(start_time, end_time)
    df_subj2 = df_subj2.between_time(start_time, end_time)
    df_subj1.name = n1
    df_subj2.name = n2
    return df_subj1, df_subj2


##### STATISTICS
def crosscorr(data_x: np.ndarray, data_y: np.ndarray, lag=0):
    """
    Lag-N cross correlation.Shifted prepped_data filled with NaNs
    :param lag: default 0
    :param data_x, data_y: pandas.Series objects of equal length
    :return: Float indicating cross-correlation
    #"""
    return data_x.corr(data_y.shift(lag))


def _fisher_z(r):
    """
    Compute Fisher's Z transformation for given correlation value.
    :param corr_arr: correlation value
    :return: Z-value
    """
    # Ceiling values to avoid inf-values
    if r >= 0.999329: return np.arctanh(0.999329)
    if r <= -0.999329: return np.arctanh(-0.999329)
    else: return np.arctanh(r)


def _inverse_fisher_z(z):
    """
    Compute inverse Fisher's Z transformation for given z value.
    :param z: Fisher-Z value
    :return: r
    """
    try:
        return np.tanh(z)
    except Exception:
        # Debugging
        print(z)
        print(type(z))
        raise


v_fisher_z = np.vectorize(_fisher_z)


##### SYNCHRONY CALCULATION
def windowed_synchrony(df_subj1: pd.DataFrame, df_subj2: pd.DataFrame,
                       interval_sec=30, test_pseudo=False, verbose=False, create_plots=False):

    df_out = pd.DataFrame(
        data={
            "Lags": np.zeros((len(df_subj1.columns),)),
            "Synchrony": np.zeros((len(df_subj1.columns),)),
            "var_subj1": df_subj1.var(),
            "var_subj2": df_subj2.var()},
        index=df_subj1.columns
    )

    # Split into 30s intervals
    s1_groups = df_subj1.resample(f'{interval_sec}s')
    s2_groups = df_subj2.resample(f'{interval_sec}s')

    # Do not iterate group directly, use list of group names.
    # Makes it possible to shuffle them and calculate pseudo-synchrony
    s1_gnames = list(s1_groups.groups.keys())
    s2_gnames = list(s2_groups.groups.keys())

    if test_pseudo:
        shuffle(s1_gnames)
        shuffle(s2_gnames)

    min_data_len = 10 * interval_sec * 0.9  # Only regard datasets that are least 90% complete (Kruzic et al)

    plts = []
    iter_sync = 0
    for s1_iter, s2_iter in zip(s1_gnames, s2_gnames):  # Iterate over intervals of both subjects at same time
        # Get by group name
        try:
            s1 = s1_groups.get_group(s1_iter)
            s2 = s2_groups.get_group(s2_iter)
        except KeyError:
            # Empty groups can not be retrieved with get_group()
            if verbose:
                print("Skipped empty interval")
            continue

        if len(s1) < min_data_len and len(s2) < min_data_len:
            if verbose:
                print(f"Skipped interval of length ({len(s1)/10}, {len(s2)/10}) seconds for being to short. "
                      f"(Minimum: {min_data_len/10}s)")
            continue

        if len(s1) != len(s2):
            if test_pseudo:  # Only allow unequal dataset length if randomly sorted.
                continue
            raise ValueError("Datasets have unequal lengths for same interval. Can not correlate.")

        if test_pseudo:
            s2.index = s1.index

        s1_start, s2_start = s1.index[0], s2.index[0]
        if s1_start != s2_start and not test_pseudo:
            raise ValueError(f"Datasets do not start the same time (s1 start: {s1_start}, s2 start {s2_start})")

        s1_end, s2_end = s1.index[-1], s2.index[-1]
        if s1_end != s2_end and not test_pseudo:
            raise ValueError(f"Datasets do not start the same time (s1 start: {s1_end}, s2 start {s2_end})")

        # Calculate synchrony for all faceshapes and add to output.
        # df index are faceshapes mapping correlation and lags
        vals, lags = pair_synchrony(s1, s2)
        df_out["Synchrony"] += vals
        df_out["Lags"] += lags
        iter_sync += 1

        if create_plots:
            col_idx = s1.columns.get_loc("Mouth_Smile")
            plts.append((s1["Mouth_Smile"], s2["Mouth_Smile"], vals[col_idx], lags[col_idx]))

    if create_plots:
        fig, axis = plt.subplots(iter_sync, figsize=(15, 15), constrained_layout=True)
        for i, plts in enumerate(plts):
            s1, s2, cc, lag = plts
            cc = _inverse_fisher_z(cc)
            s2 = s2.shift(lag)
            axis[i].plot(s1, color="red")
            axis[i].plot(s2, color="blue")
            axis[i].set_title(f"Synchony = {cc:.4f} at lag {lag}")
            axis[i].set_ylim([0.0, 1.0])
        plt.show()

    # Means for each face shape
    df_out["Lags"] = df_out["Lags"].abs()
    df_out["Synchrony"] /= iter_sync  # mean correlation
    df_out["Lags"] /= iter_sync  # mean abs lag

    # Inverse Fisher-Z after mean for actual correlations
    df_out["Synchrony"] = pd.Series(_inverse_fisher_z(df_out["Synchrony"].astype(np.float64)))

    return df_out


def pair_synchrony(df_subj1, df_subj2):
    if not all([col in df_subj2.columns for col in df_subj1.columns]):
        raise AssertionError("DataFrames must have same columns")
    if not len(df_subj1.index) == len(df_subj2.index):
        raise AssertionError(f"DataFrames must have same length ({len(df_subj1.index)} != {len(df_subj2.index)})")  # Join by timestamp?

    lags = np.arange(-50, 50, 1)
    corr_lags = []
    corr_vals = []

    # Iterate all FaceShapes
    for col in df_subj1.columns:
        rs = [crosscorr(df_subj1[col], df_subj2[col], lag) for lag in lags]  # Compute Cross-Correlation at different time lags
        rs = np.nan_to_num(rs)  # Nan values to zero -> No correlation
        rs = np.abs(rs)  # Absolute, as any synchrony counts positively to global synchrony

        corr_val = np.max(rs)  # max correlation
        corr_val = v_fisher_z(corr_val)  # normalize correlation values using Fisher-Z
        corr_vals.append(corr_val)

        corr_lag = lags[np.argmax(rs)]  # lag of max correlation
        # corr_lag /= 10  # get lag in seconds (Logging rate: 10HZ)
        corr_lags.append(corr_lag)

    # Warn if gaps in data. Should not occur after preprocessing
    nan_cols_1 = df_subj1.isna().any()
    nan_cols_2 = df_subj2.isna().any()
    if nan_cols_1.any():
        gap = df_subj1[df_subj1.isna()].index.values
        print("Gap in subject 1 data of length", (gap[-1] - gap[0]).item()/1000000000)
    if nan_cols_2.any():
        gap = df_subj2[df_subj2.isna()].index.values
        print("Gap in subject 2 data of length", (gap[-1] - gap[0]).item() / 1000000000)

    return pd.Series(corr_vals, index=df_subj1.columns), pd.Series(corr_lags, index=df_subj1.columns)
