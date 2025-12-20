# rainfall_analysis_notebook.py
# Run this as a Jupyter notebook (split into cells) or as a script.
# Requirements: pandas, numpy, matplotlib, scipy
# If running as a script, run `python rainfall_analysis_notebook.py`.
# Put your raw CSV at: data/raw/rain_raw.csv
# CSV should contain a date column and a rainfall column (daily totals in mm).
# If column names differ, update the `possible_*` detection below.

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from datetime import datetime

# ---------- CONFIG ----------
RAW_PATH = Path("data/raw/RWONYO.csv")    # change if needed
OUT_DIR = Path("outputs/rainfall_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis choices:
INTERP_LIMIT_DAYS = 3       # only interpolate gaps <= this many days
BASELINE_START = "2004-01-01"
BASELINE_END   = "2014-12-31"
PLOTS_DPI = 150
# ----------------------------

def load_and_identify_columns(raw_path):
    """Load CSV and try to identify date and rainfall columns automatically."""
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found at {raw_path}. Place your daily CSV there.")
    sample = pd.read_csv(raw_path, nrows=0)
    cols = list(sample.columns)
    # find date-like and rain-like columns
    date_col = next((c for c in cols if 'date' in c.lower()), None)
    rain_col = next((c for c in cols if any(k in c.lower() for k in ['rain','precip','ppt'])), None)
    if date_col is None or rain_col is None:
        raise ValueError(f"Could not auto-detect columns. Found columns: {cols}. Ensure there's a date column and a rain/precip column.")
    df = pd.read_csv(raw_path, parse_dates=[date_col])
    df = df.rename(columns={date_col: 'date', rain_col: 'rain_mm'})
    return df[['date','rain_mm']]

def make_continuous_daily(df):
    """Reindex to a continuous daily index (detect start/end, reindex)."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_idx)
    df.index.name = 'date'
    return df

def basic_cleaning(df):
    """Basic cleaning: convert to numeric, flag negatives, set unrealistic extremes to NaN (adjust thresholds)."""
    df['rain_mm'] = pd.to_numeric(df['rain_mm'], errors='coerce')
    # flag negative values as errors -> set to NaN (optionally set to 0 if you prefer)
    neg_count = (df['rain_mm'] < 0).sum()
    if neg_count > 0:
        print(f"Found {neg_count} negative rainfall values -> setting to NaN (flag them if you prefer to set to 0).")
        df.loc[df['rain_mm'] < 0, 'rain_mm'] = np.nan
    # Optionally cap insane high values (example threshold 500 mm/day)
    insane_count = (df['rain_mm'] > 1000).sum()
    if insane_count > 0:
        print(f"Found {insane_count} extremely large rainfall values (>1000 mm/day) -> setting to NaN.")
        df.loc[df['rain_mm'] > 1000, 'rain_mm'] = np.nan
    return df

def impute_short_gaps(df, limit_days=INTERP_LIMIT_DAYS):
    """Interpolate only short gaps to avoid inventing rainfall events across long missing periods."""
    df = df.copy()
    df['rain_mm_interp'] = df['rain_mm'].interpolate(method='time', limit=limit_days)
    # count imputed
    imputed = df['rain_mm'].isna() & df['rain_mm_interp'].notna()
    print(f"Imputed {imputed.sum()} values by time interpolation with limit={limit_days} days.")
    # leave remaining NaNs as NaN (user can choose to fill with zeros if station usually reports 0)
    df['rain_mm_fill0'] = df['rain_mm_interp'].fillna(0)  # optional fallback version
    return df

def aggregate_monthly(df_interp):
    """Aggregate to monthly totals and compute rainy-day counts and rolling sums."""
    monthly = df_interp['rain_mm_interp'].resample('M').sum().to_frame('rain_mm_total')
    monthly['rainy_days'] = df_interp['rain_mm_interp'].resample('M').apply(lambda x: (x>0).sum())
    monthly['rain_1m']  = monthly['rain_mm_total']
    monthly['rain_3m']  = monthly['rain_mm_total'].rolling(window=3, min_periods=1).sum()
    monthly['rain_6m']  = monthly['rain_mm_total'].rolling(window=6, min_periods=1).sum()
    monthly['rain_12m'] = monthly['rain_mm_total'].rolling(window=12, min_periods=1).sum()
    return monthly

def compute_climatology_and_anomalies(monthly, baseline_start=BASELINE_START, baseline_end=BASELINE_END):
    """Compute calendar-month climatology over baseline and monthly anomalies."""
    # ensure baseline lies within data
    if (monthly.index.min() <= pd.Timestamp(baseline_start)) and (monthly.index.max() >= pd.Timestamp(baseline_end)):
        baseline = monthly.loc[baseline_start:baseline_end]
    else:
        # fallback: use first 10 years present
        baseline = monthly.loc[monthly.index.min(): monthly.index.min() + pd.DateOffset(years=10) - pd.DateOffset(days=1)]
        print(f"Baseline {baseline_start}-{baseline_end} not fully available. Using fallback baseline {baseline.index.min().date()} to {baseline.index.max().date()}.")
    clim = baseline['rain_mm_total'].groupby(baseline.index.month).mean()
    monthly = monthly.copy()
    monthly['month'] = monthly.index.month
    monthly['rain_clim_monthly'] = monthly['month'].map(clim)
    monthly['rain_anom'] = monthly['rain_mm_total'] - monthly['rain_clim_monthly']
    return monthly, clim

def sen_slope(series):
    """Compute Sen's slope (non-parametric) for a series with no NaNs (or drop NaNs)."""
    s = series.dropna().values
    n = len(s)
    if n < 2:
        return np.nan
    # slopes between all pairs j>i: (xj - xi) / (j-i) where j-i measured in time steps (here months or years depending)
    # We'll use index distance in units of series frequency (assumed equally spaced)
    slopes = []
    for i in range(n-1):
        for j in range(i+1, n):
            slopes.append((s[j] - s[i]) / (j - i))
    return np.median(slopes)

def trend_test_and_slope(series, label="series"):
    """Kendall tau + Sen's slope (series is a pandas Series with datetime index)."""
    s = series.dropna()
    if len(s) < 6:
        print(f"Not enough data for trend test on {label} (n={len(s)})")
        return {}
    # Kendall tau (scipy)
    tau, p = kendalltau(np.arange(len(s)), s.values)
    # Sen's slope (units per time-step; if monthly series -> m/month; convert to m/year if needed)
    slope_per_timestep = sen_slope(s)
    return {"n": len(s), "kendall_tau": float(tau), "p_value": float(p), "sen_slope_per_timestep": float(slope_per_timestep)}

def annualize_monthly_to_annual(monthly):
    """Compute calendar year totals from monthly totals (useful for trend on annual totals)."""
    annual = monthly['rain_mm_total'].resample('A').sum()
    annual.index = annual.index.year
    return annual

def plot_and_save_monthly(monthly, outdir=OUT_DIR):
    """Produce diagnostic plots and save them."""
    # monthly totals
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(monthly.index, monthly['rain_mm_total'], marker='.', linewidth=0.7)
    ax.set_title("Monthly rainfall totals")
    ax.set_ylabel("Rain (mm)")
    fig.tight_layout()
    fig.savefig(outdir / "monthly_rain_totals.png", dpi=PLOTS_DPI)
    plt.close(fig)

    # 12-month cumulative
    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(monthly.index, monthly['rain_12m'], marker='.', linewidth=0.7)
    ax.set_title("12-month cumulative rainfall (rolling sum)")
    ax.set_ylabel("Rain (mm over 12 months)")
    fig.tight_layout()
    fig.savefig(outdir / "rain_12m_cumulative.png", dpi=PLOTS_DPI)
    plt.close(fig)

    # anomalies
    fig, ax = plt.subplots(figsize=(12,3))
    ax.bar(monthly.index, monthly['rain_anom'], width=20)
    ax.set_title("Monthly rainfall anomaly (relative to baseline climatology)")
    ax.set_ylabel("Anomaly (mm)")
    fig.tight_layout()
    fig.savefig(outdir / "monthly_rain_anomaly.png", dpi=PLOTS_DPI)
    plt.close(fig)

    # seasonal boxplot by month (visualizing monthly distribution across years)
    fig, ax = plt.subplots(figsize=(10,4))
    box_data = [monthly[monthly.index.month == m]['rain_mm_total'].dropna() for m in range(1,13)]
    ax.boxplot(box_data, labels=[str(m) for m in range(1,13)])
    ax.set_title("Monthly rainfall distribution (by calendar month)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rain (mm)")
    fig.tight_layout()
    fig.savefig(outdir / "monthly_boxplot.png", dpi=PLOTS_DPI)
    plt.close(fig)

def save_outputs(df_daily, monthly, clim, outdir=OUT_DIR):
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    df_daily.to_csv(outdir / "data" / "rain_daily_clean.csv", index=True)
    monthly.to_csv(outdir / "data" / "rain_monthly_clean.csv", index=True)
    clim.to_frame("rain_clim_mm").to_csv(outdir / "data" / "rain_monthly_climatology.csv")
    print(f"Saved cleaned daily & monthly files to {outdir / 'data'} and plots to {outdir}")

# ----------------- MAIN PIPELINE -----------------
def main():
    print("Loading raw rainfall file...")
    df_raw = load_and_identify_columns(RAW_PATH)

    print("Making a continuous daily index...")
    df_daily = make_continuous_daily(df_raw)

    print("Basic cleaning (negatives -> NaN, unrealistic extremes flagged)...")
    df_daily = basic_cleaning(df_daily)

    print("Imputing short gaps (limit days = {})...".format(INTERP_LIMIT_DAYS))
    df_daily = impute_short_gaps(df_daily, limit_days=INTERP_LIMIT_DAYS)

    print("Aggregating to monthly and computing rolling cumulative sums...")
    monthly = aggregate_monthly(df_daily)

    print("Computing baseline climatology and anomalies (baseline {} to {})...".format(BASELINE_START, BASELINE_END))
    monthly, clim = compute_climatology_and_anomalies(monthly, BASELINE_START, BASELINE_END)

    # Basic diagnostics
    print("\n--- Basic monthly summary ---")
    print(monthly['rain_mm_total'].describe().to_string())

    # Trend tests on monthly totals (Kendall + Sen's slope)
    print("\nRunning trend tests on monthly series (Kendall's tau + Sen's slope)...")
    res_monthly = trend_test_and_slope(monthly['rain_mm_total'], label="monthly_total")
    print("Monthly trend results:", res_monthly)

    # Run trend analysis on annual totals (often clearer)
    annual = annualize_monthly_to_annual(monthly)
    print("\nAnnual totals summary:")
    print(annual.describe().to_string())
    res_annual = trend_test_and_slope(annual, label='annual_total')
    print("Annual trend results:", res_annual)

    # Save outputs & plots
    print("\nGenerating plots and saving outputs...")
    plot_and_save_monthly(monthly, OUT_DIR)
    save_outputs(df_daily, monthly, clim, OUT_DIR)

    # Show last 24 months sample
    print("\nRecent 24 months of monthly results:")
    print(monthly.tail(24).to_string())

    print("\nDone. Next suggestions:")
    print(" - If you want, run the groundwater preparation pipeline next and then we'll merge monthly files to analyze lagged GW response.")
    print(" - If you need additional analyses (drought detection, SPI, wavelet analysis), tell me which.")

if __name__ == "__main__":
    main()
