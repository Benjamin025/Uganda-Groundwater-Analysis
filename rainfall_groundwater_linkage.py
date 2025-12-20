import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

# ================== PATH CONFIG ==================
BASE_DIR = Path(".")

RAIN_PATH = BASE_DIR / "outputs" / "rainfall_analysis" / "data" / "rain_monthly_clean.csv"
GW_PATH   = BASE_DIR / "outputs" / "groundwater_analysis" / "gw_monthly_clean.csv"

OUT_DIR = BASE_DIR / "outputs" / "rainfall_groundwater_linkage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = OUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PLOTS_DPI = 150

# ================== LOAD DATA ==================
rain = pd.read_csv(RAIN_PATH, parse_dates=['date'], index_col='date')
gw   = pd.read_csv(GW_PATH, parse_dates=['date'], index_col='date')

# Align time periods (common reliable window)
rain = rain.loc["2004-01":"2024-07"]
gw   = gw.loc["2004-01":"2024-07"]

print("Rain shape:", rain.shape)
print("GW shape:", gw.shape)

# ================== SELECT VARIABLES ==================
rain_sel = rain[['rain_mm_total', 'rain_3m', 'rain_6m', 'rain_12m']]
gw_sel   = gw[['gw_mean']]

df = rain_sel.join(gw_sel, how='inner')

print("Merged shape:", df.shape)
print(df.head())

# ================== STANDARDIZATION (FOR VISUALS ONLY) ==================
df_std = df.copy()

df_std['rain_std'] = (
    df['rain_mm_total'] - df['rain_mm_total'].mean()
) / df['rain_mm_total'].std()

df_std['gw_std'] = (
    df['gw_mean'] - df['gw_mean'].mean()
) / df['gw_mean'].std()

# ================== STANDARDIZED TIME SERIES PLOT ==================
fig, ax = plt.subplots(figsize=(12,5))

ax.plot(df_std.index, df_std['rain_std'],
        label='Standardized Rainfall', linewidth=1.5)

ax.plot(df_std.index, df_std['gw_std'],
        label='Standardized Groundwater Depth', linewidth=2)

ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_title("Standardized Rainfall and Groundwater Variability (Monthly)")
ax.set_ylabel("Standardized anomaly (z-score)")
ax.set_xlabel("Time")
ax.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "std_rainfall_groundwater_timeseries.png", dpi=PLOTS_DPI)
plt.close(fig)

# ================== NO-LAG CORRELATION ==================
r0, p0 = pearsonr(df['rain_mm_total'], df['gw_mean'])

no_lag_df = pd.DataFrame({
    'metric': ['pearson_r', 'p_value'],
    'value': [r0, p0]
})

no_lag_df.to_csv(DATA_DIR / "no_lag_correlation.csv", index=False)

print("\nNo-lag correlation:")
print(no_lag_df)

# ================== LAGGED CORRELATION ANALYSIS ==================
lags = range(0, 13)  # 0–12 months
results = []

for lag in lags:
    shifted_rain = df['rain_mm_total'].shift(lag)
    valid = shifted_rain.notna() & df['gw_mean'].notna()

    r, p = pearsonr(shifted_rain[valid], df.loc[valid, 'gw_mean'])
    results.append((lag, r, p))

lag_df = pd.DataFrame(results, columns=['lag_months', 'pearson_r', 'p_value'])
lag_df.to_csv(DATA_DIR / "lagged_correlation_results.csv", index=False)

print("\nLagged correlation results:")
print(lag_df)

# ================== LAGGED CORRELATION PLOT ==================
fig, ax = plt.subplots(figsize=(8,4))

ax.plot(lag_df['lag_months'], lag_df['pearson_r'], marker='o')
ax.axhline(0, color='k', linestyle='--')
ax.set_xlabel("Rainfall lag (months)")
ax.set_ylabel("Correlation coefficient (r)")
ax.set_title("Lagged Rainfall–Groundwater Correlation")

fig.tight_layout()
fig.savefig(OUT_DIR / "lagged_correlation.png", dpi=PLOTS_DPI)
plt.close(fig)

# Identify dominant recharge lag
best = lag_df.loc[lag_df['pearson_r'].idxmin()]
print("\nDominant recharge lag:")
print(best)

# ================== ANNUAL COMPARISON ==================
annual = df.resample('Y').agg({
    'rain_mm_total': 'sum',
    'gw_mean': 'mean'
})

annual.to_csv(DATA_DIR / "annual_rainfall_groundwater.csv")

fig, ax = plt.subplots(figsize=(10,4))

ax.plot(annual.index.year, annual['rain_mm_total'], label='Annual Rainfall')
ax.plot(annual.index.year, annual['gw_mean'], label='Annual Groundwater Depth')
ax.invert_yaxis()
ax.set_xlabel("Year")
ax.set_title("Annual Rainfall vs Groundwater Depth")
ax.legend()

fig.tight_layout()
fig.savefig(OUT_DIR / "annual_rainfall_vs_groundwater.png", dpi=PLOTS_DPI)
plt.close(fig)

print("\nAll plots and data saved to:", OUT_DIR)
print("Linkage analysis complete.")
