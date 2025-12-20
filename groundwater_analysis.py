import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kendalltau

# ================== PATH CONFIG ==================
RAW_PATH = Path("data/raw/Rwonyo_data_editable.csv")

OUT_DIR = Path("outputs/groundwater_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

PLOTS_DPI = 150

# ================== PARAMETERS ==================
INTERP_LIMIT_DAYS = 7
BASELINE_START = "2004-01-01"
BASELINE_END   = "2014-12-31"

# ================== LOAD & PARSE DATA ==================
df = pd.read_csv(
    RAW_PATH,
    sep=None,
    engine='python',
    header=None,
    names=['date', 'gw_m']
)

df['date'] = pd.to_datetime(
    df['date'],
    format='%d-%b-%y',
    errors='coerce'
)

df['gw_m'] = pd.to_numeric(df['gw_m'], errors='coerce')
df = df.dropna(subset=['date'])

df = df.set_index('date').sort_index()
df.index = df.index.normalize()

# Remove duplicate days (average)
df = df.groupby(df.index).mean()

# Create continuous daily index
full_index = pd.date_range(df.index.min(), df.index.max(), freq='D')
df = df.reindex(full_index)
df.index.name = 'date'

# ================== QUALITY CHECKS ==================
print("Date range:", df.index.min(), "to", df.index.max())
print("Total days:", len(df))
print(df['gw_m'].describe())
print("Missing days:", df['gw_m'].isna().sum())

# ================== PLOT 1: DAILY RAW GROUNDWATER ==================
fig, ax = plt.subplots(figsize=(12,4))
ax.plot(df.index, df['gw_m'])
ax.set_title("Daily Groundwater Level (Raw)")
ax.set_ylabel("Water level (m)")
ax.set_xlabel("Date")

fig.tight_layout()
fig.savefig(PLOTS_DIR / "daily_groundwater_raw.png", dpi=PLOTS_DPI)
plt.close(fig)

# ================== CLEAN UNREALISTIC VALUES ==================
df.loc[df['gw_m'] < 0, 'gw_m'] = np.nan
df.loc[df['gw_m'] > 300, 'gw_m'] = np.nan

print("After cleaning, missing:", df['gw_m'].isna().sum())

# ================== INTERPOLATE SHORT GAPS ==================
df['gw_interp'] = df['gw_m'].interpolate(
    method='time',
    limit=INTERP_LIMIT_DAYS
)

imputed = df['gw_m'].isna() & df['gw_interp'].notna()
print("Imputed values:", imputed.sum())

# ================== MONTHLY AGGREGATION ==================
gw_monthly = pd.DataFrame()
gw_monthly['gw_mean'] = df['gw_interp'].resample('M').mean()
gw_monthly['gw_min']  = df['gw_interp'].resample('M').min()
gw_monthly['gw_max']  = df['gw_interp'].resample('M').max()

gw_monthly = gw_monthly.loc["2004-01":"2024-07"]

# ================== PLOT 2: MONTHLY HYDROGRAPH ==================
fig, ax = plt.subplots(figsize=(12,4))

ax.plot(gw_monthly.index, gw_monthly['gw_mean'], label='Monthly mean')
ax.plot(
    gw_monthly.index,
    gw_monthly['gw_mean'].rolling(12).mean(),
    label='12-month rolling mean',
    linewidth=2
)

ax.invert_yaxis()
ax.legend()
ax.set_title("Monthly Groundwater Level (Depth to Water)")
ax.set_ylabel("Depth (m)")
ax.set_xlabel("Date")

fig.tight_layout()
fig.savefig(PLOTS_DIR / "monthly_groundwater_hydrograph.png", dpi=PLOTS_DPI)
plt.close(fig)

# ================== SEASONAL CLIMATOLOGY ==================
baseline = gw_monthly.loc[BASELINE_START:BASELINE_END]
gw_clim = baseline.groupby(baseline.index.month)['gw_mean'].mean()

# ================== PLOT 3: SEASONAL CYCLE ==================
fig, ax = plt.subplots(figsize=(8,4))

ax.plot(range(1,13), gw_clim, marker='o')
ax.invert_yaxis()
ax.set_title("Mean Seasonal Groundwater Cycle")
ax.set_xlabel("Month")
ax.set_ylabel("Depth (m)")

fig.tight_layout()
fig.savefig(PLOTS_DIR / "seasonal_groundwater_cycle.png", dpi=PLOTS_DPI)
plt.close(fig)

# ================== GROUNDWATER ANOMALIES ==================
gw_monthly['month'] = gw_monthly.index.month
gw_monthly['gw_anom'] = gw_monthly.apply(
    lambda r: r['gw_mean'] - gw_clim.loc[r['month']], axis=1
)

# ================== PLOT 4: GROUNDWATER ANOMALIES ==================
fig, ax = plt.subplots(figsize=(12,4))

ax.bar(gw_monthly.index, gw_monthly['gw_anom'])
ax.axhline(0, color='k')
ax.set_title("Groundwater Level Anomaly")
ax.set_ylabel("Anomaly (m)")
ax.set_xlabel("Date")

fig.tight_layout()
fig.savefig(PLOTS_DIR / "groundwater_anomalies.png", dpi=PLOTS_DPI)
plt.close(fig)

# ================== TREND ANALYSIS ==================
s = gw_monthly['gw_mean'].dropna()

tau, p = kendalltau(range(len(s)), s.values)

slopes = [
    (s[j] - s[i]) / (j - i)
    for i in range(len(s)) for j in range(i+1, len(s))
]
sen_slope = np.median(slopes)

print("Kendall tau:", tau)
print("p-value:", p)
print("Sen slope (m/month):", sen_slope)
print("Sen slope (m/year):", sen_slope * 12)

# ================== SAVE OUTPUT DATA ==================
df.to_csv(OUT_DIR / "gw_daily_clean.csv")
gw_monthly.to_csv(OUT_DIR / "gw_monthly_clean.csv")
gw_clim.to_frame("gw_climatology").to_csv(OUT_DIR / "gw_monthly_climatology.csv")

print("\nGroundwater analysis complete.")
print("Plots saved to:", PLOTS_DIR)
print("Cleaned datasets saved to:", OUT_DIR)
