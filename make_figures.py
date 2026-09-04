"""Регенерация графиков для reports/figures из данных и артефактов пайплайна.

Читает data/train.parquet и out/feature_importance.csv, рисует пять графиков:
timeline (DAU + дневной GMV), распределение таргета на валидационном фолде,
retention по recency-бакетам, топ-30 признаков по gain, кривая калибровки
по точкам лидерборда.

Запуск: python make_figures.py
Выход: reports/figures/{timeline,target_distribution,retention,
       feature_importance,calibration_curve}.png
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("reports/figures")
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3})

# 1) Platform timeline (daily GMV + DAU)
d = pl.read_parquet("data/train.parquet")
agg = (d.group_by("event_date").agg([pl.len().alias("dau"), pl.col("gmv").sum()])
       .sort("event_date")).to_pandas()
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax[0].plot(agg["event_date"], agg["gmv"], lw=.8); ax[0].set_title("Daily total GMV")
ax[1].plot(agg["event_date"], agg["dau"], lw=.8, color="tab:red"); ax[1].set_title("DAU")
fig.tight_layout(); fig.savefig(FIG / "timeline.png"); plt.close(fig)

# 2) 30-day target distribution on a validation fold (log scale)
t = (d.filter((pl.col("event_date") >= pl.date(2025, 12, 16)) &
              (pl.col("event_date") <= pl.date(2026, 1, 14)))
      .group_by("user_id").agg(pl.col("gmv").sum().alias("y"))).to_pandas()
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log1p(t["y"]), bins=80)
ax.set_title("30-day GMV distribution, log1p scale (46% exact zeros)")
ax.set_xlabel("log1p(GMV)"); fig.tight_layout()
fig.savefig(FIG / "target_distribution.png"); plt.close(fig)

# 3) Retention: P(target>0) and E[target] vs days since last activity
u = (d.filter(pl.col("event_date") <= pl.date(2025, 12, 15))
      .group_by("user_id").agg([pl.col("event_date").max().alias("last"),
                                pl.col("gmv").sum().alias("tot")])).to_pandas()
u = u.merge(t, on="user_id")
u["rec"] = (pd.Timestamp("2025-12-15") - pd.to_datetime(u["last"])).dt.days
u = u[u.rec <= 90]
g = u.groupby(pd.cut(u.rec, [-1, 3, 7, 14, 30, 60, 90])).agg(p=("y", lambda s: (s > 0).mean()),
                                                             m=("y", "mean"))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
g["p"].plot.bar(ax=ax[0], title="P(GMV>0) by recency bucket")
g["m"].plot.bar(ax=ax[1], title="E[GMV] by recency bucket")
fig.tight_layout(); fig.savefig(FIG / "retention.png"); plt.close(fig)

# 4) Feature importance (top-30)
fi = pd.read_csv("out/feature_importance.csv").head(30)
fig, ax = plt.subplots(figsize=(8, 7))
ax.barh(fi["feature"][::-1], fi["gain"][::-1]); ax.set_title("LightGBM feature importance (gain)")
fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.close(fig)

# 5) Calibration curve: measured LB probes + quadratic fit
pts = [(0.0, 1.6864), (np.log(1.1628), 1.7052), (np.log(0.8), 1.6766)]
xs = np.array([p[0] for p in pts]); ys = np.array([p[1] ** 2 for p in pts])
a, b, c = np.polyfit(xs, ys, 2)
k = -b / (2 * a)
grid = np.linspace(-.45, .3, 200)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.exp(grid), np.sqrt(np.polyval([a, b, c], grid)), label="quadratic fit")
ax.scatter([np.exp(x) for x, _ in pts], [y for _, y in pts], c="tab:red", zorder=3, label="LB probes")
ax.axvline(np.exp(k), ls="--", c="grey"); ax.axvline(1.0, ls=":", c="grey")
ax.set_xlabel("global multiplier k"); ax.set_ylabel("RMSLE")
ax.set_title(f"Multiplicative calibration: k* = {np.exp(k):.3f}")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "calibration_curve.png"); plt.close(fig)
print("figures ->", FIG)
