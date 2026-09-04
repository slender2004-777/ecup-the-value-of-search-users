"""Плотная недельная панель и финальный ансамбль.

Модуль строит плотную панель из ~49 недельных срезов (~11 млн строк),
обучает XGB-heavy ансамбль с весами, подобранными на честной валидации
фолда 2026-01-14 (с гэпом 31 день между train и valid), и сохраняет
набор сабмитов на сетке beta для последующей калибровки на лидерборде.

Запуск: python src/03_dense_panel.py
Выход: out/report.txt, out/formula_tuning.csv, out/sub3_*.csv,
       out/feature_importance.csv
"""
import gc
import math
import time
import traceback
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- конфиг -----------------------------------------------------------------
DATA_CANDIDATES = ["data/train.parquet", "train.parquet", "./data/train.parquet"]
SUB_CANDIDATES = ["data/sample_submit.csv", "sample_submit.csv", "./sample_submit.csv"]
S2_SUB = Path("out/sub_v2_ens_yoy_b000.csv")            # для файла-микса с пайплайном
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

CUT_STEP  = 7                      # 7 = ~49 срезов (~11M строк); 14 = быстрее
CUT_START = date(2025, 2, 13)
CUT_MAX   = date(2026, 1, 14)      # последний срез с ПОЛНЫМ 30-дн. таргетом
FINAL_CUT = date(2026, 2, 13)      # тестовый срез
VAL_CUT   = date(2026, 1, 14)      # валидационный фолд
GAP_DAYS  = 31                     # train: cutoff <= VAL_CUT - 31 день

SEASON_F_OVERRIDE = None           # None -> сезонный аналог 2025 (x1.1628)
Z_GRID    = [0.80, 0.85, 0.90, 0.95, 1.00]
BETA_GRID = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25]
EMIT_BETAS = [0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25]
MIX_S2_WEIGHT = 0.30

XGB_SEEDS = [42, 777]
LGB_SEEDS = [42, 202]

WINDOWS = [3, 7, 14, 30, 60, 90, 180, 365]
METRICS = ["gmv", "gmv_search", "gmv_cat", "to_cart", "to_ord", "searches"]
HAS_COLS = ["has_search_to_cart", "has_search_to_ord", "has_cat_to_cart", "has_cat_to_ord"]
EMA_DEFS = [("gmv", 7), ("gmv", 14), ("gmv", 30), ("gmv", 90),
            ("to_ord", 30), ("to_cart", 30), ("searches", 14), ("active", 14)]
LN2 = math.log(2.0)
EPOCH = date(1970, 1, 1)

LGB_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=96, min_child_samples=300, feature_fraction=0.65,
                  bagging_fraction=0.85, bagging_freq=1, lambda_l2=3.0,
                  max_bin=255, verbosity=-1, n_jobs=-1, seed=42)
XGB_PARAMS = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.05,
                  max_depth=8, min_child_weight=100, subsample=0.85,
                  colsample_bytree=0.65, reg_lambda=3.0, tree_method="hist",
                  nthread=-1, seed=42)

# --- утилиты ----------------------------------------------------------------
REPORT = []


def log(msg=""):
    print(msg, flush=True); REPORT.append(str(msg))


def section(t):
    log("\n" + "#" * 94); log("## " + t); log("#" * 94)


def save_report():
    (OUT_DIR / "report.txt").write_text("\n".join(REPORT), encoding="utf-8")


def find_file(cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return None


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def rmse_log(yl, pl):
    return float(np.sqrt(np.mean((yl - pl) ** 2)))


PLEN = pl.len if hasattr(pl, "len") else pl.count


def grp(df, *a, **kw):
    return df.group_by(*a, **kw) if hasattr(df, "group_by") else df.groupby(*a, **kw)


# --- данные -----------------------------------------------------------------
def load_daily():
    f = find_file(DATA_CANDIDATES)
    if f is None:
        raise FileNotFoundError("train.parquet не найден")
    data = pl.read_parquet(f)
    num_cols = [c for c in data.columns if c not in ("user_id", "event_date")]
    daily = (data.unique().group_by(["user_id", "event_date"])
             .agg([pl.col(c).sum() for c in num_cols])
             .sort(["user_id", "event_date"]))
    try:
        daily = daily.with_columns(pl.col("event_date").dt.epoch(time_unit="d").cast(pl.Int64).alias("day_idx"))
    except Exception:
        daily = daily.with_columns(pl.col("event_date").cast(pl.Int64).alias("day_idx"))
    del data; gc.collect()
    return daily


def build_platform(daily):
    p = (daily.group_by("event_date").agg(pl.col("gmv").sum().alias("gmv"))
         .sort("event_date")).to_pandas()
    p["event_date"] = pd.to_datetime(p["event_date"])
    return p.set_index("event_date")["gmv"]


def plat_mean(plat, d0, d1):
    s = plat.loc[pd.Timestamp(d0):pd.Timestamp(d1)]
    return float(s.mean()) if len(s) else float("nan")


def factor_of(plat, cut):
    return (plat_mean(plat, cut + timedelta(days=1), cut + timedelta(days=30))
            / plat_mean(plat, cut - timedelta(days=29), cut))


def seasonal_analysis(plat):
    section("1. СЕЗОННЫЙ ФАКТОР ТЕСТА")
    last30 = plat_mean(plat, FINAL_CUT - timedelta(days=29), FINAL_CUT)
    ana = plat_mean(plat, date(2025, 2, 14), date(2025, 3, 15))
    ana_prev = plat_mean(plat, date(2025, 1, 15), date(2025, 2, 13))
    F1 = ana / ana_prev
    g = (plat_mean(plat, date(2026, 1, 1), date(2026, 2, 13))
         / plat_mean(plat, date(2025, 1, 1), date(2025, 2, 13)))
    F2 = ana * g / last30
    F = float(SEASON_F_OVERRIDE) if SEASON_F_OVERRIDE else float(F1)
    log(f"Последние 30д: {last30:,.0f} GMV/день; аналог окна 2025: {ana:,.0f} (пред. 30д: {ana_prev:,.0f})")
    log(f"Сезонный фактор: метод-аналог x{F1:.4f}, YoY-метод x{F2:.4f}  ==>  SEASON_F = {F:.4f}")
    return F


def build_cuts():
    cuts, c = [], CUT_START
    while c <= CUT_MAX:
        cuts.append(c); c += timedelta(days=CUT_STEP)
    if cuts[-1] != CUT_MAX:
        cuts.append(CUT_MAX)
    return cuts


def load_libs():
    libs = {}
    for name, mod in [("lgbm", "lightgbm"), ("xgb", "xgboost")]:
        try:
            libs[name] = __import__(mod)
            log(f"  {name}: {mod} {getattr(libs[name], '__version__', '?')}")
        except Exception as e:
            log(f"  {name}: НЕДОСТУПЕН ({e})")
    return libs


# --- фичи (как в пайплайне) -------------------------------------------------
def yoy_window(cut):
    t0 = cut + timedelta(days=1)
    return t0 - timedelta(days=365), t0 + timedelta(days=30) - timedelta(days=365)


def build_fold(daily, cut, dmin, dmax):
    t0d, t1d = cut + timedelta(days=1), cut + timedelta(days=30)
    yoy0, yoy1 = yoy_window(cut)
    c_idx = (cut - EPOCH).days
    yoy_avail = yoy0 >= dmin
    hist = daily.filter(pl.col("event_date") <= cut)
    aggs = []
    for W in WINDOWS:
        cw = pl.col("day_idx") > (c_idx - W)
        for m in METRICS:
            aggs.append(pl.col(m).filter(cw).sum().alias(f"{m}_{W}"))
        aggs += [pl.col("day_idx").filter(cw).count().alias(f"active_days_{W}"),
                 pl.col("day_idx").filter(cw & (pl.col("gmv") > 0)).count().alias(f"days_gmv_{W}"),
                 pl.col("day_idx").filter(cw & (pl.col("to_ord") > 0)).count().alias(f"days_ord_{W}"),
                 pl.col("day_idx").filter(cw & (pl.col("to_cart") > 0)).count().alias(f"days_cart_{W}"),
                 pl.col("day_idx").filter(cw & (pl.col("searches") > 0)).count().alias(f"days_search_{W}"),
                 pl.col("gmv").filter(cw).max().alias(f"max_gmv_{W}")]
        if W in (30, 90):
            for h in HAS_COLS:
                aggs.append(pl.col(h).filter(cw).sum().alias(f"{h}_w{W}"))
    for m in METRICS:
        aggs.append(pl.col(m).sum().alias(f"{m}_all"))
    aggs += [pl.col("day_idx").count().alias("active_days_all"),
             pl.col("day_idx").filter(pl.col("gmv") > 0).count().alias("days_gmv_all"),
             pl.col("day_idx").filter(pl.col("to_ord") > 0).count().alias("days_ord_all"),
             pl.col("day_idx").filter(pl.col("to_cart") > 0).count().alias("days_cart_all"),
             pl.col("day_idx").filter(pl.col("searches") > 0).count().alias("days_search_all"),
             pl.col("gmv").max().alias("max_gmv_all")]
    for h in HAS_COLS:
        aggs.append(pl.col(h).sum().alias(f"{h}_all"))
    aggs += [pl.col("day_idx").min().alias("first_idx"), pl.col("day_idx").max().alias("last_idx"),
             pl.col("day_idx").filter(pl.col("gmv") > 0).max().alias("last_gmv_idx"),
             pl.col("day_idx").filter(pl.col("to_ord") > 0).max().alias("last_ord_idx"),
             pl.col("day_idx").filter(pl.col("to_cart") > 0).max().alias("last_cart_idx"),
             pl.col("day_idx").filter(pl.col("searches") > 0).max().alias("last_search_idx")]
    for m, hl in EMA_DEFS:
        w = (-(LN2 / hl) * (pl.lit(c_idx) - pl.col("day_idx"))).exp()
        aggs.append(w.sum().alias(f"ema_active_hl{hl}") if m == "active"
                    else (pl.col(m) * w).sum().alias(f"ema_{m}_hl{hl}"))
    if yoy_avail:
        cyw = ((pl.col("day_idx") >= (yoy0 - EPOCH).days)
               & (pl.col("day_idx") <= (yoy1 - EPOCH).days))
        aggs += [pl.col("gmv").filter(cyw).sum().alias("gmv_yoy"),
                 pl.col("day_idx").filter(cyw).count().alias("active_days_yoy"),
                 pl.col("day_idx").filter(cyw & (pl.col("gmv") > 0)).count().alias("days_gmv_yoy"),
                 pl.col("to_ord").filter(cyw).sum().alias("to_ord_yoy"),
                 pl.col("searches").filter(cyw).sum().alias("searches_yoy")]
    feats = grp(hist, "user_id").agg(aggs)
    has_target = t0d <= dmax
    if has_target:
        tgt = (grp(daily.filter((pl.col("event_date") >= t0d) & (pl.col("event_date") <= t1d)), "user_id")
               .agg(pl.col("gmv").sum().alias("target")))
        f = feats.join(tgt, on="user_id", how="left").with_columns(pl.col("target").fill_null(0.0))
    else:
        f = feats.with_columns(pl.lit(None, dtype=pl.Float64).alias("target"))
    return f.to_pandas(), yoy_avail, has_target


def derive(f, cut, yoy_avail):
    c_idx = (cut - EPOCH).days
    numcols = [c for c in f.columns if c not in ("user_id", "target")]
    f[numcols] = f[numcols].fillna(0.0)
    f["recency"] = c_idx - f["last_idx"]
    for col, nm in [("last_gmv_idx", "recency_gmv"), ("last_ord_idx", "recency_ord"),
                    ("last_cart_idx", "recency_cart"), ("last_search_idx", "recency_search")]:
        f[nm] = np.where(f[col] > 0, np.clip(c_idx - f[col], 0, None), 9999)
    f["tenure"] = c_idx - f["first_idx"]
    f["is_test_like"] = (f["recency"] <= 30).astype(np.int8)
    log_cols = ([f"{m}_{W}" for W in WINDOWS for m in METRICS] + [f"{m}_all" for m in METRICS]
                + [f"max_gmv_{W}" for W in WINDOWS] + ["max_gmv_all"]
                + [f"ema_{m}_hl{hl}" for m, hl in EMA_DEFS])
    for c in log_cols:
        f[c] = f[c].astype(np.float64); f[c + "_log"] = np.log1p(f[c])
    eps = 1.0
    f["gmv_ratio_7_30"] = f["gmv_7"] / (f["gmv_30"] + eps)
    f["gmv_ratio_14_30"] = f["gmv_14"] / (f["gmv_30"] + eps)
    f["gmv_ratio_30_90"] = f["gmv_30"] / (f["gmv_90"] + eps)
    f["gmv_ratio_30_180"] = f["gmv_30"] / (f["gmv_180"] + eps)
    f["gmv_ratio_30_365"] = f["gmv_30"] / (f["gmv_365"] + eps)
    f["gmv_ratio_30_all"] = f["gmv_30"] / (f["gmv_all"] + eps)
    f["gmv_ratio_90_365"] = f["gmv_90"] / (f["gmv_365"] + eps)
    f["active_ratio_7_30"] = f["active_days_7"] / (f["active_days_30"] + 1)
    f["active_ratio_30_90"] = f["active_days_30"] / (f["active_days_90"] + 1)
    f["ema_ratio_7_30"] = f["ema_gmv_hl7"] / (f["ema_gmv_hl30"] + eps)
    f["ema_ratio_14_90"] = f["ema_gmv_hl14"] / (f["ema_gmv_hl90"] + eps)
    for W in (30, 90):
        f[f"gmv_per_ad_{W}"] = f[f"gmv_{W}"] / (f[f"active_days_{W}"] + 1)
        f[f"gmv_per_dg_{W}"] = f[f"gmv_{W}"] / (f[f"days_gmv_{W}"] + 1)
        f[f"conv_ord_cart_{W}"] = f[f"to_ord_{W}"] / (f[f"to_cart_{W}"] + 1)
    f["gmv_per_ad_all"] = f["gmv_all"] / (f["active_days_all"] + 1)
    f["gmv_per_dg_all"] = f["gmv_all"] / (f["days_gmv_all"] + 1)
    f["aov_90"] = f["gmv_90"] / (f["to_ord_90"] + 1)
    f["aov_all"] = f["gmv_all"] / (f["to_ord_all"] + 1)
    f["searches_per_ds_30"] = f["searches_30"] / (f["days_search_30"] + 1)
    f["gmvS_share_30"] = f["gmv_search_30"] / (f["gmv_30"] + eps)
    f["gmvS_share_all"] = f["gmv_search_all"] / (f["gmv_all"] + eps)
    f["gmvC_share_all"] = f["gmv_cat_all"] / (f["gmv_all"] + eps)
    f["active_rate_30"] = f["active_days_30"] / 30.0
    f["active_rate_90"] = f["active_days_90"] / 90.0
    f["active_rate_all"] = f["active_days_all"] / (f["tenure"] + 1.0)
    f["month"] = cut.month
    f["dow"] = cut.weekday()
    doy = cut.timetuple().tm_yday
    f["doy_sin"] = math.sin(2 * math.pi * doy / 365.25)
    f["doy_cos"] = math.cos(2 * math.pi * doy / 365.25)
    if yoy_avail:
        f["yoy_avail"] = 1.0
        for c in ["gmv_yoy", "active_days_yoy", "days_gmv_yoy", "to_ord_yoy", "searches_yoy"]:
            f[c] = f[c].astype(np.float64)
        f["gmv_yoy_log"] = np.log1p(f["gmv_yoy"])
        f["yoy_delta_30"] = f["gmv_yoy_log"] - f["gmv_30_log"]
    else:
        f["yoy_avail"] = 0.0
        for c in ["gmv_yoy", "active_days_yoy", "days_gmv_yoy", "to_ord_yoy", "searches_yoy"]:
            f[c] = 0.0
        f["gmv_yoy_log"] = 0.0
        f["yoy_delta_30"] = 0.0
    return f


def feature_columns():
    cols = []
    for W in WINDOWS:
        cols += [f"{m}_{W}" for m in METRICS] + [f"{m}_{W}_log" for m in METRICS]
        cols += [f"active_days_{W}", f"days_gmv_{W}", f"days_ord_{W}", f"days_cart_{W}", f"days_search_{W}"]
        cols += [f"max_gmv_{W}", f"max_gmv_{W}_log"]
        if W in (30, 90):
            cols += [f"{h}_w{W}" for h in HAS_COLS]
    cols += [f"{m}_all" for m in METRICS] + [f"{m}_all_log" for m in METRICS]
    cols += ["active_days_all", "days_gmv_all", "days_ord_all", "days_cart_all",
             "days_search_all", "max_gmv_all", "max_gmv_all_log"]
    cols += [f"{h}_all" for h in HAS_COLS]
    cols += [f"ema_{m}_hl{hl}" for m, hl in EMA_DEFS] + [f"ema_{m}_hl{hl}_log" for m, hl in EMA_DEFS]
    cols += ["recency", "recency_gmv", "recency_ord", "recency_cart", "recency_search",
             "tenure", "is_test_like"]
    cols += ["gmv_ratio_7_30", "gmv_ratio_14_30", "gmv_ratio_30_90", "gmv_ratio_30_180",
             "gmv_ratio_30_365", "gmv_ratio_30_all", "gmv_ratio_90_365",
             "active_ratio_7_30", "active_ratio_30_90", "ema_ratio_7_30", "ema_ratio_14_90"]
    for W in (30, 90):
        cols += [f"gmv_per_ad_{W}", f"gmv_per_dg_{W}", f"conv_ord_cart_{W}"]
    cols += ["gmv_per_ad_all", "gmv_per_dg_all", "aov_90", "aov_all", "searches_per_ds_30",
             "gmvS_share_30", "gmvS_share_all", "gmvC_share_all",
             "active_rate_30", "active_rate_90", "active_rate_all",
             "month", "dow", "doy_sin", "doy_cos",
             "yoy_avail", "gmv_yoy", "gmv_yoy_log", "active_days_yoy", "days_gmv_yoy",
             "to_ord_yoy", "searches_yoy", "yoy_delta_30"]
    return cols


FCOLS = feature_columns()


# --- модели -----------------------------------------------------------------
def make_model(name, libs, Xtr, ytr, Xin=None, yin=None, seed=42, rounds=None):
    if name == "lgbm":
        lgb = libs["lgbm"]; params = dict(LGB_PARAMS)
        params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=FCOLS)
        if Xin is not None:
            din = lgb.Dataset(Xin, yin, reference=dtr, feature_name=FCOLS)
            m = lgb.train(params, dtr, num_boost_round=4000, valid_sets=[din],
                          callbacks=[lgb.early_stopping(120, verbose=False)])
            bi = m.best_iteration or 4000
        else:
            m = lgb.train(params, dtr, num_boost_round=int(rounds)); bi = int(rounds)
        return (lambda X: m.predict(X, num_iteration=bi)), bi, m
    if name == "xgb":
        xgb = libs["xgb"]; params = dict(XGB_PARAMS); params["seed"] = seed
        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FCOLS)
        if Xin is not None:
            din = xgb.DMatrix(Xin, label=yin, feature_names=FCOLS)
            m = xgb.train(params, dtr, num_boost_round=4000, evals=[(din, "v")],
                          early_stopping_rounds=120, verbose_eval=False)
            bi = (int(m.best_iteration) + 1) if m.best_iteration is not None else 4000
        else:
            m = xgb.train(params, dtr, num_boost_round=int(rounds)); bi = int(rounds)
        return (lambda X: m.predict(xgb.DMatrix(X, feature_names=FCOLS),
                                    iteration_range=(0, bi))), bi, m
    raise ValueError(name)


# --- валидация --------------------------------------------------------------
def validation(libs, panel, cuts, plat):
    section("3. ВАЛИДАЦИЯ НА ФОЛДЕ 2026-01-14 (честная, таргеты не пересекаются)")
    tr = [c for c in cuts if c <= VAL_CUT - timedelta(days=GAP_DAYS)]
    inner = max(tr)
    fit = [c for c in tr if c != inner]
    log(f"train-срезов: {len(fit)} ({fit[0]} .. {fit[-1]}), inner(ES)={inner}")
    Xtr = np.vstack([panel[c]["X"] for c in fit])
    ytr = np.concatenate([panel[c]["ylog"] for c in fit])
    Xin, yin = panel[inner]["X"], panel[inner]["ylog"]
    A = panel[VAL_CUT]; tl, ylog, yraw, yoy = A["tl"], A["ylog"], A["yraw"], A["yoy"]
    preds, iters = {}, {}
    model_names = [n for n in ["lgbm", "xgb"] if n in libs]
    for name in tqdm(model_names, desc="Обучение моделей"):
        t0 = time.time()
        pfn, bi, _ = make_model(name, libs, Xtr, ytr, Xin, yin)
        p = pfn(A["X"]); preds[name] = p; iters[name] = bi
        s = rmse_log(ylog[tl], p[tl])
        log(f"  {name}: RMSLE(TL)={s:.4f}  iter={bi}  "
            f"mean(pred)/mean(y)={np.expm1(p[tl]).mean()/yraw[tl].mean():.3f}  [{time.time()-t0:.0f}с]")
    del Xtr, ytr; gc.collect()
    if len(preds) < 2:
        log("  доступна только одна библиотека — вес берётся 0/1")
        wstar = 1.0 if "xgb" in preds else 0.0
    else:
        rows = [(round(float(w), 2), rmse_log(ylog[tl], (w * preds["xgb"] + (1 - w) * preds["lgbm"])[tl]))
                for w in np.arange(0.0, 1.0001, 0.05)]
        wstar = min(rows, key=lambda r: r[1])[0]
        log("  веса (w_xgb): " + ", ".join(f"{w}:{s:.4f}" for w, s in rows))
        log(f"  ==> w_xgb* = {wstar}")
    blend = (wstar * preds["xgb"] + (1 - wstar) * preds["lgbm"]) if len(preds) == 2 else list(preds.values())[0]
    Fv = factor_of(plat, VAL_CUT)
    frows = []
    combinations = []
    for mode in ["M", "C"]:
        for z in ([1.0] if mode == "M" else Z_GRID):
            for beta in BETA_GRID:
                combinations.append((mode, z, beta))
    for mode, z, beta in tqdm(combinations, desc="Перебор формул"):
        base = blend if mode == "M" else z * blend + (1 - z) * np.log1p(yoy)
        frows.append({"mode": mode, "z": z, "beta": beta,
                      "rmsle": rmsle(yraw[tl], np.expm1(base[tl]) * Fv ** beta)})
    fdf = pd.DataFrame(frows).sort_values("rmsle").reset_index(drop=True)
    fdf.to_csv(OUT_DIR / "formula_tuning.csv", index=False)
    log(f"  F(VAL)={Fv:.4f}; топ-10 формулы (mode,z,beta):")
    log(fdf.head(10).to_string(index=False))
    log("  чистая модель по beta: " + ", ".join(
        f"{b}:{rmsle(yraw[tl], np.expm1(blend[tl]) * Fv ** b):.4f}" for b in BETA_GRID))
    best = fdf.iloc[0]
    return dict(w=wstar, mode=best["mode"], z=float(best["z"]), iters=iters)


# --- сабмиты ----------------------------------------------------------------
def write_sub(base_sub, uid, pred, fname):
    out = base_sub.merge(pd.DataFrame({"user_id": uid, "predict": np.clip(pred, 0, None)}),
                         on="user_id", how="left")
    out["predict"] = out["predict"].fillna(0.0)
    assert len(out) == len(base_sub), f"рядов {len(out)} != {len(base_sub)}"
    assert out["predict"].notna().all() and (out["predict"] >= 0).all()
    out.to_csv(OUT_DIR / fname, index=False)
    p = out["predict"].values
    log(f"  {fname}: нулей(<1e-6)={(p < 1e-6).mean():.3f}; медиана={np.median(p):.1f}; "
        f"p95={np.quantile(p, .95):.0f}; max={p.max():,.0f}; сумма={p.sum():,.0f}")


# --- финал ------------------------------------------------------------------
def final_train(libs, panel, cuts, cfg, SEASON_F):
    section("4. ФИНАЛЬНОЕ ОБУЧЕНИЕ НА ВСЕХ СРЕЗАХ ПАНЕЛИ + САБМИТЫ")
    sub_file = find_file(SUB_CANDIDATES)
    if sub_file is None:
        log("sample_submit.csv не найден — сабмиты не созданы"); return
    base_sub = pd.read_csv(sub_file)[["user_id"]].astype({"user_id": "int64"})

    Xall = np.vstack([panel[c]["X"] for c in cuts])
    yall = np.concatenate([panel[c]["ylog"] for c in cuts])
    log(f"обучающих строк: {Xall.shape[0]:,}; фичей: {Xall.shape[1]}")
    for c in cuts:                       # освобождаем память (тест-срез не трогаем)
        panel[c].pop("X", None); panel[c].pop("ylog", None)
    gc.collect()
    A = panel[FINAL_CUT]
    Xf, yoy_f, uid = A["X"], A["yoy"], A["uid"]

    logs_, fi_model = {}, None
    it = cfg.get("iters", {})
    if "lgbm" in libs and "lgbm" in it:
        r = max(200, int(np.ceil(1.1 * it["lgbm"])))
        log(f"LGBM: {r} итераций × {len(LGB_SEEDS)} seed...")
        for s in tqdm(LGB_SEEDS, desc="LGBM seeds"):
            t0 = time.time()
            pfn, _, m = make_model("lgbm", libs, Xall, yall, seed=s, rounds=r)
            logs_[f"lgbm_s{s}"] = pfn(Xf); fi_model = m
            log(f"  lgbm seed {s} готов [{time.time()-t0:.0f}с]"); del pfn; gc.collect()
    if "xgb" in libs and "xgb" in it:
        r = max(200, int(np.ceil(1.1 * it["xgb"])))
        log(f"XGB: {r} итераций × {len(XGB_SEEDS)} seed...")
        for s in tqdm(XGB_SEEDS, desc="XGB seeds"):
            t0 = time.time()
            pfn, _, _ = make_model("xgb", libs, Xall, yall, seed=s, rounds=r)
            logs_[f"xgb_s{s}"] = pfn(Xf)
            log(f"  xgb seed {s} готов [{time.time()-t0:.0f}с]"); del pfn; gc.collect()
    del Xall, yall; gc.collect()
    if not logs_:
        raise RuntimeError("ни одна модель не обучена — невозможно сформировать сабмиты")

    lgb_mean = np.mean([v for k, v in logs_.items() if k.startswith("lgbm")], axis=0) \
        if any(k.startswith("lgbm") for k in logs_) else None
    xgb_mean = np.mean([v for k, v in logs_.items() if k.startswith("xgb")], axis=0) \
        if any(k.startswith("xgb") for k in logs_) else None
    w = float(cfg["w"])
    ens_log = (w * xgb_mean + (1 - w) * lgb_mean) if (lgb_mean is not None and xgb_mean is not None) \
        else (xgb_mean if xgb_mean is not None else lgb_mean)
    log(f"Ансамбль: {len(logs_)} моделей, w_xgb={w}; log-прогноз: "
        f"медиана={np.median(ens_log):.2f}, p95={np.quantile(ens_log, .95):.2f}")

    # Формула: base = z*ens + (1-z)*log1p(yoy), pred = expm1(base) × SEASON_F^beta
    mode, z = cfg["mode"], float(cfg["z"])
    base_log = (z * ens_log + (1 - z) * np.log1p(yoy_f)) if mode == "C" else ens_log.copy()
    cfg_lines = [f"SEASON_F={SEASON_F:.4f}", f"w_xgb={w}", f"mode={mode}", f"z={z}",
                 "pred = expm1(base_log) * SEASON_F^beta",
                 ("base_log = z*ens + (1-z)*log1p(gmv_yoy)" if mode == "C"
                  else "base_log = ens_log")]

    log(f"\nСабмиты (ens, mode={mode}, z={z}):")
    for b in EMIT_BETAS:
        pred = np.clip(np.expm1(base_log) * SEASON_F ** b, 0.0, None)
        nm = f"sub3_ens_yoy_b{int(round(b*100)):03d}.csv"
        write_sub(base_sub, uid, pred, nm)
        cfg_lines.append(f"{nm}: beta={b}")

    # Контрольная альтернативная формула (другой mode)
    alt_mode, alt_z = ("M", 1.0) if mode == "C" else ("C", 0.85)
    alt_log = ens_log.copy() if alt_mode == "M" else alt_z * ens_log + (1 - alt_z) * np.log1p(yoy_f)
    log(f"\nСабмиты (контроль mode={alt_mode}):")
    for b in (1.25, 1.75):
        pred = np.clip(np.expm1(alt_log) * SEASON_F ** b, 0.0, None)
        nm = f"sub3_ens_{alt_mode.lower()}_b{int(round(b*100)):03d}.csv"
        write_sub(base_sub, uid, pred, nm)
        cfg_lines.append(f"{nm}: alt mode={alt_mode}, z={alt_z}, beta={b}")

    # Микс с пайплайном (разные панели/ансамбли => диверсификация)
    if S2_SUB.exists():
        s2 = pd.read_csv(S2_SUB).rename(columns={"predict": "s2"})[["user_id", "s2"]]
        df = pd.DataFrame({"user_id": uid}).merge(s2, on="user_id", how="left")
        if df["s2"].notna().all():
            mix_log = (1 - MIX_S2_WEIGHT) * base_log \
                + MIX_S2_WEIGHT * np.log1p(df["s2"].to_numpy(np.float64))
            log(f"\nСабмиты (микс {1-MIX_S2_WEIGHT:.0%} panel + {MIX_S2_WEIGHT:.0%} pipeline):")
            for b in EMIT_BETAS:
                pred = np.clip(np.expm1(mix_log) * SEASON_F ** b, 0.0, None)
                nm = f"sub3_mix_yoy_b{int(round(b*100)):03d}.csv"
                write_sub(base_sub, uid, pred, nm)
                cfg_lines.append(f"{nm}: mix, beta={b}")
        else:
            log("файл пайплайна не покрывает всех user_id — микс пропущен")
    else:
        log(f"{S2_SUB} не найден — микс пропущен")
    (OUT_DIR / "submission_config.txt").write_text("\n".join(cfg_lines), encoding="utf-8")

    if fi_model is not None:
        try:
            fi = pd.DataFrame({"feature": FCOLS, "gain": fi_model.feature_importance("gain")}) \
                .sort_values("gain", ascending=False)
            fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
            log("\nТоп-15 признаков:"); log(fi.head(15).to_string(index=False))
        except Exception:
            pass


# --- main -------------------------------------------------------------------
def main():
    log("=" * 94)
    log("Плотная панель и финальный ансамбль | " + time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"polars={pl.__version__} pandas={pd.__version__} numpy={np.__version__}; фичей={len(FCOLS)}")
    log(f"CUT_STEP={CUT_STEP}; GAP_DAYS={GAP_DAYS}; MIX_S2_WEIGHT={MIX_S2_WEIGHT}")
    log("=" * 94)

    daily = load_daily()
    dmin, dmax = daily["event_date"].min(), daily["event_date"].max()
    plat = build_platform(daily)
    SEASON_F = seasonal_analysis(plat); save_report()

    cuts = build_cuts()
    log(f"\nСрезов панели: {len(cuts)} ({cuts[0]} .. {cuts[-1]}); тест-срез: {FINAL_CUT}")
    libs = load_libs()

    section("2. СБОРКА ПАНЕЛИ (может занять 5-8 минут)")
    panel = {}
    all_cuts = cuts + [FINAL_CUT]
    for i, cut in enumerate(tqdm(all_cuts, desc="Сборка срезов")):
        t0 = time.time()
        pdf, yoy_avail, has_target = build_fold(daily, cut, dmin, dmax)
        pdf = derive(pdf, cut, yoy_avail)
        missing = [c for c in FCOLS if c not in pdf.columns]
        assert not missing, f"нет колонок: {missing[:5]}"
        X = pdf[FCOLS].to_numpy(dtype=np.float32)
        assert np.isfinite(X).all(), f"NaN/inf в фичах, cutoff {cut}"
        ylog = (np.log1p(pdf["target"].to_numpy(dtype=np.float64)) if has_target
                else np.full(len(pdf), np.nan))
        panel[cut] = dict(X=X, ylog=ylog,
                          tl=pdf["is_test_like"].to_numpy().astype(bool),
                          yoy=pdf["gmv_yoy"].to_numpy(dtype=np.float64),
                          uid=pdf["user_id"].to_numpy(dtype=np.int64),
                          yraw=np.expm1(ylog) if has_target else None)
        if i % 10 == 0 or cut in (cuts[-1], VAL_CUT, FINAL_CUT):
            log(f"  [{i+1}/{len(all_cuts)}] cutoff {cut}: {len(pdf):,} юзеров, "
                f"yoy={yoy_avail}, target={has_target}, {time.time()-t0:.0f}с")
        del pdf; gc.collect()
    del daily; gc.collect(); save_report()

    cfg = None
    try:
        cfg = validation(libs, panel, cuts, plat); save_report()
    except Exception:
        log(traceback.format_exc()); save_report()
    if cfg is None or not cfg.get("iters"):
        raise RuntimeError("validation failed: cfg is None or empty iters; check logs above")

    try:
        final_train(libs, panel, cuts, cfg, SEASON_F); save_report()
    except Exception:
        log(traceback.format_exc()); save_report()

    log("\nГОТОВО.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        save_report()
