"""Полный пайплайн модели: ~250 признаков, ансамбль LGBM+XGB (+Cat),
сезонный множитель, YoY-бленд, файлы на сетке beta.

Модуль читает data/train.parquet и data/sample_submit.csv, строит панель
из 12 срезов с ~250 признаками на пользователя, прогоняет честную time-CV
для трёх моделей, перебирает формулу бленда (модель x YoY x сезонный множитель)
на фолде 2026-01-14, дообучает ансамбль на всех фолдах и сохраняет набор
сабмитов с разными beta для последующей калибровки.

Запуск: python src/02_pipeline.py
Выход: out/report.txt, out/{model_scores,blend_tuning_*,exp2_beta,lofo_beta,
       feature_importance}.csv, out/sub_v2_*.csv
"""
import gc
import math
import time
import traceback
import warnings
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

# --- конфиг -----------------------------------------------------------------
DATA_CANDIDATES = ["data/train.parquet", "train.parquet", "./data/train.parquet"]
SUB_CANDIDATES = ["data/sample_submit.csv", "sample_submit.csv", "./sample_submit.csv"]
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

WINDOWS = [3, 7, 14, 30, 60, 90, 180, 365]
METRICS = ["gmv", "gmv_search", "gmv_cat", "to_cart", "to_ord", "searches"]
HAS_COLS = ["has_search_to_cart", "has_search_to_ord", "has_cat_to_cart", "has_cat_to_ord"]
EMA_DEFS = [("gmv", 7), ("gmv", 14), ("gmv", 30), ("gmv", 90),
            ("to_ord", 30), ("to_cart", 30), ("searches", 14), ("active", 14)]
LN2 = math.log(2.0)
EPOCH = date(1970, 1, 1)

CUTS = [date(2025, 2, 13), date(2025, 3, 15), date(2025, 4, 15), date(2025, 5, 15),
        date(2025, 6, 15), date(2025, 7, 15), date(2025, 8, 15), date(2025, 9, 15),
        date(2025, 10, 15), date(2025, 11, 15), date(2025, 12, 15), date(2026, 1, 14)]
FINAL_CUT = date(2026, 2, 13)
FEB_FOLD = date(2025, 2, 13)
EVAL_CUTS = [date(2025, 11, 15), date(2025, 12, 15), date(2026, 1, 14)]
CAT_CV_FOLDS = [date(2026, 1, 14)]           # CatBoost валидируем только на самом тест-подобном фолде
INCLUDE_MODELS = ["lgbm", "xgb", "cat"]
RUN_LOFO = True                              # 12 доп. обучений LGBM (~50 мин)
FAST_MODE = False                            # True: только LGBM, 1 eval-фолд, без LOFO
SEASON_F_OVERRIDE = None                     # напр. 1.163; None -> сезонный аналог 2025

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
ZS = [round(float(z), 2) for z in np.arange(0.50, 1.0001, 0.05)]
FINAL_BETAS = [(0.0, "b000"), (0.5, "b050"), (1.0, "b100")]
LGB_SEEDS_FINAL = [42, 202, 777]
XGB_SEEDS_FINAL = [42, 777]

LGB_PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
                  num_leaves=96, min_child_samples=300, feature_fraction=0.65,
                  bagging_fraction=0.85, bagging_freq=1, lambda_l2=3.0,
                  max_bin=255, verbosity=-1, n_jobs=-1, seed=42)
XGB_PARAMS = dict(objective="reg:squarederror", eval_metric="rmse", eta=0.05,
                  max_depth=8, min_child_weight=100, subsample=0.85,
                  colsample_bytree=0.65, reg_lambda=3.0, tree_method="hist",
                  nthread=-1, seed=42)
CAT_PARAMS = dict(iterations=3000, learning_rate=0.07, depth=8, l2_leaf_reg=5.0,
                  loss_function="RMSE", random_seed=42, od_type="Iter", od_wait=150,
                  border_count=128, bootstrap_type="Bernoulli", subsample=0.8,
                  thread_count=-1, verbose=False)

if FAST_MODE:
    INCLUDE_MODELS = ["lgbm"]; RUN_LOFO = False
    EVAL_CUTS = [date(2026, 1, 14)]; CAT_CV_FOLDS = []
    LGB_SEEDS_FINAL = [42]; XGB_SEEDS_FINAL = [42]

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


PLEN = pl.len if hasattr(pl, "len") else pl.count


def grp(df, *a, **kw):
    return df.group_by(*a, **kw) if hasattr(df, "group_by") else df.groupby(*a, **kw)


# --- 1. данные и платформа --------------------------------------------------
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
    """Истинный фактор платформы: GMV(цель-окно)/GMV(последние 30д) для фолда с cutoff."""
    return (plat_mean(plat, cut + timedelta(days=1), cut + timedelta(days=30))
            / plat_mean(plat, cut - timedelta(days=29), cut))


def seasonal_analysis(plat):
    section("1. СЕЗОННЫЙ АНАЛИЗ ПЛАТФОРМЫ (исправленная версия)")
    last30 = plat_mean(plat, FINAL_CUT - timedelta(days=29), FINAL_CUT)
    ana = plat_mean(plat, date(2025, 2, 14), date(2025, 3, 15))
    ana_prev = plat_mean(plat, date(2025, 1, 15), date(2025, 2, 13))
    F_analog = ana / ana_prev
    g_yoy = (plat_mean(plat, date(2026, 1, 1), date(2026, 2, 13))
             / plat_mean(plat, date(2025, 1, 1), date(2025, 2, 13)))
    est_daily = ana * g_yoy
    F_yoy = est_daily / last30
    SEASON_F = float(SEASON_F_OVERRIDE) if SEASON_F_OVERRIDE else float(F_analog)
    log(f"Аналог окна таргета 2025 (14.02-15.03.2025): {ana:,.0f}; предыдущие 30д 2025: {ana_prev:,.0f}")
    log(f"Сезонный аплифт (метод 1, аналог 2025):      x{F_analog:.4f}")
    log(f"YoY-рост платформы (01.01-13.02 26/25):       x{g_yoy:.4f}")
    log(f"Сезонный аплифт (метод 2, YoY):               x{F_yoy:.4f}  (оценка окна: {est_daily:,.0f}/день)")
    log(f"==> ИСПОЛЬЗУЕМ SEASON_F = {SEASON_F:.4f}  (оба метода: x1.16-1.18 — окно таргета 'горячее')")
    Gl_test = last30 / ana                      # yoy-компонента: уровень последних 30д
    Gt_test = last30 * SEASON_F / ana           # yoy-компонента: уровень окна таргета
    log(f"Коэфф. для yoy-компоненты: G_last30={Gl_test:.4f}, G_target={Gt_test:.4f}")
    log("\nИстинные факторы фолдов (цель-окно/последние 30д):")
    for c in CUTS:
        log(f"  cutoff {c}: F = {factor_of(plat, c):.4f}")
    return dict(SEASON_F=SEASON_F, Gl_test=Gl_test, Gt_test=Gt_test, last30=last30)


# --- 2. сборка фолдов -------------------------------------------------------
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
        f[c] = f[c].astype(np.float64)
        f[c + "_log"] = np.log1p(f[c])
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


# --- 3. модели --------------------------------------------------------------
def load_libs():
    libs = {}
    for name, mod in [("lgbm", "lightgbm"), ("xgb", "xgboost"), ("cat", "catboost")]:
        try:
            libs[name] = __import__(mod)
            log(f"  {name}: {mod} {getattr(libs[name], '__version__', '?')}")
        except Exception as e:
            log(f"  {name}: НЕДОСТУПЕН ({e}) — исключаем")
    return libs


def make_model(name, libs, Xtr, ytr, Xin=None, yin=None, seed=42, rounds=None):
    """Возвращает (predict_fn, best_iter, model). Прогноз — в log1p-пространстве."""
    if name == "lgbm":
        lgb = libs["lgbm"]; params = dict(LGB_PARAMS)
        params.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
        dtr = lgb.Dataset(Xtr, ytr, feature_name=FCOLS)
        if Xin is not None:
            din = lgb.Dataset(Xin, yin, reference=dtr, feature_name=FCOLS)
            m = lgb.train(params, dtr, num_boost_round=6000, valid_sets=[din],
                          callbacks=[lgb.early_stopping(120, verbose=False)])
            bi = m.best_iteration or 6000
        else:
            m = lgb.train(params, dtr, num_boost_round=int(rounds)); bi = int(rounds)
        return (lambda X: m.predict(X, num_iteration=bi)), bi, m
    if name == "xgb":
        xgb = libs["xgb"]; params = dict(XGB_PARAMS); params["seed"] = seed
        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FCOLS)
        if Xin is not None:
            din = xgb.DMatrix(Xin, label=yin, feature_names=FCOLS)
            m = xgb.train(params, dtr, num_boost_round=6000, evals=[(din, "v")],
                          early_stopping_rounds=120, verbose_eval=False)
            bi = (int(m.best_iteration) + 1) if m.best_iteration is not None else 6000
        else:
            m = xgb.train(params, dtr, num_boost_round=int(rounds)); bi = int(rounds)
        return (lambda X: m.predict(xgb.DMatrix(X, feature_names=FCOLS),
                                    iteration_range=(0, bi))), bi, m
    if name == "cat":
        params = dict(CAT_PARAMS); params["random_seed"] = seed
        if rounds is not None:
            params["iterations"] = int(rounds)
            params.pop("od_type", None); params.pop("od_wait", None)
        m = libs["cat"].CatBoostRegressor(**params)
        if Xin is not None:
            m.fit(Xtr, ytr, eval_set=(Xin, yin), use_best_model=True)
            b = m.get_best_iteration()
            bi = int(b) + 1 if (b is not None and int(b) >= 0) else params["iterations"]
        else:
            m.fit(Xtr, ytr); bi = params["iterations"]
        return (lambda X: np.asarray(m.predict(X), dtype=np.float64)), bi, m
    raise ValueError(name)


# --- 4. тюнинг формулы ------------------------------------------------------
def tune_formula(m_log, y_raw, yoy_raw, F, Gl, Gt, tag):
    """Перебор: mode(M/C/D/E) × z × beta. pred = expm1(L) × F^beta."""
    comps = {"M": None, "C": np.log1p(yoy_raw)}
    if np.isfinite(Gl):
        comps["D"] = np.log1p(yoy_raw * Gl)
    if np.isfinite(Gt):
        comps["E"] = np.log1p(yoy_raw * Gt)
    rows = []
    for mode, comp in comps.items():
        zs = [1.0] if mode == "M" else ZS
        for z in zs:
            base = m_log if mode == "M" else z * m_log + (1 - z) * comp
            raw = np.expm1(base)
            for beta in BETAS:
                rows.append((mode, z, beta, rmsle(y_raw, raw * F ** beta)))
    df = (pd.DataFrame(rows, columns=["mode", "z", "beta", "rmsle"])
          .sort_values("rmsle").reset_index(drop=True))
    df.to_csv(OUT_DIR / f"blend_tuning_{tag}.csv", index=False)
    return df


def beta_table(m_log, y_raw, F, tag):
    rows = [(b, rmsle(y_raw, np.expm1(m_log) * F ** b)) for b in BETAS]
    df = pd.DataFrame(rows, columns=["beta", "rmsle"])
    log(f"  [{tag}] F={F:.4f}; RMSLE по beta: "
        + ", ".join(f"b={b:.2f}:{r:.4f}" for b, r in rows))
    return df


# --- 5. эксперименты --------------------------------------------------------
def exp1_cv(libs, panel, plat):
    section("3. ЭКСПЕРИМЕНТ 1: честная time-CV (LGBM+XGB [+Cat на 14.01.2026])")
    active = [m for m in INCLUDE_MODELS if m in libs]
    cv_preds, cv_scores, best_iters = {}, [], defaultdict(list)
    for ev in EVAL_CUTS:
        earlier = [c for c in CUTS if c < ev]
        inner = max(earlier)
        fit_cuts = [c for c in earlier if c != inner]
        Xtr = np.vstack([panel[c]["X"] for c in fit_cuts])
        ytr = np.concatenate([panel[c]["ylog"] for c in fit_cuts])
        Xin, yin = panel[inner]["X"], panel[inner]["ylog"]
        A = panel[ev]; Xev, ylog, tl, yoy = A["X"], A["ylog"], A["tl"], A["yoy"]
        yraw = A["yraw"]
        log(f"\n-- eval {ev}: fit={[str(c) for c in fit_cuts]}, inner(ES)={inner}, строк {Xtr.shape[0]:,}")
        preds = {}
        for name in active:
            if name == "cat" and ev not in CAT_CV_FOLDS:
                continue
            t0 = time.time()
            pfn, bi, _ = make_model(name, libs, Xtr, ytr, Xin, yin)
            p = pfn(Xev)
            s_tl = float(np.sqrt(np.mean((p[tl] - ylog[tl]) ** 2)))
            s_all = float(np.sqrt(np.mean((p - ylog) ** 2)))
            preds[name] = p; best_iters[name].append(bi)
            cv_scores.append({"cutoff": ev, "model": name, "rmsle_tl": s_tl, "rmsle_all": s_all, "best_iter": bi})
            log(f"   {name:>5s}: RMSLE(TL)={s_tl:.4f}  all={s_all:.4f}  iter={bi}  [{time.time()-t0:.0f}с]")
            del pfn, p
        ens = np.mean(list(preds.values()), axis=0)
        s_ens = float(np.sqrt(np.mean((ens[tl] - ylog[tl]) ** 2)))
        cv_scores.append({"cutoff": ev, "model": "ENSEMBLE", "rmsle_tl": s_ens,
                          "rmsle_all": float(np.sqrt(np.mean((ens - ylog) ** 2))), "best_iter": 0})
        log(f"   ENSEMBLE: RMSLE(TL)={s_ens:.4f}")
        F = factor_of(plat, ev)
        yoy0, _ = yoy_window(ev)
        Gl = plat_mean(plat, ev - timedelta(days=29), ev) / plat_mean(plat, yoy0, yoy0 + timedelta(days=29))
        Gt = (plat_mean(plat, ev + timedelta(days=1), ev + timedelta(days=30))
              / plat_mean(plat, yoy0, yoy0 + timedelta(days=29)))
        cv_preds[ev] = dict(preds=preds, ens=ens, lgbm=preds.get("lgbm", ens),
                            ylog=ylog, tl=tl, yoy=yoy, yraw=yraw, F=F, Gl=Gl, Gt=Gt)
        beta_table(ens[tl], yraw[tl], F, f"multiplier {ev}")
        del Xtr, ytr; gc.collect()
    pd.DataFrame(cv_scores).to_csv(OUT_DIR / "model_scores.csv", index=False)
    log("\nСводка RMSLE(TL):")
    log(pd.DataFrame(cv_scores).pivot(index="model", columns="cutoff", values="rmsle_tl")
        .to_string(float_format=lambda x: f"{x:.4f}"))
    return cv_preds, best_iters


def exp2_feb(libs, panel, plat):
    section("4. ЭКСПЕРИМЕНТ 2: февральский аналог (модель без фолда 13.02.2025 -> предсказать его)")
    if "lgbm" not in libs:
        log("нет lightgbm — пропуск"); return None
    inner = date(2026, 1, 14)
    fit_cuts = [c for c in CUTS if c not in (FEB_FOLD, inner)]
    Xtr = np.vstack([panel[c]["X"] for c in fit_cuts])
    ytr = np.concatenate([panel[c]["ylog"] for c in fit_cuts])
    pfn, bi, _ = make_model("lgbm", libs, Xtr, ytr, panel[inner]["X"], panel[inner]["ylog"])
    A = panel[FEB_FOLD]; p = pfn(A["X"]); tl, yraw = A["tl"], A["yraw"]
    F = factor_of(plat, FEB_FOLD)
    base = rmsle(yraw[tl], np.expm1(p[tl]))
    log(f"  F(фолда)={F:.4f}; базовый RMSLE(TL)={base:.4f}; "
        f"mean(pred)/mean(target)={np.expm1(p[tl]).mean()/yraw[tl].mean():.3f}")
    rows = [(b, rmsle(yraw[tl], np.expm1(p[tl]) * F ** b)) for b in BETAS]
    rows += [(f"k={k:.3f}", rmsle(yraw[tl], np.expm1(p[tl]) * k))
             for k in np.arange(1.0, 1.351, 0.025)]
    for b, r in rows:
        log(f"    множитель {b}: RMSLE={r:.4f}")
    pd.DataFrame(rows, columns=["mult", "rmsle"]).to_csv(OUT_DIR / "exp2_beta.csv", index=False)
    del Xtr, ytr; gc.collect()
    return rows


def exp3_lofo(libs, panel, plat, best_iters):
    section("5. ЭКСПЕРИМЕНТ 3: LOFO по 12 фолдам (оценка beta для множителя, LGBM)")
    if not RUN_LOFO or "lgbm" not in libs:
        log("пропущен (RUN_LOFO=False)"); return None
    rows = []
    for fc in CUTS:
        others = [c for c in CUTS if c != fc]
        inner = max(others)
        fit_cuts = [c for c in others if c != inner]
        Xtr = np.vstack([panel[c]["X"] for c in fit_cuts])
        ytr = np.concatenate([panel[c]["ylog"] for c in fit_cuts])
        pfn, _, _ = make_model("lgbm", libs, Xtr, ytr, panel[inner]["X"], panel[inner]["ylog"])
        A = panel[fc]; p = pfn(A["X"]); tl, yraw = A["tl"], A["yraw"]
        F = factor_of(plat, fc)
        for b in BETAS:
            rows.append({"cutoff": str(fc), "F": round(F, 4), "beta": b,
                         "rmsle": rmsle(yraw[tl], np.expm1(p[tl]) * F ** b)})
        best = min((r for r in rows if r["cutoff"] == str(fc)), key=lambda r: r["rmsle"])
        log(f"  fold {fc}: F={F:.3f}, лучший beta={best['beta']} (RMSLE={best['rmsle']:.4f})")
        del Xtr, ytr; gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "lofo_beta.csv", index=False)
    agg = df.groupby("beta")["rmsle"].mean()
    strong = df[df["F"].apply(lambda x: abs(np.log(x)) > 0.07)].groupby("beta")["rmsle"].mean()
    log("\nСредний RMSLE по beta (все фолды / фолды с сильным фактором):")
    for b in BETAS:
        log(f"  beta={b:.2f}: {agg.get(b, float('nan')):.4f} / {strong.get(b, float('nan')):.4f}")
    return agg


# --- 6. финал и сабмиты -----------------------------------------------------
def apply_formula(base_log, yoy_raw, mode, z, beta, F, Gl, Gt):
    if mode == "M":
        L = base_log
    else:
        comp = {"C": np.log1p(yoy_raw), "D": np.log1p(yoy_raw * Gl),
                "E": np.log1p(yoy_raw * Gt)}[mode]
        L = z * base_log + (1 - z) * comp
    return np.clip(np.expm1(L) * F ** beta, 0.0, None)


def write_sub(base_sub, uid, pred, fname):
    out = base_sub.merge(pd.DataFrame({"user_id": uid, "predict": np.clip(pred, 0, None)}),
                         on="user_id", how="left")
    out["predict"] = out["predict"].fillna(0.0)
    assert len(out) == len(base_sub) and out["predict"].notna().all() and (out["predict"] >= 0).all()
    out.to_csv(OUT_DIR / fname, index=False)
    p = out["predict"].values
    log(f"  {fname}: нулей(<1e-6)={(p < 1e-6).mean():.3f}; медиана={np.median(p):.1f}; "
        f"p95={np.quantile(p, .95):.0f}; max={p.max():,.0f}; сумма={p.sum():,.0f}")


# --- main -------------------------------------------------------------------
def main():
    log("=" * 94)
    log("Полный пайплайн | " + time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"polars={pl.__version__} pandas={pd.__version__} numpy={np.__version__}; "
        f"фичей={len(FCOLS)}; фолдов={len(CUTS)}+test; FAST_MODE={FAST_MODE}")
    log("=" * 94)
    daily = load_daily()
    dmin, dmax = daily["event_date"].min(), daily["event_date"].max()
    plat = build_platform(daily)
    sea = seasonal_analysis(plat); save_report()

    section("2. СБОРКА ПАНЕЛИ ФОЛДОВ")
    libs = load_libs()
    panel = {}
    for cut in CUTS + [FINAL_CUT]:
        t0 = time.time()
        pdf, yoy_avail, has_target = build_fold(daily, cut, dmin, dmax)
        pdf = derive(pdf, cut, yoy_avail)
        missing = [c for c in FCOLS if c not in pdf.columns]
        assert not missing, f"нет колонок: {missing[:5]}"
        X = pdf[FCOLS].to_numpy(dtype=np.float32)
        assert np.isfinite(X).all()
        ylog = (np.log1p(pdf["target"].to_numpy(dtype=np.float64)) if has_target
                else np.full(len(pdf), np.nan))
        panel[cut] = dict(X=X, ylog=ylog, tl=pdf["is_test_like"].to_numpy().astype(bool),
                          yoy=pdf["gmv_yoy"].to_numpy(dtype=np.float64),
                          uid=pdf["user_id"].to_numpy(dtype=np.int64),
                          yraw=(np.expm1(ylog) if has_target else None))
        log(f"  cutoff {cut}: {len(pdf):,} юзеров, yoy={yoy_avail}, target={has_target}, "
            f"test-like={panel[cut]['tl'].mean():.1%}, {time.time()-t0:.0f}с")
        del pdf; gc.collect()
    del daily; gc.collect(); save_report()

    cv_preds, best_iters = {}, defaultdict(list)
    try:
        cv_preds, best_iters = exp1_cv(libs, panel, plat); save_report()
    except Exception:
        log(traceback.format_exc()); save_report()

    # Тюнинг финальной формулы на фолде 14.01.2026 (там есть YoY).
    winner = dict(mode="M", z=1.0, beta=0.0)
    winner_l = dict(mode="M", z=1.0, beta=0.0)
    try:
        section("3a. ТЮНИНГ ФОРМУЛЫ (бленд с YoY x сезонный множитель), фолд 14.01.2026")
        ev = date(2026, 1, 14)
        if ev in cv_preds:
            st = cv_preds[ev]; tl = st["tl"]
            df_t = tune_formula(st["ens"][tl], st["yraw"][tl], st["yoy"][tl],
                                st["F"], st["Gl"], st["Gt"], "ens")
            log("Топ-10 (ансамбль):"); log(df_t.head(10).to_string(index=False))
            winner = dict(df_t.iloc[0][["mode", "z", "beta"]])
            df_l = tune_formula(st["lgbm"][tl], st["yraw"][tl], st["yoy"][tl],
                                st["F"], st["Gl"], st["Gt"], "lgbm")
            winner_l = dict(df_l.iloc[0][["mode", "z", "beta"]])
            log(f"==> победитель (ансамбль): mode={winner['mode']}, z={winner['z']}, beta={winner['beta']}")
            log(f"==> победитель (только LGBM): mode={winner_l['mode']}, z={winner_l['z']}, beta={winner_l['beta']}")
        save_report()
    except Exception:
        log(traceback.format_exc()); save_report()

    try:
        exp2_feb(libs, panel, plat); save_report()
    except Exception:
        log(traceback.format_exc()); save_report()
    try:
        exp3_lofo(libs, panel, plat, best_iters); save_report()
    except Exception:
        log(traceback.format_exc()); save_report()

    section("6. ФИНАЛЬНЫЕ МОДЕЛИ НА ВСЕХ 12 ФОЛДАХ + САБМИТЫ")
    sub_file = find_file(SUB_CANDIDATES)
    if sub_file is None:
        log("sample_submit.csv не найден — сабмиты не созданы"); save_report(); return
    base_sub = pd.read_csv(sub_file)[["user_id"]].astype({"user_id": "int64"})
    Xall = np.vstack([panel[c]["X"] for c in CUTS])
    yall = np.concatenate([panel[c]["ylog"] for c in CUTS])
    Afin = panel[FINAL_CUT]; Xfin, yoy_fin, uid_fin = Afin["X"], Afin["yoy"], Afin["uid"]
    logs_, fi_model = {}, None
    use = [m for m in INCLUDE_MODELS if m in libs and best_iters.get(m)]
    if "lgbm" in use:
        r = max(200, int(np.ceil(1.1 * np.mean(best_iters["lgbm"]))))
        log(f"LGBM: {r} итераций × {len(LGB_SEEDS_FINAL)} seed...")
        for s in LGB_SEEDS_FINAL:
            pfn, _, m = make_model("lgbm", libs, Xall, yall, seed=s, rounds=r)
            logs_[f"lgbm_s{s}"] = pfn(Xfin); fi_model = m
    if "xgb" in use:
        r = max(200, int(np.ceil(1.1 * np.mean(best_iters["xgb"]))))
        log(f"XGB: {r} итераций x {len(XGB_SEEDS_FINAL)} seed...")
        for s in XGB_SEEDS_FINAL:
            pfn, _, _ = make_model("xgb", libs, Xall, yall, seed=s, rounds=r)
            logs_[f"xgb_s{s}"] = pfn(Xfin)
    if "cat" in use:
        ms = pd.read_csv(OUT_DIR / "model_scores.csv") if (OUT_DIR / "model_scores.csv").exists() else None
        ok = True
        if ms is not None:
            d = ms[(ms.cutoff == "2026-01-14")]
            cat_sc = d[d.model == "cat"]["rmsle_tl"]; lgb_sc = d[d.model == "lgbm"]["rmsle_tl"]
            ok = (len(cat_sc) > 0 and len(lgb_sc) > 0
                  and float(cat_sc.iloc[0]) <= float(lgb_sc.iloc[0]) + 0.010)
        if ok:
            r = max(300, int(np.ceil(1.1 * np.mean(best_iters["cat"]))))
            log(f"CatBoost: {r} итераций...")
            pfn, _, _ = make_model("cat", libs, Xall, yall, rounds=r)
            logs_["cat"] = pfn(Xfin)
        else:
            log("CatBoost хуже LGBM на валидации >0.01 — в ансамбль не включаем")
    ens = np.mean(list(logs_.values()), axis=0)
    lgbm_mean = np.mean([v for k, v in logs_.items() if k.startswith("lgbm")], axis=0)
    log(f"Ансамбль из {len(logs_)} моделей. Прогнозы (log): медиана={np.median(ens):.2f}, "
        f"p95={np.quantile(ens,.95):.2f}")

    F, Gl, Gt = sea["SEASON_F"], sea["Gl_test"], sea["Gt_test"]
    cfg = [f"SEASON_F={F:.4f} Gl={Gl:.4f} Gt={Gt:.4f}",
           f"ens_winner: {winner}", f"lgbm_winner: {winner_l}",
           "pred = expm1(z*model + (1-z)*log1p(yoy*G)) * F^beta"]
    for b, nm in FINAL_BETAS:
        pred = apply_formula(ens, yoy_fin, winner["mode"], winner["z"], b, F, Gl, Gt)
        write_sub(base_sub, uid_fin, pred, f"sub_v2_ens_yoy_{nm}.csv")
        cfg.append(f"sub_v2_ens_yoy_{nm}.csv: ens, mode={winner['mode']}, z={winner['z']}, beta={b}")
    pred = apply_formula(lgbm_mean, yoy_fin, winner_l["mode"], winner_l["z"], 0.5, F, Gl, Gt)
    write_sub(base_sub, uid_fin, pred, "sub_v2_lgbm_yoy_b050.csv")
    cfg.append(f"sub_v2_lgbm_yoy_b050.csv: lgbm-only, mode={winner_l['mode']}, z={winner_l['z']}, beta=0.5")
    pred = apply_formula(ens, yoy_fin, "M", 1.0, 0.5, F, Gl, Gt)
    write_sub(base_sub, uid_fin, pred, "sub_v2_ens_noyoy_b050.csv")
    cfg.append("sub_v2_ens_noyoy_b050.csv: чистый ансамбль, beta=0.5")
    (OUT_DIR / "submission_config.txt").write_text("\n".join(cfg), encoding="utf-8")

    if fi_model is not None:
        fi = pd.DataFrame({"feature": FCOLS, "gain": fi_model.feature_importance("gain")}) \
            .sort_values("gain", ascending=False)
        fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
        log("\nТоп-25 признаков:"); log(fi.head(25).to_string(index=False))

    save_report(); log("\nГОТОВО.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc()); save_report()
