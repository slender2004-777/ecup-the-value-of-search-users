"""Аудит исходных логов и первичные бейзлайны для задачи 30-дневного LTV.

Модуль читает data/train.parquet, проверяет инварианты колонок, считает
глобальную сезонность и портрет тестовой популяции, собирает временную панель
фолдов с таргетом на 30 дней вперёд, прогоняет наивные бейзлайны и quick-модель
LightGBM, сохраняет сабмиты-якоря.

Запуск: python src/01_eda.py
Выход: out/report.txt, out/{baseline_scores,model_scores,feature_importance,
       daily_agg,user_stats}.csv, out/sub_*.csv, out/timeline.png
"""
import gc
import time
import traceback
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")

# --- конфиг -----------------------------------------------------------------
DATA_CANDIDATES = ["data/train.parquet", "train.parquet",
                   "./data/train.parquet", "input/train.parquet"]
SUB_CANDIDATES = ["data/sample_submit.csv", "sample_submit.csv",
                  "./sample_submit.csv", "input/sample_submit.csv"]
OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

METRICS = ["gmv", "gmv_search", "gmv_cat", "to_cart", "to_ord", "searches"]
WINDOWS = [7, 30, 90, 180, 365]

# Cutoff = последний день истории; таргет = сумма gmv за (cut+1 .. cut+30).
CUTS = [date(2025, 2, 13),  # сезонный аналог теста (таргет 14.02-15.03.2025)
        date(2025, 5, 15), date(2025, 6, 15), date(2025, 7, 15), date(2025, 8, 15),
        date(2025, 9, 15), date(2025, 10, 15), date(2025, 11, 15),
        date(2025, 12, 15), date(2026, 1, 14)]
EVAL_CUTS = [date(2025, 2, 13), date(2025, 12, 15), date(2026, 1, 14)]
MODEL_EVAL_CUTS = [date(2025, 11, 15), date(2025, 12, 15), date(2026, 1, 14)]
FINAL_CUT = date(2026, 2, 13)

LGB_ROUNDS = 1200
LGB_ES = 60

# --- утилиты ----------------------------------------------------------------
REPORT = []


def log(msg=""):
    print(msg, flush=True)
    REPORT.append(str(msg))


def section(t):
    log("\n" + "#" * 94)
    log("## " + t)
    log("#" * 94)


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
    """RMSE в log-пространстве равен RMSLE исходных величин."""
    return float(np.sqrt(np.mean((yl - pl) ** 2)))


PLEN = pl.len if hasattr(pl, "len") else pl.count


def grp(df, *args, **kwargs):
    """Поларсовский group_by с fallback на старое имя groupby."""
    if hasattr(df, "group_by"):
        return df.group_by(*args, **kwargs)
    return df.groupby(*args, **kwargs)


# --- 1-2. загрузка ----------------------------------------------------------
def load_and_clean():
    f = find_file(DATA_CANDIDATES)
    if f is None:
        raise FileNotFoundError(
            "train.parquet не найден: положите в ./data/ или поправьте DATA_CANDIDATES")
    t0 = time.time()
    data = pl.read_parquet(f)
    log(f"Файл: {f}")
    log(f"Размер: {data.height:,} строк x {data.width} колонок (загрузка {time.time()-t0:.1f}с)")
    log(f"Колонки: {data.columns}")
    nulls = data.null_count().to_dicts()[0]
    log(f"Null-ы: { {k: v for k, v in nulls.items() if v} or 'нет' }")
    log(f"Даты: {data['event_date'].min()} .. {data['event_date'].max()}")
    n0 = data.height
    data_u = data.unique()
    log(f"Точных дубликатов строк: {n0 - data_u.height:,} ({(n0 - data_u.height)/n0:.4%})")
    pc = grp(data_u, ["user_id", "event_date"]).agg(PLEN().alias("cnt"))
    n_dup = pc.filter(pl.col("cnt") > 1).height
    log(f"Пар (user_id,event_date) с >1 строкой после exact-dedup: {n_dup:,}"
        + ("  <-- агрегация суммой" if n_dup else ""))
    num_cols = [c for c in data.columns if c not in ("user_id", "event_date")]
    daily = (grp(data_u, ["user_id", "event_date"])
             .agg([pl.col(c).sum() for c in num_cols])
             .sort(["user_id", "event_date"]))
    log(f"Дневная таблица (userxday): {daily.height:,} строк; пользователей: {daily['user_id'].n_unique():,}")
    del data, data_u
    gc.collect()
    return daily, num_cols


# --- 3. инварианты ----------------------------------------------------------
def invariants(daily, num_cols):
    section("3. ИНВАРИАНТЫ И КАЧЕСТВО КОЛОНОК")
    def sh(expr):
        return daily.select(expr.cast(pl.Float64).mean()).item()

    checks = [
        ("gmv == gmv_search + gmv_cat",
         (pl.col("gmv") - pl.col("gmv_search") - pl.col("gmv_cat")).abs() <= 1e-6),
        ("to_cart == search_to_cart + cat_to_cart",
         pl.col("to_cart") == pl.col("search_to_cart") + pl.col("cat_to_cart")),
        ("to_ord == search_to_ord + cat_to_ord",
         pl.col("to_ord") == pl.col("search_to_ord") + pl.col("cat_to_ord")),
        ("search == (searches > 0)", pl.col("search") == (pl.col("searches") > 0).cast(pl.Int64)),
        ("search==1 & searches==0 (странно)", (pl.col("search") == 1) & (pl.col("searches") == 0)),
        ("cat==1: только просмотр (без корзины/заказа)",
         (pl.col("cat") == 1) & (pl.col("cat_to_cart") == 0) & (pl.col("cat_to_ord") == 0)),
        ("gmv>0 & to_ord==0", (pl.col("gmv") > 0) & (pl.col("to_ord") == 0)),
        ("to_ord>0 & gmv==0", (pl.col("to_ord") > 0) & (pl.col("gmv") == 0)),
        ("строка вообще без активности",
         (pl.col("searches") == 0) & (pl.col("to_cart") == 0) & (pl.col("to_ord") == 0)
         & (pl.col("gmv") == 0) & (pl.col("search") == 0) & (pl.col("cat") == 0)),
        ("has_search_to_cart == (search_to_cart > 0)",
         pl.col("has_search_to_cart") == (pl.col("search_to_cart") > 0).cast(pl.Int64)),
        ("has_search_to_ord == (search_to_ord > 0)",
         pl.col("has_search_to_ord") == (pl.col("search_to_ord") > 0).cast(pl.Int64)),
        ("has_cat_to_cart == (cat_to_cart > 0)",
         pl.col("has_cat_to_cart") == (pl.col("cat_to_cart") > 0).cast(pl.Int64)),
        ("has_cat_to_ord == (cat_to_ord > 0)",
         pl.col("has_cat_to_ord") == (pl.col("cat_to_ord") > 0).cast(pl.Int64)),
    ]
    for name, e in checks:
        log(f"  {name:<52s}: {sh(e):.6f}")
    mins = daily.select([pl.col(c).min().alias(c) for c in num_cols]).to_dicts()[0]
    log(f"Минимумы по колонкам (нет ли отрицательных): {mins}")
    qs = [0.5, 0.9, 0.99, 0.999]
    gz = daily.filter(pl.col("gmv") > 0)
    gq = gz.select([pl.col("gmv").quantile(q).alias(f"p{q}") for q in qs]
                   + [pl.col("gmv").max().alias("max")]).to_dicts()[0]
    log("Дневной gmv (только gmv>0): " + ", ".join(f"{k}={v:,.0f}" for k, v in gq.items())
        + f"; доля дней с gmv>0: {gz.height/daily.height:.4f}")
    aov = daily.filter(pl.col("to_ord") > 0).with_columns((pl.col("gmv") / pl.col("to_ord")).alias("aov"))
    oq = aov.select([pl.col("aov").quantile(q).alias(f"p{q}") for q in qs]).to_dicts()[0]
    log("AOV = gmv/to_ord (to_ord>0): " + ", ".join(f"{k}={v:,.0f}" for k, v in oq.items()))


# --- 4. сезонность ----------------------------------------------------------
def seasonality(daily):
    section("4. ГЛОБАЛЬНАЯ СЕЗОННОСТЬ, ПРАЗДНИКИ, YoY-РОСТ ПЛАТФОРМЫ")
    daily_agg = (grp(daily, "event_date").agg(
        PLEN().alias("dau"),
        pl.col("gmv").sum().alias("gmv"),
        pl.col("to_cart").sum().alias("to_cart"),
        pl.col("to_ord").sum().alias("orders"),
        pl.col("searches").sum().alias("searches"),
    ).sort("event_date"))
    daily_agg.write_csv(OUT_DIR / "daily_agg.csv")
    da = daily_agg.to_pandas()
    log(f"Дней в данных: {len(da)}; суммарный GMV за период: {da['gmv'].sum():,.0f}")

    mon = (da.assign(y=da["event_date"].dt.year, m=da["event_date"].dt.month)
             .groupby(["y", "m"])
             .agg(days=("gmv", "size"), dau_avg=("dau", "mean"),
                  gmv_per_day=("gmv", "mean"), orders_per_day=("orders", "mean"),
                  searches_per_day=("searches", "mean")).reset_index())
    log("\nПомесячная статистика (среднее за один день):")
    log(mon.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    dw = (da.assign(dow=da["event_date"].dt.weekday)
             .groupby("dow").agg(gmv_per_day=("gmv", "mean"),
                                 dau_avg=("dau", "mean")).reset_index())
    dw["dow"] = dw["dow"].map({0: "Пн", 1: "Вт", 2: "Ср", 3: "Чт", 4: "Пт", 5: "Сб", 6: "Вс"})
    log("\nПо дню недели (среднее за день):")
    log(dw.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))

    top = da.nlargest(12, "gmv")[["event_date", "dau", "gmv", "orders", "searches"]]
    log("\nТоп-12 дней по суммарному GMV (ищем распродажи/праздники):")
    log(top.to_string(index=False))
    bot = da.nsmallest(8, "gmv")[["event_date", "dau", "gmv"]]
    log("\nАнти-топ-8 дней по GMV (провалы/каникулы):")
    log(bot.to_string(index=False))

    def davg(d0, d1, col="gmv"):
        m = (da["event_date"] >= pd.Timestamp(d0)) & (da["event_date"] <= pd.Timestamp(d1))
        return float(da.loc[m, col].mean())

    log("\nКлючевые календарные факты:")
    ny = davg(date(2026, 1, 1), date(2026, 1, 8)); dec = davg(date(2025, 12, 20), date(2025, 12, 31))
    jan = davg(date(2026, 1, 9), date(2026, 1, 20))
    log(f"  Дневной GMV: 01-08.01.2026={ny:,.0f}; 20-31.12.2025={dec:,.0f}; 09-20.01.2026={jan:,.0f}")
    G = davg(date(2026, 1, 1), date(2026, 2, 13)) / davg(date(2025, 1, 1), date(2025, 2, 13))
    Gd = davg(date(2026, 1, 1), date(2026, 2, 13), "dau") / davg(date(2025, 1, 1), date(2025, 2, 13), "dau")
    Go = davg(date(2026, 1, 1), date(2026, 2, 13), "orders") / davg(date(2025, 1, 1), date(2025, 2, 13), "orders")
    log(f"  YoY-рост платформы (01.01-13.02 2026 vs 2025): GMV x{G:.3f}, DAU x{Gd:.3f}, заказы x{Go:.3f}")
    a = davg(date(2025, 2, 14), date(2025, 3, 15)); b = davg(date(2025, 1, 15), date(2025, 2, 13))
    c = davg(date(2025, 12, 16), date(2026, 1, 14)); d = davg(date(2025, 11, 15), date(2025, 12, 14))
    log(f"  Дневной GMV: окно-аналог таргета 14.02-15.03.2025={a:,.0f}; 15.01-13.02.2025={b:,.0f}; "
        f"15.11-14.12.2025={d:,.0f}; последние 30д 15.01-13.02.2026={c:,.0f}")
    log(f"  Сезонный аплифт окна таргета: (аналог/15.01-13.02.25) x{a/b:.3f}; (аналог/15.11-14.12.25) x{a/d:.3f}")
    log(f"  Грубая оценка дневного GMV в окне таргета: {c:,.0f} x {a/b:.2f} = {c*a/b:,.0f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax[0].plot(da["event_date"], da["gmv"], lw=0.8); ax[0].set_title("Daily total GMV")
        ax[1].plot(da["event_date"], da["dau"], lw=0.8, color="tab:red"); ax[1].set_title("Daily active users")
        for a_ in ax:
            a_.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT_DIR / "timeline.png", dpi=110); plt.close(fig)
        log("Сохранён график: out/timeline.png")
    except Exception as e:
        log(f"(график пропущен: {e})")


# --- 5. пользователи --------------------------------------------------------
def user_stats(daily):
    section("5. ПОЛЬЗОВАТЕЛИ: АКТИВНОСТЬ, DORMANCY, ПРИТОК, КОНЦЕНТРАЦИЯ, САБМИТ")
    users = (grp(daily, "user_id").agg(
        PLEN().alias("n_days"),
        pl.col("event_date").min().alias("first_date"),
        pl.col("event_date").max().alias("last_date"),
        pl.col("gmv").sum().alias("gmv_total"),
        (pl.col("gmv") > 0).sum().alias("days_with_gmv"),
        (pl.col("searches") > 0).any().alias("ever_search"),
        (pl.col("cat") == 1).any().alias("ever_cat"),
        (pl.col("to_ord") > 0).any().alias("ever_order"),
    ))
    u = users.to_pandas()
    n = len(u)

    def qs(s, name):
        q = s.quantile([.01, .1, .25, .5, .75, .9, .99])
        log(f"  {name}: " + ", ".join(f"p{int(k*100)}={v:,.1f}" for k, v in q.items())
            + f", mean={s.mean():,.1f}")

    qs(u["n_days"], "активных дней (вся история)")
    qs(u["gmv_total"], "суммарный GMV (вся история)")
    qs(u["days_with_gmv"], "дней с gmv>0")
    log(f"  Пользователей с нулевым GMV за всю историю: {(u['gmv_total']==0).mean():.4f}")
    log(f"  Пользователей с <5 активных дней: {(u['n_days']<5).mean():.4f}")
    log(f"  ever_search (хоть 1 поиск): {u['ever_search'].mean():.4f}; "
        f"ever_cat (хоть 1 день каталога): {u['ever_cat'].mean():.4f}; "
        f"ever_order (хоть 1 заказ): {u['ever_order'].mean():.4f}")

    rec = (pd.Timestamp(FINAL_CUT) - pd.to_datetime(u["last_date"])).dt.days
    bins = [-1, 3, 7, 14, 30, 60, 90, 180, 10**6]
    labels = ["0-3", "4-7", "8-14", "15-30", "31-60", "61-90", "91-180", "180+"]
    t = pd.cut(rec, bins=bins, labels=labels).value_counts().reindex(labels)
    log(f"\nRecency на {FINAL_CUT} (портрет тестовой выборки):")
    for lab in labels:
        log(f"  {lab:>6s}: {int(t[lab]):>7,} ({t[lab]/n:.2%})")
    log(f"  Активны за последние 30 дней: {(rec<=30).mean():.4f}; за 90: {(rec<=90).mean():.4f}")

    fm = pd.to_datetime(u["first_date"]).dt.to_period("M").value_counts().sort_index()
    fm = fm.reindex(pd.period_range("2025-01", "2026-02", freq="M"), fill_value=0)
    log("\nПервое появление по месяцам (янв.2025 цензурирован началом данных):")
    for k, v in fm.items():
        log(f"  {k}: {int(v):,}")

    gs = np.sort(u["gmv_total"].values)[::-1]
    cs = np.cumsum(gs) / max(gs.sum(), 1.0)
    for p in [0.001, 0.01, 0.05, 0.10]:
        log(f"  Топ-{p:.1%} пользователей дают {cs[max(int(n*p)-1,0)]:.2%} всего GMV")
    u.to_csv(OUT_DIR / "user_stats.csv", index=False)

    sf = find_file(SUB_CANDIDATES)
    if sf is not None:
        sub = pd.read_csv(sf)
        log(f"\nsample_submit: {len(sub):,} строк; колонки {list(sub.columns)}")
        sset, tset = set(sub["user_id"].astype(int)), set(u["user_id"].astype(int))
        log(f"  Покрытие: пересечение {len(sset & tset):,}; в сабмите нет в train {len(sset-tset):,}; "
            f"в train нет в сабмите {len(tset-sset):,}")
        p = sub["predict"].values
        pos_q = (", ".join(f"p{int(k*100)}={v:,.1f}"
                           for k, v in pd.Series(p[p > 0]).quantile([.1, .5, .9, .99]).items())
                 if (p > 0).any() else "—")
        log(f"  predict==0: {(p==0).mean():.4f}; predict>0 квантили: {pos_q}")
    else:
        log("\nsample_submit.csv не найден (проверка покрытия пропущена)")


# --- 6a. сборка фолдов ------------------------------------------------------
def build_fold(daily, cut):
    """Признаки на конец дня `cut` + таргет = сумма gmv за (cut+1 .. cut+30)."""
    t0d, t1d = cut + timedelta(days=1), cut + timedelta(days=30)
    yoy0, yoy1 = t0d - timedelta(days=365), t1d - timedelta(days=365)
    hist = daily.filter(pl.col("event_date") <= cut)
    aggs = []
    for w in WINDOWS:
        cw = pl.col("event_date") > (cut - timedelta(days=w))
        for m in METRICS:
            aggs.append(pl.col(m).filter(cw).sum().alias(f"{m}_{w}"))
        aggs.append(pl.col("event_date").filter(cw).count().alias(f"active_days_{w}"))
        aggs.append(pl.col("gmv").filter(cw & (pl.col("gmv") > 0)).count().alias(f"days_gmv_{w}"))
    for m in METRICS:
        aggs.append(pl.col(m).sum().alias(f"{m}_all"))
    aggs += [
        pl.col("event_date").count().alias("active_days_all"),
        pl.col("gmv").filter(pl.col("gmv") > 0).count().alias("days_gmv_all"),
        pl.col("event_date").max().alias("last_date"),
        pl.col("event_date").min().alias("first_date"),
        pl.col("event_date").filter(pl.col("gmv") > 0).max().alias("last_gmv_date"),
        pl.col("event_date").filter(pl.col("to_ord") > 0).max().alias("last_order_date"),
        pl.col("gmv").filter((pl.col("event_date") >= yoy0) & (pl.col("event_date") <= yoy1)).sum().alias("gmv_yoy"),
        pl.col("event_date").filter((pl.col("event_date") >= yoy0) & (pl.col("event_date") <= yoy1)).count().alias("active_days_yoy"),
    ]
    feats = grp(hist, "user_id").agg(aggs)
    tgt = (grp(daily.filter((pl.col("event_date") >= t0d) & (pl.col("event_date") <= t1d)), "user_id")
           .agg(pl.col("gmv").sum().alias("target")))
    f = (feats.join(tgt, on="user_id", how="left")
               .with_columns(pl.col("target").fill_null(0.0))
               .with_columns(pl.lit(cut).alias("cut")))
    pdf = f.to_pandas()
    date_cols = ["last_date", "first_date", "last_gmv_date", "last_order_date", "cut"]
    for c in date_cols:
        pdf[c] = pd.to_datetime(pdf[c])
    num = [c for c in pdf.columns if c not in date_cols + ["user_id"]]
    pdf[num] = pdf[num].fillna(0.0)
    for c in num:
        if c != "target":
            pdf[c] = pdf[c].astype("float32")
    pdf["target"] = pdf["target"].astype("float64")
    return pdf, (yoy0 >= date(2025, 1, 1))


def derive(f, cut, yoy_avail):
    cut_ts = pd.Timestamp(cut)
    f = f.copy()
    f["recency"] = (cut_ts - f["last_date"]).dt.days
    f["is_test_like"] = (f["recency"] <= 30).astype(np.int8)
    f["recency_gmv"] = (cut_ts - f["last_gmv_date"]).dt.days.fillna(9999).clip(0, 9999)
    f["recency_order"] = (cut_ts - f["last_order_date"]).dt.days.fillna(9999).clip(0, 9999)
    f["tenure"] = (cut_ts - f["first_date"]).dt.days
    eps = 1.0
    f["gmv_ratio_30_90"] = f["gmv_30"] / (f["gmv_90"] + eps)
    f["gmv_ratio_30_365"] = f["gmv_30"] / (f["gmv_365"] + eps)
    f["gmv_ratio_30_all"] = f["gmv_30"] / (f["gmv_all"] + eps)
    f["gmvS_share"] = f["gmv_search_all"] / (f["gmv_all"] + eps)
    f["gmvC_share"] = f["gmv_cat_all"] / (f["gmv_all"] + eps)
    f["gmv_per_ad_90"] = f["gmv_90"] / (f["active_days_90"] + 1)
    f["gmv_per_ad_all"] = f["gmv_all"] / (f["active_days_all"] + 1)
    f["gmv_per_dg_all"] = f["gmv_all"] / (f["days_gmv_all"] + 1)
    f["searches_per_ad_90"] = f["searches_90"] / (f["active_days_90"] + 1)
    f["conv_cart_ord_90"] = f["to_ord_90"] / (f["to_cart_90"] + 1)
    f["active_rate_90"] = f["active_days_90"] / 90.0
    f["active_rate_all"] = f["active_days_all"] / (f["tenure"] + 1.0)
    f["yoy_avail"] = 1.0 if yoy_avail else 0.0
    if not yoy_avail:
        f["gmv_yoy"] = 0.0
        f["active_days_yoy"] = 0.0
    log_cols = ([f"{m}_{w}" for w in WINDOWS for m in METRICS]
                + [f"active_days_{w}" for w in WINDOWS] + [f"days_gmv_{w}" for w in WINDOWS]
                + [f"{m}_all" for m in METRICS]
                + ["active_days_all", "days_gmv_all", "gmv_yoy", "active_days_yoy"])
    for c in log_cols:
        f[c + "_log1p"] = np.log1p(f[c].astype(np.float64))
    f["month"] = cut.month
    f["dow"] = cut.weekday()
    return f


# --- 6. бейзлайны -----------------------------------------------------------
def baselines(folds):
    section("6. ФОЛДЫ, НАИВНЫЕ БЕЙЗЛАЙНЫ (RMSLE), КОРРЕЛЯЦИИ, RETENTION")
    log("Тест-популяция = юзеры, активные за последние 30 дней до cutoff.")
    log("Основная метрика — на test-like подвыборке (recency<=30); 'all' приводится справочно.")
    rows = []
    for cut in CUTS:
        f = folds[cut]
        y = f["target"].values
        tl = f["is_test_like"].values == 1
        ytl = y[tl]
        log(f"\n-- cutoff {cut} | таргет {cut+timedelta(days=1)}..{cut+timedelta(days=30)} | "
            f"юзеров {len(f):,} | test-like {tl.mean():.1%} | нулевой таргет(TL) {(ytl==0).mean():.2%} | "
            f"средний(TL) {ytl.mean():,.1f} | p95(TL) {np.quantile(ytl,.95):,.1f}")
        last30 = f["gmv_30"].values.astype(np.float64)
        preds = {
            "0_всегда": np.zeros(len(f)),
            "last30 (=sample организаторов)": last30,
            "mean90x30": f["gmv_90"].values.astype(np.float64) * (30/90),
            "mean180x30": f["gmv_180"].values.astype(np.float64) * (30/180),
            "mean365x30": f["gmv_365"].values.astype(np.float64) * (30/365),
            "mean_all_x30(ценз.)": f["gmv_all"].values.astype(np.float64) * 30.0 / (f["tenure"].values + 1),
            "ms_avg(7/30/90)": ((0.5*f["gmv_7"]/7 + 0.3*f["gmv_30"]/30 + 0.2*f["gmv_90"]/90)
                                .values.astype(np.float64)) * 30,
        }
        if int(f["yoy_avail"].iloc[0]) == 1:
            yoy = f["gmv_yoy"].values.astype(np.float64)
            preds["yoy_год_назад"] = yoy
            preds["logblend_.5(last30,yoy)"] = np.expm1(0.5*np.log1p(last30) + 0.5*np.log1p(yoy))
        for name, p in preds.items():
            rows.append({"cutoff": cut, "baseline": name,
                         "rmsle_tl": rmsle(ytl, p[tl]), "rmsle_all": rmsle(y, p)})
        alphas = np.arange(0.10, 1.51, 0.05)
        sc = [rmsle(ytl, a * last30[tl]) for a in alphas]
        i = int(np.argmin(sc))
        rows.append({"cutoff": cut, "baseline": f"last30 x {alphas[i]:.2f} (opt,TL)",
                     "rmsle_tl": sc[i], "rmsle_all": np.nan})
        scl = [rmsle(ytl, np.expm1(a * np.log1p(last30[tl]))) for a in alphas]
        j = int(np.argmin(scl))
        rows.append({"cutoff": cut, "baseline": f"logshrink(last30) a={alphas[j]:.2f} (TL)",
                     "rmsle_tl": scl[j], "rmsle_all": np.nan})
        rows.append({"cutoff": cut, "baseline": "const=expm1(mean_log) (TL)",
                     "rmsle_tl": rmsle(ytl, np.full(len(ytl), np.expm1(np.mean(np.log1p(ytl))))),
                     "rmsle_all": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "baseline_scores.csv", index=False)

    piv = df.pivot(index="baseline", columns="cutoff", values="rmsle_tl")
    piv = piv[sorted(piv.columns)]
    piv["MEAN_TL"] = piv.mean(axis=1)
    piv = piv.sort_values("MEAN_TL")
    log("\nRMSLE наивных бейзлайнов на TEST-LIKE подвыборке (сорт по среднему):")
    log(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    piv2 = df.pivot(index="baseline", columns="cutoff", values="rmsle_all")
    piv2 = piv2[sorted(piv2.columns)]
    piv2["MEAN_ALL"] = piv2.mean(axis=1)
    piv2 = piv2.sort_values("MEAN_ALL")
    log("\n(справочно) RMSLE на ВСЕХ юзерах фолда:")
    log(piv2.to_string(float_format=lambda x: f"{x:.4f}"))

    log("\nКорреляции Пирсона log1p(target) ~ признак, на test-like:")
    corr_feats = ["gmv_7", "gmv_30", "gmv_90", "gmv_365", "gmv_all", "gmv_yoy",
                  "to_cart_30", "to_ord_30", "searches_30",
                  "active_days_30", "active_days_90", "days_gmv_30",
                  "recency", "recency_gmv"]
    table = {}
    for cut in CUTS:
        f = folds[cut]
        ftl = f.loc[f["is_test_like"].values == 1]
        ly = np.log1p(ftl["target"].values)
        row = {}
        for c in corr_feats:
            x = ftl[c].values.astype(np.float64)
            x = np.clip(x, 0, 400) if c.startswith("recency") else np.log1p(x)
            row[c] = np.corrcoef(x, ly)[0, 1] if np.std(x) > 0 else np.nan
        table[cut] = row
    log(pd.DataFrame(table).round(3).to_string())

    log("\nP(target>0) и E[target] по recency-бакетам (все юзеры фолда):")
    bins = [-1, 3, 7, 14, 30, 60, 90, 180, 10**6]
    labels = ["0-3", "4-7", "8-14", "15-30", "31-60", "61-90", "91-180", "180+"]
    for cut in EVAL_CUTS:
        f = folds[cut]
        b = pd.cut(f["recency"], bins=bins, labels=labels)
        g = f.groupby(b, observed=True).agg(n=("target", "size"),
                                            p_pos=("target", lambda s: float((s > 0).mean())),
                                            mean_gmv=("target", "mean"))
        log(f"\n  cutoff {cut}:")
        for lab, r in g.iterrows():
            log(f"    recency {lab:>6s}: n={int(r['n']):>7,}  P(gmv>0)={r['p_pos']:.3f}  E[gmv]={r['mean_gmv']:,.1f}")

    import itertools
    log("\nУстойчивость: корреляция log1p(таргетов) между фолдами (общие test-like юзеры):")
    for c1, c2 in itertools.combinations(EVAL_CUTS, 2):
        a = folds[c1].loc[folds[c1]["is_test_like"].values == 1, ["user_id", "target"]]
        b_ = folds[c2].loc[folds[c2]["is_test_like"].values == 1, ["user_id", "target"]]
        m = a.merge(b_, on="user_id", suffixes=("_1", "_2"))
        if len(m) > 100:
            r = np.corrcoef(np.log1p(m["target_1"].values), np.log1p(m["target_2"].values))[0, 1]
            log(f"  {c1} vs {c2}: n={len(m):,}, corr={r:.3f}")
    return df


# --- 7. quick-модель --------------------------------------------------------
def feature_columns(f):
    cols = []
    for w in WINDOWS:
        cols += [f"{m}_{w}_log1p" for m in METRICS]
        cols += [f"active_days_{w}", f"days_gmv_{w}"]
    cols += [f"{m}_all_log1p" for m in METRICS]
    cols += ["active_days_all", "days_gmv_all"]
    cols += ["gmv_yoy_log1p", "active_days_yoy", "yoy_avail"]
    cols += ["recency", "recency_gmv", "recency_order", "tenure",
             "gmv_ratio_30_90", "gmv_ratio_30_365", "gmv_ratio_30_all",
             "gmvS_share", "gmvC_share", "gmv_per_ad_90", "gmv_per_ad_all",
             "gmv_per_dg_all", "searches_per_ad_90", "conv_cart_ord_90",
             "active_rate_90", "active_rate_all", "month", "dow"]
    return [c for c in cols if c in f.columns]


def run_model(folds, base_df):
    section("7. QUICK-МОДЕЛЬ: LightGBM на log1p(таргет) — оценка достижимого скора")
    try:
        import lightgbm as lgb
    except Exception:
        log("lightgbm не установлен (pip install lightgbm) — секция пропущена")
        return None, None
    fcols = feature_columns(folds[FINAL_CUT])
    log(f"Признаков: {len(fcols)}")
    log("Честная time-CV: обучение только на срезах СТРОГО раньше eval-среза;")
    log("inner-фолд для early stopping из обучения исключён.")
    log("Основная метрика — RMSLE на test-like подвыборке eval-фолда.\n")

    params = dict(objective="regression", metric="rmse", learning_rate=0.07,
                  num_leaves=63, min_child_samples=300, feature_fraction=0.7,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbosity=-1, n_jobs=-1, seed=42)

    def xy(f):
        return f[fcols].astype(np.float32).values, np.log1p(f["target"].values)

    results = {"all_rows": [], "test_like": []}
    yoy_diag = {"z": 1.0, "score": None, "pure": None}
    for ev in MODEL_EVAL_CUTS:
        train_cuts = [c for c in CUTS if c < ev]
        if not train_cuts:
            log(f"\ncutoff {ev}: нет более ранних срезов — обучение невозможно (только бейзлайны)")
            continue
        inner = max(train_cuts)
        fit_cuts = [c for c in train_cuts if c != inner]
        tr = pd.concat([folds[c] for c in fit_cuts], ignore_index=True)
        f_ev = folds[ev]
        tl_ev = f_ev["is_test_like"].values == 1
        Xev, yev = xy(f_ev)
        log(f"eval cutoff {ev}: fit-срезы {[str(c) for c in fit_cuts]}, inner-valid {inner}")
        for variant in ["all_rows", "test_like"]:
            t0 = time.time()
            tr_v = tr if variant == "all_rows" else tr.loc[tr["is_test_like"].values == 1]
            inner_f = folds[inner]
            if variant == "test_like":
                inner_f = inner_f.loc[inner_f["is_test_like"].values == 1]
            Xtr, ytr = xy(tr_v)
            Xin, yin = xy(inner_f)
            dtr = lgb.Dataset(Xtr, ytr, feature_name=fcols)
            din = lgb.Dataset(Xin, yin, reference=dtr, feature_name=fcols)
            model = lgb.train(params, dtr, num_boost_round=LGB_ROUNDS, valid_sets=[din],
                              callbacks=[lgb.early_stopping(LGB_ES, verbose=False)])
            bi = model.best_iteration or LGB_ROUNDS
            p_log = model.predict(Xev, num_iteration=bi)
            s_tl = rmse_log(yev[tl_ev], p_log[tl_ev])
            s_all = rmse_log(yev, p_log)
            bstr = ""
            try:
                bb = base_df[(base_df["cutoff"] == ev) & (~base_df["rmsle_tl"].isna())]
                bn = bb.loc[bb["rmsle_tl"].idxmin()]
                bstr = f"; лучший наивный(TL): {bn['baseline']} = {bn['rmsle_tl']:.4f}"
            except Exception:
                pass
            log(f"  [{variant:>9s}] RMSLE(TL)={s_tl:.4f} | RMSLE(all)={s_all:.4f} | iter={bi} | "
                f"{len(tr_v):,} строк | {time.time()-t0:.0f}с{bstr}")
            results[variant].append({"cutoff": ev, "rmsle_tl": s_tl, "rmsle_all": s_all, "best_iter": bi})

            if ev == date(2026, 1, 14):
                if variant == "all_rows":
                    # Диагностическая двухэтапная модель: P(gmv>0) × E[log1p|gmv>0].
                    ybin = (tr_v["target"].values > 0).astype(int)
                    clf = lgb.train(dict(params, objective="binary", metric="auc"),
                                    lgb.Dataset(Xtr, ybin, feature_name=fcols), num_boost_round=600)
                    pos = tr_v["target"].values > 0
                    reg = lgb.train(params, lgb.Dataset(Xtr[pos], ytr[pos], feature_name=fcols),
                                    num_boost_round=min(bi, 800))
                    mix_log = clf.predict(Xev) * np.clip(reg.predict(Xev), 0, None)
                    log(f"    two-stage (P(gmv>0) x E[log1p|gmv>0]): RMSLE(TL)="
                        f"{rmse_log(yev[tl_ev], mix_log[tl_ev]):.4f}")
                yoy_log = np.log1p(f_ev["gmv_yoy"].values.astype(np.float64))
                best = (1.0, rmse_log(yev[tl_ev], p_log[tl_ev]))
                for z in np.arange(0.0, 1.0001, 0.05):
                    sc = rmse_log(yev[tl_ev], (z * p_log + (1 - z) * yoy_log)[tl_ev])
                    if sc < best[1]:
                        best = (float(z), sc)
                pure = rmse_log(yev[tl_ev], p_log[tl_ev])
                log(f"    бленд модель+YoY: z={best[0]:.2f} -> RMSLE(TL)={best[1]:.4f} "
                    f"(чистая модель {pure:.4f}; прирост {100*(1-best[1]/pure):.1f}%)")
                if yoy_diag["score"] is None or best[1] < yoy_diag["score"]:
                    yoy_diag = {"z": best[0], "score": best[1], "pure": pure}
        log("")

    rdf = pd.DataFrame([{"variant": v, **r} for v, lst in results.items() for r in lst])
    if rdf.empty:
        return None, None
    rdf.to_csv(OUT_DIR / "model_scores.csv", index=False)
    summary = rdf.groupby("variant").agg(mean_tl=("rmsle_tl", "mean"),
                                         mean_all=("rmsle_all", "mean"),
                                         mean_iter=("best_iter", "mean")).reset_index()
    log("Итог по вариантам обучения:")
    log(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    rec_variant = summary.sort_values("mean_tl").iloc[0]["variant"]
    log(f"Рекомендованный вариант для финала: {rec_variant}")

    all_folds = pd.concat([folds[c] for c in CUTS], ignore_index=True)
    Xf, _ = xy(folds[FINAL_CUT])
    preds_final, imps = {}, {}
    for variant in ["all_rows", "test_like"]:
        iters = rdf[rdf["variant"] == variant]["best_iter"].mean()
        rounds_final = int(np.clip(iters if not pd.isna(iters) else 600, 200, LGB_ROUNDS))
        tr_v = all_folds if variant == "all_rows" else all_folds.loc[all_folds["is_test_like"].values == 1]
        Xa, ya = xy(tr_v)
        log(f"\nФинальная модель [{variant}]: {len(tr_v):,} строк, {rounds_final} итераций...")
        m = lgb.train(params, lgb.Dataset(Xa, ya, feature_name=fcols),
                      num_boost_round=rounds_final)
        p = np.clip(np.expm1(m.predict(Xf)), 0, None)
        preds_final[variant] = p
        imps[variant] = m.feature_importance("gain")
        log(f"  предсказания: нули(<1e-6) {(p<1e-6).mean():.3f}; медиана {np.median(p):,.1f}; "
            f"p95 {np.quantile(p,.95):,.1f}; max {p.max():,.0f}")

    fi = pd.DataFrame({"feature": fcols, "gain_all_rows": imps["all_rows"],
                       "gain_test_like": imps["test_like"]})
    fi["gain_rec"] = fi[f"gain_{rec_variant}"]
    fi = fi.sort_values("gain_rec", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    log("\nТоп-25 признаков (gain, рекомендованный вариант):")
    log(fi.head(25)[["feature", "gain_rec"]].to_string(index=False))
    return preds_final, {"variant": rec_variant, "yoy": yoy_diag}


# --- 8. сабмиты-якоря -------------------------------------------------------
def make_submissions(folds, sub_file, preds_final, model_info):
    section("8. САБМИТЫ-ЯКОРЯ ДЛЯ ЛИДЕРБОРДА")
    if sub_file is None:
        log("sample_submit.csv не найден — сабмиты не созданы")
        return
    sub = pd.read_csv(sub_file)
    base = sub[["user_id"]].astype({"user_id": "int64"})
    f = folds[FINAL_CUT]

    # Параметры бленда last30 + YoY подобраны на фолде 2026-01-14 (test-like).
    fv = folds[date(2026, 1, 14)]
    tl = fv["is_test_like"].values == 1
    yy = fv["target"].values[tl]
    l30 = np.log1p(fv["gmv_30"].values.astype(np.float64))[tl]
    yoy = np.log1p(fv["gmv_yoy"].values.astype(np.float64))[tl]
    grid = []
    for w in np.arange(0.0, 1.01, 0.1):
        bl = w * l30 + (1 - w) * yoy
        for a in np.arange(0.10, 1.51, 0.05):
            grid.append((rmsle(yy, np.expm1(a * bl)), round(float(w), 2), round(float(a), 2)))
    grid.sort()
    s_best, w_best, a_best = grid[0]
    s_l30 = min(g[0] for g in grid if g[1] == 1.0)
    s_yoy = min(g[0] for g in grid if g[1] == 0.0)
    log("Фолд 2026-01-14 (test-like), бленд w*last30 + (1-w)*yoy со сжатием a:")
    log(f"  лучший: w={w_best:.2f}, a={a_best:.2f} -> RMSLE={s_best:.4f}")
    log(f"  только last30 (opt a): {s_l30:.4f}; только yoy (opt a): {s_yoy:.4f}")

    blf = (w_best * np.log1p(f["gmv_30"].values.astype(np.float64))
           + (1 - w_best) * np.log1p(f["gmv_yoy"].values.astype(np.float64)))
    pred = np.clip(np.expm1(a_best * blf), 0, None)
    out = base.merge(pd.DataFrame({"user_id": f["user_id"].astype("int64"), "predict": pred}),
                     on="user_id", how="left")
    out["predict"] = out["predict"].fillna(0.0)
    out.to_csv(OUT_DIR / "sub_naive_blend.csv", index=False)
    log(f"sub_naive_blend.csv: {len(out):,} строк; нулей {(out['predict']==0).mean():.3f}")

    if preds_final:
        rec = (model_info or {}).get("variant")
        for variant, p in preds_final.items():
            name = "sub_lgbm_testlike.csv" if variant == "test_like" else "sub_lgbm_allrows.csv"
            o = base.merge(pd.DataFrame({"user_id": f["user_id"].astype("int64"),
                                         "predict": np.clip(p, 0, None)}), on="user_id", how="left")
            o["predict"] = o["predict"].fillna(0.0)
            o.to_csv(OUT_DIR / name, index=False)
            log(f"{name}: {len(o):,} строк; нулей {(o['predict']==0).mean():.3f}"
                + ("  <-- рекомендованный" if variant == rec else ""))
        yd = (model_info or {}).get("yoy") or {}
        if yd.get("z") is not None and yd["z"] < 0.999:
            z = yd["z"]
            p_log = np.log1p(np.clip(preds_final[rec], 0, None))
            yoy_log = np.log1p(f["gmv_yoy"].values.astype(np.float64))
            pred_b = np.expm1(z * p_log + (1 - z) * yoy_log)
            o = base.merge(pd.DataFrame({"user_id": f["user_id"].astype("int64"), "predict": pred_b}),
                           on="user_id", how="left")
            o["predict"] = o["predict"].fillna(0.0)
            o.to_csv(OUT_DIR / "sub_lgbm_yoyblend.csv", index=False)
            log(f"sub_lgbm_yoyblend.csv (z={z:.2f}, вариант {rec}): {len(o):,} строк; "
                f"нулей {(o['predict']==0).mean():.3f}")
        else:
            log("YoY-бленд поверх модели не дал прироста на CV — sub_lgbm_yoyblend.csv не создан")


# --- main -------------------------------------------------------------------
def main():
    log("=" * 94)
    log("EDA: аудит данных, гипотезы, валидация, бейзлайны, quick-модель")
    log(f"Старт: {time.strftime('%Y-%m-%d %H:%M:%S')} | polars={pl.__version__} "
        f"pandas={pd.__version__} numpy={np.__version__}")
    log("=" * 94)
    try:
        section("1-2. ЗАГРУЗКА И ОЧИСТКА ДАННЫХ")
        daily, num_cols = load_and_clean()
        save_report()
    except Exception:
        log(traceback.format_exc()); save_report(); return

    for fn, args in [(invariants, (daily, num_cols)), (seasonality, (daily,)),
                     (user_stats, (daily,))]:
        try:
            fn(*args); save_report()
        except Exception:
            log(traceback.format_exc()); save_report()

    folds = {}
    try:
        section("6a. СБОРКА ФОЛДОВ: ОКНА ПРИЗНАКОВ + ТАРГЕТ НА 30 ДНЕЙ ВПЕРЁД")
        for cut in CUTS + [FINAL_CUT]:
            t = time.time()
            pdf, yoy_avail = build_fold(daily, cut)
            folds[cut] = derive(pdf, cut, yoy_avail)
            log(f"  cutoff {cut}: {len(pdf):,} юзеров, "
                f"test-like {(folds[cut]['is_test_like']==1).mean():.1%}, "
                f"yoy_avail={yoy_avail}, {time.time()-t:.0f}с")
            if cut == FINAL_CUT:
                ff = folds[cut]
                log(f"  [тест-срез] gmv_30==0: {(ff['gmv_30']==0).mean():.3f} "
                    f"(нули sample_submit: 0.459); активны за 30д: {(ff['active_days_30']>0).mean():.3f}")
        del daily
        gc.collect()
        save_report()
    except Exception:
        log(traceback.format_exc()); save_report(); return

    base_df = None
    try:
        base_df = baselines(folds); save_report()
    except Exception:
        log(traceback.format_exc()); save_report()

    preds_final, model_info = None, None
    try:
        preds_final, model_info = run_model(folds, base_df); save_report()
    except Exception:
        log(traceback.format_exc()); save_report()

    try:
        make_submissions(folds, find_file(SUB_CANDIDATES), preds_final, model_info)
        save_report()
    except Exception:
        log(traceback.format_exc()); save_report()

    log("\nГОТОВО.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        save_report()
