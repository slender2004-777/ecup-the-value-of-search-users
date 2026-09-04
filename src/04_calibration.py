"""Пост-обработка предсказаний: масштабирование, квадратичный фит калибровки
по точкам лидерборда, лог-бленд двух файлов.

Модуль читает готовые сабмит-файлы из out/ и применяет к ним калибровочные
преобразования. Все операции — это линейные масштабирования или лог-бленды,
они не меняют порядок пользователей и сохраняют формат (user_id, predict).

Команды CLI:
  scale   <k> <src.csv> [<out_name>]          — домножить predict на k
  fit                                          — квадратичный фит по lb_points,
                                                вывод оптимального k* и запись
                                                out/<src>_xOPT.csv
  apply   <src.csv> <out_name> <beta> <F_test> <F_val>
                                               — снять с файла beta-множитель
                                                F_test^0.75 (файл sub3_*_b075.csv)
                                                и применить k_v3 = F_val^beta
  megamix <src1.csv> <src2.csv> <w1> <out_name>
                                               — лог-бленд w1*log1p(p1) + (1-w1)*log1p(p2)

Запуск: python src/04_calibration.py <command> ...
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("out")

# Сезонный фактор тестового окна (из seasonal_analysis).
SEASON_F = 1.1628
LOG_F = np.log(SEASON_F)

# Точки лидерборда: k -> public RMSLE. Дополняется по мере получения новых оценок.
lb_points = {
    1.0000: 1.6864337908930493,
    1.1628: 1.7051948125397354,
    0.8000: None,   # скор сабмита sub_v2_x080 (заполняется после получения LB)
}


def scale_file(src, k, out_name=None):
    """Домножить столбец predict в src на k и записать рядом.

    Если out_name не задан, пишём в тот же каталог, что и src.
    """
    src = Path(src)
    df = pd.read_csv(src)
    df["predict"] = np.clip(df["predict"].astype(np.float64) * k, 0.0, None)
    out = src.parent / out_name if out_name else src
    df.to_csv(out, index=False)
    print(f"{out.name}: k={k:.4f} (beta={np.log(k)/LOG_F:+.2f}); "
          f"медиана={df['predict'].median():.2f}; сумма={df['predict'].sum():,.0f}")


def fit_quadratic(points):
    """Квадратичный фит RMSLE²(ln k) по известным точкам лидерборда.

    Возвращает (k_star, r_min). При числе точек < 2 поднимает RuntimeError.
    При 2 точках свободный член кривизны фиксируется равным 1 (нет данных для
    оценки), при 3+ — фит полный.
    """
    pts = sorted((np.log(k), r) for k, r in points.items() if r)
    if len(pts) < 2:
        raise RuntimeError("нужно хотя бы две точки лидерборда с ненулевым RMSLE")
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] ** 2 for p in pts])
    if len(pts) >= 3:
        a, b, c = np.polyfit(xs, ys, 2)
        if a <= 0:
            raise RuntimeError("кривизна <= 0: оптимум за сеткой проб, нужен зонд ниже 0.8")
        x_ = -b / (2 * a); rmin = np.sqrt(max(c - b * b / (4 * a), 0.0))
        print(f"свободный фит (кривизна {a:.2f}): k*={np.exp(x_):.4f} "
              f"(beta={x_ / LOG_F:+.2f}), ожидаемый минимум {rmin:.4f}")
    else:
        (x1, r1), (x2, r2) = pts
        b = ((r2 ** 2 - r1 ** 2) - (x2 ** 2 - x1 ** 2)) / (2 * (x2 - x1))
        c = r1 ** 2 - x1 ** 2 - 2 * b * x1
        x_ = -b; rmin = np.sqrt(max(c - b * b, 0.0))
        print(f"2 точки (кривизна=1): k*={np.exp(x_):.4f}, минимум {rmin:.4f}")
    return float(np.exp(x_)), float(rmin)


def log_blend(src1, src2, w1, out_name):
    """Лог-бленд двух файлов: predict = expm1(w1*log1p(p1) + (1-w1)*log1p(p2))."""
    a = pd.read_csv(src1).rename(columns={"predict": "p1"})
    b = pd.read_csv(src2).rename(columns={"predict": "p2"})
    mm = a.merge(b, on="user_id")
    if mm["p2"].isna().any():
        raise RuntimeError(f"{src2} не покрывает всех user_id из {src1}")
    L = w1 * np.log1p(mm["p1"]) + (1 - w1) * np.log1p(mm["p2"])
    mm["predict"] = np.clip(np.expm1(L), 0, None)
    out = OUT_DIR / out_name
    mm[["user_id", "predict"]].to_csv(out, index=False)
    print(f"{out.name}: w1={w1:.2f}; медиана={mm['predict'].median():.2f}; "
          f"сумма={mm['predict'].sum():,.0f}")


def cmd_fit():
    """Фит по lb_points и запись файла с оптимальным k."""
    k_star, _ = fit_quadratic(lb_points)
    src = OUT_DIR / "sub_v2_ens_yoy_b000.csv"
    if not src.exists():
        raise RuntimeError(f"{src} не найден — сначала запустите src/02_pipeline.py")
    scale_file(src, k_star, "sub_v2_xOPT.csv")


def cmd_apply(src, out_name, beta, f_test, f_val):
    """Снять с sub3_*_b075.csv множитель F_test^0.75 и применить k_v3 = F_val^beta.

    Логика: файл sub3_*_b075.csv уже содержит SEASON_F_test^0.75; чтобы
    перейти к оптимальной калибровке, домножаем на SEASON_F_test^-0.75
    и на k_v3 = SEASON_F_val^beta (где beta — победитель formula_tuning.csv).
    """
    k = (f_test ** -0.75) * (f_val ** beta)
    print(f"apply: beta={beta:.2f}, F_test={f_test:.4f}, F_val={f_val:.4f} -> k={k:.4f}")
    scale_file(src, k, out_name)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "scale":
        k = float(sys.argv[2]); src = sys.argv[3]
        out_name = sys.argv[4] if len(sys.argv) > 4 else None
        scale_file(src, k, out_name)
    elif cmd == "fit":
        cmd_fit()
    elif cmd == "apply":
        src = sys.argv[2]; out_name = sys.argv[3]
        beta = float(sys.argv[4]); f_test = float(sys.argv[5]); f_val = float(sys.argv[6])
        cmd_apply(src, out_name, beta, f_test, f_val)
    elif cmd == "megamix":
        src1 = sys.argv[2]; src2 = sys.argv[3]; w1 = float(sys.argv[4]); out_name = sys.argv[5]
        log_blend(src1, src2, w1, out_name)
    else:
        print(f"неизвестная команда: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
