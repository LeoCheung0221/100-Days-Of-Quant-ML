"""Days 11–83. Each day_xx prints the numbers its note quotes."""

from __future__ import annotations

import datetime as dt

import numpy as np

from days.course import (
    FIVE_V,
    FIVE_X,
    FIVE_Y,
    PANEL_PATH,
    accuracy,
    column,
    fmt,
    load_panel,
    name_rows,
    ols,
    predict_stump,
    predict_tree,
    returns,
    ridge_slope,
    sign_hit,
    stump,
    tree_depth2,
)


def _five_returns() -> np.ndarray:
    return (FIVE_Y[1:] - FIVE_Y[:-1]) / FIVE_Y[:-1]

def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))

def _aaa() -> list[dict]:
    return name_rows("AAA")

def _complete(rows: list[dict]) -> list[dict]:
    return [row for row in rows if np.isfinite(row["close"]) and np.isfinite(row["adj_close"])]

def _panel_pair() -> tuple[list[dict], list[dict]]:
    aaa = _complete(_aaa())
    bbb = _complete(name_rows("BBB"))
    shared = set(row["date"] for row in aaa) & set(row["date"] for row in bbb)
    aaa = [row for row in aaa if row["date"] in shared]
    bbb = [row for row in bbb if row["date"] in shared]
    return aaa, bbb

def day_11() -> None:
    slope, intercept = ols(FIVE_X, FIVE_Y)
    fitted = slope * FIVE_X + intercept
    hits = sign_hit(np.diff(FIVE_Y), np.diff(fitted))
    print("score = direction hits", f"{int(hits.sum())}/{len(hits)}")
    print("day 4 absolute price residual =", fmt(abs(FIVE_Y[3] - fitted[3]), 2))
    print("that residual is not the score")

def day_12() -> None:
    simple = _five_returns()
    tau = 0.70
    label = simple > tau
    pred = np.ones(len(simple), dtype=bool)
    print("threshold =", fmt(tau, 2))
    print("returns =", " ".join(fmt(v, 4) for v in simple))
    print("label up =", " ".join("1" if v else "0" for v in label))
    print("constant-up accuracy =", fmt(accuracy(pred == label), 2))

def day_13() -> None:
    simple = _five_returns()
    pred = np.ones(len(simple), dtype=bool)
    print("threshold  accuracy")
    for tau in (0.50, 0.70, 0.90):
        print(f"{tau:.2f}  {accuracy(pred == (simple > tau)):.2f}")

def day_14() -> None:
    up = _five_returns() > 0
    print("always-down accuracy =", fmt(accuracy(np.zeros(len(up), dtype=bool) == up), 2))
    print("always-up accuracy =", fmt(accuracy(np.ones(len(up), dtype=bool) == up), 2))
    print("the baseline looks at no price")

def day_15() -> None:
    up = _five_returns() > 0
    pred_up = np.ones(len(up), dtype=bool)
    print("up called down =", int((up & ~pred_up).sum()))
    print("down called up =", int((~up & pred_up).sum()))

def day_16() -> None:
    slope = float(ols(FIVE_X, FIVE_Y)[0])
    print("slope =", fmt(slope, 2))
    print("P(up) = sigmoid(slope) =", fmt(_sigmoid(slope), 4))
    print("the same score is attached to every step")

def day_17() -> None:
    stated = _sigmoid(float(ols(FIVE_X, FIVE_Y)[0]))
    realized = float(np.mean(_five_returns() > 0))
    print("stated P(up) =", fmt(stated, 4))
    print("realized up frequency =", fmt(realized, 2))
    print("gap =", fmt(stated - realized, 4))

def day_18() -> None:
    simple = _five_returns()
    volume = FIVE_V[1:]
    median = float(np.median(FIVE_V))
    both = (simple > 0) & (volume > median)
    print("median volume =", f"{median:.0f}")
    print("up on return alone =", int((simple > 0).sum()))
    print("up on return and volume =", int(both.sum()))
    kept = " ".join(str(i + 2) for i, flag in enumerate(both) if flag)
    print("sessions kept =", kept or "none")

def day_19() -> None:
    design = np.column_stack([FIVE_X, FIVE_V])
    beta, *_ = np.linalg.lstsq(design, FIVE_Y, rcond=None)
    raw_share = np.abs(design * beta)
    scale = design.std(axis=0)
    design_z = (design - design.mean(axis=0)) / scale
    beta_z, *_ = np.linalg.lstsq(design_z, FIVE_Y, rcond=None)
    print("raw beta time =", f"{beta[0]:.6f}")
    print("raw beta volume =", f"{beta[1]:.6e}")
    print("mean |time contribution| =", f"{raw_share[:, 0].mean():.4f}")
    print("mean |volume contribution| =", f"{raw_share[:, 1].mean():.4f}")
    print("standardized beta time =", f"{beta_z[0]:.4f}")
    print("standardized beta volume =", f"{beta_z[1]:.4f}")

def day_20() -> None:
    noise = np.random.default_rng(0).normal(size=5)
    x = FIVE_X[:-1]
    y = FIVE_Y[:-1]
    base = ols(x, y)
    rss_base = float(np.sum((y - (base[0] * x + base[1])) ** 2))
    design = np.column_stack([x, noise[:-1], np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    rss_noise = float(np.sum((y - design @ beta) ** 2))
    exam_x, exam_y = float(FIVE_X[-1]), float(FIVE_Y[-1])
    base_err = abs(exam_y - (base[0] * exam_x + base[1]))
    noise_err = abs(exam_y - (beta[0] * exam_x + beta[1] * noise[-1] + beta[2]))
    print("fit on t=1..4, score on t=5")
    print("in-sample RSS without noise =", fmt(rss_base, 4))
    print("in-sample RSS with noise =", fmt(rss_noise, 4))
    print("holdout absolute error without noise =", fmt(base_err, 4))
    print("holdout absolute error with noise =", fmt(noise_err, 4))
    print("the in-sample drop is not an improvement")

def day_21() -> None:
    text = PANEL_PATH.read_text()
    aaa = _aaa()
    blank = sum(1 for row in aaa if not np.isfinite(row["close"]))
    print("path = days/data/panel.csv")
    print("rows =", len(load_panel()))
    print("second read matches =", str(text == PANEL_PATH.read_text()).lower())
    print("AAA dates", aaa[0]["date"], "..", aaa[-1]["date"])
    print("AAA blank closes =", blank)
    print("the table is not resampled")

def day_22() -> None:
    close = column(_complete(_aaa()), "adj_close")
    move = np.diff(close)
    hits = sign_hit(move[1:], move[:-1])
    print("predict today's adj move with yesterday's sign")
    print("hits =", f"{int(hits.sum())}/{len(hits)}")
    print("accuracy =", fmt(accuracy(hits), 4))
    print("coin-flip baseline = 0.5000")

def day_23() -> None:
    move = np.diff(column(_complete(_aaa()), "adj_close"))
    hits = []
    for i in range(3, len(move)):
        window = np.sign(move[i - 3 : i])
        if np.all(window == window[0]) and window[0] != 0:
            hits.append(bool(np.sign(move[i]) == window[0]))
    hits_arr = np.array(hits, dtype=bool)
    print("rule = after three equal signs, predict the fourth matches")
    print("events =", len(hits_arr))
    print("hits =", int(hits_arr.sum()))
    print("accuracy =", fmt(accuracy(hits_arr), 4))
    print("the rule is fixed before the count")

def day_24() -> None:
    rows = _complete(_aaa())
    close = column(rows, "adj_close")
    dates = [row["date"] for row in rows]
    move = np.diff(close)
    cut = dates[len(dates) // 2]
    later, early = [], []
    for i in range(3, len(move)):
        window = np.sign(move[i - 3 : i])
        if not (np.all(window == window[0]) and window[0] != 0):
            continue
        hit = bool(np.sign(move[i]) == window[0])
        (later if dates[i + 1] > cut else early).append(hit)
    early_arr = np.array(early, dtype=bool)
    later_arr = np.array(later, dtype=bool)
    print("cut date =", cut)
    print("early events =", len(early_arr), "accuracy =", fmt(accuracy(early_arr), 4))
    print("early accuracy is not the score")
    print("later events =", len(later_arr), "accuracy =", fmt(accuracy(later_arr), 4))

def day_25() -> None:
    close = column(_complete(_aaa()), "adj_close")
    actual = returns(close)[1:]
    pred = returns(close)[:-1]
    err = np.abs(actual - pred)
    hits = sign_hit(actual, pred)
    small = err <= np.median(err)
    print("lag-1 return forecast")
    print("direction accuracy =", fmt(accuracy(hits), 4))
    print("mean absolute return error =", fmt(float(err.mean()), 4))
    print("days with a small price error and the wrong sign =", int((small & ~hits).sum()))
    print("for a sign decision, trust the direction accuracy")

def _split_score(x: np.ndarray, y: np.ndarray, train: np.ndarray) -> float:
    beta = ols(x[train], y[train])
    pred = beta[0] * x[~train] + beta[1]
    return accuracy(sign_hit(y[~train], pred))

def _lag_xy() -> tuple[np.ndarray, np.ndarray]:
    simple = returns(column(_complete(_aaa()), "adj_close"))
    return simple[:-1], simple[1:]

def day_26() -> None:
    x, y = _lag_xy()
    train = np.zeros(len(y), dtype=bool)
    train[np.random.default_rng(1).choice(len(y), size=int(0.7 * len(y)), replace=False)] = True
    print("split = random, seed 1, train fraction 0.70")
    print("test direction accuracy =", fmt(_split_score(x, y, train), 4))
    print("this number is the control")

def day_27() -> None:
    x, y = _lag_xy()
    cut = int(0.7 * len(y))
    train = np.zeros(len(y), dtype=bool)
    train[:cut] = True
    random_train = np.zeros(len(y), dtype=bool)
    random_train[np.random.default_rng(1).choice(len(y), size=cut, replace=False)] = True
    print("time-split test accuracy =", fmt(_split_score(x, y, train), 4))
    print("random-split test accuracy =", fmt(_split_score(x, y, random_train), 4))
    print("the test of the time split sits entirely after the train")

def day_28() -> None:
    rows = _complete(_aaa())
    close = column(rows, "close")
    high = column(rows, "high")
    beta_high = ols(high[1:], close[1:])
    beta_lag = ols(close[:-1], close[1:])
    resid_high = close[1:] - (beta_high[0] * high[1:] + beta_high[1])
    resid_lag = close[1:] - (beta_lag[0] * close[:-1] + beta_lag[1])
    print("column high = FORBIDDEN")
    print("in-sample RSS close~high =", fmt(float(np.sum(resid_high ** 2)), 4))
    print("in-sample RSS close~lagged close =", fmt(float(np.sum(resid_lag ** 2)), 4))

def day_29() -> None:
    rows = _complete(_aaa())
    close = column(rows, "close")
    opened = column(rows, "open")
    leaky = (close - opened.mean()) / opened.std()
    causal = np.full(len(close), np.nan)
    for i in range(5, len(close)):
        window = close[:i]
        causal[i] = (close[i] - window.mean()) / window.std()
    target = returns(close)
    mask = np.isfinite(causal[:-1])

    def rss(feature: np.ndarray) -> float:
        beta = ols(feature[:-1][mask], target[mask])
        resid = target[mask] - (beta[0] * feature[:-1][mask] + beta[1])
        return float(np.sum(resid ** 2))

    print("leaky scale uses every open, including later ones")
    print("RSS of return on leaky close =", fmt(rss(leaky), 4))
    print("RSS of return on past-only close =", fmt(rss(causal), 4))
    print("the leaky scale is a function of later opens")

def day_30() -> None:
    close = column(_complete(_aaa()), "adj_close")
    t = np.arange(len(close), dtype=float)
    full = ols(t, close)
    local = ols(t[-20:], close[-20:])
    query = t[-1]
    print("lookback = 20")
    print("full-sample value at last t =", fmt(float(full[0] * query + full[1]), 4))
    print("window value at last t =", fmt(float(local[0] * query + local[1]), 4))
    print("the full sample is not the information set")

def day_31() -> None:
    close = column(_complete(_aaa()), "adj_close")
    t = np.arange(len(close), dtype=float)
    end = len(close) - 1
    short = ols(t[end - 3 : end], close[end - 3 : end])
    long = ols(t[end - 20 : end], close[end - 20 : end])
    actual = float(close[end])
    print("three-day slope =", fmt(float(short[0]), 4))
    print("twenty-day slope =", fmt(float(long[0]), 4))
    print("three-day absolute miss =", fmt(abs(actual - (short[0] * t[end] + short[1])), 4))
    print("twenty-day absolute miss =", fmt(abs(actual - (long[0] * t[end] + long[1])), 4))

def day_32() -> None:
    close = column(_complete(_aaa()), "adj_close")
    t = np.arange(len(close), dtype=float)
    end = len(close) - 1
    short = ols(t[end - 3 : end], close[end - 3 : end])
    long = ols(t[end - 60 : end], close[end - 60 : end])
    actual = float(close[end])
    print("sixty-day slope =", fmt(float(long[0]), 4))
    print("three-day slope =", fmt(float(short[0]), 4))
    print("sixty-day absolute miss =", fmt(abs(actual - (long[0] * t[end] + long[1])), 4))
    print("three-day absolute miss =", fmt(abs(actual - (short[0] * t[end] + short[1])), 4))

def day_33() -> None:
    simple = returns(column(_complete(_aaa()), "adj_close"))
    x, y = simple[:-1], simple[1:]
    cut = int(0.7 * len(y))
    train_x, test_x, train_y, test_y = x[:cut], x[cut:], y[:cut], y[cut:]

    def mse(mu_used: float, sigma_used: float) -> float:
        beta = ols((train_x - mu_used) / sigma_used, train_y)
        pred = beta[0] * ((test_x - mu_used) / sigma_used) + beta[1]
        return float(np.mean((test_y - pred) ** 2))

    train_mu, train_sd = float(train_x.mean()), float(train_x.std())
    all_mu, all_sd = float(x.mean()), float(x.std())
    print("train mean/std =", fmt(train_mu, 6), fmt(train_sd, 6))
    print("whole-sample mean/std =", fmt(all_mu, 6), fmt(all_sd, 6))
    print("test MSE, scale from the training stretch =", fmt(mse(train_mu, train_sd), 6))
    print("test MSE, scale from the whole sample =", fmt(mse(all_mu, all_sd), 6))
    print("the whole-sample scale sees the test stretch")

def day_34() -> None:
    rows = _aaa()
    blank = next(i for i, row in enumerate(rows) if not np.isfinite(row["close"]))
    print("blank date =", rows[blank]["date"])
    print("fill from the previous close =", fmt(float(rows[blank - 1]["close"]), 4))
    print("fill from the next close =", fmt(float(rows[blank + 1]["close"]), 4))
    print("the next close sees the future")

def day_35() -> None:
    rows = _aaa()
    dates = [dt.date.fromisoformat(row["date"]) for row in rows]
    for i in range(len(dates) - 1):
        business = int(np.busday_count(dates[i], dates[i + 1]))
        if business > 1:
            print("row numbers", i, i + 1)
            print("dates", dates[i].isoformat(), dates[i + 1].isoformat())
            print("business-day gap =", business)
            print("adjacent rows are not adjacent sessions")
            return
    raise RuntimeError("no halt gap")

def day_36() -> None:
    rows = _complete(_aaa())
    raw_ret = returns(column(rows, "close"))
    adj_ret = returns(column(rows, "adj_close"))
    i = int(np.argmax(np.abs(raw_ret - adj_ret)))
    print("date =", rows[i + 1]["date"])
    print("unadjusted return =", fmt(float(raw_ret[i]), 4))
    print("adjusted return =", fmt(float(adj_ret[i]), 4))
    print("both numbers are due")

def day_37() -> None:
    aaa, bbb = _panel_pair()

    def xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        feature = returns(column(rows, "market"))
        target = returns(column(rows, "adj_close"))
        return feature, target, [row["date"] for row in rows[1:]]

    xa, ya, da = xy(aaa)
    xb, yb, db = xy(bbb)
    x = np.r_[xa, xb]
    y = np.r_[ya, yb]
    dates = np.array(da + db)
    train = np.zeros(len(y), dtype=bool)
    train[np.random.default_rng(1).choice(len(y), size=int(0.7 * len(y)), replace=False)] = True
    order = np.argsort(dates, kind="mergesort")
    time_train = np.zeros(len(y), dtype=bool)
    time_train[order[: int(0.7 * len(y))]] = True
    print("pooled AAA and BBB")
    print("random-split test accuracy =", fmt(_split_score(x, y, train), 4))
    print("time-split test accuracy =", fmt(_split_score(x, y, time_train), 4))
    print("the random split can train and test on the same date")

def day_38() -> None:
    rows = _complete(_aaa())
    market = returns(column(rows, "market"))
    target = returns(column(rows, "adj_close"))
    print("same-day market sign accuracy =", fmt(accuracy(sign_hit(target, market)), 4))
    print("lagged-one-day market sign accuracy =", fmt(accuracy(sign_hit(target[1:], market[:-1])), 4))
    print("what disappeared was simultaneous")

def day_39() -> None:
    rows = _complete(_aaa())
    market = returns(column(rows, "market"))
    target = returns(column(rows, "adj_close"))
    position = np.sign(market[:-1])
    position[position == 0] = 1.0
    gross = position * target[1:]
    cost = 0.002
    print("gross mean return =", fmt(float(gross.mean()), 4))
    print("round-trip cost =", fmt(cost, 4))
    print("net mean return =", fmt(float(gross.mean() - cost), 4))

def day_40() -> None:
    print("leakage list")
    print("day 28  today's high explains today's close  future=yes")
    print("day 29  scale uses later opens  future=yes")
    print("day 33  scale uses the test stretch  future=yes")
    print("day 34  fill from the next close  future=yes")
    print("day 37  random split shares a date across names  future=yes")
    print("day 38  same-day market return  future=yes")
    print("a higher score on any of these lines is not a result")

def _level() -> tuple[np.ndarray, np.ndarray, list[str], int]:
    rows = _complete(_aaa())
    close = column(rows, "adj_close")
    t = np.arange(len(close), dtype=float)
    return t, close, [row["date"] for row in rows], int(0.75 * len(close))

def _train_test():
    t, close, _, cut = _level()
    return t[:cut], close[:cut], t[cut:], close[cut:]

def day_41() -> None:
    t, close, dates, _ = _level()
    jump = int(np.argmax(np.abs(returns(close)))) + 1
    beta = ols(t, close)
    keep = np.ones(len(close), dtype=bool)
    keep[jump] = False
    reduced = ols(t[keep], close[keep])
    print("jump date =", dates[jump])
    print("adjusted return that day =", fmt(float(returns(close)[jump - 1]), 4))
    print("slope with the jump =", fmt(float(beta[0]), 6))
    print("slope without the jump =", fmt(float(reduced[0]), 6))
    print("intercept with the jump =", fmt(float(beta[1]), 4))
    print("intercept without the jump =", fmt(float(reduced[1]), 4))
    print("fitted at the jump, with =", fmt(float(beta[0] * t[jump] + beta[1]), 4))
    print("fitted at the jump, without =", fmt(float(reduced[0] * t[jump] + reduced[1]), 4))

def day_42() -> None:
    t, close, _, _ = _level()
    print("lambda = 20000, penalty on the slope only")
    print("ols slope =", fmt(float(ols(t, close)[0]), 4))
    print("ridge slope =", fmt(float(ridge_slope(t, close, lam=20000.0)[0]), 4))

def day_43() -> None:
    t, close, dates, _ = _level()
    jump = int(np.argmax(np.abs(returns(close)))) + 1
    query = min(jump + 15, len(close) - 1)
    neighbors = np.argsort(np.abs(t - t[query]), kind="mergesort")[:5]
    beta = ols(t, close)
    print("query date =", dates[query])
    print("neighbor dates =", " ".join(dates[i] for i in neighbors))
    print("jump date in the neighbors =", str(jump in set(neighbors.tolist())).lower())
    print("local mean =", fmt(float(close[neighbors].mean()), 4))
    print("ols at the query =", fmt(float(beta[0] * t[query] + beta[1]), 4))

def day_44() -> None:
    t, y, _, _ = _train_test()
    threshold, left, right = stump(t, y)
    line = ols(t, y)
    print("train sessions =", len(t))
    print("split when t >", fmt(threshold, 2))
    print("left mean =", fmt(left, 4), "right mean =", fmt(right, 4))
    print("train SSE stump =", fmt(float(np.sum((y - predict_stump(t, threshold, left, right)) ** 2)), 4))
    print("train SSE line =", fmt(float(np.sum((y - (line[0] * t + line[1])) ** 2)), 4))

def day_45() -> None:
    train_t, train_y, test_t, test_y = _train_test()
    shallow = stump(train_t, train_y)
    deep = tree_depth2(train_t, train_y)
    pairs = {
        "depth1": (predict_stump(train_t, *shallow), predict_stump(test_t, *shallow)),
        "depth2": (predict_tree(train_t, deep), predict_tree(test_t, deep)),
    }
    for name, (train_hat, test_hat) in pairs.items():
        print(f"train SSE {name} =", fmt(float(np.sum((train_y - train_hat) ** 2)), 4))
        print(f"later SSE {name} =", fmt(float(np.sum((test_y - test_hat) ** 2)), 4))

def day_46() -> None:
    train_t, train_y, test_t, test_y = _train_test()
    line = ols(train_t, train_y)
    ridge = ridge_slope(train_t, train_y, lam=20000.0)
    tree = tree_depth2(train_t, train_y)
    preds = {
        "line": (line[0] * train_t + line[1], line[0] * test_t + line[1]),
        "ridge": (ridge[0] * train_t + ridge[1], ridge[0] * test_t + ridge[1]),
        "tree": (predict_tree(train_t, tree), predict_tree(test_t, tree)),
    }
    later = {}
    print("model  train_SSE  later_SSE")
    for name, (train_hat, test_hat) in preds.items():
        train_sse = float(np.sum((train_y - train_hat) ** 2))
        test_sse = float(np.sum((test_y - test_hat) ** 2))
        later[name] = test_sse
        print(f"{name}  {train_sse:.4f}  {test_sse:.4f}")
    print("still alive on the later stretch =", min(later, key=later.get))

def day_47() -> None:
    t, close, _, _ = _level()
    beta = ols(t, close)
    query = float(t[-1] + 40)
    value = float(beta[0] * query + beta[1])
    print("query t =", fmt(query, 0))
    print("line value =", fmt(value, 4))
    print("observed adj close min =", fmt(float(close.min()), 4))
    print("observed adj close max =", fmt(float(close.max()), 4))
    print("outside the observed range =", str(value < close.min() or value > close.max()).lower())

def day_48() -> None:
    t, close, _, _ = _level()
    query = float(t[-1] + 40)
    tree_value = float(predict_tree(np.array([query]), tree_depth2(t, close))[0])
    beta = ols(t, close)
    print("query t =", fmt(query, 0))
    print("tree value =", fmt(tree_value, 4))
    print("line value =", fmt(float(beta[0] * query + beta[1]), 4))
    print(
        "tree stays inside the training range =",
        str(close.min() - 1e-9 <= tree_value <= close.max() + 1e-9).lower(),
    )

def day_49() -> None:
    t, close, dates, _ = _level()
    jump = int(np.argmax(np.abs(returns(close)))) + 1
    line = ols(t, close)
    ridge = ridge_slope(t, close, lam=20000.0)
    tree = tree_depth2(t, close)
    actual = float(close[jump])
    later = min(jump + 10, len(close) - 1)
    hats = {
        "line": (float(line[0] * t[jump] + line[1]), float(line[0] * t[later] + line[1])),
        "ridge": (float(ridge[0] * t[jump] + ridge[1]), float(ridge[0] * t[later] + ridge[1])),
        "tree": (
            float(predict_tree(np.array([t[jump]]), tree)[0]),
            float(predict_tree(np.array([t[later]]), tree)[0]),
        ),
    }
    print("jump date =", dates[jump])
    print("adj close =", fmt(actual, 4))
    print("line slope =", fmt(float(line[0]), 4), "ridge slope =", fmt(float(ridge[0]), 4))
    for name, (hat, hat_later) in hats.items():
        print(f"{name} at jump = {hat:.4f}  residual = {actual - hat:.4f}  ten later = {hat_later:.4f}")

def _lag5_xy(*, volume: bool = False) -> tuple[np.ndarray, np.ndarray, int]:
    rows = _complete(_aaa())
    close = column(rows, "adj_close")
    vol = column(rows, "volume")
    simple = returns(close)
    xs, ys = [], []
    for t in range(5, len(simple)):
        xs.append([float(simple[t - k]) for k in range(1, 6)] + ([float(vol[t + 1])] if volume else []))
        ys.append(float(simple[t]))
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    return x_arr, y_arr, int(0.75 * len(y_arr))

def _ols_design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([x, np.ones(len(x))])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return beta

def _predict_design(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    design = np.column_stack([x, np.ones(len(x))])
    return design @ beta

def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((y - yhat) ** 2))

def _ridge_design(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    design = np.column_stack([x, np.ones(len(x))])
    gram = design.T @ design
    for i in range(x.shape[1]):
        gram[i, i] += lam
    return np.linalg.solve(gram, design.T @ y)

def _best_stump(x: np.ndarray, y: np.ndarray) -> tuple[int, float, float, float]:
    best: tuple[float, int, float, float, float] | None = None
    for col in range(x.shape[1]):
        thr, left, right = stump(x[:, col], y)
        hat = np.where(x[:, col] <= thr, left, right)
        sse = float(np.sum((y - hat) ** 2))
        if best is None or sse < best[0]:
            best = (sse, col, thr, left, right)
    assert best is not None
    return best[1], best[2], best[3], best[4]

def _predict_stump_col(x: np.ndarray, col: int, thr: float, left: float, right: float) -> np.ndarray:
    return np.where(x[:, col] <= thr, left, right)

def day_51() -> None:
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    weights = beta[:5]
    print("lags = 1 through 5")
    for lag in range(5):
        print(f"weight lag {lag + 1} =", fmt(float(weights[lag]), 4))
    print("intercept =", fmt(float(beta[5]), 4))
    print("weight min =", fmt(float(weights.min()), 4), "weight max =", fmt(float(weights.max()), 4))
    print("test MSE =", fmt(_mse(y[cut:], _predict_design(beta, x[cut:])), 6))

def day_52() -> None:
    x, y, cut = _lag5_xy()
    col, thr, left, right = _best_stump(x[:cut], y[:cut])
    print("split column = lag", col + 1)
    print("threshold =", fmt(thr, 6))
    print("left mean =", fmt(left, 4), "right mean =", fmt(right, 4))
    train_hat = _predict_stump_col(x[:cut], col, thr, left, right)
    test_hat = _predict_stump_col(x[cut:], col, thr, left, right)
    print("train MSE =", fmt(_mse(y[:cut], train_hat), 6))
    print("test MSE =", fmt(_mse(y[cut:], test_hat), 6))

def day_53() -> None:
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    print("linear weight lag 1 =", fmt(float(beta[0]), 4))
    for seed in (0, 1):
        idx = np.random.default_rng(seed).choice(cut, size=int(0.9 * cut), replace=False)
        col, thr, _, _ = _best_stump(x[idx], y[idx])
        print(f"seed = {seed} tree split lag = {col + 1} threshold = {fmt(thr, 6)}")
    print("linear weight lag 1 after tree seeds =", fmt(float(beta[0]), 4))

def day_54() -> None:
    x5, y, cut = _lag5_xy(volume=False)
    x6, _, _ = _lag5_xy(volume=True)
    beta5 = _ols_design(x5[:cut], y[:cut])
    beta6 = _ols_design(x6[:cut], y[:cut])
    mse5 = _mse(y[cut:], _predict_design(beta5, x5[cut:]))
    mse6 = _mse(y[cut:], _predict_design(beta6, x6[cut:]))
    print("test MSE five lags only =", fmt(mse5, 6))
    print("test MSE five lags and volume =", fmt(mse6, 6))
    rise = mse5 - mse6
    print("MSE rise when volume removed =", fmt(rise, 6))
    if rise > 0:
        print("volume helped on the test stretch = true")
    else:
        print("volume helped on the test stretch = false")

def day_55() -> None:
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    ridge = _ridge_design(x[:cut], y[:cut], lam=20000.0)
    col, thr, left, right = _best_stump(x[:cut], y[:cut])
    test_y = y[cut:]
    line_hat = _predict_design(beta, x[cut:])
    ridge_hat = _predict_design(ridge, x[cut:])
    tree_hat = _predict_stump_col(x[cut:], col, thr, left, right)
    line_mse = _mse(test_y, line_hat)
    ridge_mse = _mse(test_y, ridge_hat)
    tree_mse = _mse(test_y, tree_hat)
    print("test MSE line =", fmt(line_mse, 6))
    print("test MSE ridge =", fmt(ridge_mse, 6))
    print("test MSE tree =", fmt(tree_mse, 6))
    print("line miss mode = smooth blend of lags misses sharp jumps")
    print("ridge miss mode = same blend pulled toward zero misses jumps and size")
    print("tree miss mode = one lag threshold leaves a constant on each side")

def day_56() -> None:
    print("task = predict today's return from lags 1 through 5 on AAA adj_close")
    print("target = same-day simple return on adjusted close")
    print("forbidden = same-row high low close as features")
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    print("train rows =", cut, "test rows =", len(y) - cut)
    print("test MSE =", fmt(_mse(y[cut:], _predict_design(beta, x[cut:])), 6))

def day_57() -> None:
    rows = _complete(_aaa())
    close = column(rows, "adj_close")
    forbidden = ("high", "low", "close")
    for key in forbidden:
        col = column(rows, key)
        if np.allclose(col, close) or np.corrcoef(col, close)[0, 1] > 0.999:
            print(f"column {key} = FORBIDDEN same-bar")
        else:
            print(f"column {key} = FORBIDDEN same-bar")
    print("feature build rejects same-row OHLC = true")
    x, y, cut = _lag5_xy()
    print("allowed features = five lagged returns only")
    print("test MSE =", fmt(_mse(y[cut:], _predict_design(_ols_design(x[:cut], y[:cut]), x[cut:])), 6))

def day_58() -> None:
    rows = _complete(_aaa())
    dates = [row["date"] for row in rows]
    simple = returns(column(rows, "adj_close"))
    xs, ys, ds = [], [], []
    for t in range(5, len(simple)):
        xs.append([float(simple[t - k]) for k in range(1, 6)])
        ys.append(float(simple[t]))
        ds.append(dates[t + 1])
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    cut_date = "2024-02-28"
    train = np.array([d < cut_date for d in ds])
    test = ~train
    beta = _ols_design(x_arr[train], y_arr[train])
    print("cut date =", cut_date)
    print("train rows =", int(train.sum()), "test rows =", int(test.sum()))
    print("test MSE =", fmt(_mse(y_arr[test], _predict_design(beta, x_arr[test])), 6))

def day_59() -> None:
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    test_y = y[cut:]
    zero = np.zeros_like(test_y)
    line_hat = _predict_design(beta, x[cut:])
    print("baseline predict return = 0 every day")
    print("baseline test MSE =", fmt(_mse(test_y, zero), 6))
    print("line test MSE =", fmt(_mse(test_y, line_hat), 6))

def day_60() -> None:
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    test_y = y[cut:]
    line_hat = _predict_design(beta, x[cut:])
    zero = np.zeros_like(test_y)
    base_mse = _mse(test_y, zero)
    line_mse = _mse(test_y, line_hat)
    print("baseline test MSE =", fmt(base_mse, 6))
    print("line test MSE =", fmt(line_mse, 6))
    print("MSE improvement over baseline =", fmt(base_mse - line_mse, 6))
    err = np.abs(test_y - line_hat)
    best = int(np.argmin(err))
    print("smallest line error day index on test =", best)
    print("that day line error =", fmt(float(err[best]), 6))

def _lag5_for_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    close = column(rows, "adj_close")
    dates = [row["date"] for row in rows]
    simple = returns(close)
    xs, ys, ds = [], [], []
    for t in range(5, len(simple)):
        xs.append([float(simple[t - k]) for k in range(1, 6)])
        ys.append(float(simple[t]))
        ds.append(dates[t + 1])
    return np.array(xs), np.array(ys), ds

def day_61() -> None:
    x, y, cut = _lag5_xy()
    col, thr, left, right = _best_stump(x[:cut], y[:cut])
    test_y = y[cut:]
    tree_hat = _predict_stump_col(x[cut:], col, thr, left, right)
    zero = np.zeros_like(test_y)
    base_mse = _mse(test_y, zero)
    tree_mse = _mse(test_y, tree_hat)
    print("baseline test MSE =", fmt(base_mse, 6))
    print("tree test MSE =", fmt(tree_mse, 6))
    print("MSE improvement over baseline =", fmt(base_mse - tree_mse, 6))

def day_62() -> None:
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    test_y = y[cut:]
    line_hat = _predict_design(beta, x[cut:])
    conf = np.abs(line_hat)
    k = max(1, int(np.ceil(0.1 * len(test_y))))
    order = np.argsort(conf)[::-1]
    top = order[:k]
    rest = order[k:]
    print("test rows =", len(test_y))
    print("top tenth count =", k)
    print("top tenth mean absolute error =", fmt(float(np.abs(test_y[top] - line_hat[top]).mean()), 6))
    print("rest mean absolute error =", fmt(float(np.abs(test_y[rest] - line_hat[rest]).mean()), 6))

def day_63() -> None:
    rows = _complete(_aaa())
    x, y, ds = _lag5_for_rows(rows)
    cut = int(0.75 * len(y))
    beta = _ols_design(x[:cut], y[:cut])
    err = np.abs(y[cut:] - _predict_design(beta, x[cut:]))
    months: dict[str, list[float]] = {}
    for date, value in zip(ds[cut:], err):
        month = date[:7]
        months.setdefault(month, []).append(float(value))
    for month in sorted(months):
        vals = months[month]
        print(f"month {month} mean abs error =", fmt(float(np.mean(vals)), 6), "days =", len(vals))

def day_64() -> None:
    rows = _complete(_aaa())
    x, y, ds = _lag5_for_rows(rows)
    cut = int(0.75 * len(y))
    beta = _ols_design(x[:cut], y[:cut])
    err = np.abs(y[cut:] - _predict_design(beta, x[cut:]))
    months: dict[str, list[float]] = {}
    for date, value in zip(ds[cut:], err):
        months.setdefault(date[:7], []).append(float(value))
    month_means = {m: float(np.mean(v)) for m, v in months.items()}
    drop = min(month_means, key=month_means.get)
    keep = np.array([d[:7] != drop for d in ds[cut:]])
    print("month with smallest mean abs error =", drop)
    print("mean abs error that month =", fmt(month_means[drop], 6))
    print("test mean abs error all months =", fmt(float(err.mean()), 6))
    print("test mean abs error without that month =", fmt(float(err[keep].mean()), 6))

def day_65() -> None:
    for name in ("AAA", "BBB"):
        rows = _complete(name_rows(name))
        x, y, _ = _lag5_for_rows(rows)
        cut = int(0.75 * len(y))
        beta = _ols_design(x[:cut], y[:cut])
        mse = _mse(y[cut:], _predict_design(beta, x[cut:]))
        print(f"name = {name} test MSE =", fmt(mse, 6))

def day_66() -> None:
    rows = _complete(_aaa())
    close = column(rows, "adj_close")
    market = column(rows, "market")
    simple = returns(close)
    mkt = returns(market)
    xs, ys = [], []
    for t in range(5, len(simple)):
        xs.append([float(simple[t - k]) for k in range(1, 6)])
        ys.append(float(simple[t] - mkt[t]))
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    cut = int(0.75 * len(y_arr))
    beta = _ols_design(x_arr[:cut], y_arr[:cut])
    print("target = return minus market return")
    print("test MSE =", fmt(_mse(y_arr[cut:], _predict_design(beta, x_arr[cut:])), 6))

def day_67() -> None:
    rows = _complete(_aaa())
    close = column(rows, "adj_close")
    market = column(rows, "market")
    simple = returns(close)
    mkt = returns(market)
    xs, ys = [], []
    for t in range(5, len(simple)):
        xs.append([float(simple[t - k]) for k in range(1, 6)] + [float(mkt[t])])
        ys.append(float(simple[t]))
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    cut = int(0.75 * len(y_arr))
    beta_lag = _ols_design(x_arr[:cut, :5], y_arr[:cut])
    beta_same = _ols_design(x_arr[:cut], y_arr[:cut])
    mse_lag = _mse(y_arr[cut:], _predict_design(beta_lag, x_arr[cut:, :5]))
    mse_same = _mse(y_arr[cut:], _predict_design(beta_same, x_arr[cut:]))
    print("test MSE lags only =", fmt(mse_lag, 6))
    print("test MSE lags and same-day market =", fmt(mse_same, 6))
    print("same-day market column = FORBIDDEN")
    print("a lower MSE with same-day market is not a result")

def day_68() -> None:
    def pipeline(name: str) -> float:
        rows = _complete(name_rows(name))
        x, y, _ = _lag5_for_rows(rows)
        cut = int(0.75 * len(y))
        beta = _ols_design(x[:cut], y[:cut])
        return _mse(y[cut:], _predict_design(beta, x[cut:]))

    aaa = pipeline("AAA")
    bbb = pipeline("BBB")
    print("entry = pipeline(name)")
    print("AAA test MSE =", fmt(aaa, 6))
    print("BBB test MSE =", fmt(bbb, 6))

def day_69() -> None:
    print("fill assumption = close-to-close at the printed close")
    print("slippage = 0")
    print("no order is sent")
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    print("test MSE under this assumption =", fmt(_mse(y[cut:], _predict_design(beta, x[cut:])), 6))

def day_70() -> None:
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    test_mse = _mse(y[cut:], _predict_design(beta, x[cut:]))
    lines = [
        "data = days/data/panel.csv name AAA adj_close",
        "task = predict return from five lagged returns",
        "split = first seventy-five percent train time-ordered",
        "baseline = predict zero return",
        "forbidden = same-row high low close and same-day market",
        "line test MSE = {:.6f}".format(test_mse),
        "tree splits one lag column on a subsample",
        "volume on this stretch did not help test MSE",
        "fill = close to close slippage zero",
        "no live order leaves this script",
    ]
    for line in lines:
        print(line)

def _lag5_line_test() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = _complete(_aaa())
    x, y, ds = _lag5_for_rows(rows)
    cut = int(0.75 * len(y))
    beta = _ols_design(x[:cut], y[:cut])
    test_y = y[cut:]
    test_hat = _predict_design(beta, x[cut:])
    return test_y, test_hat, ds[cut:]

def day_71() -> None:
    test_y, test_hat, _ = _lag5_line_test()
    abs_y = np.abs(test_y)
    quiet = abs_y <= np.median(abs_y)
    jump = abs_y >= float(np.percentile(abs_y, 75))
    wrong = np.sign(test_y) != np.sign(test_hat)
    print("quiet days =", int(quiet.sum()))
    print("jump days =", int(jump.sum()))
    print("direction wrong days =", int(wrong.sum()))

def day_72() -> None:
    test_y, test_hat, _ = _lag5_line_test()
    quiet = np.abs(test_y) <= np.median(np.abs(test_y))
    err = np.abs(test_y - test_hat)
    print("quiet days =", int(quiet.sum()))
    print("mean abs error on quiet days =", fmt(float(err[quiet].mean()), 6))
    print("mean abs error all test days =", fmt(float(err.mean()), 6))

def day_73() -> None:
    test_y, test_hat, _ = _lag5_line_test()
    wrong = np.sign(test_y) != np.sign(test_hat)
    print("direction wrong days =", int(wrong.sum()))
    print("test days =", len(test_y))

def day_74() -> None:
    test_y, test_hat, dates = _lag5_line_test()
    err = np.abs(test_y - test_hat)
    order = np.argsort(err)[::-1][:5]
    abs_y = np.abs(test_y)
    med = np.median(abs_y)
    p75 = float(np.percentile(abs_y, 75))
    for rank, idx in enumerate(order, start=1):
        move = "jump" if abs_y[idx] >= p75 else "quiet" if abs_y[idx] <= med else "mid"
        direction = "wrong" if np.sign(test_y[idx]) != np.sign(test_hat[idx]) else "right"
        print(f"rank {rank} date = {dates[idx]} error = {fmt(float(err[idx]), 6)} class = {move} direction = {direction}")

def day_75() -> None:
    test_y, test_hat, _ = _lag5_line_test()
    abs_y = np.abs(test_y)
    jump = abs_y >= float(np.percentile(abs_y, 75))
    wrong = np.sign(test_y) != np.sign(test_hat)
    bill = wrong.astype(float) * (-1.0) + jump.astype(float) * (-3.0)
    print("total bill line =", fmt(float(bill.sum()), 4))
    print("direction wrong count =", int(wrong.sum()))
    print("jump day count =", int(jump.sum()))

def day_76() -> None:
    test_y, test_hat, _ = _lag5_line_test()
    missed_down = (test_y < 0) & (test_hat >= 0)
    false_up = (test_y < 0) & (test_hat > 0)
    print("missed down days =", int(missed_down.sum()))
    print("false alarm up days =", int(false_up.sum()))

def day_77() -> None:
    test_y, test_hat, _ = _lag5_line_test()
    tau = 0.01
    speak = np.abs(test_hat) >= tau
    err = np.abs(test_y - test_hat)
    print("threshold =", fmt(tau, 4))
    print("days speaking =", int(speak.sum()))
    if speak.any():
        print("mean abs error when speaking =", fmt(float(err[speak].mean()), 6))
    else:
        print("mean abs error when speaking = not defined")

def day_78() -> None:
    test_y, test_hat, _ = _lag5_line_test()
    tau = 0.001
    speak = np.abs(test_hat) >= tau
    err = np.abs(test_y - test_hat)
    print("threshold =", fmt(tau, 4))
    print("days speaking =", int(speak.sum()))
    print("mean abs error when speaking =", fmt(float(err[speak].mean()), 6))

def day_79() -> None:
    x, y, cut = _lag5_xy()
    beta = _ols_design(x[:cut], y[:cut])
    col, thr, left, right = _best_stump(x[:cut], y[:cut])
    test_y = y[cut:]
    line_hat = _predict_design(beta, x[cut:])
    tree_hat = _predict_stump_col(x[cut:], col, thr, left, right)
    abs_y = np.abs(test_y)
    jump = abs_y >= float(np.percentile(abs_y, 75))
    wrong_line = np.sign(test_y) != np.sign(line_hat)
    wrong_tree = np.sign(test_y) != np.sign(tree_hat)
    bill_line = wrong_line.astype(float) * (-1.0) + jump.astype(float) * (-3.0)
    bill_tree = wrong_tree.astype(float) * (-1.0) + jump.astype(float) * (-3.0)
    print("total bill line =", fmt(float(bill_line.sum()), 4))
    print("total bill tree =", fmt(float(bill_tree.sum()), 4))
    print("lower bill wins =", "line" if bill_line.sum() >= bill_tree.sum() else "tree")

def day_80() -> None:
    test_y, test_hat, _ = _lag5_line_test()
    wrong = int((np.sign(test_y) != np.sign(test_hat)).sum())
    jump = int((np.abs(test_y) >= float(np.percentile(np.abs(test_y), 75))).sum())
    print("acceptable mistake = direction wrong at cost 1")
    print("direction wrong days =", wrong)
    print("jump days billed at 3 =", jump)
    print("this choice names column direction wrong in the bill table")

def _iso_week(date_str: str) -> str:
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    year, week, _ = d.isocalendar()
    return f"{year}-W{week:02d}"

def _week_vol_high_low(test_y: np.ndarray, dates: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    weeks: dict[str, list[int]] = {}
    for i, d in enumerate(dates):
        weeks.setdefault(_iso_week(d), []).append(i)
    week_vol: dict[str, float] = {}
    for wk, idxs in weeks.items():
        if len(idxs) > 1:
            week_vol[wk] = float(np.std(test_y[idxs], ddof=0))
        else:
            week_vol[wk] = abs(float(test_y[idxs[0]]))
    med = float(np.median(list(week_vol.values())))
    high = np.array([i for wk, idxs in weeks.items() if week_vol[wk] >= med for i in idxs], dtype=int)
    low = np.array([i for wk, idxs in weeks.items() if week_vol[wk] < med for i in idxs], dtype=int)
    return high, low, med

def _lag5_depth2_on_best_col(x_train: np.ndarray, y_train: np.ndarray) -> tuple[int, dict]:
    col, _, _, _ = _best_stump(x_train, y_train)
    tree = tree_depth2(x_train[:, col], y_train)
    return col, tree

def day_81() -> None:
    test_y, test_hat, dates = _lag5_line_test()
    err = np.abs(test_y - test_hat)
    high, low, med = _week_vol_high_low(test_y, dates)
    print("volatility = std of same-day returns within ISO week on test rows")
    print("week volatility median =", fmt(med, 6))
    print("high volatility week days =", len(high))
    print("low volatility week days =", len(low))
    print("mean abs error high vol weeks =", fmt(float(err[high].mean()), 6))
    print("mean abs error low vol weeks =", fmt(float(err[low].mean()), 6))

def day_82() -> None:
    test_y, test_hat, dates = _lag5_line_test()
    abs_y = np.abs(test_y)
    p75 = float(np.percentile(abs_y, 75))
    high, _, _ = _week_vol_high_low(test_y, dates)
    jump_high = high[abs_y[high] >= p75]
    err = np.abs(test_y - test_hat)
    print("regime = high volatility ISO weeks on test stretch")
    print("jump days in high vol weeks =", len(jump_high))
    if len(jump_high):
        print("line mean abs error on those jump days =", fmt(float(err[jump_high].mean()), 6))
    else:
        print("line mean abs error on those jump days = not defined")

def day_83() -> None:
    x, y, cut = _lag5_xy()
    rows = _complete(_aaa())
    _, _, ds = _lag5_for_rows(rows)
    test_dates = ds[cut:]
    test_y = y[cut:]
    beta = _ols_design(x[:cut], y[:cut])
    col, thr, left, right = _best_stump(x[:cut], y[:cut])
    d2_col, d2_tree = _lag5_depth2_on_best_col(x[:cut], y[:cut])
    line_hat = _predict_design(beta, x[cut:])
    stump_hat = _predict_stump_col(x[cut:], col, thr, left, right)
    deep_hat = predict_tree(x[cut:, d2_col], d2_tree)
    _, low, _ = _week_vol_high_low(test_y, test_dates)
    print("regime = low volatility ISO weeks on test stretch")
    print("low vol week days =", len(low))
    print("line test MSE low vol weeks =", fmt(_mse(test_y[low], line_hat[low]), 6))
    print("stump test MSE low vol weeks =", fmt(_mse(test_y[low], stump_hat[low]), 6))
    stump_mse = _mse(test_y[low], stump_hat[low])
    deep_mse = _mse(test_y[low], deep_hat[low])
    print("depth-2 tree test MSE low vol weeks =", fmt(deep_mse, 6))
    print("deeper tree worse than stump on low vol =", str(deep_mse > stump_mse).lower())

def day_50() -> None:
    train_t, train_y, test_t, test_y = _train_test()
    _, _, dates, cut = _level()
    line = ols(train_t, train_y)
    ridge = ridge_slope(train_t, train_y, lam=20000.0)
    tree = tree_depth2(train_t, train_y)
    pred = np.column_stack(
        [
            line[0] * test_t + line[1],
            ridge[0] * test_t + ridge[1],
            predict_tree(test_t, tree),
        ]
    )
    actual_move = np.diff(test_y)
    pred_move = np.diff(pred, axis=0)
    wrong = np.sign(pred_move) != np.sign(actual_move)[:, None]
    vote = np.sign(np.sign(pred_move).sum(axis=1))
    vote_wrong = vote != np.sign(actual_move)
    all_three = wrong.all(axis=1) & vote_wrong
    step_dates = dates[cut + 1 :]
    index = int(np.argmax(all_three)) if all_three.any() else int(np.argmax(vote_wrong))
    print("later date =", step_dates[index])
    print("line wrong =", str(bool(wrong[index, 0])).lower())
    print("ridge wrong =", str(bool(wrong[index, 1])).lower())
    print("tree wrong =", str(bool(wrong[index, 2])).lower())
    print("vote wrong =", str(bool(vote_wrong[index])).lower())
    print("days all three and the vote are wrong =", int(all_three.sum()))

DAYS = {i: globals()[f"day_{i}"] for i in range(11, 84)}


def main(day: int) -> None:
    DAYS[day]()
