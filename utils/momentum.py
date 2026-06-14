"""动量计算核心 — 向量化版本（基于收盘价直接计算累积收益）"""

import numpy as np
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
H5_DAILY = os.path.join(PROJECT_ROOT, 'data', 'stock_daily_data_csi300.h5')
H5_MCAP = os.path.join(PROJECT_ROOT, 'data', 'market_cap_annual.h5')

# 持有期 → 交易日
HOLD_DAYS = {'1W': 5, '1M': 21, '3M': 63, '6M': 126}


# ============================================================
# 数据加载
# ============================================================

def _load_close_prices(stock_codes):
    """从 HDF5 加载收盘价 DataFrame (dates × codes)，一次加载。"""
    prices = {}
    with pd.HDFStore(H5_DAILY, 'r') as store:
        for code in stock_codes:
            key = f'data/{code.replace(".", "_")}'
            if key not in store:
                continue
            df = store[key]
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            prices[code] = df.set_index('trade_date')['close']
    df = pd.DataFrame(prices).sort_index()
    return df


def _load_industry_map():
    csv_path = os.path.join(PROJECT_ROOT, 'output', 'stocks', 'CSI300_union.csv')
    df = pd.read_csv(csv_path).dropna(subset=['ts_code', 'l1_name'])
    return dict(zip(df['ts_code'], df['l1_name']))


def _load_mcap_for_date(date, stock_codes_list):
    """获取 date 对应的年末市值 Series。按年报披露滞后规则。"""
    year, month = date.year, date.month
    mcap_year = year - 2 if month <= 4 else year - 1
    try:
        with pd.HDFStore(H5_MCAP, 'r') as store:
            key = f'market_cap/{mcap_year}'
            if key not in store:
                return pd.Series(dtype=float)
            df = store[key].copy()
            df['total_mv'] = pd.to_numeric(df['total_mv'], errors='coerce') * 10000  # 万元→元
            df = df.set_index('ts_code')['total_mv']
            return df.reindex(stock_codes_list).fillna(1.0)
    except (KeyError, FileNotFoundError):
        return pd.Series(dtype=float)


# ============================================================
# 动量计算（向量化）
# ============================================================

def _compute_all_momentum(close_df, sector_map, mom_type, formation_days, holding,
                          skip_recent=False):
    """通用向量化动量计算。

    参数
    ----
    close_df : DataFrame (dates × ts_code)
    sector_map : {ts_code: l1_name}
    mom_type : 'cross' | 'ts'
    formation_days : int  形成期交易日数
    holding : str  持有期 '1W'|'1M'|'3M'|'6M'
    skip_recent : bool  是否跳过最近1个月

    返回
    ----
    {sector: {'dates': [...], 'momentum': [...]}}
    """
    codes = close_df.columns.tolist()
    valid_codes = [c for c in codes if c in sector_map]
    close_df = close_df[valid_codes]
    dates = close_df.index
    n_days = len(dates)

    # ---- 板块分组 ----
    sector_stocks = {}
    for code in valid_codes:
        sec = sector_map[code]
        sector_stocks.setdefault(sec, []).append(code)
    sectors = sorted(sector_stocks.keys())

    # ---- 向量化: 形成期累积收益 (全矩阵) ----
    lookback = formation_days + (21 if skip_recent else 0)
    if skip_recent:
        # e.g. 6M(126d) skip last 21d: cum = close[t-21] / close[t-126] - 1
        form_ret_all = close_df.shift(21) / close_df.shift(lookback) - 1.0
    else:
        form_ret_all = close_df / close_df.shift(lookback) - 1.0

    # ---- 向量化: 持有期收益 (每个持有月份长度) ----
    hold_days = HOLD_DAYS[holding]
    hold_ret_all = close_df.shift(-hold_days) / close_df - 1.0  # 注意负号: 用未来价格

    # ---- 预加载市值 (每50个交易日缓存一次，避免重复读HDF5) ----
    mcap_cache = {}
    def get_mcap(d):
        month_key = (d.year, d.month)
        if month_key not in mcap_cache:
            mcap_cache[month_key] = _load_mcap_for_date(d, valid_codes)
        return mcap_cache[month_key]

    # ---- 转为 numpy 数组加速内层循环 ----
    close_arr = close_df.values  # (n_days, n_codes)
    form_arr = form_ret_all.values  # 形成期累积收益
    hold_arr = hold_ret_all.values  # 持有期收益
    code_list = close_df.columns.tolist()
    code_to_idx = {c: i for i, c in enumerate(code_list)}

    # 预计算板块索引数组
    sector_idx_arrays = {}
    for s in sectors:
        idx = np.array([code_to_idx[c] for c in sector_stocks[s] if c in code_to_idx])
        if len(idx) > 0:
            sector_idx_arrays[s] = idx

    # 预加载全时段市值到数组 (按月份)
    all_months = pd.date_range(dates[0], dates[-1], freq='MS')
    mcap_matrix = np.full((len(code_list), len(all_months)), 1.0)
    for mi, month_dt in enumerate(all_months):
        mcap = _load_mcap_for_date(month_dt, code_list)
        if len(mcap) > 0:
            mcap_matrix[:, mi] = mcap.values

    results = {s: {'dates': [], 'momentum': []} for s in sectors}

    for t in range(lookback, n_days - hold_days):
        form_date = dates[t]
        month_idx = (form_date.year - all_months[0].year) * 12 + (form_date.month - all_months[0].month)
        month_idx = max(0, min(month_idx, len(all_months) - 1))

        for s, idx in sector_idx_arrays.items():
            vals = form_arr[t, idx]
            valid_mask = ~np.isnan(vals)
            if valid_mask.sum() < 2:
                continue
            valid_idx = idx[valid_mask]
            valid_vals = vals[valid_mask]

            if mom_type == 'cross':
                if len(valid_vals) >= 5:
                    q_hi = np.quantile(valid_vals, 0.8)
                    q_lo = np.quantile(valid_vals, 0.2)
                    long_mask = valid_vals >= q_hi
                    short_mask = valid_vals <= q_lo
                else:
                    long_mask = np.zeros(len(valid_vals), dtype=bool)
                    long_mask[np.argmax(valid_vals)] = True
                    short_mask = np.zeros(len(valid_vals), dtype=bool)
                    short_mask[np.argmin(valid_vals)] = True
            else:
                long_mask = valid_vals > 0
                short_mask = valid_vals < 0

            if not long_mask.any() or not short_mask.any():
                continue

            long_i = valid_idx[long_mask]
            short_i = valid_idx[short_mask]

            # 市值权重 (从预加载矩阵取)
            long_w_raw = mcap_matrix[long_i, month_idx]
            short_w_raw = mcap_matrix[short_i, month_idx]
            long_w = long_w_raw / long_w_raw.sum() if long_w_raw.sum() > 0 else np.ones(len(long_i)) / len(long_i)
            short_w = short_w_raw / short_w_raw.sum() if short_w_raw.sum() > 0 else np.ones(len(short_i)) / len(short_i)

            # 持有期收益
            long_ret = np.nansum(hold_arr[t, long_i] * long_w)
            short_ret = np.nansum(hold_arr[t, short_i] * short_w)
            ls_return = float(long_ret - short_ret)

            if not np.isnan(ls_return):
                results[s]['dates'].append(form_date)
                results[s]['momentum'].append(ls_return)

    return results


# ============================================================
# 全市场聚合动量（不分子行业，对所有股票统一做多空）
# ============================================================

def compute_aggregate_momentum(stock_codes, formation='1M', holding_periods=None, mom_type='cross'):
    """全市场动量策略：对所有股票按形成期收益排序，选 top/bottom 20% 形成多空组合。

    返回 {'dates': [...], 'momentum': [...]}
    """
    if holding_periods is None:
        holding_periods = [1]

    formation_days = {'1W': 5, '1M': 21, '6M': 126, '12M': 252}[formation]
    skip_recent = formation in ('6M', '12M')
    lookback = formation_days + (21 if skip_recent else 0)
    hold_days = HOLD_DAYS[holding_periods[0]]

    close_df = _load_close_prices(stock_codes)
    # 去掉列名可能导致的重复
    close_df = close_df.loc[:, ~close_df.columns.duplicated()]
    dates = close_df.index
    n_days = len(dates)
    code_list = close_df.columns.tolist()

    if skip_recent:
        form_arr = (close_df.shift(21) / close_df.shift(lookback) - 1.0).values
    else:
        form_arr = (close_df / close_df.shift(lookback) - 1.0).values
    hold_arr = (close_df.shift(-hold_days) / close_df - 1.0).values

    # 预加载市值
    mcap_matrix = np.full((len(code_list), 1), 1.0)

    results = {'dates': [], 'momentum': []}
    n_codes = len(code_list)

    for t in range(lookback, n_days - hold_days):
        ret_row = form_arr[t]
        valid_mask = ~np.isnan(ret_row)
        n_valid = valid_mask.sum()
        if n_valid < 5:
            continue

        ret_valid = ret_row[valid_mask]
        idx_valid = np.where(valid_mask)[0]

        if mom_type == 'cross':
            q_hi = np.quantile(ret_valid, 0.8)
            q_lo = np.quantile(ret_valid, 0.2)
            long_mask = ret_valid >= q_hi
            short_mask = ret_valid <= q_lo
        else:
            long_mask = ret_valid > 0
            short_mask = ret_valid < 0

        if not long_mask.any() or not short_mask.any():
            continue

        long_i = idx_valid[long_mask]
        short_i = idx_valid[short_mask]

        # 等权（全市场不需要市值分层）
        long_w = np.ones(len(long_i)) / len(long_i)
        short_w = np.ones(len(short_i)) / len(short_i)

        long_ret = np.nansum(hold_arr[t, long_i] * long_w)
        short_ret = np.nansum(hold_arr[t, short_i] * short_w)
        ls = float(long_ret - short_ret)

        if not np.isnan(ls):
            results['dates'].append(dates[t])
            results['momentum'].append(ls)

    return results


def compute_cross_sectional_momentum(stock_codes, formation='1M', holding_periods=None):
    if holding_periods is None:
        holding_periods = [1]
    formation_days = {'1W': 5, '1M': 21, '6M': 126, '12M': 252}[formation]
    skip_recent = formation in ('6M', '12M')

    close_df = _load_close_prices(stock_codes)
    sector_map = _load_industry_map()

    results = {}
    for h in holding_periods:
        results[h] = _compute_all_momentum(
            close_df, sector_map, 'cross', formation_days, h, skip_recent)
    return results


def compute_time_series_momentum(stock_codes, formation='1M', holding_periods=None):
    if holding_periods is None:
        holding_periods = [1]
    formation_days = {'1W': 5, '1M': 21, '6M': 126, '12M': 252}[formation]
    skip_recent = formation in ('6M', '12M')

    close_df = _load_close_prices(stock_codes)
    sector_map = _load_industry_map()

    results = {}
    for h in holding_periods:
        results[h] = _compute_all_momentum(
            close_df, sector_map, 'ts', formation_days, h, skip_recent)
    return results
