"""数据加载层 — pickle 文件读取、板块指标计算、NetworkShock 数据封装"""

import pickle
import sys
import types
import numpy as np
from collections import defaultdict


# ============================================================
# 注册 GL.NetworkData 存根（与 RShiny reticulate 中的 hack 相同）
# ============================================================
class NetworkData:
    def __init__(self, time=None, jaccard_index=None, A=None, Theta=None,
                 l1_penalty=None):
        self.time = time
        self.jaccard_index = jaccard_index
        self.A = A
        self.Theta = Theta
        self.l1_penalty = l1_penalty


_gl_stub = types.ModuleType('GL')
_gl_stub.NetworkData = NetworkData
sys.modules['GL'] = _gl_stub


# ============================================================
# 基础加载
# ============================================================

def load_network_pkl(filepath):
    """加载 SGL/FGL pkl 文件，返回解析后的数据结构。

    Returns
    -------
    dict with keys:
        names: list[str]        股票名称
        l1_groups: list[str]    行业分类
        times: list[datetime.date]
        jaccard: np.ndarray (T,)
        l1_penalty: np.ndarray (T,)
        A: np.ndarray (T, p, p)  邻接矩阵
        Theta: np.ndarray (T, p, p)  精度矩阵
    """
    with open(filepath, 'rb') as f:
        raw = pickle.load(f)

    names = list(raw['name'])
    l1_groups = list(raw.get('l1_groups', []))
    data_arr = raw['data_array']

    T = len(data_arr)
    p = len(names)

    times = [d.time for d in data_arr]
    jaccard = np.array([float(d.jaccard_index) for d in data_arr])
    l1_pen = np.array([float(d.l1_penalty) for d in data_arr])
    A_stack = np.stack([np.asarray(d.A) for d in data_arr])
    Theta_stack = np.stack([np.asarray(d.Theta) for d in data_arr])

    return {
        'names': names,
        'l1_groups': l1_groups,
        'times': times,
        'jaccard': jaccard,
        'l1_penalty': l1_pen,
        'A': A_stack,
        'Theta': Theta_stack,
        'p': p,
        'T': T,
    }


# ============================================================
# 板块指标计算（从 app.R compute_sector_metrics 移植）
# ============================================================

def compute_sector_metrics(A_stack, Theta_stack, l1_groups):
    """计算逐板块的密度、中心性、跨板块连接指标。

    返回 dict 包含所有指标矩阵，以及 sectors 列表和全局 jaccard/l1。
    """
    if not l1_groups:
        return None

    sectors = sorted(set(l1_groups))
    n_sectors = len(sectors)
    T = len(A_stack)
    p = A_stack.shape[1]

    # 板块 -> 资产索引
    sector_to_idx = {}
    for s in sectors:
        sector_to_idx[s] = [i for i, g in enumerate(l1_groups) if g == s]

    # 预分配
    density_disc = np.zeros((n_sectors, T))
    density_cont = np.zeros((n_sectors, T))
    centrality_disc = np.zeros((n_sectors, T))
    centrality_cont = np.zeros((n_sectors, T))
    cross_conn_disc = np.zeros((n_sectors, n_sectors, T))
    cross_conn_cont = np.zeros((n_sectors, n_sectors, T))

    for t in range(T):
        A = A_stack[t]
        Theta = Theta_stack[t]

        for si, s in enumerate(sectors):
            idx = sector_to_idx[s]
            n_s = len(idx)
            ext_idx = [i for i in range(p) if i not in idx]
            n_ext = len(ext_idx)

            # 内部密度
            if n_s > 1:
                sub_A = A[np.ix_(idx, idx)]
                n_edges = (sub_A.sum() - np.trace(sub_A)) / 2.0
                max_edges = n_s * (n_s - 1) / 2.0
                density_disc[si, t] = n_edges / max_edges if max_edges > 0 else 0.0

                sub_T = Theta[np.ix_(idx, idx)].copy()
                np.fill_diagonal(sub_T, 0.0)
                density_cont[si, t] = np.abs(sub_T).sum() / 2.0 / max_edges if max_edges > 0 else 0.0
            else:
                density_disc[si, t] = 0.0
                density_cont[si, t] = 0.0

            # 外部 degree centrality
            if n_ext > 0 and n_s > 0:
                ext_A = A[np.ix_(idx, ext_idx)]
                centrality_disc[si, t] = ext_A.sum() / (n_s * n_ext)

                ext_T = Theta[np.ix_(idx, ext_idx)]
                centrality_cont[si, t] = np.abs(ext_T).sum() / (n_s * n_ext)
            else:
                centrality_disc[si, t] = 0.0
                centrality_cont[si, t] = 0.0

        # 板块间连接性
        for si, s1 in enumerate(sectors):
            for sj in range(si + 1, n_sectors):
                s2 = sectors[sj]
                idx1 = sector_to_idx[s1]
                idx2 = sector_to_idx[s2]
                n1, n2 = len(idx1), len(idx2)

                if n1 > 0 and n2 > 0:
                    cross_A = A[np.ix_(idx1, idx2)]
                    cross_conn_disc[si, sj, t] = cross_A.sum() / (n1 * n2)

                    cross_T = Theta[np.ix_(idx1, idx2)]
                    cross_conn_cont[si, sj, t] = np.abs(cross_T).sum() / (n1 * n2)

    return {
        'sectors': sectors,
        'n_sectors': n_sectors,
        'density_disc': density_disc,
        'density_cont': density_cont,
        'centrality_disc': centrality_disc,
        'centrality_cont': centrality_cont,
        'cross_conn_disc': cross_conn_disc,
        'cross_conn_cont': cross_conn_cont,
    }


# ============================================================
# NetworkShock 实时计算（五分量 + 复合指标）
# ============================================================

def compute_networkshock_from_data(A_stack, Theta_stack, jaccard_seq, l1_groups):
    """从已加载的 A/Theta 矩阵序列实时计算 NetworkShock 五分量。

    参数
    ----
    A_stack : ndarray (T, p, p)  邻接矩阵
    Theta_stack : ndarray (T, p, p)  精度矩阵
    jaccard_seq : ndarray (T,)  Jaccard 指数序列
    l1_groups : list[str]  行业分类

    返回
    ----
    dict with keys:
        times_idx: np.arange(1, T) — 对应 time index (t 对比 t-1)
        1-Jaccard, L1, EdgeTurnover, DeltaDensity, CentDisp — 原始分量
        z_{name} — z-score 标准化分量
        NetworkShock — 五分量 z-score 等权加总
    """
    from collections import defaultdict

    T, p, _ = A_stack.shape
    total_edges = p * (p - 1) // 2

    # 预计算密度
    densities = np.array([A.sum() / 2 / total_edges for A in A_stack])

    # 行业分组
    sector_to_idx = defaultdict(list)
    for i, s in enumerate(l1_groups):
        sector_to_idx[s].append(i)
    sectors = list(sector_to_idx.keys())

    results = {
        'jaccard': np.full(T - 1, np.nan),
        'L1': np.full(T - 1, np.nan),
        'edge_turnover': np.full(T - 1, np.nan),
        'delta_density': np.full(T - 1, np.nan),
        'centrality_dispersion': np.full(T - 1, np.nan),
    }

    for t in range(1, T):
        tp = t - 1

        # 1 - Jaccard
        jac = jaccard_seq[t]
        results['jaccard'][tp] = 1.0 - jac if not np.isnan(jac) else np.nan

        # L1 distance (normalized)
        results['L1'][tp] = np.sum(np.abs(Theta_stack[t] - Theta_stack[tp])) / (p * p)

        # Edge turnover
        A_prev, A_cur = A_stack[tp], A_stack[t]
        sym_diff = np.triu(A_prev != A_cur, 1).sum()
        both_exist = (A_prev > 0) & (A_cur > 0)
        sign_flip = np.triu(
            both_exist & (np.sign(Theta_stack[t]) != np.sign(Theta_stack[tp])), 1
        ).sum()
        results['edge_turnover'][tp] = (sym_diff + sign_flip) / total_edges

        # Delta density
        results['delta_density'][tp] = densities[t] - densities[tp]

        # Centrality dispersion (行业间 CV of mean degree)
        degrees = A_cur.sum(axis=1)
        sector_means = np.array([degrees[sector_to_idx[s]].mean() for s in sectors])
        mean_s = sector_means.mean()
        std_s = sector_means.std(ddof=1) if len(sectors) > 1 else 0.0
        results['centrality_dispersion'][tp] = std_s / mean_s if mean_s > 0 else 0.0

    # z-score 标准化
    z_components = {}
    ns = np.zeros(T - 1)
    for k in ['jaccard', 'L1', 'edge_turnover', 'delta_density', 'centrality_dispersion']:
        vals = results[k]
        valid = ~np.isnan(vals)
        if valid.sum() > 0:
            mu = np.mean(vals[valid])
            sigma = np.std(vals[valid])
            if sigma < 1e-10:
                sigma = 1.0
            z = np.full_like(vals, np.nan)
            z[valid] = (vals[valid] - mu) / sigma
        else:
            z = np.zeros_like(vals)
        z_components[k] = z
        ns += np.nan_to_num(z, nan=0.0)

    # 构建返回 dict
    out = {
        'times_idx': np.arange(1, T),
        '1-Jaccard': results['jaccard'],
        'L1': results['L1'],
        'EdgeTurnover': results['edge_turnover'],
        'DeltaDensity': results['delta_density'],
        'CentDisp': results['centrality_dispersion'],
        'z_1-Jaccard': z_components['jaccard'],
        'z_L1': z_components['L1'],
        'z_EdgeTurnover': z_components['edge_turnover'],
        'z_DeltaDensity': z_components['delta_density'],
        'z_CentDisp': z_components['centrality_dispersion'],
        'NetworkShock': ns,
    }
    return out


# ============================================================
# 便捷: 一次性加载 pkl + sector metrics + NetworkShock
# ============================================================

def load_full(filepath):
    """加载 pkl 并计算板块指标，返回合并后的 dict。"""
    net = load_network_pkl(filepath)
    if net['l1_groups']:
        sm = compute_sector_metrics(net['A'], net['Theta'], net['l1_groups'])
        net.update(sm)
    else:
        net['sectors'] = []
        net['n_sectors'] = 0
    return net
