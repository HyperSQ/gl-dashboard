"""Tab 4: 动量轮动 — RotationScore = ΔDensity + ΔCentrality + Momentum
动量数据从预计算缓存 momentum_cache.pkl 加载。
"""

from dash import html, dcc, Input, Output, callback, State
import numpy as np
import pandas as pd
import pickle
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_network_pkl, compute_sector_metrics
from components.plots import empty_figure
from utils.momentum import HOLD_DAYS
from data.loader import compute_networkshock_from_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'momentum_cache.pkl')

FORMATION_OPTIONS = [
    {'label': '1周 (5日)', 'value': '1W'},
    {'label': '1月 (21日)', 'value': '1M'},
    {'label': '6月 (剔除近1月)', 'value': '6M'},
    {'label': '12月 (剔除近1月)', 'value': '12M'},
]
HOLDING_OPTIONS = [
    {'label': '1周', 'value': '1W'},
    {'label': '1个月', 'value': '1M'},
    {'label': '3个月', 'value': '3M'},
    {'label': '6个月', 'value': '6M'},
]
MOM_TYPE_OPTIONS = [
    {'label': '横截面动量', 'value': 'cross'},
    {'label': '时序动量', 'value': 'ts'},
]
FREQ_OPTIONS_ALL = [
    {'label': '日度', 'value': 'D'},
    {'label': '周度', 'value': 'W'},
    {'label': '月度', 'value': 'M'},
]
FREQ_ORDER = {'D': 0, 'W': 1, 'M': 2}  # D 最细, M 最粗


def _allowed_freqs(native_freq):
    """返回不低于 pkl 原生频率的选项。例: W → 周度、月度。"""
    min_level = FREQ_ORDER.get(native_freq, 0)
    return [o for o in FREQ_OPTIONS_ALL if FREQ_ORDER[o['value']] >= min_level]
UNIVERSE_OPTIONS = [
    {'label': '当前组合 (95只)', 'value': 'network'},
    {'label': 'CSI300全市场 (690只)', 'value': 'union'},
]


def _resample(data_2d, times, freq, agg='last'):
    """将 (n_sectors, T) 数据按频率重采样。

    freq: 'D' (不变), 'W' (周五), 'M' (月末)
    agg: 'last' | 'mean'
    返回 (resampled_data, new_times)
    """
    if freq == 'D':
        return data_2d, times

    df = pd.DataFrame(data_2d.T, index=pd.DatetimeIndex(times))
    if freq == 'W':
        grouper = pd.Grouper(freq='W-FRI')
    else:
        grouper = pd.Grouper(freq='ME')

    if agg == 'last':
        resampled = df.groupby(grouper).last()
    else:
        resampled = df.groupby(grouper).mean()

    # 去掉全 NaN 的行
    resampled = resampled.dropna(how='all')
    new_times = resampled.index
    return resampled.values.T, list(new_times)


def _load_momentum_cache():
    """加载预计算的动量缓存，转换 epoch days → date 字符串。"""
    try:
        with open(CACHE_FILE, 'rb') as f:
            cache = pickle.load(f)
    except FileNotFoundError:
        return None

    # 转换 epoch days → ISO 日期字符串
    for k in cache:
        for sector, sv in cache[k].items():
            if isinstance(sv, dict) and 'dates' in sv:
                d = sv['dates']
                if hasattr(d, 'dtype') and d.dtype == np.dtype('int32'):
                    cache[k][sector]['dates'] = [str(np.datetime64(int(x), 'D')) for x in d]
                sv['momentum'] = np.array(sv['momentum'], dtype=float)
    return cache


def _make_ns_compare(net, mom_cache, agg_key, freq, cache_key):
    """NetworkShock 五分量 + 动量 时序叠加对比。"""
    if not net or not net.get('l1_groups'):
        return None
    # 先对原始矩阵按频率重采样，再计算五分量
    times_all = net['times']
    A_all = net['A']       # (T, p, p)
    Theta_all = net['Theta']
    jaccard_all = net['jaccard']

    if freq == 'D':
        A_use, Theta_use = A_all, Theta_all
        j_use = jaccard_all
        idx_keep = list(range(len(times_all)))
    else:
        dates = pd.DatetimeIndex(times_all)
        grouper = pd.Grouper(freq='W-FRI') if freq == 'W' else pd.Grouper(freq='ME')
        s = pd.Series(range(len(dates)), index=dates)
        idx_keep = sorted(s.groupby(grouper).last().dropna().astype(int).values)
        A_use = A_all[idx_keep]
        Theta_use = Theta_all[idx_keep]
        j_use = jaccard_all[idx_keep]

    ns = compute_networkshock_from_data(A_use, Theta_use, j_use, net['l1_groups'])
    ns_times = [times_all[i] for i in idx_keep[1:]]

    if agg_key not in mom_cache:
        return None
    agg_md = mom_cache[agg_key]
    agg_dates = pd.to_datetime(agg_md['dates'])
    agg_vals = np.array(agg_md['momentum'])

    if freq != 'D':
        df_a = pd.DataFrame({'v': agg_vals}, index=agg_dates)
        grouper = pd.Grouper(freq='W-FRI') if freq == 'W' else pd.Grouper(freq='ME')
        rs_a = df_a.groupby(grouper).last().dropna()
        agg_dates = rs_a.index; agg_vals = rs_a['v'].values

    # 对齐日期：取 NS 和 动量 的日期交集
    ns_date_map = {pd.Timestamp(d).date() if hasattr(d, 'date') else d: i
                   for i, d in enumerate(ns_times)}
    mom_date_map = {}
    for i, d in enumerate(agg_dates):
        dk = pd.Timestamp(d).date() if hasattr(d, 'date') else d
        if dk in ns_date_map:
            mom_date_map[dk] = (ns_date_map[dk], agg_vals[i])

    if len(mom_date_map) < 10:
        return None

    shared_dates_sorted = sorted(mom_date_map.keys())
    ns_idx = [mom_date_map[d][0] for d in shared_dates_sorted]
    mom_shared = np.array([mom_date_map[d][1] for d in shared_dates_sorted])

    fig = go.Figure()

    components = [
        ('NetworkShock', 'NetworkShock', 'black', 2.0),
        ('z_1-Jaccard', '1-Jaccard', '#E41A1C', 0.8),
        ('z_L1', 'L1', '#377EB8', 0.8),
        ('z_EdgeTurnover', 'EdgeTurnover', '#4DAF4A', 0.8),
        ('z_DeltaDensity', 'DeltaDensity', '#FF7F00', 0.8),
        ('z_CentDisp', 'CentDisp', '#984EA3', 0.8),
    ]
    for key, label, color, width in components:
        raw = np.array(ns[key])[ns_idx] if key in ns else None
        if raw is None:
            continue
        z = (raw - np.nanmean(raw)) / max(np.nanstd(raw), 1e-10)
        fig.add_trace(go.Scatter(
            x=shared_dates_sorted, y=z, mode='lines', name=label,
            line=dict(color=color, width=width)))

    z_mom = (mom_shared - np.nanmean(mom_shared)) / max(np.nanstd(mom_shared), 1e-10)
    fig.add_trace(go.Scatter(
        x=shared_dates_sorted, y=z_mom, mode='lines', name='动量',
        line=dict(color='#00CED1', width=2.0, dash='dash')))

    fig.add_hline(y=0, line=dict(color='gray', dash='dot'))
    fig.update_layout(
        title='NetworkShock 五分量 + 动量 (z-score)',
        hovermode='x unified', margin=dict(l=10, r=10, t=35, b=10),
        template='plotly_white', legend=dict(orientation='h', y=1.05))
    return fig


def layout():
    return html.Div([
        html.H4('动量与板块轮动', className='mt-3 mb-3'),

        html.Div([
            html.Div([
                html.Label('数据文件'),
                dcc.Dropdown(id='rot-file-dropdown', clearable=False, className='mb-2'),
            ], style={'flex': '2'}),
            html.Div([
                html.Label('动量类型'),
                dcc.Dropdown(id='rot-mom-type', options=MOM_TYPE_OPTIONS,
                             value='cross', clearable=False, className='mb-2'),
            ], style={'flex': '1', 'padding-left': '10px'}),
            html.Div([
                html.Label('形成期'),
                dcc.Dropdown(id='rot-formation', options=FORMATION_OPTIONS,
                             value='12M', clearable=False, className='mb-2'),
            ], style={'flex': '1', 'padding-left': '10px'}),
            html.Div([
                html.Label('持有期'),
                dcc.Dropdown(id='rot-holding', options=HOLDING_OPTIONS,
                             value='6M', clearable=False, className='mb-2'),
            ], style={'flex': '1', 'padding-left': '10px'}),
            html.Div([
                html.Label('股票范围'),
                dcc.Dropdown(id='rot-universe', options=UNIVERSE_OPTIONS, value='network',
                             clearable=False, className='mb-2'),
            ], style={'flex': '1', 'padding-left': '10px'}),
            html.Div([
                html.Label('频率'),
                dcc.Dropdown(id='rot-freq', options=FREQ_OPTIONS_ALL, value='W',
                             clearable=False, className='mb-2'),
            ], style={'flex': '1', 'padding-left': '10px'}),
            html.Div([
                html.Label('板块'),
                dcc.Dropdown(id='rot-sector', clearable=False, className='mb-2'),
            ], style={'flex': '1', 'padding-left': '10px'}),
        ], style={'display': 'flex', 'align-items': 'flex-end'}),

        html.Div(id='rot-status', style={'text-align': 'center', 'color': 'gray',
                                          'margin': '10px'}),

        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='rot-momentum-plot', style={'height': '350px'}), type='circle'), style={'flex': '1'}),
            html.Div(dcc.Loading(dcc.Graph(id='rot-rs-plot', style={'height': '350px'}), type='circle'), style={'flex': '1'}),
        ], style={'display': 'flex'}),

        dcc.Loading(dcc.Graph(id='rot-cumret-plot', style={'height': '320px'}), type='circle'),

        dcc.Loading(dcc.Graph(id='rot-ns-compare-plot', style={'height': '420px'}), type='circle'),

        dcc.Loading(dcc.Graph(id='rot-heatmap', style={'height': '450px'}), type='circle'),

        dcc.Store(id='rot-metrics-store'),
    ])


def register_callbacks(app, pkl_files, pkl_meta_map):

    @callback(
        Output('rot-file-dropdown', 'options'),
        Output('rot-file-dropdown', 'value'),
        Input('rot-file-dropdown', 'id'),
    )
    def init_dropdown(_):
        options = [{'label': pkl_meta_map[f]['label'], 'value': f} for f in pkl_files]
        # 默认选有PC1的周频
        default = pkl_files[0] if pkl_files else None
        for f in pkl_files:
            if 'W_W=1' in f and 'noPC1' not in f:
                default = f; break
        return options, default

    # ---- 文件变化 → 计算板块指标 ----
    @callback(
        Output('rot-metrics-store', 'data'),
        Output('rot-sector', 'options'),
        Output('rot-sector', 'value'),
        Output('rot-freq', 'options'),
        Output('rot-freq', 'value'),
        Output('rot-status', 'children'),
        Input('rot-file-dropdown', 'value'),
        State('rot-sector', 'value'),
        State('rot-freq', 'value'),
    )
    def on_file_change(filepath, old_sector, old_freq):
        # 根据 pkl 原生频率限制可选频率
        from config import parse_filename
        pkl_meta = parse_filename(filepath) if filepath else {}
        native_f = pkl_meta.get('freq', 'D')
        freq_opts = _allowed_freqs(native_f)
        use_freq = old_freq if old_freq and any(o['value'] == old_freq for o in freq_opts) else freq_opts[0]['value']

        if not filepath:
            return None, [], None, freq_opts, use_freq, ''
        net = load_network_pkl(filepath)
        if not net['l1_groups']:
            return None, [], None, freq_opts, use_freq, '当前文件不含行业分类信息。'
        metrics = compute_sector_metrics(net['A'], net['Theta'], net['l1_groups'])
        sectors = metrics['sectors']
        metrics_serial = {
            'sectors': sectors,
            'density_disc': metrics['density_disc'].tolist(),
            'centrality_disc': metrics['centrality_disc'].tolist(),
            'T': net['T'],
        }
        sector_opts = [{'label': s, 'value': s} for s in sectors]
        sel = old_sector if old_sector in sectors else sectors[0]
        return metrics_serial, sector_opts, sel, freq_opts, use_freq, \
            f'{len(sectors)} 个板块, {net["T"]} 个时间点'

    # ---- 计算 RotationScore (从缓存加载动量) ----
    @callback(
        Output('rot-momentum-plot', 'figure'),
        Output('rot-rs-plot', 'figure'),
        Output('rot-cumret-plot', 'figure'),
        Output('rot-ns-compare-plot', 'figure'),
        Output('rot-heatmap', 'figure'),
        Input('rot-file-dropdown', 'value'),
        Input('rot-mom-type', 'value'),
        Input('rot-formation', 'value'),
        Input('rot-holding', 'value'),
        Input('rot-universe', 'value'),
        Input('rot-freq', 'value'),
        Input('rot-sector', 'value'),
        State('rot-metrics-store', 'data'),
    )
    def compute_all(filepath, mom_type, formation, holding, universe, freq, sel_sector, metrics_data):
        if not filepath or not metrics_data:
            return empty_figure(), empty_figure(), empty_figure(), empty_figure()

        net = load_network_pkl(filepath)
        times = net['times']
        T = net['T']
        sectors = metrics_data['sectors']

        # ---- 板块指标差分 ----
        density = np.array(metrics_data['density_disc'])
        centrality = np.array(metrics_data['centrality_disc'])
        delta_density = np.diff(density, axis=1)
        delta_centrality = np.diff(centrality, axis=1)
        diff_times = times[1:]

        # ---- 从缓存加载动量 ----
        cache_key = f'{universe}_{mom_type}_{formation}_H{holding}'
        mom_cache = _load_momentum_cache()

        if mom_cache is None:
            return (empty_figure('动量缓存不可用，请先运行 analysis/precompute_momentum.py'),
                    empty_figure(), empty_figure(), empty_figure())

        if cache_key not in mom_cache:
            return (empty_figure(f'缓存中无 {cache_key}，请运行 precompute_momentum.py'),
                    empty_figure(), empty_figure(), empty_figure())

        mom_sector_data = mom_cache[cache_key]

        # 对齐动量到 diff_times
        momentum_aligned = np.full((len(sectors), T - 1), np.nan)
        has_momentum = np.zeros(len(sectors), dtype=bool)
        for si, sec in enumerate(sectors):
            if sec not in mom_sector_data:
                continue
            md = mom_sector_data[sec]
            mom_dates = pd.to_datetime(md['dates'])
            mom_vals = np.array(md['momentum'])
            for ti, d in enumerate(diff_times):
                mask = mom_dates == pd.Timestamp(d)
                if mask.any():
                    momentum_aligned[si, ti] = mom_vals[mask][0]
                    has_momentum[si] = True

        # ---- 频率重采样 (统一用 times 对齐) ----
        if freq != 'D':
            density_rs, rd_times = _resample(density, times, freq, agg='last')
            centrality_rs, _ = _resample(centrality, times, freq, agg='last')
            delta_density = np.diff(density_rs, axis=1)
            delta_centrality = np.diff(centrality_rs, axis=1)
            # 动量也用同一时间网格重采样
            momentum_aligned, _ = _resample(momentum_aligned, diff_times, freq, agg='last')
            # 截断/补齐到同长度
            n_dd = delta_density.shape[1]
            n_mom = momentum_aligned.shape[1]
            n = min(n_dd, n_mom)
            delta_density = delta_density[:, :n]
            delta_centrality = delta_centrality[:, :n]
            momentum_aligned = momentum_aligned[:, :n]
            has_momentum = ~np.all(np.isnan(momentum_aligned), axis=1)
            diff_times = rd_times[1:1+n]

        # ---- z-score (仅用有动量数据的板块) ----
        def rz(arr):
            mu = np.nanmean(arr, axis=1, keepdims=True)
            sigma = np.nanstd(arr, axis=1, keepdims=True)
            sigma[sigma < 1e-10] = 1.0
            return (arr - mu) / sigma

        z_dd = rz(delta_density)
        z_dc = rz(delta_centrality)
        z_mom = np.where(has_momentum[:, None], rz(np.where(has_momentum[:, None], momentum_aligned, 0.0)), np.nan)
        rotation_score = (z_dd + z_dc + z_mom) / 3.0

        # 过滤有动量数据的板块
        valid_sectors = [s for i, s in enumerate(sectors) if has_momentum[i]]
        valid_indices = [i for i, s in enumerate(sectors) if has_momentum[i]]

        if not valid_sectors:
            return (empty_figure('所有板块均无足够动量数据'),
                    empty_figure(), empty_figure(), empty_figure())

        # 确保选中的板块有数据，否则用第一个有效板块
        plot_sector = sel_sector if sel_sector in valid_sectors else valid_sectors[0]
        psi = sectors.index(plot_sector)

        # ---- 图1: 板块动量 + 全市场聚合动量 ----
        fig_mom = go.Figure()

        # 全市场聚合 (从缓存加载)
        agg_key = f'{cache_key}_aggregate'
        if agg_key in mom_cache:
            agg_md = mom_cache[agg_key]
            agg_dates = pd.to_datetime(agg_md['dates'])
            agg_vals = np.array(agg_md['momentum'])
            if freq != 'D':
                df_agg = pd.DataFrame({'v': agg_vals}, index=agg_dates)
                grouper = pd.Grouper(freq='W-FRI') if freq == 'W' else pd.Grouper(freq='ME')
                rs = df_agg.groupby(grouper).last().dropna()
                agg_dates = rs.index; agg_vals = rs['v'].values
            agg_valid = ~np.isnan(agg_vals)
            fig_mom.add_trace(go.Scatter(
                x=agg_dates[agg_valid], y=agg_vals[agg_valid],
                mode='lines', name=f'全市场策略 ({universe})',
                line=dict(color='black', width=2.5)
            ))

        # 选中板块
        md = mom_sector_data[plot_sector]
        mom_dates = pd.to_datetime(md['dates'])
        mom_vals = np.array(md['momentum'])
        if freq != 'D':
            df_mom = pd.DataFrame({'v': mom_vals}, index=mom_dates)
            rs = df_mom.groupby(grouper).last().dropna()
            mom_dates = rs.index
            mom_vals = rs['v'].values
        valid = ~np.isnan(mom_vals)
        fig_mom.add_trace(go.Scatter(
            x=mom_dates[valid], y=mom_vals[valid],
            mode='lines', name=plot_sector, line=dict(color='#377EB8', width=1.2)
        ))
        fig_mom.add_hline(y=0, line=dict(color='gray', dash='dot'))
        fig_mom.update_layout(
            title=f'{plot_sector} 动量 (+ 全市场聚合)',
            hovermode='x unified', margin=dict(l=10, r=10, t=35, b=10), template='plotly_white')

        # ---- 图2: RotationScore ----
        fig_rs = go.Figure()
        for name, vals, color in [('z(ΔDensity)', z_dd[psi], '#E41A1C'),
                                   ('z(ΔCentrality)', z_dc[psi], '#377EB8'),
                                   ('z(Momentum)', z_mom[psi], '#4DAF4A')]:
            fig_rs.add_trace(go.Scatter(
                x=diff_times, y=vals, mode='lines', name=name,
                line=dict(color=color, width=1, dash='dot')))
        fig_rs.add_trace(go.Scatter(
            x=diff_times, y=rotation_score[psi], mode='lines',
            name='RotationScore', line=dict(color='black', width=2)))
        fig_rs.add_hline(y=0, line=dict(color='gray', dash='dot'))
        fig_rs.update_layout(
            title=f'{plot_sector} RotationScore',
            hovermode='x unified', margin=dict(l=10, r=10, t=35, b=10), template='plotly_white')

        # ---- 图3: 累积收益 (重叠并行投资) ----
        fig_cum = go.Figure()

        hold_d = HOLD_DAYS.get(holding, 21)
        freq_days = {'D': 1, 'W': 5, 'M': 21}.get(freq, 1)

        def _overlap_cumret(dates, vals, hold_days, step_days):
            """重叠并行累积收益。每个信号日均摊持有期收益到每日，叠加并累加。"""
            if len(vals) == 0 or step_days <= 0:
                return [], []
            n = len(dates)
            hold_positions = max(1, int(hold_days / step_days))
            daily_pnl = np.zeros(n)
            for i in range(n):
                if np.isnan(vals[i]):
                    continue
                daily_ret = vals[i] / hold_positions
                end = min(i + hold_positions, n)
                daily_pnl[i:end] += daily_ret
            cum = np.cumsum(daily_pnl)
            return list(dates), list(cum)

        # 板块
        cd, cv = _overlap_cumret(mom_dates[valid], mom_vals[valid], hold_d, freq_days)
        fig_cum.add_trace(go.Scatter(
            x=cd, y=cv, mode='lines',
            name=plot_sector, line=dict(color='#377EB8', width=1.5)
        ))

        # 全市场
        if agg_key in mom_cache:
            agg_md2 = mom_cache[agg_key]
            agg_d2 = pd.to_datetime(agg_md2['dates'])
            agg_v2 = np.array(agg_md2['momentum'])
            if freq != 'D':
                df2 = pd.DataFrame({'v': agg_v2}, index=agg_d2)
                rs2 = df2.groupby(grouper).last().dropna()
                agg_d2 = rs2.index; agg_v2 = rs2['v'].values
            av2 = ~np.isnan(agg_v2)
            cd2, cv2 = _overlap_cumret(agg_d2[av2], agg_v2[av2], hold_d, freq_days)
            fig_cum.add_trace(go.Scatter(
                x=cd2, y=cv2, mode='lines',
                name=f'全市场 ({universe})', line=dict(color='black', width=2.0)
            ))

        fig_cum.add_hline(y=0, line=dict(color='gray', dash='dot'))
        fig_cum.update_layout(
            title=f'累积收益 (重叠并行, {holding}持有)',
            hovermode='x unified', margin=dict(l=10, r=10, t=35, b=10), template='plotly_white')

        # ---- 图4: 动量 vs NetworkShock 对比 ----
        fig_ns = _make_ns_compare(net, mom_cache, agg_key, freq, cache_key)
        # 如果 NS 不可用则跳过
        if fig_ns is None:
            fig_ns = empty_figure('NetworkShock 数据不可用')

        # ---- 图5: 热力图 (仅显示有数据的板块) ----
        rs_filtered = rotation_score[valid_indices]
        fig_hm = go.Figure(data=go.Heatmap(
            z=rs_filtered, x=[str(d)[:10] for d in diff_times], y=valid_sectors,
            colorscale='RdBu_r', zmid=0, colorbar=dict(title='RS')))
        fig_hm.update_layout(
            title=f'RotationScore 板块×时间 ({len(valid_sectors)}/{len(sectors)} 板块有动量)',
            height=450, margin=dict(l=10, r=10, t=35, b=80), template='plotly_white',
            xaxis=dict(tickangle=-45))

        return fig_mom, fig_rs, fig_cum, fig_ns, fig_hm
