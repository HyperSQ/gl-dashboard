"""Tab 5: 整协分析 — ADF / Johansen / EG 检验"""

from dash import html, dcc, Input, Output, callback, State, dash_table
import numpy as np
import pandas as pd
import pickle
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_network_pkl, compute_sector_metrics, compute_networkshock_from_data
from utils.momentum import HOLD_DAYS, compute_aggregate_momentum
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import plotly.graph_objects as go


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
UNIVERSE_OPTIONS = [
    {'label': '当前组合 (95只)', 'value': 'network'},
    {'label': 'CSI300全市场 (690只)', 'value': 'union'},
]
FREQ_OPTIONS_ALL = [
    {'label': '日度', 'value': 'D'},
    {'label': '周度', 'value': 'W'},
    {'label': '月度', 'value': 'M'},
]
FREQ_ORDER = {'D': 0, 'W': 1, 'M': 2}

SERIES_OPTIONS = [
    {'label': 'NetworkShock (复合)', 'value': 'NetworkShock'},
    {'label': '1 - Jaccard', 'value': '1-Jaccard'},
    {'label': 'L1 Distance', 'value': 'L1'},
    {'label': 'Edge Turnover', 'value': 'EdgeTurnover'},
    {'label': 'Delta Density', 'value': 'DeltaDensity'},
    {'label': 'Centrality Dispersion', 'value': 'CentDisp'},
    {'label': '全市场动量', 'value': 'Momentum'},
]


def _allowed_freqs(native_freq):
    min_level = FREQ_ORDER.get(native_freq, 0)
    return [o for o in FREQ_OPTIONS_ALL if FREQ_ORDER[o['value']] >= min_level]


def _load_momentum_cache():
    try:
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def _get_mom_sector_data(cache, key):
    """从缓存取动量数据，int32 日期 → datetime 按需转换。"""
    if cache is None or key not in cache:
        return None
    entry = cache[key]
    if isinstance(entry, dict) and 'dates' in entry and 'momentum' in entry:
        d = entry['dates']
        dates = pd.to_datetime([str(np.datetime64(int(x), 'D')) for x in d]) if hasattr(d, 'dtype') and d.dtype == np.dtype('int32') else pd.to_datetime(d)
        return {'dates': dates, 'momentum': np.array(entry['momentum'], dtype=float)}
    result = {}
    for sector, sv in entry.items():
        if isinstance(sv, dict) and 'dates' in sv:
            d = sv['dates']
            dates = pd.to_datetime([str(np.datetime64(int(x), 'D')) for x in d]) if hasattr(d, 'dtype') and d.dtype == np.dtype('int32') else pd.to_datetime(sv['dates'])
            result[sector] = {'dates': dates, 'momentum': np.array(sv['momentum'], dtype=float)}
    return result


def layout():
    return html.Div([
        html.H4('整协分析', className='mt-3 mb-3'),

        # 控制行1
        html.Div([
            html.Div([html.Label('数据文件'),
                      dcc.Dropdown(id='coin-file-dropdown', clearable=False, className='mb-2')],
                     style={'flex': '2'}),
            html.Div([html.Label('动量类型'),
                      dcc.Dropdown(id='coin-mom-type', options=MOM_TYPE_OPTIONS, value='cross', clearable=False)],
                     style={'flex': '1', 'padding-left': '10px'}),
            html.Div([html.Label('形成期'),
                      dcc.Dropdown(id='coin-formation', options=FORMATION_OPTIONS, value='1M', clearable=False)],
                     style={'flex': '1', 'padding-left': '10px'}),
            html.Div([html.Label('持有期'),
                      dcc.Dropdown(id='coin-holding', options=HOLDING_OPTIONS, value='1M', clearable=False)],
                     style={'flex': '1', 'padding-left': '10px'}),
        ], style={'display': 'flex', 'margin-bottom': '10px'}),

        # 控制行2
        html.Div([
            html.Div([html.Label('股票范围'),
                      dcc.Dropdown(id='coin-universe', options=UNIVERSE_OPTIONS, value='network', clearable=False)],
                     style={'flex': '1'}),
            html.Div([html.Label('频率'),
                      dcc.Dropdown(id='coin-freq', options=FREQ_OPTIONS_ALL, value='D', clearable=False)],
                     style={'flex': '1', 'padding-left': '10px'}),
            html.Div([html.Label('板块'),
                      dcc.Dropdown(id='coin-sector', clearable=False)],
                     style={'flex': '1', 'padding-left': '10px'}),
        ], style={'display': 'flex', 'margin-bottom': '10px'}),

        # 序列选择
        html.Label('参与检验的序列 (≥2):'),
        dcc.Checklist(id='coin-series', options=SERIES_OPTIONS,
                      value=['NetworkShock', 'Momentum'], inline=True,
                      className='mb-3'),

        # ---- ADF 表 ----
        html.H6('ADF 单位根检验'),
        html.Div(id='coin-adf-table', className='mb-3'),

        # ---- Johansen ----
        html.H6('Johansen 协整检验'),
        html.Div(id='coin-johansen-table', className='mb-3'),

        # ---- 残差图 ----
        dcc.Loading(dcc.Graph(id='coin-resid-plot', style={'height': '300px'}), type='circle'),

        # ---- EG 配对热力图 ----
        html.H6('Engle-Granger 配对检验 (p值)'),
        dcc.Loading(dcc.Graph(id='coin-eg-heatmap', style={'height': '400px'}), type='circle'),

        dcc.Store(id='coin-data-store'),
    ])


def register_callbacks(app, pkl_files, pkl_meta_map):

    @callback(
        Output('coin-file-dropdown', 'options'),
        Output('coin-file-dropdown', 'value'),
        Input('coin-file-dropdown', 'id'),
    )
    def init_dropdown(_):
        options = [{'label': pkl_meta_map[f]['label'], 'value': f} for f in pkl_files]
        return options, pkl_files[0] if pkl_files else None

    # ---- 文件变化 → 更新板块 + 频率 ----
    @callback(
        Output('coin-sector', 'options'),
        Output('coin-sector', 'value'),
        Output('coin-freq', 'options'),
        Output('coin-freq', 'value'),
        Input('coin-file-dropdown', 'value'),
        State('coin-sector', 'value'),
        State('coin-freq', 'value'),
    )
    def on_file_change(filepath, old_sector, old_freq):
        from config import parse_filename
        pkl_meta = parse_filename(filepath) if filepath else {}
        native_f = pkl_meta.get('freq', 'D')
        freq_opts = _allowed_freqs(native_f)
        use_freq = old_freq if old_freq and any(o['value'] == old_freq for o in freq_opts) else freq_opts[0]['value']

        if not filepath:
            return [], None, freq_opts, use_freq
        net = load_network_pkl(filepath)
        if not net['l1_groups']:
            return [], None, freq_opts, use_freq
        sectors = sorted(set(net['l1_groups']))
        opts = [{'label': s, 'value': s} for s in sectors]
        sel = old_sector if old_sector in sectors else sectors[0]
        return opts, sel, freq_opts, use_freq

    # ---- 主计算回调 ----
    @callback(
        Output('coin-adf-table', 'children'),
        Output('coin-johansen-table', 'children'),
        Output('coin-resid-plot', 'figure'),
        Output('coin-eg-heatmap', 'figure'),
        Input('coin-file-dropdown', 'value'),
        Input('coin-mom-type', 'value'),
        Input('coin-formation', 'value'),
        Input('coin-holding', 'value'),
        Input('coin-universe', 'value'),
        Input('coin-freq', 'value'),
        Input('coin-sector', 'value'),
        Input('coin-series', 'value'),
    )
    def run_analysis(filepath, mom_type, formation, holding, universe, freq,
                     sel_sector, selected_series):
        if not filepath or not selected_series or len(selected_series) < 2:
            return '', '', {}, {}

        # 确保频率不低于 pkl 原生频率
        from config import parse_filename
        native_f = parse_filename(filepath).get('freq', 'D')
        if FREQ_ORDER.get(freq, 0) < FREQ_ORDER.get(native_f, 0):
            freq = native_f

        net = load_network_pkl(filepath)
        times_all = net['times']

        # ---- 频率降采样 ----
        if freq == 'D':
            idx_keep = list(range(len(times_all)))
        else:
            dates = pd.DatetimeIndex(times_all)
            grouper = pd.Grouper(freq='W-FRI') if freq == 'W' else pd.Grouper(freq='ME')
            s = pd.Series(range(len(dates)), index=dates)
            idx_keep = sorted(s.groupby(grouper).last().dropna().astype(int).values)

        A_use = net['A'][idx_keep]
        Theta_use = net['Theta'][idx_keep]
        j_use = net['jaccard'][idx_keep]
        ns = compute_networkshock_from_data(A_use, Theta_use, j_use, net['l1_groups'])
        ns_times = [times_all[i] for i in idx_keep[1:]]

        # ---- 动量 ----
        mom_cache = _load_momentum_cache()
        agg_key = f'{universe}_{mom_type}_{formation}_H{holding}_aggregate'
        mom_vals_aligned = np.full(len(ns_times), np.nan)
        if mom_cache and agg_key in mom_cache:
            agg = _get_mom_sector_data(mom_cache, agg_key)
            agg_dates = agg['dates']
            agg_vals = agg['momentum']
            if freq != 'D':
                df_a = pd.DataFrame({'v': agg_vals}, index=agg_dates)
                grouper = pd.Grouper(freq='W-FRI') if freq == 'W' else pd.Grouper(freq='ME')
                rs_a = df_a.groupby(grouper).last().dropna()
                agg_dates = rs_a.index; agg_vals = rs_a['v'].values
            mom_date_set = {str(pd.Timestamp(ad).date()): v for ad, v in zip(agg_dates, agg_vals)}
            for i, d in enumerate(ns_times):
                dk = str(pd.Timestamp(d).date())
                if dk in mom_date_set:
                    mom_vals_aligned[i] = mom_date_set[dk]

        # ---- 构建序列字典 ----
        series_dict = {}
        for key in selected_series:
            if key == 'Momentum':
                series_dict[key] = mom_vals_aligned
            elif key == 'NetworkShock':
                series_dict[key] = ns['NetworkShock']
            elif key == '1-Jaccard':
                series_dict[key] = ns['1-Jaccard']
            elif key == 'L1':
                series_dict[key] = ns['L1']
            elif key == 'EdgeTurnover':
                series_dict[key] = ns['EdgeTurnover']
            elif key == 'DeltaDensity':
                series_dict[key] = ns['DeltaDensity']
            elif key == 'CentDisp':
                series_dict[key] = ns['CentDisp']

        # 剔除 NaN
        valid = np.ones(len(ns_times), dtype=bool)
        for v in series_dict.values():
            valid &= ~np.isnan(v)
        if valid.sum() < 30:
            return '有效数据点不足 (<30)', '', {}, {}

        data_matrix = np.column_stack([series_dict[k][valid] for k in selected_series])
        n_obs = data_matrix.shape[0]

        # ==================== ADF ====================
        adf_rows = []
        for i, name in enumerate(selected_series):
            try:
                result = adfuller(data_matrix[:, i], autolag='AIC', maxlag=int(np.sqrt(n_obs)))
                adf_rows.append({
                    '序列': name,
                    'ADF统计量': f'{result[0]:.4f}',
                    'p值': f'{result[1]:.4f}',
                    '1%': f'{result[4]["1%"]:.4f}',
                    '5%': f'{result[4]["5%"]:.4f}',
                    '10%': f'{result[4]["10%"]:.4f}',
                    '结论': 'I(0)' if result[1] < 0.05 else '疑似I(1)',
                })
            except Exception as e:
                adf_rows.append({'序列': name, 'ADF统计量': 'N/A', 'p值': 'N/A', '1%': 'N/A', '5%': 'N/A', '10%': 'N/A', '结论': f'失败:{str(e)[:30]}'})

        adf_table = dash_table.DataTable(
            data=adf_rows,
            columns=[{'name': c, 'id': c} for c in ['序列', 'ADF统计量', 'p值', '1%', '5%', '10%', '结论']],
            style_cell={'textAlign': 'center', 'fontSize': '12px'},
            style_header={'fontWeight': 'bold'},
        )

        # ==================== Johansen ====================
        johansen_text = ''
        fig_resid = go.Figure()
        try:
            jres = coint_johansen(data_matrix, det_order=0, k_ar_diff=1)
            trace_rows = []
            for r in range(len(selected_series)):
                trace_rows.append({
                    '秩 r': str(r),
                    '迹统计量': f'{jres.lr1[r]:.3f}',
                    '5%临界值': f'{jres.cvt[r, 1]:.3f}',
                    '最大特征值': f'{jres.lr2[r]:.3f}',
                    '5%临界值(λmax)': f'{jres.cvm[r, 1]:.3f}',
                })
            johansen_table = dash_table.DataTable(
                data=trace_rows,
                columns=[{'name': c, 'id': c} for c in ['秩 r', '迹统计量', '5%临界值', '最大特征值', '5%临界值(λmax)']],
                style_cell={'textAlign': 'center', 'fontSize': '12px'},
                style_header={'fontWeight': 'bold'},
            )
            # 残差: 第一个协整向量
            cv = jres.evec[:, 0]
            spread = data_matrix @ cv
            spread = spread / np.std(spread)
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(
                x=list(range(len(spread))), y=spread, mode='lines',
                name='协整残差', line=dict(color='#377EB8', width=1.2)))
            fig_resid.add_hline(y=2, line=dict(color='red', dash='dot'))
            fig_resid.add_hline(y=-2, line=dict(color='red', dash='dot'))
            fig_resid.add_hline(y=0, line=dict(color='gray', dash='dot'))
            fig_resid.update_layout(
                title=f'第一个协整向量残差 (±2σ)',
                margin=dict(l=10, r=10, t=35, b=10), template='plotly_white')
        except Exception as e:
            johansen_text = f'Johansen 检验失败: {e}'
            johansen_table = html.Div(johansen_text)

        # ==================== EG 配对 ====================
        n_series = len(selected_series)
        eg_pvals = np.ones((n_series, n_series))
        for i in range(n_series):
            for j in range(i + 1, n_series):
                try:
                    _, pval, _ = coint(data_matrix[:, i], data_matrix[:, j])
                    eg_pvals[i, j] = pval
                    eg_pvals[j, i] = pval
                except Exception:
                    pass
            eg_pvals[i, i] = np.nan

        fig_eg = go.Figure(data=go.Heatmap(
            z=eg_pvals, x=selected_series, y=selected_series,
            colorscale='RdYlGn_r', zmin=0, zmax=1,
            text=np.where(np.isnan(eg_pvals), '', np.round(eg_pvals, 3)),
            texttemplate='%{text}', colorbar=dict(title='p值'),
        ))
        fig_eg.update_layout(
            title='Engle-Granger 配对协整检验 (p值: 绿<0.05 显著)',
            margin=dict(l=10, r=10, t=35, b=10), template='plotly_white')

        return adf_table, johansen_table, fig_resid, fig_eg
