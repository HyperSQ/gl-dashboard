"""Tab 2: 板块分析 — 密度 / 中心性 / 跨板块连接（无滑块/无线版）"""

from dash import html, dcc, Input, Output, callback, State
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_network_pkl, compute_sector_metrics
from components.plots import (
    make_sector_density_plot, make_sector_centrality_plot,
    make_cross_conn_plot, empty_figure
)


def layout():
    return html.Div([
        html.H4('板块分析', className='mt-3 mb-3'),

        html.Div([
            html.Div([
                html.Label('数据文件'),
                dcc.Dropdown(id='sec-file-dropdown', clearable=False, className='mb-2'),
            ], style={'flex': '1'}),
            html.Div([
                html.Label('板块 A'),
                dcc.Dropdown(id='sec-sector-a', clearable=False, className='mb-2'),
            ], style={'flex': '1', 'padding-left': '10px'}),
            html.Div([
                html.Label('板块 B'),
                dcc.Dropdown(id='sec-sector-b', clearable=False, className='mb-2'),
            ], style={'flex': '1', 'padding-left': '10px'}),
        ], style={'display': 'flex'}),

        html.Div(id='sec-no-sectors-msg', style={'color': 'gray', 'text-align': 'center'}),

        # Row 1: 密度 A + B
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='sec-density-a', style={'height': '280px'}), type='circle'), style={'flex': '1'}),
            html.Div(dcc.Loading(dcc.Graph(id='sec-density-b', style={'height': '280px'}), type='circle'), style={'flex': '1'}),
        ], style={'display': 'flex'}),

        # Row 2: 中心性 A + B
        html.Div([
            html.Div(dcc.Loading(dcc.Graph(id='sec-cent-a', style={'height': '280px'}), type='circle'), style={'flex': '1'}),
            html.Div(dcc.Loading(dcc.Graph(id='sec-cent-b', style={'height': '280px'}), type='circle'), style={'flex': '1'}),
        ], style={'display': 'flex'}),

        # Row 3: 跨板块连接
        dcc.Loading(dcc.Graph(id='sec-cross-conn', style={'height': '300px'}), type='circle'),

        dcc.Store(id='sec-metrics-store'),
    ])


def register_callbacks(app, pkl_files, pkl_meta_map):

    @callback(
        Output('sec-file-dropdown', 'options'),
        Output('sec-file-dropdown', 'value'),
        Input('sec-file-dropdown', 'id'),
    )
    def init_dropdown(_):
        options = [{'label': pkl_meta_map[f]['label'], 'value': f} for f in pkl_files]
        return options, pkl_files[0] if pkl_files else None

    # ---- 文件变化 → 加载数据 & 更新板块下拉框 ----
    @callback(
        Output('sec-metrics-store', 'data'),
        Output('sec-sector-a', 'options'),
        Output('sec-sector-a', 'value'),
        Output('sec-sector-b', 'options'),
        Output('sec-sector-b', 'value'),
        Output('sec-no-sectors-msg', 'children'),
        Input('sec-file-dropdown', 'value'),
        State('sec-sector-a', 'value'),
        State('sec-sector-b', 'value'),
    )
    def on_file_change(filepath, old_a, old_b):
        if not filepath:
            return None, [], None, [], None, ''

        net = load_network_pkl(filepath)
        if not net['l1_groups']:
            return None, [], None, [], None, \
                '当前文件不含行业分类信息 (l1_groups)，板块分析不可用。'

        metrics = compute_sector_metrics(net['A'], net['Theta'], net['l1_groups'])
        sectors = metrics['sectors']

        metrics_serial = {
            'sectors': sectors,
            'n_sectors': metrics['n_sectors'],
            'density_disc': metrics['density_disc'].tolist(),
            'density_cont': metrics['density_cont'].tolist(),
            'centrality_disc': metrics['centrality_disc'].tolist(),
            'centrality_cont': metrics['centrality_cont'].tolist(),
            'cross_conn_disc': metrics['cross_conn_disc'].tolist(),
            'cross_conn_cont': metrics['cross_conn_cont'].tolist(),
        }

        sector_opts = [{'label': s, 'value': s} for s in sectors]
        sel_a = old_a if old_a in sectors else sectors[0]
        sel_b = old_b if old_b in sectors else sectors[min(1, len(sectors) - 1)]

        return metrics_serial, sector_opts, sel_a, sector_opts, sel_b, None

    # ---- 密度 A (依赖: 文件 + 板块A) ----
    @callback(
        Output('sec-density-a', 'figure'),
        Input('sec-file-dropdown', 'value'),
        Input('sec-sector-a', 'value'),
        State('sec-metrics-store', 'data'),
    )
    def update_density_a(filepath, sector, metrics):
        return _build_sector_figure(filepath, metrics, sector, 'density', make_sector_density_plot)

    # ---- 密度 B ----
    @callback(
        Output('sec-density-b', 'figure'),
        Input('sec-file-dropdown', 'value'),
        Input('sec-sector-b', 'value'),
        State('sec-metrics-store', 'data'),
    )
    def update_density_b(filepath, sector, metrics):
        return _build_sector_figure(filepath, metrics, sector, 'density', make_sector_density_plot)

    # ---- 中心性 A ----
    @callback(
        Output('sec-cent-a', 'figure'),
        Input('sec-file-dropdown', 'value'),
        Input('sec-sector-a', 'value'),
        State('sec-metrics-store', 'data'),
    )
    def update_cent_a(filepath, sector, metrics):
        return _build_sector_figure(filepath, metrics, sector, 'centrality', make_sector_centrality_plot)

    # ---- 中心性 B ----
    @callback(
        Output('sec-cent-b', 'figure'),
        Input('sec-file-dropdown', 'value'),
        Input('sec-sector-b', 'value'),
        State('sec-metrics-store', 'data'),
    )
    def update_cent_b(filepath, sector, metrics):
        return _build_sector_figure(filepath, metrics, sector, 'centrality', make_sector_centrality_plot)

    # ---- 跨板块连接 (依赖: 文件 + 板块A + 板块B) ----
    @callback(
        Output('sec-cross-conn', 'figure'),
        Input('sec-file-dropdown', 'value'),
        Input('sec-sector-a', 'value'),
        Input('sec-sector-b', 'value'),
        State('sec-metrics-store', 'data'),
    )
    def update_cross(filepath, sec_a, sec_b, metrics):
        if not filepath or not metrics or not sec_a or not sec_b:
            return empty_figure()

        net = load_network_pkl(filepath)
        sectors = metrics['sectors']
        si = sectors.index(sec_a)
        sj = sectors.index(sec_b)
        if si > sj:
            si, sj = sj, si

        disc = np.array(metrics['cross_conn_disc'][si][sj])
        cont = np.array(metrics['cross_conn_cont'][si][sj])
        return make_cross_conn_plot(net['times'], disc, cont, None, sec_a, sec_b)


def _build_sector_figure(filepath, metrics, sector, metric_type, plot_func):
    if not filepath or not metrics or not sector:
        return empty_figure()

    net = load_network_pkl(filepath)
    sectors = metrics['sectors']
    if sector not in sectors:
        return empty_figure()

    si = sectors.index(sector)
    disc = np.array(metrics[f'{metric_type}_disc'][si])
    cont = np.array(metrics[f'{metric_type}_cont'][si])
    return plot_func(net['times'], disc, cont, None, sector)
