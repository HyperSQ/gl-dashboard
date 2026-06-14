"""Tab 3: 矩阵热力图 — 邻接矩阵 A + 精度矩阵 Theta"""

from dash import html, dcc, Input, Output, callback, State
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_network_pkl
from components.plots import make_adjacency_heatmap, make_theta_heatmap, empty_figure


def layout():
    return html.Div([
        html.H4('矩阵热力图', className='mt-3 mb-3'),

        html.Div([
            html.Div([
                html.Label('数据文件'),
                dcc.Dropdown(id='mat-file-dropdown', clearable=False, className='mb-2'),
            ], style={'flex': '3'}),
            html.Div([
                html.Label('时间进度'),
                dcc.Slider(id='mat-time-slider', min=0, max=100, value=0, step=1,
                           tooltip={'placement': 'bottom'}),
            ], style={'flex': '7', 'padding-left': '20px'}),
        ], style={'display': 'flex', 'align-items': 'center'}),

        html.Div(id='mat-time-label', style={'text-align': 'center', 'font-weight': 'bold',
                                              'margin-bottom': '15px'}),

        # 双图并排
        html.Div([
            html.Div(dcc.Loading(
                dcc.Graph(id='mat-adj-plot', style={'height': '500px'},
                          config={'scrollZoom': False}),
                type='circle'
            ), style={'flex': '1'}),
            html.Div(dcc.Loading(
                dcc.Graph(id='mat-theta-plot', style={'height': '500px'},
                          config={'scrollZoom': False}),
                type='circle'
            ), style={'flex': '1'}),
        ], style={'display': 'flex'}),

        # 存储
        dcc.Store(id='mat-filepath-store'),
        dcc.Store(id='mat-meta-store'),  # 存 names, l1_groups, times
    ])


def register_callbacks(app, pkl_files, pkl_meta_map):

    @callback(
        Output('mat-file-dropdown', 'options'),
        Output('mat-file-dropdown', 'value'),
        Input('mat-file-dropdown', 'id'),
    )
    def init_dropdown(_):
        options = [{'label': pkl_meta_map[f]['label'], 'value': f} for f in pkl_files]
        return options, pkl_files[0] if pkl_files else None

    # ---- 文件变化 → 加载元数据 ----
    @callback(
        Output('mat-filepath-store', 'data'),
        Output('mat-meta-store', 'data'),
        Output('mat-time-slider', 'max'),
        Output('mat-time-slider', 'value'),
        Input('mat-file-dropdown', 'value'),
        State('mat-time-slider', 'value'),
    )
    def on_file_change(filepath, old_slider):
        if not filepath:
            return None, None, 100, 0
        net = load_network_pkl(filepath)
        T = net['T']
        new_val = min(old_slider or 0, T - 1)

        meta = {
            'names': net['names'],
            'l1_groups': net['l1_groups'],
            'T': T,
            'p': net['p'],
        }
        return filepath, meta, T - 1, new_val

    # ---- 时间标签 ----
    @callback(
        Output('mat-time-label', 'children'),
        Input('mat-time-slider', 'value'),
        State('mat-filepath-store', 'data'),
        prevent_initial_call=True,
    )
    def update_label(idx, filepath):
        if not filepath:
            return ''
        idx = int(idx)
        net = load_network_pkl(filepath)
        if idx < net['T']:
            return f'当前时间: {str(net["times"][idx])[:10]}'
        return ''

    # ---- 邻接矩阵 (懒加载：仅当前时间点) ----
    @callback(
        Output('mat-adj-plot', 'figure'),
        Input('mat-time-slider', 'value'),
        State('mat-filepath-store', 'data'),
        State('mat-meta-store', 'data'),
    )
    def update_adj(idx, filepath, meta):
        if not filepath:
            return empty_figure()
        idx = int(idx)
        net = load_network_pkl(filepath)
        if idx >= net['T']:
            return empty_figure('索引越界')

        A = net['A'][idx]
        Theta = net['Theta'][idx]
        date_str = str(net['times'][idx])[:10]
        return make_adjacency_heatmap(A, Theta, net['l1_groups'],
                                      f'邻接矩阵 A ({date_str})')

    # ---- 精度矩阵 ----
    @callback(
        Output('mat-theta-plot', 'figure'),
        Input('mat-time-slider', 'value'),
        State('mat-filepath-store', 'data'),
        State('mat-meta-store', 'data'),
    )
    def update_theta(idx, filepath, meta):
        if not filepath:
            return empty_figure()
        idx = int(idx)
        net = load_network_pkl(filepath)
        if idx >= net['T']:
            return empty_figure('索引越界')

        Theta = net['Theta'][idx]
        date_str = str(net['times'][idx])[:10]
        return make_theta_heatmap(Theta, net['l1_groups'],
                                  f'精度矩阵 Theta ({date_str})')
