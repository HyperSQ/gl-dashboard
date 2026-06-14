"""Tab 1: 时间序列总览 — Jaccard/L1 + NetworkShock + 5分量 (实时计算)"""

from dash import html, dcc, Input, Output, callback, State
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.loader import load_network_pkl, compute_networkshock_from_data
from components.plots import (
    make_jaccard_l1_plot, make_component_subplots, empty_figure
)
import plotly.graph_objects as go


def layout():
    return html.Div([
        html.H4('时间序列总览', className='mt-3 mb-3'),

        html.Div([
            html.Label('数据文件'),
            dcc.Dropdown(id='ov-file-dropdown', clearable=False, className='mb-3',
                         style={'max-width': '500px'}),
        ]),

        # 图 1: Jaccard & L1
        dcc.Loading(dcc.Graph(id='ov-jaccard-l1-plot', style={'height': '280px'}), type='circle'),

        # 图 2: NetworkShock 复合
        html.H5('NetworkShock — 五分量 z-score 等权加总', className='mt-3'),
        dcc.Loading(dcc.Graph(id='ov-ns-composite-plot', style={'height': '280px'}), type='circle'),

        # 图 3: 5 分量子图
        html.H5('五分量细节 (z-score 标准化)', className='mt-3'),
        dcc.Loading(dcc.Graph(id='ov-components-plot', style={'height': '620px'}), type='circle'),

        dcc.Store(id='ov-ns-data-store'),
    ])


def register_callbacks(app, pkl_files, pkl_meta_map):

    @callback(
        Output('ov-file-dropdown', 'options'),
        Output('ov-file-dropdown', 'value'),
        Input('ov-file-dropdown', 'id'),
    )
    def init_dropdown(_):
        options = [{'label': pkl_meta_map[f]['label'], 'value': f} for f in pkl_files]
        return options, pkl_files[0] if pkl_files else None

    # ---- 文件选择 → 加载数据 + 计算 NetworkShock ----
    @callback(
        Output('ov-ns-data-store', 'data'),
        Output('ov-jaccard-l1-plot', 'figure'),
        Output('ov-ns-composite-plot', 'figure'),
        Output('ov-components-plot', 'figure'),
        Input('ov-file-dropdown', 'value'),
    )
    def on_file_change(filepath):
        if not filepath:
            return None, empty_figure(), empty_figure(), empty_figure()

        net = load_network_pkl(filepath)
        times = net['times']
        T = net['T']

        # Jaccard & L1
        fig_jl = make_jaccard_l1_plot(times, net['jaccard'], net['l1_penalty'])

        # NetworkShock 实时计算
        ns = compute_networkshock_from_data(
            net['A'], net['Theta'], net['jaccard'], net['l1_groups']
        )
        ns_serial = {k: v.tolist() if hasattr(v, 'tolist') else v
                     for k, v in ns.items()}
        ns_times = times[1:]  # NS 从 t=1 开始

        # NetworkShock 复合
        ns_vals = np.array(ns['NetworkShock'])
        fig_ns = go.Figure()
        fig_ns.add_trace(go.Scatter(
            x=ns_times, y=ns_vals, mode='lines', name='NetworkShock',
            line=dict(color='#E41A1C', width=1.2)
        ))
        fig_ns.add_hline(y=0, line=dict(color='gray', dash='dot', width=0.5))
        fig_ns.update_layout(
            title='NetworkShock (五分量 z-score 等权加总)',
            hovermode='x unified',
            margin=dict(l=10, r=10, t=35, b=10),
            template='plotly_white',
        )

        # 五分量
        components = {
            '1-Jaccard': np.array(ns['z_1-Jaccard']),
            'L1': np.array(ns['z_L1']),
            'EdgeTurnover': np.array(ns['z_EdgeTurnover']),
            'DeltaDensity': np.array(ns['z_DeltaDensity']),
            'CentDisp': np.array(ns['z_CentDisp']),
        }
        fig_comp = make_component_subplots(ns_times, components)

        return ns_serial, fig_jl, fig_ns, fig_comp
