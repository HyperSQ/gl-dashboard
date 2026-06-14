"""Plotly 图表工厂函数 — 所有图表在此构建"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def empty_figure(message='无数据'):
    return go.Figure().add_annotation(
        text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=14))


# ============================================================
# 通用: 双Y轴时间序列 + 竖线
# ============================================================

def make_dual_axis_plot(dates, v1, v2, curr_date, title='',
                        name1='', name2='', color1='#2b8cbe', color2='#d95f02'):
    """双 Y 轴折线图，带当前时间竖线。"""
    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(
        go.Scatter(x=dates, y=v1, mode='lines', name=name1,
                   line=dict(color=color1, width=1.2)),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=dates, y=v2, mode='lines', name=name2,
                   line=dict(color=color2, width=1.2)),
        secondary_y=True
    )

    # 竖线
    if curr_date is not None:
        fig.add_vline(x=curr_date, line=dict(color='#de2d26', dash='dash', width=1.5))

    fig.update_layout(
        title=title,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=35, b=10),
        template='plotly_white',
    )
    fig.update_yaxes(title_text=name1, title_font_color=color1, secondary_y=False)
    fig.update_yaxes(title_text=name2, title_font_color=color2, secondary_y=True)

    return fig


# ============================================================
# Jaccard & L1 (兼容原 app)
# ============================================================

def make_jaccard_l1_plot(dates, jaccard, l1, curr_date=None):
    return make_dual_axis_plot(
        dates, jaccard, l1, curr_date,
        title='Jaccard Index & L1 Distance',
        name1='Jaccard', name2='L1', color1='#2b8cbe', color2='#d95f02')


# ============================================================
# NetworkShock 5 分量面板
# ============================================================

def make_component_subplots(times, components, curr_date=None):
    """5 行子图: (1-Jaccard, L1, edge_turnover, delta_density, centrality_dispersion)"""
    comp_keys = ['1-Jaccard', 'L1', 'EdgeTurnover', 'DeltaDensity', 'CentDisp']
    comp_names = {
        '1-Jaccard': '1 - Jaccard',
        'L1': 'L1 Distance',
        'EdgeTurnover': 'Edge Turnover',
        'DeltaDensity': 'Delta Density',
        'CentDisp': 'Centrality Dispersion',
    }
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3']

    fig = make_subplots(rows=5, cols=1, shared_xaxes=True,
                        vertical_spacing=0.04,
                        subplot_titles=[comp_names[k] for k in comp_keys])

    for i, (key, color) in enumerate(zip(comp_keys, colors)):
        vals = components.get(key)
        if vals is None:
            continue
        fig.add_trace(
            go.Scatter(x=times, y=vals, mode='lines', name=comp_names[key],
                       line=dict(color=color, width=0.8), showlegend=False),
            row=i + 1, col=1
        )
        fig.add_hline(y=0, line=dict(color='gray', dash='dot', width=0.5),
                      row=i + 1, col=1)
        if curr_date is not None:
            fig.add_vline(x=curr_date, line=dict(color='#de2d26', dash='dash',
                          width=1.0), row=i + 1, col=1)

    fig.update_layout(
        height=600,
        margin=dict(l=10, r=10, t=30, b=10),
        template='plotly_white',
        hovermode='x unified',
    )
    return fig


# ============================================================
# NetworkShock 复合指标 (多配置叠加)
# ============================================================

def make_networkshock_plot(ns_data, curr_date=None):
    """NetworkShock 复合指标时序图 (C1-C4 叠加)"""
    fig = go.Figure()

    configs = [
        ('C1_min_D_W5', 'min→D W=5', '#E41A1C'),
        ('C2_min_W_W1', 'min→W W=1', '#377EB8'),
        ('C3_DK_W_W12', 'DK→W W=12', '#4DAF4A'),
        ('C4_min_W_W12', 'min→W W=12', '#FF7F00'),
    ]

    for cfg_key, label, color in configs:
        ns_key = f'{cfg_key}_ns'
        times_key = f'{cfg_key}_times'
        if ns_key in ns_data and times_key in ns_data:
            fig.add_trace(go.Scatter(
                x=ns_data[times_key], y=ns_data[ns_key],
                mode='lines', name=label,
                line=dict(color=color, width=1.0, dash='solid'),
                opacity=0.8
            ))

    fig.add_hline(y=0, line=dict(color='gray', dash='dot', width=0.5))
    if curr_date is not None:
        fig.add_vline(x=curr_date, line=dict(color='#de2d26', dash='dash', width=1.5))

    fig.update_layout(
        title='NetworkShock (4 Configurations)',
        hovermode='x unified',
        legend=dict(orientation='h', y=1.02),
        margin=dict(l=10, r=10, t=35, b=10),
        template='plotly_white',
    )
    return fig


# ============================================================
# 板块: 密度 / 中心性 / 跨连接
# ============================================================

def make_sector_density_plot(times, density_disc, density_cont, curr_date,
                             sector_name=''):
    return make_dual_axis_plot(
        times, density_disc, density_cont, curr_date,
        title=f'{sector_name} 内部连接密度',
        name1='离散 (A)', name2='连续 (|Theta|)',
        color1='#e41a1c', color2='#377eb8')


def make_sector_centrality_plot(times, cent_disc, cent_cont, curr_date,
                                sector_name=''):
    return make_dual_axis_plot(
        times, cent_disc, cent_cont, curr_date,
        title=f'{sector_name} 外部 Degree Centrality',
        name1='离散 (A)', name2='连续 (|Theta|)',
        color1='#e41a1c', color2='#377eb8')


def make_cross_conn_plot(times, disc, cont, curr_date,
                         sector_a='', sector_b=''):
    return make_dual_axis_plot(
        times, disc, cont, curr_date,
        title=f'{sector_a} × {sector_b} 板块间连接性',
        name1='离散 (A)', name2='连续 (|Theta|)',
        color1='#e41a1c', color2='#377eb8')


# ============================================================
# 矩阵热力图
# ============================================================

def make_adjacency_heatmap(A, Theta, l1_groups, title='邻接矩阵 A'):
    """邻接矩阵 A 热力图: 红=正边(Theta>0), 蓝=负边(Theta<0), 白=无边 (需要 Theta 提供符号)"""
    return _make_matrix_heatmap(A, Theta, l1_groups, title, discrete=True)


def make_theta_heatmap(Theta, l1_groups, title='精度矩阵 Theta'):
    """精度矩阵 Theta 热力图: 蓝-白-红 连续色标"""
    global_max = float(np.max(np.abs(Theta)))
    return _make_matrix_heatmap(Theta, None, l1_groups, title, discrete=False,
                                zmax=global_max)


def _make_matrix_heatmap(M, Theta_sign, l1_groups, title, discrete=False, zmax=None):
    """通用矩阵热力图。

    参数
    ----
    M : ndarray (p, p)  — 要显示的矩阵
    Theta_sign : ndarray or None  — discrete 模式下提供符号信息 (红/蓝)
    l1_groups : list[str]  — 行业分类
    discrete : bool  — True=邻接矩阵 (0/1), False=Theta (连续值)
    """
    p = M.shape[0]

    if discrete:
        # A 矩阵: 利用 Theta 符号决定颜色
        z_display = np.zeros((p, p))
        if Theta_sign is not None:
            # Theta<0 → 正偏相关 → 红色 (+1); Theta>0 → 负偏相关 → 蓝色 (-1)
            pos = (M == 1) & (Theta_sign < 0)
            neg = (M == 1) & (Theta_sign > 0)
            z_display[pos] = 1
            z_display[neg] = -1

        fig = go.Figure(data=go.Heatmap(
            z=z_display,
            colorscale=[[0, 'blue'], [0.5, 'white'], [1, 'red']],
            zmin=-1, zmid=0, zmax=1,
            showscale=False,
            hoverongaps=False,
            xgap=0, ygap=0,
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=M,
            colorscale='RdBu_r',
            zmid=0,
            zmin=-zmax if zmax else None,
            zmax=zmax,
            colorbar=dict(title='Theta'),
            hoverongaps=False,
            xgap=0, ygap=0,
        ))

    # 仅框对角板块块 (黑色矩形)
    if l1_groups:
        starts = [0]
        for i in range(1, p):
            if l1_groups[i] != l1_groups[i - 1]:
                starts.append(i)
        starts.append(p)

        for k in range(len(starts) - 1):
            x0 = starts[k] - 0.5
            x1 = starts[k + 1] - 0.5
            # 对角块: y 范围与 x 相同
            fig.add_shape(
                type='rect',
                x0=x0, x1=x1, y0=x0, y1=x1,
                line=dict(color='black', width=1.2),
                fillcolor='rgba(0,0,0,0)',
            )

    fig.update_layout(
        title=title,
        xaxis=dict(
            scaleanchor='y', scaleratio=1,
            showticklabels=False, showgrid=False, zeroline=False,
            side='top',  # (0,0) 放左上
        ),
        yaxis=dict(
            autorange='reversed',  # y 轴翻转使 (0,0) 在左上
            showticklabels=False, showgrid=False, zeroline=False,
        ),
        margin=dict(l=10, r=10, t=35, b=10),
        template='plotly_white',
        plot_bgcolor='white',
    )
    return fig
