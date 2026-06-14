"""
Graph LASSO Network Analysis Dashboard
======================================
Python Dash + Plotly 重写版
Tab 1: 时间序列总览 (Jaccard/L1 + NetworkShock + 5分量)
Tab 2: 板块分析 (密度/中心性/跨连接)
Tab 3: 矩阵热力图 (A / Theta)
Tab 4: 动量轮动 (Phase C — 待实现)
"""

import os
import sys

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

from data.cache import init_cache
from config import scan_all_pkl, parse_filename, CACHE_DIR, OUTPUT_DIR

# ---- 扫描数据文件 ----
pkl_files = scan_all_pkl()
pkl_meta_map = {f: parse_filename(f) for f in pkl_files}
print(f'扫描到 {len(pkl_files)} 个 pkl 文件')

# 确保 output/ 下的文件也在可达范围内（用于网络数据加载）
# 添加项目根目录以便 loader.py 能访问 GL.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---- 初始化 Dash App ----
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title='GL Network Dashboard',
)
server = app.server
init_cache(app, cache_dir=CACHE_DIR, timeout=3600)

# ============================================================
# 导航栏
# ============================================================
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink('总览', href='/', active='exact')),
        dbc.NavItem(dbc.NavLink('板块分析', href='/sector', active='exact')),
        dbc.NavItem(dbc.NavLink('矩阵', href='/matrices', active='exact')),
        dbc.NavItem(dbc.NavLink('动量与板块轮动', href='/rotation', active='exact')),
        dbc.NavItem(dbc.NavLink('整协分析(未完善)', href='/cointegration', active='exact')),
    ],
    brand='Graph LASSO Network Analysis',
    brand_href='/',
    color='dark',
    dark=True,
    className='mb-2',
)

# ============================================================
# 页面路由
# ============================================================
app.layout = dbc.Container([
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
], fluid=True, className='px-3')

# ---- 导入页面 ----
from pages.overview import layout as ov_layout, register_callbacks as ov_register
from pages.sector import layout as sec_layout, register_callbacks as sec_register
from pages.matrices import layout as mat_layout, register_callbacks as mat_register
from pages.rotation import layout as rot_layout, register_callbacks as rot_register
from pages.cointegration import layout as coin_layout, register_callbacks as coin_register

# 注册各页面的回调
ov_register(app, pkl_files, pkl_meta_map)
sec_register(app, pkl_files, pkl_meta_map)
mat_register(app, pkl_files, pkl_meta_map)
rot_register(app, pkl_files, pkl_meta_map)
coin_register(app, pkl_files, pkl_meta_map)


# ---- 页面路由回调 ----
@app.callback(
    dash.Output('page-content', 'children'),
    dash.Input('url', 'pathname'),
)
def display_page(pathname):
    if pathname == '/sector':
        return sec_layout()
    elif pathname == '/matrices':
        return mat_layout()
    elif pathname == '/rotation':
        return rot_layout()
    elif pathname == '/cointegration':
        return coin_layout()
    else:
        return ov_layout()


# ============================================================
# 入口
# ============================================================
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
