"""常量与配置"""

import os
import re
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
CACHE_DIR = os.path.join(BASE_DIR, '.cache')
SECTOR_CACHE_DIR = os.path.join(OUTPUT_DIR, 'sector_metrics_cache')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SECTOR_CACHE_DIR, exist_ok=True)

# ---- 扫描数据文件 ----
def _scan_pkl(directory):
    return sorted(glob.glob(os.path.join(directory, '*.pkl')))

PKL_FILES_LOCAL = _scan_pkl(DATA_DIR)

def scan_all_pkl():
    """返回所有可用 pkl 文件路径 (仅本地 data/ 目录)"""
    return sorted(PKL_FILES_LOCAL)

def parse_filename(filepath):
    """从 pkl 文件名提取元数据: method, freq, W, delta, noPC1"""
    basename = os.path.basename(filepath).replace('.pkl', '')
    info = {'path': filepath, 'basename': basename}

    info['method'] = 'SGL' if basename.startswith('SGL') else 'FGL' if basename.startswith('FGL') else 'Unknown'

    m = re.search(r'_W[=_](\d+)', basename)
    info['W'] = int(m.group(1)) if m else None

    m = re.search(r'_delta[=_](\d+)', basename)
    info['delta'] = int(m.group(1)) if m else None

    # 频率
    if '_minute_D_' in basename or '_minute_D_' in basename.replace('_W=', '_W_'):
        info['freq'] = 'D'
    elif '_minute_W_' in basename:
        info['freq'] = 'W'
    elif '_minute_M_' in basename:
        info['freq'] = 'M'
    elif '_1W_' in basename:
        info['freq'] = 'W'
    elif '_W_' in basename and 'W=' in basename:
        info['freq'] = 'W'
    else:
        info['freq'] = 'M'  # 默认月频

    info['noPC1'] = 'noPC1' in basename
    info['label'] = f"{info['method']} | {info['freq']} | W={info['W']}"
    if info.get('delta'):
        info['label'] += f" | δ={info['delta']}"
    if info['noPC1']:
        info['label'] += ' (no PC1)'

    return info

# ---- NetworkShock 配置 ----
CONFIGS = {
    'C1_min_D_W5': {
        'label': 'min→D W=5',
        'pkl': os.path.join(OUTPUT_DIR, 'SGL_CSI300_S_minute_D_W=5_delta=5.pkl'),
        'freq': 'D',
    },
    'C2_min_W_W1': {
        'label': 'min→W W=1',
        'pkl': os.path.join(OUTPUT_DIR, 'SGL_CSI300_S_minute_W_W=1_delta=5.pkl'),
        'freq': 'W',
    },
    'C3_DK_W_W12': {
        'label': 'DK→W W=12',
        'pkl': os.path.join(OUTPUT_DIR, 'SGL_CSI300_S_1W_W=12.pkl'),
        'freq': 'W',
    },
    'C4_min_W_W12': {
        'label': 'min→W W=12',
        'pkl': os.path.join(OUTPUT_DIR, 'SGL_CSI300_S_minute_W_W=12_delta=5.pkl'),
        'freq': 'W',
    },
}
