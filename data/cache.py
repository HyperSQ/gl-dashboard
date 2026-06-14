"""Flask-Caching + diskcache 配置"""

from flask_caching import Cache

cache = Cache()


def init_cache(dash_app, cache_dir, timeout=3600):
    """初始化缓存（FileSystem 后端）。注意：传入 Dash app，内部使用 app.server (Flask)。"""
    cache.init_app(dash_app.server, config={
        'CACHE_TYPE': 'FileSystemCache',
        'CACHE_DIR': cache_dir,
        'CACHE_DEFAULT_TIMEOUT': timeout,
        'CACHE_THRESHOLD': 200,
    })
    return cache
