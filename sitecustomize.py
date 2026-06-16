"""
站点定制入口（Python 启动时由 site.py 的 execsitecustomize 加载）。

关键：sitecustomize 是**无条件**加载的，而 usercustomize 受 ENABLE_USER_SITE
限制。某些环境（venv、spack python 等）的 ENABLE_USER_SITE=False，会导致
usercustomize.py 不会被自动加载，平台适配钩子静默失效。

因此由本文件间接 import usercustomize，确保平台适配钩子在任何环境下都能安装。
本文件需在 PYTHONPATH 上（由 env.sh 把仓库根加入 PYTHONPATH 保证）。
"""
try:
    import usercustomize  # noqa: F401  安装平台适配 import hook
except Exception:
    pass
