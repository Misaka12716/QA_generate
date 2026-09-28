"""方案对照台：读取实验目录并提供只读 API。"""

from .app import create_app, serve

__all__ = ["create_app", "serve"]
