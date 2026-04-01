"""日志工具。"""

from __future__ import annotations

import logging


LOGGER_NAME = "paper_reading"


def configure_logging(verbose: bool = False) -> logging.Logger:
    """配置项目日志输出。

    默认输出简洁的 INFO 日志到控制台，便于命令行下观察当前执行阶段。
    """
    logger = logging.getLogger(LOGGER_NAME)
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)

    for handler in logger.handlers:
        handler.setLevel(level)
    return logger


def get_logger() -> logging.Logger:
    """获取项目 logger。"""
    return logging.getLogger(LOGGER_NAME)
