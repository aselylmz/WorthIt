"""
Logging altyapısı.

Proje genelinde tutarlı ve yapılandırılabilir logging sağlar.
"""

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_initialized: bool = False


def setup_logging(level: Optional[str] = None) -> None:
    """Uygulama genelinde logging'i yapılandır.

    Parameters
    ----------
    level : str, optional
        Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Verilmezse config'den okunur.

    Notes
    -----
    Bu fonksiyon birden fazla çağrılsa bile sadece ilk çağrıda
    handler ekler (duplicate log önlemi).
    """
    global _initialized
    if _initialized:
        return

    if level is None:
        # Döngüsel import'u önlemek için burada import ediyoruz
        from time_to_afford.config.settings import get_settings

        level = get_settings().log_level

    root_logger = logging.getLogger("time_to_afford")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root_logger.addHandler(handler)
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Modül-spesifik logger oluştur.

    Parameters
    ----------
    name : str
        Logger adı. Genellikle ``__name__`` geçilir.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(f"time_to_afford.{name}")
