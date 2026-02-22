from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Iterator


def configure_runtime(quiet: bool) -> None:
    """Reduce noisy warnings/logs from model runtime dependencies."""
    if not quiet:
        return

    warnings.filterwarnings("ignore")
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["MODELSCOPE_LOG_LEVEL"] = "40"

    for logger_name in ("modelscope", "transformers"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        logger.disabled = True
        for handler in logger.handlers:
            handler.setLevel(logging.ERROR)


@contextmanager
def suppress_output(enabled: bool) -> Iterator[None]:
    """Mute stdout/stderr during model initialization when requested."""
    if not enabled:
        yield
        return

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield
