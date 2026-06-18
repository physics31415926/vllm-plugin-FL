# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Local copy of vllm_metax/utils/__init__.py (relevant parts)

from . import envs as mx_envs  # noqa: F401
from .mccl import find_mccl_library  # noqa: F401


def import_pymxsml():
    """Import pymxsml (MetaX equivalent of pynvml)."""
    try:
        import pymxsml
        return pymxsml
    except ImportError:
        raise ImportError(
            "pymxsml is not available. Please install it or check your environment."
        )
