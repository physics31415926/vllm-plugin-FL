# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Local copy of vllm_metax/envs.py to eliminate external dependency.

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    VLLM_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: str | None
    NVCC_THREADS: str | None
    VLLM_USE_PRECOMPILED: bool = False
    CMAKE_BUILD_TYPE: str | None
    VERBOSE: bool = False
    USE_PRECOMPILED_KERNEL: bool = True
    VLLM_METAX_OPTIMIZED_DP_ALL2ALL: bool = False
    MACA_VLLM_ENABLE_MCTLASS_PYTHON_API: bool = True
    MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE: bool = False
    USE_VLLM_TRITON_EXPERT: bool = False
    VLLM_METAX_ENABLE_FA_SPLIT_FORWARD: bool = True
    VLLM_FUSED_MOE_CHUNK_SIZE: int = 16 * 1024
    VLLM_METAX_USE_FP8_SPARSE_ATTN_INDEXER: bool = False
    VLLM_METAX_USE_SGL_FUSED_MOE_GROUPED_TOPK: bool = False
    VLLM_MCCL_SO_PATH: str | None

environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_TARGET_DEVICE": lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda"),
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    "VLLM_MCCL_SO_PATH": lambda: os.environ.get("VLLM_MCCL_SO_PATH", None),
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    "USE_PRECOMPILED_KERNEL": lambda: bool(
        int(os.environ.get("USE_PRECOMPILED_KERNEL", "1"))
    ),
    "MACA_VLLM_ENABLE_MCTLASS_PYTHON_API": lambda: bool(
        int(os.getenv("MACA_VLLM_ENABLE_MCTLASS_PYTHON_API", "1"))
    ),
    "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE": lambda: bool(
        int(os.getenv("MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE", "0"))
    ),
    "VLLM_METAX_OPTIMIZED_DP_ALL2ALL": lambda: bool(
        int(os.environ.get("VLLM_METAX_OPTIMIZED_DP_ALL2ALL", "0"))
    ),
    "VLLM_METAX_ENABLE_FA_SPLIT_FORWARD": lambda: bool(
        int(os.environ.get("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", "1"))
    ),
    "VLLM_FUSED_MOE_CHUNK_SIZE": lambda: int(
        os.getenv("VLLM_FUSED_MOE_CHUNK_SIZE", str(16 * 1024))
    ),
    "VLLM_METAX_USE_FP8_SPARSE_ATTN_INDEXER": lambda: bool(
        int(os.environ.get("VLLM_METAX_USE_FP8_SPARSE_ATTN_INDEXER", "0"))
    ),
    "VLLM_METAX_USE_SGL_FUSED_MOE_GROUPED_TOPK": lambda: bool(
        int(os.getenv("VLLM_METAX_USE_SGL_FUSED_MOE_GROUPED_TOPK", "0"))
    ),
    "USE_VLLM_TRITON_EXPERT": lambda: bool(
        int(os.getenv("USE_VLLM_TRITON_EXPERT", "0"))
    ),
}


def override_vllm_env(env_name: str, value: Any, reason: str | None) -> None:
    from vllm import envs
    from vllm.logger import logger

    if not isinstance(env_name, str):
        raise TypeError("env_name must be a string")

    if env_name not in envs.environment_variables:
        raise KeyError(f"{env_name} is not a recognized vLLM environment variable")

    logger.info("Plugin sets %s to %s. Reason: %s", env_name, value, reason)
    envs.environment_variables[env_name] = lambda v=value: v

    if value is None:
        os.environ.pop(env_name, None)
    else:
        if isinstance(value, bool):
            os.environ[env_name] = "1" if value else "0"
        else:
            os.environ[env_name] = str(value)


def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
