# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Local copy of vllm_metax/utils/mccl.py

import torch
from vllm.utils.nccl import logger
from . import envs as mx_envs


def find_mccl_library() -> str:
    so_file = mx_envs.VLLM_MCCL_SO_PATH

    if so_file:
        logger.info(
            "Found mccl from environment variable VLLM_NCCL_SO_PATH=%s", so_file
        )
    else:
        if torch.version.cuda is not None:
            so_file = "libmccl.so"
        else:
            raise ValueError("MCCL only supports MACA backends.")
        logger.info_once("Found mccl from library %s", so_file)
    return so_file
