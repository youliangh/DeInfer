# SPDX-License-Identifier: Apache-2.0
"""CacheEngine class for managing the KV cache."""
import pickle
import itertools
from typing import List, Dict, Callable

import numpy as np
import torch

from vllm import envs
from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        align_to_256bytes, get_dtype_size,
                        is_pin_memory_available)

logger = init_logger(__name__)

_variadic_kv_cache_config = {}


def _init_with_env_var():
    import os
    global _variadic_kv_cache_config
    model_type = os.environ["MODEL_TYPE"].lower()
    if "llama" in model_type:
        # example: model.layers.26.self_attn.q_proj
        compose_submodule = lambda idx, submodule: f"model.layers.{idx}.self_attn.{submodule}"
        layer_idx_pos = 2
    elif "opt" in model_type:
        compose_submodule = lambda idx, submodule: f"model.decoder.layers.{idx}.self_attn.{submodule}"
        layer_idx_pos = 3
    else:
        raise NotImplementedError(f"Model type {model_type} not supported yet.")

    with open(os.environ["DECOMPOSE_CONFIG"], "rb") as f:
        variadic_kv_cache_config_raw = pickle.load(f)

    _variadic_kv_cache_config = _init_variadic_kv_cache_config(variadic_kv_cache_config_raw, layer_idx_pos,
                                                               compose_submodule)


def _init_variadic_kv_cache_config(variadic_kv_cache_config_raw: Dict, layer_idx_pos: int,
                                   compose_submodule: Callable):
    variadic_kv_cache_config = {}

    # adaptive layer_idx_pos
    try:
        grouped_config = itertools.groupby(variadic_kv_cache_config_raw, lambda x: int(x.split(".")[layer_idx_pos]))
    except ValueError:
        grouped_config = itertools.groupby(variadic_kv_cache_config_raw, lambda x: int(x.split(".")[layer_idx_pos+1]))

    for layer_idx, matrices_dict_obj in grouped_config:
        # matrices_dict_obj: itertools._grouper
        k_cache_size = variadic_kv_cache_config_raw[compose_submodule(layer_idx, "k_proj")]
        v_cache_size = variadic_kv_cache_config_raw[compose_submodule(layer_idx, "v_proj")]
        assert k_cache_size == v_cache_size, "Only support same kv size for now."
        variadic_kv_cache_config[layer_idx] = k_cache_size
    return variadic_kv_cache_config


class VariadicCacheEngine:
    """Manages the KV cache, allowing each layer has kv size of different size.

    Note: Modified to 0.7.2 release
    Known bugs: Trigger JSONDecodeError of cpuinfo package

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        # MODIFICATION START
        self.hf_config = self.model_config.hf_config
        model_type_list = getattr(self.hf_config, "architectures", None)
        assert model_type_list, "Model type not found in the config."
        assert len(model_type_list) == 1, "Only support single model type for now."
        model_type = model_type_list[0].lower()

        variadic_kv_cache_config_raw = None
        self.variadic_kv_cache_config = {}
        self.enable_variadic_kv_cache = getattr(self.hf_config, "enable_variadic_kv_cache", False)
        variadic_kv_cache_config_fp = getattr(self.hf_config, "variadic_kv_cache_config", None)

        if self.enable_variadic_kv_cache and variadic_kv_cache_config_fp:
            with open(variadic_kv_cache_config_fp, "rb") as f:
                variadic_kv_cache_config_raw = pickle.load(f)

        # intra-layer evenly-compression only for now
        # Only support LLaMA for now.
        if self.enable_variadic_kv_cache:
            assert variadic_kv_cache_config_raw is not None
            assert any(["opt" in model_type, "llama" in model_type])
            # llama and opt for illustration
            if "llama" in model_type:
                # example: model.layers.26.self_attn.q_proj
                compose_submodule = lambda idx, submodule: f"model.layers.{idx}.self_attn.{submodule}"
                layer_idx_pos = 2
            elif "opt" in model_type:
                compose_submodule = lambda idx, submodule: f"model.decoder.layers.{idx}.self_attn.{submodule}"
                layer_idx_pos = 3
            else:
                raise NotImplementedError(f"Model type {model_type} not supported yet.")

            self.variadic_kv_cache_config = _init_variadic_kv_cache_config(variadic_kv_cache_config_raw,
                                                                           layer_idx_pos,
                                                                           compose_submodule)
            global _variadic_kv_cache_config
            _variadic_kv_cache_config = self.variadic_kv_cache_config

        assert self.model_config.use_mla is False, "MLA not supported yet."
        # MODIFICATION END

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=model_config.use_mla)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")
        logger.info("Using VariadicCacheEngine.")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        for layer_idx in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            # (2, num_blocks, block_size, kv_cache_size)
            kv_cache_size = self.variadic_kv_cache_config[layer_idx]
            alloc_shape = (2, num_blocks, self.block_size, kv_cache_size)
            layer_kv_cache = torch.zeros(alloc_shape,
                                         dtype=self.dtype,
                                         pin_memory=pin_memory,
                                         device=device)

            # view back to (TOTAL_PAGES, PAGE_SIZE, entry_shape...) for cases
            # when entry_shape is higher than 1D
            kv_cache.append(layer_kv_cache)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)


    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        if len(_variadic_kv_cache_config) == 0:
            _init_with_env_var()

        assert len(_variadic_kv_cache_config) != 0, "Uninitialized variadic_kv_cache_config."

        total = 0
        for layer_idx in range(num_attention_layers):
            kv_cache_entry_size = _variadic_kv_cache_config[layer_idx]
            total += cache_config.block_size * kv_cache_entry_size * 2

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
