import os
import pickle
from typing import Any, Optional, Dict, Union, Set

import torch

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (ReplicatedLinear,
                                               ColumnParallelLinear,
                                               RowParallelLinear,
                                               MergedColumnParallelLinear)
from vllm.model_executor.models.llama import LlamaMLP as NativeLlamaMLP
from vllm.model_executor.models.llama import LlamaAttention as NativeLlamaAttention
from vllm.model_executor.models.opt import OPTAttention as NativeOPTAttention
from vllm.distributed import get_tensor_model_parallel_world_size

from effide.vllm_plugin.model_executor.layers.linear import (MergedBatchedLinear, 
                                                             OrderedMergedColumnParallelLinear,
                                                             QKVOrderedMergedColumnParallelLinear,
                                                             WithOutputColumnParallelLinear)

from transformers import LlamaConfig
from vllm.config import CacheConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.attention import Attention, AttentionMetadata
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.activation import get_act_fn
from transformers import OPTConfig

from effide.vllm_plugin.attention.layer import CustomizedAttention


def _loading_truncation_config(config_path: Union[str, os.PathLike]) -> Optional[Dict[str, int]]:
    if config_path is None:
        return None

    # use pickle to load the config
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    return config


def _filtering_truncation_config(config: Optional[Dict[str, int]]) -> Optional[Set[str]]:
    if config is None:
        return None

    # filtering according to the list length
    # example:
    # self_attn: model.layers.26.self_attn.q_proj (len=5)
    # mlp: model.layers.26.mlp.gate_proj (len=5)
    # note: layer-wise for now
    prefix_set = set()
    for key in config.keys():
        component_list = key.split(".")
        if len(component_list) > 4:
            prefix = ".".join(component_list[:4])
            prefix_set.add(prefix)
    return prefix_set


def _filtering_truncation_config_opt(config: Optional[Dict[str, int]]) -> Optional[Set[str]]:
    if config is None:
        return None

    # filtering according to the list length
    # opt
    # model.decoder.layers.0.self_attn.q_proj (len = 6)
    # model.decoder.layers.0.fc1 (len=5)
    # note: layer-wise for now
    prefix_set = set()
    for key in config.keys():
        component_list = key.split(".")
        if len(component_list) >= 5:
            prefix = ".".join(component_list[:5])
            prefix_set.add(prefix)
    return prefix_set

TRUNCATION_CONFIG: Optional[Dict[str, int]] = _loading_truncation_config(os.environ.get("DECOMPOSE_CONFIG", None))
PREFIX_SET: Optional[Set[str]] = _filtering_truncation_config(TRUNCATION_CONFIG)  # all layers to be decomposed
PREFIX_SET_OPT: Optional[Set[str]] = _filtering_truncation_config_opt(TRUNCATION_CONFIG)  # all layers to be decomposed


class SerialNaiveLlamaMLP(torch.nn.Module):
    submodule_name = ["gate_proj", "up_proj", "down_proj"]

    stacked_params_mapping = []

    decomposed_params_mapping = [
        (".gate_proj_v", ".gate_proj.v"),
        (".gate_proj_u", ".gate_proj.u"),
        (".up_proj_v", ".up_proj.v"),
        (".up_proj_u", ".up_proj.u"),
        (".down_proj_v", ".down_proj.v"),
        (".down_proj_u", ".down_proj.u"),
    ]

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            bias: bool = False,
            prefix: str = "",
            trunc_dict: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if trunc_dict is None:
            trunc_dict = TRUNCATION_CONFIG

        # Get the truncation positions
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in self.submodule_name:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = trunc_dict[layer_name]

        # inp @ v @ u
        self.up_proj_v = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.trunc_config[proj_name("up_proj")],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj_v",
        )

        self.up_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("up_proj")],
            output_size=intermediate_size,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj_u",
        )

        self.gate_proj_v = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.trunc_config[proj_name("gate_proj")],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj_v",
        )
        self.gate_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("gate_proj")],
            output_size=intermediate_size,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj_u",
        )

        self.down_proj_v = ColumnParallelLinear(
            input_size=intermediate_size,
            output_size=self.trunc_config[proj_name("down_proj")],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj_v",
        )

        self.down_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("down_proj")],
            output_size=hidden_size,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj_u",
        )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, inp):
        up, _ = self.up_proj_v(inp)
        up, _ = self.up_proj_u(up)

        gate, _ = self.gate_proj_v(inp)
        gate, _ = self.gate_proj_u(gate)

        x = self.act_fn(torch.concat([gate, up], dim=-1))  # must be [gate, up]
        x, _ = self.down_proj_v(x)
        x, _ = self.down_proj_u(x)
        return x


class ParallelNaiveLlamaMLP(torch.nn.Module):
    submodule_name = ["gate_proj", "up_proj", "down_proj"]

    stacked_params_mapping = [
        (".gate_up_proj_v", ".gate_proj.v", 0),
        (".gate_up_proj_v", ".up_proj.v", 1),
        (".gate_up_proj_u", ".gate_proj.u", 0),
        (".gate_up_proj_u", ".up_proj.u", 1),
    ]

    decomposed_params_mapping = [
        (".down_proj_v", ".down_proj.v"),
        (".down_proj_u", ".down_proj.u"),
    ]

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            bias: bool = False,
            prefix: str = "",
            trunc_dict: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if trunc_dict is None:
            trunc_dict = TRUNCATION_CONFIG

        self.world_size = get_tensor_model_parallel_world_size()
        assert self.world_size == 2, "Only support 2 GPUs for now."
        # Get the truncation positions
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in self.submodule_name:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = trunc_dict[layer_name]

        # inp @ v @ u
        self.gate_up_proj_v = OrderedMergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[self.trunc_config[proj_name(key)] for key in ["gate_proj", "up_proj"]],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj_v",
        )

        self.gate_up_proj_u = OrderedMergedColumnParallelLinear(
            input_size=[self.trunc_config[proj_name(key)] for key in ["gate_proj", "up_proj"]][0],
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            gather_output=True,
            prefix=f"{prefix}.gate_up_proj_u",
        )

        self.down_proj_v = ColumnParallelLinear(
            input_size=intermediate_size,
            output_size=self.trunc_config[proj_name("down_proj")],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj_v",
        )

        self.down_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("down_proj")],
            output_size=hidden_size,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj_u",
        )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj_v(x)  # x.shape == [seq_len, hidden_size]
        x, _ = self.gate_up_proj_u(x)  # x.shape == [seq_len, intermediate_size * 2]

        x = self.act_fn(x)
        x, _ = self.down_proj_v(x)
        x, _ = self.down_proj_u(x)
        return x


class ParallelLlamaMLP(torch.nn.Module):
    submodule_name = ["gate_proj", "up_proj", "down_proj"]

    stacked_params_mapping = [
        (".gate_up_proj_v", ".gate_proj.v", 0),
        (".gate_up_proj_v", ".up_proj.v", 1),
        (".gate_up_proj_u", ".gate_proj.u", 0),
        (".gate_up_proj_u", ".up_proj.u", 1),
    ]

    decomposed_params_mapping = [
        (".down_proj_v", ".down_proj.v"),
        (".down_proj_u", ".down_proj.u"),
    ]

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            bias: bool = False,
            prefix: str = "",
            trunc_dict: Optional[dict] = None,
    ) -> None:
        super().__init__()

        if trunc_dict is None:
            trunc_dict = TRUNCATION_CONFIG

        self.world_size = get_tensor_model_parallel_world_size()
        # Get the truncation positions
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in self.submodule_name:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = trunc_dict[layer_name]

        # inp @ v @ u
        self.gate_up_proj_v = OrderedMergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[self.trunc_config[proj_name(key)] for key in ["gate_proj", "up_proj"]],
            bias=bias,
            quant_config=quant_config,
            gather_output=True,
            prefix=f"{prefix}.gate_up_proj_v",
        )

        self.gate_up_proj_u = MergedBatchedLinear(
            input_size=[self.trunc_config[proj_name(key)] for key in ["gate_proj", "up_proj"]][0],
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj_u",
        )

        self.down_proj_v = RowParallelLinear(
            input_size=intermediate_size,
            output_size=self.trunc_config[proj_name("down_proj")],
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj_v",
        )

        self.down_proj_u = ReplicatedLinear(
            input_size=self.trunc_config[proj_name("down_proj")],
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj_u",
        )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj_v(x)  # x.shape == [seq_len, 2*L]
        x, _ = self.gate_up_proj_u(x)  # x.shape == [seq_len, intermediate_size // tp_size]

        x = self.act_fn(x)
        x, _ = self.down_proj_v(x)
        x, _ = self.down_proj_u(x)
        return x


class NaiveLlamaAttention(torch.nn.Module):
    '''
    Naive Attention for the low-rank decomposed attention layer.
    '''
    
    submodule_name = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    stacked_params_mapping = []

    decomposed_params_mapping = [
        (".q_proj_v", ".q_proj.v"),
        (".q_proj_u", ".q_proj.u"),
        (".k_proj_v", ".k_proj.v"),
        (".k_proj_u", ".k_proj.u"),
        (".v_proj_v", ".v_proj.v"),
        (".v_proj_u", ".v_proj.u"),
        (".o_proj_v", ".o_proj.v"),
        (".o_proj_u", ".o_proj.u"),
    ]
    
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        trunc_dict: Optional[dict] = None,
    ) -> None:
        super().__init__()
        
        if trunc_dict is None:
            trunc_dict = TRUNCATION_CONFIG

        # Get the truncation positions
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in self.submodule_name:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = trunc_dict[layer_name]

        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.q_proj_v = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.trunc_config[proj_name("q_proj")],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_v",
        )
        self.q_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("q_proj")],
            output_size=hidden_size,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_u",
        )

        self.k_proj_v = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.trunc_config[proj_name("k_proj")],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj_v",
        )
        self.k_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("k_proj")],
            output_size=self.hidden_size//self.num_kv_groups,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj_u",
        )

        self.v_proj_v = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.trunc_config[proj_name("v_proj")],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj_v",
        )
        self.v_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("v_proj")],
            output_size=self.hidden_size//self.num_kv_groups,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj_u",
        )
        
        self.o_proj_v = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.trunc_config[proj_name("o_proj")],
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_v",
        )
        self.o_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("o_proj")],
            output_size=hidden_size,
            bias=bias_o_proj,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_u",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.attn = Attention(
            self.total_num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.total_num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        q, _ = self.q_proj_v(hidden_states)
        q, _ = self.q_proj_u(q)
        k, _ = self.k_proj_v(hidden_states)
        k, _ = self.k_proj_u(k)
        v, _ = self.v_proj_v(hidden_states)
        v, _ = self.v_proj_u(v)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj_v(attn_output)
        output, _ = self.o_proj_u(output)
        return output


class ParallelLlamaAttention(torch.nn.Module):
    '''
    Multiple GPUs are used for the low-rank decomposed attention layer.
    '''
    
    submodule_name = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    stacked_params_mapping = [
        (".qkv_proj_v", ".q_proj.v", 'q'),
        (".q_proj_u", ".q_proj.u", 'q'),
        (".qkv_proj_v", ".k_proj.v", 'k'),
        (".k_proj_u", ".k_proj.u", 'k'),
        (".qkv_proj_v", ".v_proj.v", 'v'),
        (".v_proj_u", ".v_proj.u", 'v'),
    ]

    decomposed_params_mapping = [
        (".o_proj_v", ".o_proj.v"),
        (".o_proj_u", ".o_proj.u"),
    ]

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
        trunc_dict: Optional[dict] = None,
    ) -> None:
        super().__init__()
        
        if trunc_dict is None:
            trunc_dict = TRUNCATION_CONFIG
            
        # Get the truncation positions
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in self.submodule_name:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = trunc_dict[layer_name]
            
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        
        kv_groups = self.total_num_heads // self.total_num_kv_heads
        
        self.q_rank = self.trunc_config[proj_name("q_proj")]
        self.k_rank = self.trunc_config[proj_name("k_proj")]
        self.v_rank = self.trunc_config[proj_name("v_proj")]

        assert self.k_rank == self.v_rank, "Support for different ranks for k and v is not implemented yet."
        self.kv_cache_size = self.k_rank

        self.qkv_proj_v = QKVOrderedMergedColumnParallelLinear(
            ranks=[self.trunc_config[proj_name(key)] for key in ["q_proj", "k_proj", "v_proj"]],
            hidden_size=hidden_size,
            bias=bias,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj_v",
        )
        
        self.q_proj_u = ColumnParallelLinear(
            input_size=self.trunc_config[proj_name("q_proj")],
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_u",
        )

        self.k_proj_u = WithOutputColumnParallelLinear(
            input_size=self.trunc_config[proj_name("k_proj")],
            output_size=hidden_size//kv_groups,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj_u",
        )
        
        self.v_proj_u = WithOutputColumnParallelLinear(
            input_size=self.trunc_config[proj_name("v_proj")],
            output_size=hidden_size//kv_groups,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj_u",
        )

        self.o_proj_v = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.trunc_config[proj_name("o_proj")],
            bias=bias_o_proj,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_v",
        )
        self.o_proj_u = ReplicatedLinear(
            input_size=self.trunc_config[proj_name("o_proj")],
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj_u",
        )
        
        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.enable_low_rank_cache = os.environ.get("ENABLE_LOW_RANK_CACHE", "OFF") == "ON"
        if self.enable_low_rank_cache:
            self.attn = CustomizedAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                self.q_proj_u,
                self.k_proj_u,
                self.v_proj_u,
                self.kv_cache_size,
                rotary_emb=self.rotary_emb,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                per_layer_sliding_window=sliding_window,
                prefix=f"{prefix}.attn",
            )
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                per_layer_sliding_window=sliding_window,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv_v, _ = self.qkv_proj_v(hidden_states)
        q_v = qkv_v[:, :self.q_rank]
        k_v = qkv_v[:, self.q_rank:self.q_rank + self.k_rank]
        v_v = qkv_v[:, self.q_rank + self.k_rank:]
        if isinstance(self.attn, CustomizedAttention):
            attn_output = self.attn(q_v, k_v, v_v, kv_cache, positions, attn_metadata)
        else:
            q, _ = self.q_proj_u(q_v)
            k, _ = self.k_proj_u(k_v)
            v, _ = self.v_proj_u(v_v)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj_v(attn_output)
        output, _ = self.o_proj_u(output)
        return output


class NaiveOPTAttention(torch.nn.Module):
    '''
    Naive Attention for the low-rank decomposed attention layer of OPT Model.
    '''
    
    submodule_name = ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    stacked_params_mapping = []

    decomposed_params_mapping = [
        (".q_proj_v", ".q_proj.v"),
        (".q_proj_u", ".q_proj.u"),
        (".k_proj_v", ".k_proj.v"),
        (".k_proj_u", ".k_proj.u"),
        (".v_proj_v", ".v_proj.v"),
        (".v_proj_u", ".v_proj.u"),
        (".out_proj_v", ".out_proj.v"),
        (".out_proj_u", ".out_proj.u"),
    ]
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        trunc_dict: Optional[dict] = None,
    ) -> None:
        super().__init__()
        
        if trunc_dict is None:
            trunc_dict = TRUNCATION_CONFIG

        # Get the truncation positions
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in self.submodule_name:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = trunc_dict[layer_name]

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj_v = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=self.trunc_config[proj_name("q_proj")],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_v",
        )
        self.q_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("q_proj")],
            output_size=embed_dim,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_u",
        )

        self.k_proj_v = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=self.trunc_config[proj_name("k_proj")],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj_v",
        )
        self.k_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("k_proj")],
            output_size=embed_dim,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj_u",
        )
        
        self.v_proj_v = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=self.trunc_config[proj_name("v_proj")],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj_v",
        )
        self.v_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("v_proj")],
            output_size=embed_dim,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj_u",
        )
        
        self.out_proj_v = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=self.trunc_config[proj_name("out_proj")],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj_v",
        )
        self.out_proj_u = RowParallelLinear(
            input_size=self.trunc_config[proj_name("out_proj")],
            output_size=embed_dim,
            bias=bias,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj_u",
        )

        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              scale=self.scaling,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        q, _ = self.q_proj_v(hidden_states)
        q, _ = self.q_proj_u(q)
        k, _ = self.k_proj_v(hidden_states)
        k, _ = self.k_proj_u(k)
        v, _ = self.v_proj_v(hidden_states)
        v, _ = self.v_proj_u(v)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.out_proj_v(attn_output)
        output, _ = self.out_proj_u(output)
        return output


class ParallelOPTAttention(torch.nn.Module):
    '''
    Multiple GPUs are used for the low-rank decomposed attention layer of OPT Model.
    '''
    
    submodule_name = ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    stacked_params_mapping = [
        (".qkv_proj_v", ".q_proj.v", 'q'),
        (".q_proj_u", ".q_proj.u", 'q'),
        (".qkv_proj_v", ".k_proj.v", 'k'),
        (".k_proj_u", ".k_proj.u", 'k'),
        (".qkv_proj_v", ".v_proj.v", 'v'),
        (".v_proj_u", ".v_proj.u", 'v'),
    ]

    decomposed_params_mapping = [
        (".out_proj_v", ".out_proj.v"),
        (".out_proj_u", ".out_proj.u"),
    ]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        trunc_dict: Optional[dict] = None,
    ) -> None:
        super().__init__()
        
        if trunc_dict is None:
            trunc_dict = TRUNCATION_CONFIG
            
        # Get the truncation positions
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in self.submodule_name:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = trunc_dict[layer_name]
            
        self.embed_dim = embed_dim
        tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        total_num_heads = num_heads
        assert num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = total_num_heads // tensor_model_parallel_world_size
        self.head_dim = embed_dim // total_num_heads
        self.scaling = self.head_dim**-0.5
        
        self.q_rank = self.trunc_config[proj_name("q_proj")]
        self.k_rank = self.trunc_config[proj_name("k_proj")]
        self.v_rank = self.trunc_config[proj_name("v_proj")]

        assert self.k_rank == self.v_rank, "Support for different ranks for k and v is not implemented yet."
        self.kv_cache_size = self.k_rank

        self.qkv_proj_v = QKVOrderedMergedColumnParallelLinear(
            ranks=[self.trunc_config[proj_name(key)] for key in ["q_proj", "k_proj", "v_proj"]],
            hidden_size=embed_dim,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj_v",
        )
        
        self.q_proj_u = ColumnParallelLinear(
            input_size=self.trunc_config[proj_name("q_proj")],
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_u",
        )
        
        self.k_proj_u = ColumnParallelLinear(
            input_size=self.trunc_config[proj_name("k_proj")],
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj_u",
        )
        
        self.v_proj_u = ColumnParallelLinear(
            input_size=self.trunc_config[proj_name("v_proj")],
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj_u",
        )

        self.out_proj_v = RowParallelLinear(
            input_size=embed_dim,
            output_size=self.trunc_config[proj_name("out_proj")],
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj_v",
        )
        self.out_proj_u = ReplicatedLinear(
            input_size=self.trunc_config[proj_name("out_proj")],
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj_u",
        )

        self.enable_low_rank_cache = os.environ.get("ENABLE_LOW_RANK_CACHE", "OFF") == "ON"
        if self.enable_low_rank_cache:
            self.attn = CustomizedAttention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                self.q_proj_u,
                self.k_proj_u,
                self.v_proj_u,
                self.kv_cache_size,
                rotary_emb=None,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )
        else:
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                scale=self.scaling,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv_v, _ = self.qkv_proj_v(hidden_states)
        q_v = qkv_v[:, :self.q_rank]
        k_v = qkv_v[:, self.q_rank:self.q_rank + self.k_rank]
        v_v = qkv_v[:, self.q_rank + self.k_rank:]
        if isinstance(self.attn, CustomizedAttention):
            attn_output = self.attn(q_v, k_v, v_v, kv_cache, positions, attn_metadata)
        else:
            q, _ = self.q_proj_u(q_v)
            k, _ = self.k_proj_u(k_v)
            v, _ = self.v_proj_u(v_v)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.out_proj_v(attn_output)
        output, _ = self.out_proj_u(output)
        return output


class NativeOPTFC(torch.nn.Module):
    def __init__(self,
                 config: OPTConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""
                 ):
        super().__init__()
        
        self.embed_dim = config.hidden_size
        self.fc1 = ColumnParallelLinear(
            self.embed_dim,
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.activation_fn = get_act_fn(config.activation_function)
        self.fc2 = RowParallelLinear(
            config.ffn_dim,
            self.embed_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )
        
    def forward(self, hidden_states):
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states
        

class NaiveSVDOPTFC(torch.nn.Module):
    '''
    prefix: It is [model.decoder.layers.1] instead of [model.decoder.layers.1.fc1/fc2]
    '''
    
    decomposed_params_mapping = [
        (".fc1_v", ".fc1.v"),
        (".fc1_u", ".fc1.u"),
        (".fc2_v", ".fc2.v"),
        (".fc2_u", ".fc2.u"),
    ]
    
    def __init__(self,
                 config: OPTConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""
                 ):
        super().__init__()
        
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in ["fc1", "fc2"]:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = TRUNCATION_CONFIG[layer_name]
        
        self.embed_dim = config.hidden_size
        
        self.fc1_v = ColumnParallelLinear(
                self.embed_dim,
                self.trunc_config[proj_name("fc1")],
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1_v",
        )
        self.fc1_u = RowParallelLinear(
            self.trunc_config[proj_name("fc1")],
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_u",
        )
        
        self.activation_fn = get_act_fn(config.activation_function)
        
        self.fc2_v = ColumnParallelLinear(
            config.ffn_dim,
            self.trunc_config[proj_name("fc2")],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_v",
        )
        self.fc2_u = RowParallelLinear(
            self.trunc_config[proj_name("fc2")],
            self.embed_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_u",
        )
        
    def forward(self, hidden_states):
        hidden_states, _ = self.fc1_v(hidden_states)
        hidden_states, _ = self.fc1_u(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2_v(hidden_states)
        hidden_states, _ = self.fc2_u(hidden_states)
        return hidden_states


class ParallelSVDOPTFC(torch.nn.Module):
    '''
    Low-rank data transmission
    prefix: It is [model.decoder.layers.1] instead of [model.decoder.layers.1.fc1/fc2]
    '''
    
    decomposed_params_mapping = [
        (".fc1_v", ".fc1.v"),
        (".fc1_u", ".fc1.u"),
        (".fc2_v", ".fc2.v"),
        (".fc2_u", ".fc2.u"),
    ]
    
    def __init__(self,
                 config: OPTConfig,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""
                 ):
        super().__init__()
        
        self.trunc_config = {}
        proj_name = lambda x: f"{prefix}.{x}"
        for submodule in ["fc1", "fc2"]:
            layer_name = f"{prefix}.{submodule}"
            self.trunc_config[layer_name] = TRUNCATION_CONFIG[layer_name]
        
        self.embed_dim = config.hidden_size
        
        self.fc1_v = ColumnParallelLinear(
                self.embed_dim,
                self.trunc_config[proj_name("fc1")],
                bias=False,
                gather_output=True,
                quant_config=quant_config,
                prefix=f"{prefix}.fc1_v",
        )
        self.fc1_u = ColumnParallelLinear(
            self.trunc_config[proj_name("fc1")],
            config.ffn_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1_u",
        )
        
        self.activation_fn = get_act_fn(config.activation_function)
        
        self.fc2_v = RowParallelLinear(
            config.ffn_dim,
            self.trunc_config[proj_name("fc2")],
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_v",
        )
        self.fc2_u = ReplicatedLinear(
            self.trunc_config[proj_name("fc2")],
            self.embed_dim,
            bias=config.enable_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2_u",
        )
        
    def forward(self, hidden_states):
        hidden_states, _ = self.fc1_v(hidden_states)
        hidden_states, _ = self.fc1_u(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2_v(hidden_states)
        hidden_states, _ = self.fc2_u(hidden_states)
        return hidden_states
