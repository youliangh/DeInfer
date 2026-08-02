import itertools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               adjust_scalar_to_fused_array,
                                               adjust_marlin_shard,
                                               adjust_bitsandbytes_4bit_shard)
from vllm.logger import init_logger
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              split_tensor_along_last_dim,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.linear import QuantizationConfig, UnquantizedLinearMethod
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.parameter import (BasevLLMParameter, PackedvLLMParameter, PackedColumnParameter,
                                           PerTensorScaleParameter, RowvLLMParameter)

logger = init_logger(__name__)


# monkey patching
def weight_loader_merged_column_parallel(self,
                                         param: Parameter,
                                         loaded_weight: torch.Tensor,
                                         loaded_shard_id: Optional[int] = None):
    # TODO: support GGUF and other special cases
    # Special case for GGUF
    # initialize GGUF param after we know the quantize type
    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
    if is_gguf_weight_type:
        raise NotImplementedError("GGUF weight type not implemented yet.")
        param.data[loaded_shard_id].copy_(loaded_weight)
        param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
        return

    if is_gguf_weight:
        raise NotImplementedError("GGUF weight type not implemented yet.")
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        output_dim = getattr(param, "output_dim", None)
        shard_size = loaded_weight.size(output_dim) // tp_size
        start_idx = tp_rank * shard_size

        loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                             shard_size)

        param.shard_id.append(loaded_shard_id)
        param.shard_id_map[loaded_shard_id] = len(param.data_container)
        param.data_container.append(loaded_weight)
        if len(param.data_container) == 2:
            self.qweight = param.materialize_nested()
        return

    param_data = param.data
    output_dim = getattr(param, "output_dim", None)
    shard_dim = getattr(param, "shard_dim", None)
    # Special case for AQLM codebooks.
    is_metadata = getattr(param, "is_metadata", False)
    # Special case for per-tensor scale to load scalar into fused array.
    needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

    if loaded_shard_id is None:
        raise NotImplementedError("Loaded shard id is None.")
        # Loaded weight is already fused on disk (mlp).
        # (e.g., Phi-3's gate_up_proj).
        if output_dim is None:
            if needs_scalar_to_array:
                param_data, loaded_weight = adjust_scalar_to_fused_array(
                    param_data, loaded_weight, 0)

            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
            return
        current_shard_offset = 0
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                        False)
        shard_offsets: List[Tuple[int, int, int]] = []
        for i, output_size in enumerate(self.output_sizes):
            shard_offsets.append((i, current_shard_offset, output_size))
            current_shard_offset += output_size
        packed_dim = getattr(param, "packed_dim", None)
        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor
                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            if use_bitsandbytes_4bit:
                index = list(itertools.accumulate([0] + self.output_sizes))
                orig_offsets = {
                    str(i): (index[i], size)
                    for i, size in enumerate(self.output_sizes)
                }
                orig_offsets["total"] = (self.output_size, 0)
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_offsets, str(shard_id))

            loaded_weight_shard = loaded_weight.narrow(
                output_dim, shard_offset, shard_size)
            self.weight_loader(param, loaded_weight_shard, shard_id)
        return

    assert loaded_shard_id < len(self.output_sizes)
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    if output_dim is not None:
        # Note:
        block_size = sum(self.output_sizes) // tp_size  # the same size but on different devices
        shard_size = self.output_sizes[loaded_shard_id] // tp_size

        # our presumption: all the shards have the same sizes
        assert all(self.output_sizes[0] == size for size in self.output_sizes)
        assert sum(self.output_sizes) % tp_size == 0
        assert self.output_sizes[loaded_shard_id] % tp_size == 0

        # Special case for quantization.
        # If quantized, we need to adjust the offset and size to account
        # for the packing.
        packed_dim = getattr(param, "packed_dim", None)
        if packed_dim == output_dim:
            raise NotImplementedError("Packed dim is not implemented.")
            shard_size = shard_size // param.pack_factor
            shard_offset = shard_offset // param.pack_factor
            # Special case for Marlin.
            shard_size, shard_offset = adjust_marlin_shard(
                param, shard_size, shard_offset)

        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                        False)
        if use_bitsandbytes_4bit:
            raise NotImplementedError("Bitsandbytes 4bit quantization is not implemented.")
            shard_size = loaded_weight.shape[output_dim]
            shard_offset = loaded_weight.shape[output_dim] * loaded_shard_id

        # only load a small fraction of the weight/
        param_data = param_data.narrow(shard_dim, loaded_shard_id, 1).squeeze(0)
        start_idx = tp_rank * shard_size
        # bitsandbytes loads the weights of the specific portion
        # no need to narrow here
        if not use_bitsandbytes_4bit:
            # NOTE: DIMENSION IS HARDCODED HERE, BEWARE OF HOW WEIGHTS ARE BEING STORED IN PYTORCH
            # loaded_weight = loaded_weight.narrow(output_dim, start_idx,
            #                                      shard_size)
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size).t()
    # Special case for AQLM codebooks.
    elif is_metadata:
        raise NotImplementedError("Metadata is not implemented.")
        # metadata indicates fixed size concatenated along dim 0
        shard_size = loaded_weight.shape[0]
        shard_offset = loaded_shard_id * shard_size
        param_data = param_data.narrow(0, shard_offset, shard_size)

    # Special case for per-tensor scales in fused case.
    elif needs_scalar_to_array:
        raise NotImplementedError("Scalar to array is not implemented.")
        param_data, loaded_weight = adjust_scalar_to_fused_array(
            param_data, loaded_weight, loaded_shard_id)

    else:
        ignore_warning = getattr(param, "ignore_warning", False)
        if not ignore_warning:
            logger.warning(
                "Loading a weight without `output_dim` attribute in "
                "MergedColumnParallelLinear, assume the weight is "
                "the same for all partitions.")
    assert param_data.shape == loaded_weight.shape, f"{param_data.shape=}, {loaded_weight.shape=}"
    param_data.copy_(loaded_weight)


class OrderedMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Packed linear layers with column parallelism.

    Similar to ColumnParallelLinear, but the weight matrix is concatenated
    along the output dimension. When the weight matrix is loaded, the
    different partitions are sharded separately.

    Modification: The order of output is guaranteed.

    Args:
        input_size: input dimension of the linear layer.
        output_sizes: list of output dimensions of the linear layer.
        bias: If true, add bias.
        gather_output: If true, call all-gather on output and make the output
                       available to all GPUs, otherwise, every GPU will have
                       its own output.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
        prefix: The name of the layer in the state dict, including all parents
                        (e.g. model.layers.0.qkv_proj)
    """
    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):

        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            if loaded_shard_id is not None:
                param.data[loaded_shard_id].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {
                    i: loaded_weight.item()
                    for i, _ in enumerate(self.output_sizes)
                }
            return

        if is_gguf_weight:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()

            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // tp_size
            start_idx = tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                if len(param.data_container) == 2:
                    self.qweight = param.materialize_nested()
                return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (mlp).
            # (e.g., Phi-3's gate_up_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                if use_bitsandbytes_4bit:
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id))

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        assert len(self.output_sizes) == 2, "Only support 2 shards for now."
        should_load_shard_id = 0 if (tp_rank + 1) - tp_size // len(self.output_sizes) <= 0 else 1
        if should_load_shard_id != loaded_shard_id:
            return  # skip

        if output_dim is not None:
            local_parallelism = tp_size // len(self.output_sizes)
            block_size = sum(self.output_sizes) // tp_size  # the same size but on different devices

            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                raise NotImplementedError("Packed dim is not implemented.")

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                raise NotImplementedError("Bitsandbytes 4bit quantization is not implemented.")

            # param_data = param_data.narrow(output_dim, 0, block_size)
            local_tp_rank = tp_rank % local_parallelism
            start_idx = local_tp_rank * block_size
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, block_size)

        # Special case for AQLM codebooks.
        elif is_metadata:
            raise NotImplementedError("Metadata is not implemented.")

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            raise NotImplementedError("Scalar to array is not implemented.")

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class MergedBatchedLinear(MergedColumnParallelLinear):

    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 bias: bool = True,
                 gather_output: bool = False,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        super().__init__(input_size=input_size,
                         output_sizes=output_sizes,
                         bias=bias,
                         gather_output=gather_output,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

        # monkey patch to modify the forward method
        if quant_config is not None:
            raise NotImplementedError("Quantization is not implemented.")

        self.world_size = get_tensor_model_parallel_world_size()
        self.shard_nums = len(output_sizes)

        # view the weight as a 3D tensor
        transposed_weight = self.weight.view(self.shard_nums, -1, input_size)  # [2, N, L]
        self.weight = torch.nn.Parameter(transposed_weight.permute(0, 2, 1))
        set_weight_attrs(self.weight, {"shard_dim": 0, "input_dim": 1, "output_dim": 2, "weight_loader": self.weight_loader})

    def forward(self, input_):
        bias = self.bias if not self.skip_bias_add else None
        assert bias is None

        # Takeover the matrix multiplication.
        # output_parallel = self.quant_method.apply(self, input_, bias)

        # Batched GEMM
        input_shape = input_.shape  # [M, 2*D]  [8192, 4544]
        assert len(input_shape) == 2  # two dimensions
        input_ = input_.view(input_shape[0], self.shard_nums, -1).permute(1, 0, 2)  # [2, M, D] w/o movement

        output = torch.bmm(input_, self.weight).permute(1, 0, 2).reshape(input_shape[0], -1)  # [M, 2*N] w. movement

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    # monkey patch
    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):
        weight_loader_merged_column_parallel(self, param, loaded_weight, loaded_shard_id)


class QKVOrderedMergedColumnParallelLinear(MergedColumnParallelLinear):
    def __init__(self,
                 ranks: List[int],
                 hidden_size: int,
                 bias: bool = True,
                 gather_output: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.hidden_size = hidden_size
        self.ranks = ranks
        
        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()
        
        assert len(ranks) == 3
        assert sum(ranks) % tp_size == 0, f"{sum(ranks)=}, {tp_size=}, {sum(ranks)%tp_size=}"
        
        input_size = self.hidden_size
        self.output_sizes = self.ranks
        
        super().__init__(input_size=input_size,
                         output_sizes=self.output_sizes,
                         bias=bias,
                         gather_output=gather_output,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)
    
    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):
        if type(loaded_shard_id) == str:
            idx_map = {"q": 0, "k": 1, "v": 2}
            loaded_shard_id = idx_map[loaded_shard_id]
        
        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            raise NotImplementedError("GGUF weight type not implemented yet.")
            if loaded_shard_id is not None:
                param.data[loaded_shard_id].copy_(loaded_weight)
                param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            else:
                param.shard_weight_type = {
                    i: loaded_weight.item()
                    for i, _ in enumerate(self.output_sizes)
                }
            return

        if is_gguf_weight:
            raise NotImplementedError("GGUF weight type not implemented yet.")
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()

            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // tp_size
            start_idx = tp_rank * shard_size

            if loaded_shard_id is not None:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)
                param.shard_id.append(loaded_shard_id)
                param.shard_id_map[loaded_shard_id] = len(param.data_container)
                param.data_container.append(loaded_weight)
                if len(param.data_container) == 2:
                    self.qweight = param.materialize_nested()
                return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)
        # Special case for per-tensor scale to load scalar into fused array.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            raise NotImplementedError("Loaded shard id is None.")
            # Loaded weight is already fused on disk (mlp).
            # (e.g., Phi-3's gate_up_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            current_shard_offset = 0
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            shard_offsets: list[tuple[int, int, int]] = []
            for i, output_size in enumerate(self.output_sizes):
                shard_offsets.append((i, current_shard_offset, output_size))
                current_shard_offset += output_size
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantization.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                if use_bitsandbytes_4bit:
                    index = list(itertools.accumulate([0] + self.output_sizes))
                    orig_offsets = {
                        str(i): (index[i], size)
                        for i, size in enumerate(self.output_sizes)
                    }
                    orig_offsets["total"] = (self.output_size, 0)
                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_offsets, str(shard_id))

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id < len(self.output_sizes)
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        
        # Load Balancing
        block_size = sum(self.output_sizes) // tp_size
        
        '''Determine which shard and which data the current device is loading'''
        # The global start/end of the current shard
        shard_start = sum(self.output_sizes[:loaded_shard_id])
        shard_end = shard_start + self.output_sizes[loaded_shard_id] - 1
        
        # Block start/end for the current device
        block_start = tp_rank * block_size
        block_end = block_start + block_size - 1

        # Calculate the intersection area between shards and blocks, 
        # and dynamically decide whether to load; where to load
        intersect_start = max(shard_start, block_start)
        intersect_end = min(shard_end, block_end)
        if intersect_start > intersect_end:
            return  # No intersection, no loading
        
        if output_dim is not None:
            # Calculate the offset and length within the shard
            shard_offset = intersect_start - shard_start
            copy_length = intersect_end - intersect_start + 1
            
            # Calculate the offset within the block
            block_offset = intersect_start - block_start

            # Special case for quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                raise NotImplementedError("Packed dim is not implemented.")

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            is_sharded_weight = getattr(param, "is_sharded_weight", False)
            # bitsandbytes loads the weights of the specific portion
            # no need to narrow
            is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit

            if use_bitsandbytes_4bit:
                raise NotImplementedError("Bitsandbytes 4bit quantization is not implemented.")

            param_data = param_data.narrow(output_dim, block_offset, copy_length)
            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, shard_offset, copy_length)

        # Special case for AQLM codebooks.
        elif is_metadata:
            raise NotImplementedError("Metadata is not implemented.")

        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            raise NotImplementedError("Scalar to array is not implemented.")

        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "MergedColumnParallelLinear, assume the weight is "
                    "the same for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class QKVSVDParallelLinear(ColumnParallelLinear):
    """Linear layers for MHA low-rank decomposed QKV transformation.

    Args:
        ranks: decomposition ranks for Q, K, and V, multiple of 16.
        chunk_size: parallel granularity for the decomposition dimension, multiple of 16.
    """

    def __init__(self,
                 ranks: List[int],
                 chunk_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 skip_bias_add: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = ""):
        self.hidden_size = hidden_size

        self.ranks = ranks

        # Divide the weight matrix along the last dimension.
        tp_size = get_tensor_model_parallel_world_size()

        def find_proper_chunk_size(rank, tp_size, chunk_size):
            '''
            Ensure that
            (1) rank is divisible by the chunk_size,
            (2) (rank // chunk_size) is divisible by the tp_size
            '''

            def is_valid(chunk):
                return rank % chunk == 0 and (rank // tp_size // chunk) % 2 == 0

            if is_valid(chunk_size):
                return chunk_size
            lower_chunk = chunk_size
            upper_chunk = chunk_size
            while True:
                lower_chunk -= 1
                upper_chunk += 1
                if lower_chunk > 0 and is_valid(lower_chunk):
                    return lower_chunk
                if is_valid(upper_chunk):
                    return upper_chunk

        self.chunk_size = find_proper_chunk_size(self.ranks[0], tp_size, chunk_size)

        # total number of decomposed attention query/key/value chunks.
        self.total_num_q_chunks = self.ranks[0] // self.chunk_size
        self.total_num_k_chunks = self.ranks[1] // self.chunk_size
        self.total_num_v_chunks = self.ranks[2] // self.chunk_size

        assert len(ranks) == 3
        assert self.chunk_size > 0
        assert self.chunk_size <= min(ranks)
        assert (ranks[0] // self.chunk_size // tp_size) % 2 == 0, f"{ranks[0]=}, {self.chunk_size=}, {tp_size=}"
        assert (ranks[1] // self.chunk_size // tp_size) % 2 == 0, f"{ranks[1]=}, {self.chunk_size=}, {tp_size=}"
        assert (ranks[2] // self.chunk_size // tp_size) % 2 == 0, f"{ranks[2]=}, {self.chunk_size=}, {tp_size=}"

        self.num_q_chunks = divide(self.total_num_q_chunks, tp_size)
        if tp_size >= self.total_num_k_chunks:
            self.num_k_chunks = 1
            self.num_k_chunk_replicas = divide(tp_size, self.total_num_k_chunks)
        else:
            self.num_k_chunks = divide(self.total_num_k_chunks, tp_size)
            self.num_k_chunk_replicas = 1
        if tp_size >= self.total_num_v_chunks:
            self.num_v_chunks = 1
            self.num_v_chunk_replicas = divide(tp_size, self.total_num_v_chunks)
        else:
            self.num_v_chunks = divide(self.total_num_v_chunks, tp_size)
            self.num_v_chunk_replicas = 1
        input_size = self.hidden_size

        output_size = (self.num_q_chunks + self.num_k_chunks +
                       self.num_v_chunks) * tp_size * self.chunk_size
        self.output_sizes = [
            self.num_q_chunks * self.chunk_size * tp_size,  # q_proj
            self.num_k_chunks * self.chunk_size * tp_size,  # k_proj
            self.num_v_chunks * self.chunk_size * tp_size,  # v_proj
        ]

        super().__init__(input_size=input_size,
                         output_size=output_size,
                         bias=bias,
                         gather_output=False,
                         skip_bias_add=skip_bias_add,
                         params_dtype=params_dtype,
                         quant_config=quant_config,
                         prefix=prefix)

    '''
    Determine the starting position and size of q/k/v in the concatenated weight matrix
    '''

    def _get_shard_offset_mapping(self, loaded_shard_id: str):
        shard_offset_mapping = {
            "q": 0,
            "k": self.num_q_chunks * self.chunk_size,
            "v": (self.num_q_chunks + self.num_k_chunks) * self.chunk_size,
            "total": (self.num_q_chunks + self.num_k_chunks + self.num_v_chunks) * self.chunk_size
        }
        return shard_offset_mapping.get(loaded_shard_id)

    def _get_shard_size_mapping(self, loaded_shard_id: str):
        shard_size_mapping = {
            "q": self.num_q_chunks * self.chunk_size,
            "k": self.num_k_chunks * self.chunk_size,
            "v": self.num_v_chunks * self.chunk_size,
        }
        return shard_size_mapping.get(loaded_shard_id)

    def _load_fused_module_from_checkpoint(self, param: BasevLLMParameter,
                                           loaded_weight: torch.Tensor):
        """
        Handle special case for models where QKV layers are already
        fused on disk. In this case, we have no shard id. This function
        determmines the shard id by splitting these layers and then calls
        the weight loader using the shard id.

        An example of a model with these fused layers:
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
        """
        shard_offsets = [
            # (shard_id, shard_offset, shard_size)
            ("q", 0, self.total_num_q_chunks * self.chunk_size),
            ("k", self.total_num_q_chunks * self.chunk_size,
             self.total_num_k_chunks * self.chunk_size),
            ("v",
             (self.total_num_q_chunks + self.total_num_k_chunks) * self.chunk_size,
             self.total_num_v_chunks * self.chunk_size),
        ]

        for shard_id, shard_offset, shard_size in shard_offsets:
            # Special case for Quantization.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            if isinstance(param, (PackedColumnParameter, PackedvLLMParameter
                                  )) and param.packed_dim == param.output_dim:
                shard_size, shard_offset = \
                    param.adjust_shard_indexes_for_packing(
                        shard_size=shard_size, shard_offset=shard_offset)

            loaded_weight_shard = loaded_weight.narrow(param.output_dim,
                                                       shard_offset,
                                                       shard_size)
            self.weight_loader_v2(param, loaded_weight_shard, shard_id)

    def weight_loader_v2(self,
                         param: BasevLLMParameter,
                         loaded_weight: torch.Tensor,
                         loaded_shard_id: Optional[str] = None):
        if loaded_shard_id is None:  # special case for certain models
            if isinstance(param, PerTensorScaleParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight, shard_id=0)
                return
            elif type(param) in (RowvLLMParameter, BasevLLMParameter):
                param.load_qkv_weight(loaded_weight=loaded_weight)
                return
            # TODO: @dsikka - move to parameter.py
            self._load_fused_module_from_checkpoint(param, loaded_weight)
            return

        assert loaded_shard_id in ["q", "k", "v"]

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)

        if loaded_shard_id == "k":
            replicas = self.num_k_chunk_replicas
        elif loaded_shard_id == "v":
            replicas = self.num_v_chunk_replicas
        else:  # 'q'
            replicas = 1
        param.load_qkv_weight(loaded_weight=loaded_weight,
                              num_heads=replicas,
                              shard_id=loaded_shard_id,
                              shard_offset=shard_offset,
                              shard_size=shard_size)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[str] = None):
        # Special case for GGUF
        # initialize GGUF param after we know the quantize type
        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type and loaded_shard_id is not None:
            idx_map = {"q": 0, "k": 1, "v": 2}
            param.data[idx_map[loaded_shard_id]].copy_(loaded_weight)
            param.shard_weight_type[loaded_shard_id] = loaded_weight.item()
            return

        if is_gguf_weight:
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank = get_tensor_model_parallel_rank()

            output_dim = getattr(param, "output_dim", None)
            shard_size = loaded_weight.size(output_dim) // tp_size
            start_idx = tp_rank * shard_size

            loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                 shard_size)

            param.shard_id.append(loaded_shard_id)
            param.shard_id_map[loaded_shard_id] = len(param.data_container)
            param.data_container.append(loaded_weight)
            if len(param.data_container) == 3:
                self.qweight = param.materialize_nested()
            return

        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)

        # Special case for per-tensor scales in fused case.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv).
            # (e.g., Phi-3's qkv_proj).
            if output_dim is None:
                if needs_scalar_to_array:
                    param_data, loaded_weight = adjust_scalar_to_fused_array(
                        param_data, loaded_weight, 0)

                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                ("q", 0, self.total_num_q_chunks * self.chunk_size),
                ("k", self.total_num_q_chunks * self.chunk_size,
                 self.total_num_k_chunks * self.chunk_size),
                ("v", (self.total_num_q_chunks + self.total_num_k_chunks) *
                 self.chunk_size, self.total_num_v_chunks * self.chunk_size),
            ]
            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)

            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # Special case for Quantized Weights.
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # Special case for Marlin.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                if use_bitsandbytes_4bit:
                    orig_qkv_offsets = {
                        "q": (0, self.total_num_q_chunks * self.chunk_size),
                        "k": (self.total_num_q_chunks * self.chunk_size,
                              self.total_num_k_chunks * self.chunk_size),
                        "v":
                            ((self.total_num_q_chunks + self.total_num_k_chunks) *
                             self.chunk_size,
                             self.total_num_v_chunks * self.chunk_size),
                        "total":
                            ((self.total_num_q_chunks + self.total_num_k_chunks
                              + self.total_num_v_chunks) * self.chunk_size, 0)
                    }

                    shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                        param, orig_qkv_offsets, shard_id)

                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size)
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        tp_rank = get_tensor_model_parallel_rank()
        assert loaded_shard_id in ["q", "k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "q":
                shard_offset = 0
                shard_size = self.num_q_chunks * self.chunk_size
            elif loaded_shard_id == "k":
                shard_offset = self.num_q_chunks * self.chunk_size
                shard_size = self.num_k_chunks * self.chunk_size
            elif loaded_shard_id == "v":
                shard_offset = (self.num_q_chunks + self.num_k_chunks) * self.chunk_size
                shard_size = self.num_v_chunks * self.chunk_size
            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset)

            use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit",
                                            False)
            if use_bitsandbytes_4bit:
                orig_qkv_offsets = {
                    "q": (0, self.num_q_chunks * self.chunk_size),
                    "k": (self.num_q_chunks * self.chunk_size,
                          self.num_k_chunks * self.chunk_size),
                    "v":
                        ((self.num_q_chunks + self.num_k_chunks) * self.chunk_size,
                         self.num_v_chunks * self.chunk_size),
                    "total":
                        ((self.num_q_chunks + self.num_k_chunks + self.num_v_chunks) * self.chunk_size,
                         0)
                }
                shard_size, shard_offset = adjust_bitsandbytes_4bit_shard(
                    param, orig_qkv_offsets, loaded_shard_id)

            param_data = param_data.narrow(output_dim, shard_offset,
                                           shard_size)
            if loaded_shard_id == "q":
                shard_id = tp_rank
            elif loaded_shard_id == "k":
                shard_id = tp_rank // self.num_k_chunk_replicas
            elif loaded_shard_id == "v":
                shard_id = tp_rank // self.num_v_chunk_replicas
            start_idx = shard_id * shard_size

            # bitsandbytes loads the weights of the specific portion
            # no need to narrow here
            if not use_bitsandbytes_4bit:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)

        # Special case for for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_index = ["q", "k", "v"].index(loaded_shard_id)
            param_data = param_data.narrow(0, shard_index * shard_size,
                                           shard_size)
        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id)
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions.")

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


class WithOutputColumnParallelLinear(ColumnParallelLinear):

    def forward(self, input_, out=None) -> None:
        bias = self.bias if not self.skip_bias_add else None

        # Matrix multiply.
        assert self.quant_method is not None
        assert not self.gather_output  # disable gather
        assert isinstance(self.quant_method, UnquantizedLinearMethod)  # assume there is no quantization

        if out is not None:
            # TODO: replace with cublas call
            torch.matmul(input_, self.weight.t(), out=out)
            if bias is not None:
                out += bias
            return

        output = self.quant_method.apply(self, input_, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

        # output_parallel = F.linear(input_, self.weight, bias, output=)  #  self.quant_method.apply(self, input_, bias)