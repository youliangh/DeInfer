import os
from typing import Optional, Tuple, Dict, List
import torch

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_current_path = os.path.dirname(__file__)

# Check if builds
if os.path.exists(os.path.join(_current_path, "build", "effide_ops")):
    raise FileNotFoundError(f"Cannot find custom operators, please build first")

from effide.ops.build import effide_ops

paged_apply_rotary_embeds = effide_ops.paged_apply_rotary_embeds
paged_copy = effide_ops.paged_copy
cache_copy = effide_ops.cache_copy

if os.environ.get("ENABLE_OPTIMIZED_OP", "ON") != "ON":
    from .pytorch_implement import torch_paged_copy
    paged_copy = torch_paged_copy

kMaxBatchSize = int(os.environ.get("BUFFER_MAX_BATCH_SIZE", 256))
kMaxSeqLen = int(os.environ.get("BUFFER_MAX_SEQ_LEN", 512))

_kv_cache_buffer: Optional[torch.Tensor] = None
_kv_buffer: Optional[torch.Tensor] = None


def init_kv_cache_buffer(buffer_init_configs: Optional[Dict[str, int]],
                         cache_size: int, dtype: torch.dtype, device: torch.device):
    global _kv_cache_buffer

    # _key_cache_buffer is None, _value_cache_buffer is None,
    not_initialized = all([_kv_cache_buffer is None])
    if not_initialized:
        print(f"Initializing key and value buffers...")
        if buffer_init_configs is None or len(buffer_init_configs) == 0:
            buffer_init_configs = {}
            print("No buffer_init_configs is provided, using default values.")
    else:
        return

    max_batch_size = buffer_init_configs.pop('max_batch_size', kMaxBatchSize)
    max_seq_len = buffer_init_configs.pop('max_seq_len', kMaxSeqLen)
    block_size = 16

    if _kv_cache_buffer is None:
        _kv_cache_buffer = torch.empty((2, max_batch_size * max_seq_len // block_size, block_size, cache_size),
                                        dtype=dtype, device=device)

    if not_initialized:
        memory_size = _kv_cache_buffer.element_size() * _kv_cache_buffer.nelement() / 1024 / 1024  # in MB
        print(f"KV cache buffers: {_kv_cache_buffer.shape}, {max_batch_size=}, {max_seq_len=}, {cache_size=}, memory: {memory_size: .2f} MB")


def get_kv_cache_buffer_all() -> torch.Tensor:
    global _kv_cache_buffer
    if _kv_cache_buffer is None:
        raise RuntimeError('KV buffer has not been initialized')
    return _kv_cache_buffer


def get_kv_cache_buffer() -> Tuple[torch.Tensor, torch.Tensor]:
    global _kv_cache_buffer
    if _kv_cache_buffer is None:
        raise RuntimeError('Cache buffer has not been initialized')
    return _kv_cache_buffer[0], _kv_cache_buffer[1]


def init_kv_buffer(buffer_init_configs: Optional[Dict[str, int]],
                   kv_size: int, dtype: torch.dtype, device: torch.device):
    global _key_buffer, _value_buffer, _kv_buffer

    not_initialized = all([_kv_buffer is None])
    if not_initialized:
        print("Initializing key and value buffers...")
        if buffer_init_configs is None or len(buffer_init_configs) == 0:
            buffer_init_configs = {}
            print("No buffer_init_configs is provided, using default values.")

    max_batch_size = buffer_init_configs.pop('max_batch_size', kMaxBatchSize)
    max_seq_len = buffer_init_configs.pop('max_seq_len', kMaxSeqLen)
    block_size = 16

    if _kv_buffer is None:
        _kv_buffer = torch.empty((2, max_batch_size * max_seq_len // block_size, block_size, kv_size),
                                    dtype=dtype, device=device)

    if not_initialized:
        memory_size = _kv_buffer.element_size() * _kv_buffer.nelement() / 1024 / 1024  # in MB
        print(f"KV buffer shape: {_kv_buffer.shape}, {max_batch_size=}, {max_seq_len=}, {kv_size=}, memory: {memory_size: .2f} MB")


def get_kv_buffer_all() -> torch.Tensor:
    global _kv_buffer
    if _kv_buffer is None:
        raise RuntimeError('KV buffer has not been initialized')
    return _kv_buffer


def get_kv_buffer() -> Tuple[torch.Tensor, torch.Tensor]:
    global _kv_buffer
    if _kv_buffer is None:
        raise RuntimeError('KV buffer has not been initialized')
    return _kv_buffer[0], _kv_buffer[1]


kNumStreams = 3
_stream_list: List[torch.cuda.Stream] = []


def init_stream():
    global _stream_list
    if len(_stream_list) == 0:
        for _ in range(kNumStreams):
            _stream_list.append(torch.cuda.Stream())


def get_stream(num_stream: int) -> List[torch.cuda.Stream]:
    global _stream_list
    if num_stream > kNumStreams:
        raise ValueError(f"Number of streams requested: {num_stream} is greater preset number of streams: {kNumStreams}")
    return _stream_list[:num_stream]
