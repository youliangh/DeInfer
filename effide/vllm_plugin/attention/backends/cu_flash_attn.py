# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union, Callable
from contextlib import contextmanager
import functools
import copy
import time
import os

import numpy as np
import torch
import nvtx

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import (
    PAD_SLOT_ID, CommonAttentionState, compute_slot_mapping,
    compute_slot_mapping_start_idx, get_num_prefill_decode_query_kv_tokens,
    get_seq_len_block_table_args, is_all_cross_attn_metadata_set,
    is_all_encoder_attn_metadata_set, is_block_tables_empty)
from vllm.envs import VLLM_FLASH_ATTN_VERSION
from vllm.logger import init_logger
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.platforms import current_platform
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm.vllm_flash_attn import (fa_version_unsupported_reason,
                                  flash_attn_varlen_func,
                                  flash_attn_with_kvcache,
                                  is_fa_version_supported)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)

logger = init_logger(__name__)

ENABLE_CONTIGUOUS_COPY = os.environ.get("ENABLE_CONTIGUOUS_COPY", "OFF")
if ENABLE_CONTIGUOUS_COPY not in ["ON", "OFF"]:
    print(f"Invalid value for ENABLE_CONTIGUOUS_COPY: {ENABLE_CONTIGUOUS_COPY}, specify 'ON' or 'OFF'.")
    exit(-1)
else:
    if ENABLE_CONTIGUOUS_COPY == "OFF":
        ENABLE_CONTIGUOUS_COPY = False
    else:
        ENABLE_CONTIGUOUS_COPY = True


def nvtx_wrapper(fn: Callable, desc: str, color: str = "red"):
    def wrapper(*args, **kwargs):
        with nvtx.annotate(message=desc, color=color):
            return fn(*args, **kwargs)
    return wrapper


def scan_contiguous_blocks(block_list: Union[List[int], np.ndarray]) -> List[int]:
    if len(block_list) == 0:
        return []

    start_idx: int = -1
    prev_idx: int = -1
    contiguous_block_list: List[int] = []
    for i, block_idx in enumerate(block_list):
        if start_idx == -1:
            start_idx = block_idx
            prev_idx = block_idx
            continue

        if block_idx - prev_idx == 1:
            prev_idx = block_idx
        else:
            contiguous_block_list += [start_idx, prev_idx - start_idx + 1]
            start_idx = block_idx
            prev_idx = block_idx

        if i == len(block_list) - 1:
            contiguous_block_list += [start_idx, prev_idx - start_idx + 1]

    if ENABLE_CONTIGUOUS_COPY:
        assert contiguous_block_list[0] == 0 and len(contiguous_block_list) == 2, "convention breaks"
    return contiguous_block_list


class CustomizedFlashAttentionState(CommonAttentionState):
    def __init__(self, runner):
        self.runner = runner
        self._is_graph_capturing = False

        # hack runner for injecting a new field remapping_table_blocks
        if not hasattr(self.runner, "remapping_block_tables"):
            setattr(self.runner, "remapping_block_tables", np.zeros_like(self.runner.graph_block_tables))
        if not hasattr(self.runner, "contiguous_block_list"):
            setattr(self.runner, "contiguous_block_list", np.array([0]).astype(np.int32))
        if not hasattr(self.runner, "block_per_seq"):
            setattr(self.runner, "block_per_seq", np.zeros(self.runner.graph_block_tables.shape[0]).astype(np.int32))

        super().__init__(runner)

    @contextmanager
    def graph_capture(self, max_batch_size: int):

        self._is_graph_capturing = True

        self._graph_slot_mapping = torch.full((max_batch_size, ),
                                              PAD_SLOT_ID,
                                              dtype=torch.long,
                                              device=self.runner.device)
        self._graph_seq_lens = torch.ones(max_batch_size,
                                          dtype=torch.int32,
                                          device=self.runner.device)
        self._graph_block_tables = torch.from_numpy(
            self.runner.graph_block_tables).to(device=self.runner.device)
        setattr(self, "_graph_remapping_block_tables",
                torch.from_numpy(self.runner.remapping_block_tables).to(device=self.runner.device)
        )
        setattr(self, "_contiguous_block_list",
                torch.from_numpy(self.runner.contiguous_block_list).to(device=self.runner.device)
        )
        setattr(self, "_block_per_seq",
                torch.from_numpy(self.runner.block_per_seq).to(device=self.runner.device)
        )

        yield

        self._is_graph_capturing = False
        del self._graph_slot_mapping
        del self._graph_seq_lens
        del self._graph_block_tables
        delattr(self, "_graph_remapping_block_tables")
        delattr(self, "_contiguous_block_list")
        delattr(self, "_block_per_seq")

    def get_graph_input_buffers(
            self,
            attn_metadata,
            is_encoder_decoder_model: bool = False) -> Dict[str, Any]:
        input_buffers = {
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
            "remapping_block_tables": attn_metadata.decode_metadata.remapping_block_tables,
            "contiguous_block_list": attn_metadata.decode_metadata.contiguous_block_list,
            "block_per_seq": attn_metadata.decode_metadata.block_per_seq,
        }
        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers and
            # Flash Attention backend. Assert the same.
            assert self.runner.attn_backend.get_name() in\
                ["XFORMERS", "FLASH_ATTN"], \
                f"Expected attn_backend name to be either 'XFORMERS' or "\
                f"'FLASH_ATTN', but "\
                f"got '{self.runner.attn_backend.get_name()}'"
            self._add_additonal_input_buffers_for_enc_dec_model(
                attn_metadata=attn_metadata, input_buffers=input_buffers)
        return input_buffers

    def prepare_graph_input_buffers(
            self,
            input_buffers,
            attn_metadata,
            is_encoder_decoder_model: bool = False) -> None:
        input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        input_buffers["remapping_block_tables"].copy_(
            attn_metadata.decode_metadata.remapping_block_tables, non_blocking=True)
        input_buffers["contiguous_block_list"].copy_(
            attn_metadata.decode_metadata.contiguous_block_list, non_blocking=True)

        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers and
            # Flash Attention backend. Assert the same.
            assert self.runner.attn_backend.get_name() in\
                ["XFORMERS", "FLASH_ATTN"], \
                f"Expected attn_backend name to be either 'XFORMERS' or "\
                f"'FLASH_ATTN', but "\
                f"got '{self.runner.attn_backend.get_name()}'"
            self._prepare_input_buffers_for_enc_dec_model(
                attn_metadata, input_buffers)

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        assert self._is_graph_capturing
        attn_metadata = self.runner.attn_backend.make_metadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
            slot_mapping=self._graph_slot_mapping[:batch_size],
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens=None,
            seq_lens_tensor=self._graph_seq_lens[:batch_size],
            max_query_len=1,
            max_decode_query_len=1,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.runner.max_seq_len_to_capture,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self._graph_block_tables[:batch_size],
            use_cuda_graph=True,
            # Newly added fields
            remapping_block_tables=self._graph_remapping_block_tables[:batch_size],
            contiguous_block_list=self._contiguous_block_list,
            block_per_seq=self._block_per_seq[:batch_size],
        )
        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers and
            # Flash Attention backend. Assert the same.
            assert self.runner.attn_backend.get_name() in \
                   ["XFORMERS", "FLASH_ATTN"], \
                f"Expected attn_backend name to be either 'XFORMERS' or " \
                f"'FLASH_ATTN', but " \
                f"got '{self.runner.attn_backend.get_name()}'"
            self._update_captured_metadata_for_enc_dec_model(
                batch_size=batch_size, attn_metadata=attn_metadata)

        return attn_metadata


class CustomizedFlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["CustomizedFlashAttentionImpl"]:
        return CustomizedFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return CustomizedFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["CustomizedFlashAttentionMetadataBuilder"]:
        return CustomizedFlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CustomizedFlashAttentionState"]:
        return CustomizedFlashAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]

        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class CustomizedFlashAttentionMetadata(AttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.

    use_cuda_graph: bool

    # Maximum query length in the batch.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    _cached_prefill_metadata: Optional["FlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["FlashAttentionMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    encoder_seq_start_loc: Optional[torch.Tensor] = None
    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None
    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    # Newly added fields for decomposed model decoding
    remapping_block_tables: Optional[torch.Tensor] = None
    contiguous_block_list: Optional[torch.Tensor] = None
    block_per_seq: Optional[torch.Tensor] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        self._cached_prefill_metadata = CustomizedFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        self._cached_decode_metadata = CustomizedFlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            # Batch may be composed of prefill|decodes, adjust query start
            # indices to refer to the start of decodes. E.g.
            # in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
            query_start_loc=(self.query_start_loc[self.num_prefills:] -
                             self.query_start_loc[self.num_prefills])
            if self.query_start_loc is not None else None,
            seq_start_loc=self.seq_start_loc[self.num_prefills:]
            if self.seq_start_loc is not None else None,
            context_lens_tensor=None,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            # Newly added fields for decomposed model decoding
            remapping_block_tables=self.remapping_block_tables,
            contiguous_block_list=self.contiguous_block_list,
            block_per_seq=self.block_per_seq)
        return self._cached_decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        assert False
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        if turn_prefills_into_decodes:
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step_flashattn(num_seqs=num_seqs,
                                   num_queries=num_queries,
                                   block_size=block_size,
                                   input_tokens=model_input.input_tokens,
                                   sampled_token_ids=sampled_token_ids,
                                   input_positions=model_input.input_positions,
                                   seq_lens=self.seq_lens_tensor,
                                   slot_mapping=self.slot_mapping,
                                   block_tables=self.block_tables)


class CustomizedFlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[CustomizedFlashAttentionMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def prepare(self):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)

            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

    def _get_graph_runner_block_tables(
            self, num_seqs: int,
            block_tables: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, List]:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[
                        i, :max_blocks] = block_table[:max_blocks]

        # Get contiguous block list
        contiguous_block_list = scan_contiguous_blocks(sorted(list(chain(*block_tables))))
        assert len(contiguous_block_list) % 2 == 0

        # Construct remapping block tables
        remapping_block_tables = graph_block_tables  # padded just as graph_block_tables

        # General solution (disable for benchmarking)
        if ENABLE_CONTIGUOUS_COPY:
            assert len(contiguous_block_list) == 2 and contiguous_block_list[0] == 0, "convention breaks, try disable uncheck copy instead"
        else:
            remapping_block_tables = copy.deepcopy(graph_block_tables)  # padded just as graph_block_tables
            total_pads = prev_bound = 0
            for i in range(0, len(contiguous_block_list), 2):
                start_idx = contiguous_block_list[i]
                block_len = contiguous_block_list[i+1]
                mask = (start_idx <= remapping_block_tables) & (remapping_block_tables < start_idx + block_len)

                if i != 0:
                    total_pads += start_idx - prev_bound

                remapping_block_tables[mask] -= total_pads
                prev_bound = start_idx + block_len

            if len(contiguous_block_list) > 0 and contiguous_block_list[0] != 0:
                remapping_block_tables -= contiguous_block_list[0]

        graph_block_tables = torch.from_numpy(graph_block_tables)
        remapping_block_tables = torch.from_numpy(remapping_block_tables)

        if self.runner.pin_memory:
            graph_block_tables.pin_memory()
            remapping_block_tables.pin_memory()

        return (graph_block_tables.to(device=self.runner.device, non_blocking=True),
                remapping_block_tables.to(device=self.runner.device, non_blocking=True),
                contiguous_block_list)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))
        block_per_seq_list = list(map(lambda x: np.ceil(x / self.block_size).astype(np.int32), seq_lens))

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables, remapping_block_tables, contiguous_block_list = \
                self._get_graph_runner_block_tables(num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
            contiguous_block_list = scan_contiguous_blocks(sorted(list(chain(*self.block_tables))))
            assert len(contiguous_block_list) % 2 == 0

            remapping_block_tables = block_tables.clone()
            if remapping_block_tables.shape[1] == 0:
                pass
            else:
                # General solution (disable for benchmarking)
                total_pads = prev_bound = 0
                for i in range(0, len(contiguous_block_list), 2):
                    start_idx = contiguous_block_list[i]
                    block_len = contiguous_block_list[i + 1]
                    mask = (start_idx <= remapping_block_tables) & (remapping_block_tables < start_idx + block_len)

                    if i != 0:
                        total_pads += start_idx - prev_bound

                    remapping_block_tables[mask] -= total_pads
                    prev_bound = start_idx + block_len

                if len(contiguous_block_list) > 0 and contiguous_block_list[0] != 0:
                    remapping_block_tables -= contiguous_block_list[0]

        # it's observed that the block manager tries to keep blocks continuous
        if len(contiguous_block_list) != 0:
            contiguous_block_list = [contiguous_block_list[1]]
        else:
            contiguous_block_list = [1]  # placeholder

        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)

        contiguous_block_list = async_tensor_h2d(contiguous_block_list, torch.int32,
                                                 device, self.runner.pin_memory)
        block_per_seq = async_tensor_h2d(block_per_seq_list, torch.int32,
                                         device, self.runner.pin_memory)

        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }

        return CustomizedFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            # Newly added arguments
            remapping_block_tables=remapping_block_tables,
            contiguous_block_list=contiguous_block_list,
            block_per_seq=block_per_seq,
        )


class CustomizedFlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:
    |<----------------- num_decode_tokens ------------------>|
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        #
        q_u_proj: torch.nn.Module,
        k_u_proj: torch.nn.Module,
        v_u_proj: torch.nn.Module,
        kv_cache_size: int,
        buffer_init_configs: Optional[Dict[str, int]] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
        #
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window - 1,
                                0) if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = CustomizedFlashAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")
        self.attn_type = attn_type

        # if hopper default to FA3, otherwise stick to FA2 for now
        # TODO(lucas): profile FA3 on ampere to see if it makes sense to
        #  use FA3 as default for both
        if current_platform.get_device_capability()[0] >= 9:
            self.fa_version = 3 if is_fa_version_supported(3) else 2
        else:
            self.fa_version = 2

        if VLLM_FLASH_ATTN_VERSION is not None:
            assert VLLM_FLASH_ATTN_VERSION in [2, 3]
            self.fa_version = VLLM_FLASH_ATTN_VERSION

        if not is_fa_version_supported(self.fa_version):
            logger.error("Cannot use FA version %d is not supported due to %s",
                         self.fa_version,
                         fa_version_unsupported_reason(self.fa_version))

        assert is_fa_version_supported(self.fa_version)

        # Reminder: remove self.k_u_proj and self.v_u_proj in OPT
        self.q_u_proj = q_u_proj
        self.k_u_proj = k_u_proj
        self.v_u_proj = v_u_proj
        self.kv_cache_size = kv_cache_size
        self.rotary_embed = rotary_emb

        if q_u_proj.bias is not None:
            self.q_fn = nvtx_wrapper(functools.partial(torch.addmm, q_u_proj.bias),
                                     desc="q_u_proj")
        else:
            self.q_fn = nvtx_wrapper(torch.matmul, desc="q_u_proj")

        if k_u_proj.bias is not None:
            self.k_fn = nvtx_wrapper(functools.partial(torch.addmm, k_u_proj.bias),
                                     desc="k_u_proj")
        else:
            self.k_fn = nvtx_wrapper(torch.matmul, desc="k_u_proj")

        if v_u_proj.bias is not None:
            self.v_fn = nvtx_wrapper(functools.partial(torch.addmm, v_u_proj.bias),
                                     desc="v_u_proj")
        else:
            self.v_fn = nvtx_wrapper(torch.matmul, desc="v_u_proj")

        self.cache_size = k_u_proj.weight.shape[1]

        self.cache_head = self.cache_size // 16
        self.cache_head_size = 16
        # try to make cache_head size as 128

        # import here to avoid starting compiling during process bootstrap
        from effide.ops import (paged_apply_rotary_embeds, cache_copy,
                                init_kv_cache_buffer, get_kv_cache_buffer, get_kv_cache_buffer_all,
                                init_kv_buffer, get_kv_buffer, get_kv_buffer_all,
                                init_stream, get_stream,
                                paged_copy)
        init_kv_buffer(
            buffer_init_configs,
            kv_size=k_u_proj.weight.shape[0],
            dtype=k_u_proj.weight.dtype,
            device=k_u_proj.weight.device,
        )
        init_kv_cache_buffer(
            buffer_init_configs,
            cache_size=self.cache_size,
            dtype=k_u_proj.weight.dtype,
            device=k_u_proj.weight.device,
        )
        init_stream()
        self.paged_apply_rotary_embeds = paged_apply_rotary_embeds
        self.paged_copy = paged_copy
        self.contiguous_cache_copy = cache_copy
        self.get_kv_buffer_all = get_kv_buffer_all
        self.get_kv_buffer = get_kv_buffer
        self.get_kv_cache_buffer_all = get_kv_cache_buffer_all
        self.get_kv_cache_buffer = get_kv_cache_buffer

        # dynamic graph
        self.query_stream, self.key_stream, self.value_stream = get_stream(3)
        self.sync_event = torch.cuda.Event()
        self.q_u_event = torch.cuda.Event()

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,        # in low-rank space
        key: torch.Tensor,          # in low-rank space
        value: torch.Tensor,        # in low-rank space
        kv_cache: torch.Tensor,     # in low-rank space
        positions: torch.Tensor,
        attn_metadata: CustomizedFlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention. (modified)

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, kv_cache_size]
            value: shape = [num_tokens, kv_cache_size]
            output: shape = [num_tokens, num_heads, head_size]
            kv_cache = [2, num_blocks, block_size, kv_cache_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            positions: shape = [num_tokens]  Note: indexes of latest tokens
            attn_metadata: Metadata for attention.
        NOTE: It in-place updates the output tensor.
        """
        # NOTE(woosuk): FlashAttention does not support FP8 KV cache.
        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0, (
            "key/v_scale is not supported in FlashAttention.")

        assert output is not None, "Output tensor must be provided."

        attn_type = self.attn_type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: Optional[torch.Tensor] = self.alibi_slopes
        logits_soft_cap: Optional[float] = self.logits_soft_cap

        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]  # .view(-1, block_size, self.kv_cache_size)
            value_cache = kv_cache[1]  # .view(-1, block_size, self.kv_cache_size)
            # We skip updating the KV cache under two conditions:
            #  a. When the Attention Type is ENCODER. In this phase, we compute
            #     only the encoder attention without updating the cache.
            #  b. When both Key and Value are None. This occurs during
            #     cross-attention computation in the decoding phase, where the
            #     KV cache is already populated with the cross-attention
            #     tensor. Thus, we skip cache updates during this time.
            if (attn_type != AttentionType.ENCODER) and (key is not None) and (
                    value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                # Reshape the input keys and values and store them in the cache.
                # If kv_cache is not provided, the new key and value tensors are
                # not cached. This happens during the initial memory
                # profiling run.
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    key.view(-1, self.cache_head, self.cache_head_size),
                    value.view(-1, self.cache_head, self.cache_head_size),
                    kv_cache[0],
                    kv_cache[1],
                    updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                    kv_cache_dtype,
                    layer._k_scale,
                    layer._v_scale,
                )

        (num_prefill_query_tokens, num_prefill_kv_tokens,
        num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        # QKV for prefill.

        if prefill_meta := attn_metadata.prefill_metadata:

            prefill_output = output[:num_prefill_query_tokens]

            # Prompt run.
            if (kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                q_seq_start_loc, q_seq_len, k_seq_start_loc, k_seq_len = \
                    _get_query_key_seq_metadata(prefill_meta, True, attn_type)

                query, _ = self.q_u_proj(query)
                key, _ = self.k_u_proj(key)
                value, _ = self.v_u_proj(value)

                if self.rotary_embed is not None:
                    query, key = self.rotary_embed(positions, query, key)

                query = query.view(-1, self.num_heads, self.head_size)
                key = key.view(-1, self.num_kv_heads, self.head_size)
                value = value.view(-1, self.num_kv_heads, self.head_size)

                query = query[:num_prefill_query_tokens]
                key = key[:num_prefill_kv_tokens]
                value = value[:num_prefill_kv_tokens]

                assert query.shape[0] == num_prefill_query_tokens

                flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=q_seq_start_loc,
                    cu_seqlens_k=k_seq_start_loc,
                    max_seqlen_q=q_seq_len,
                    max_seqlen_k=k_seq_len,
                    softmax_scale=softmax_scale,
                    causal=_get_causal_option(attn_type),
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    out=prefill_output,
                    fa_version=self.fa_version,
                )
            else:
                # prefix-enabled attention
                assert False, "Prefix-enabled attention is not supported for now"
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support prefix caching")
                assert prefill_meta.seq_lens is not None
                max_seq_len = max(prefill_meta.seq_lens)
                flash_attn_varlen_func(  # noqa
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=prefill_meta.query_start_loc,
                    max_seqlen_q=prefill_meta.max_query_len,
                    seqused_k=prefill_meta.seq_lens_tensor,
                    max_seqlen_k=max_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    block_table=prefill_meta.block_tables,
                    softcap=logits_soft_cap,
                    out=prefill_output,
                    fa_version=self.fa_version,
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.

            decode_output = output[num_prefill_query_tokens:]

            assert decode_meta.max_decode_query_len is not None
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert False, "Speculative decoding not supported for now"
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support max_decode_query_len > 1"
                )

                decode_query = query[num_prefill_query_tokens:]
                assert decode_query.shape[0] == num_decode_query_tokens
                flash_attn_varlen_func(
                    q=decode_query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_decode_query_len,
                    seqused_k=decode_meta.seq_lens_tensor,
                    max_seqlen_k=decode_meta.max_decode_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    block_table=decode_meta.block_tables,
                    out=decode_output,
                    fa_version=self.fa_version,
                )
            else:
                # Use flash_attn_with_kvcache for normal decoding.
                (
                    seq_lens_arg,
                    _,
                    block_tables_arg,
                ) = get_seq_len_block_table_args(decode_meta, False, attn_type)

                # seq_lens_arg: torch.Tensor, with len = batch_size

                # Initialize temporary key and value tensors and establish re-mapping
                block_size = key_cache.shape[1]  # [num_blocks, block_size, num_kv_heads (placeholder), head_size]

                key_buffer, value_buffer = self.get_kv_buffer()
                kv_cache_buffer = self.get_kv_cache_buffer_all()
                key_cache_buffer, value_cache_buffer = self.get_kv_cache_buffer()

                remapping_block_tables = decode_meta.remapping_block_tables
                contiguous_block_list = decode_meta.contiguous_block_list
                block_per_seq = decode_meta.block_per_seq

                self.query_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.query_stream):
                    query = self.q_fn(query, self.q_u_proj.weight.t())
                    self.q_u_event.record()

                if ENABLE_CONTIGUOUS_COPY:
                    self.contiguous_cache_copy(kv_cache, kv_cache_buffer, contiguous_block_list)
                    self.sync_event.record()
                    self.key_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.key_stream):
                        torch.cuda.current_stream().wait_event(self.sync_event)
                        self.k_fn(key_cache_buffer.view(-1, self.kv_cache_size), self.k_u_proj.weight.t(),
                                  out=key_buffer)

                    self.value_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.value_stream):
                        torch.cuda.current_stream().wait_event(self.sync_event)
                        self.v_fn(value_cache_buffer.view(-1, self.kv_cache_size), self.v_u_proj.weight.t(),
                                  out=value_buffer)
                else:
                    self.key_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.key_stream):
                        self.paged_copy(key_cache, block_tables_arg, block_per_seq, remapping_block_tables, key_cache_buffer)
                        self.k_fn(key_cache_buffer.view(-1, self.kv_cache_size), self.k_u_proj.weight.t(),
                                  out=key_buffer)

                    self.value_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.value_stream):
                        self.paged_copy(value_cache, block_tables_arg, block_per_seq, remapping_block_tables, value_cache_buffer)
                        self.v_fn(value_cache_buffer.view(-1, self.kv_cache_size), self.v_u_proj.weight.t(),
                                  out=value_buffer)

                # Stage-2: apply rotary embeddings
                torch.cuda.current_stream().wait_event(self.q_u_event)
                decode_query = query.view(-1, self.num_heads, self.head_size)[num_prefill_query_tokens:]
                torch.cuda.current_stream().wait_stream(self.key_stream)
                key_buffer = key_buffer.view(-1, block_size, self.num_kv_heads, self.head_size)
                if self.rotary_embed is not None:
                    self.paged_apply_rotary_embeds(
                        positions, decode_query, remapping_block_tables, seq_lens_arg, key_buffer,
                        self.head_size, self.rotary_embed.cos_sin_cache, self.rotary_embed.is_neox_style
                    )

                torch.cuda.current_stream().wait_stream(self.value_stream)

                value_buffer = value_buffer.view(-1, block_size, self.num_kv_heads, self.head_size)

                # Stage-3: use flash_attn_with_kvcache to perform attention
                flash_attn_with_kvcache(
                    q=decode_query.unsqueeze(1),
                    k_cache=key_buffer,
                    v_cache=value_buffer,
                    block_table=remapping_block_tables,
                    cache_seqlens=seq_lens_arg,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    out=decode_output.unsqueeze(1),
                    fa_version=self.fa_version,
                )
        return output


def _get_query_key_seq_metadata(
    attn_metadata,
    is_prompt: bool,
    attn_type: str,
) -> tuple:
    """
    Returns sequence metadata for key and query based on the specified
    attention type and whether input is a prompt.

    This function computes the starting locations and maximum sequence lengths
    for key and query sequences for different attention types.

    Args:
        attn_metadata: The attention metadata object
        is_prompt (bool): A flag indicating if the input is a prompt
        attn_type (AttentionType): The type of attention being used.

    Returns:
        tuple: A tuple containing four integers:
            - Starting location for the query sequence.
            - Maximum sequence length for the query sequence.
            - Starting location for the key sequence.
            - Maximum sequence length for the key sequence.

    Raises:
        AttributeError: If an invalid attention type is provided.
    """
    if attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_start_loc, max_seq_len,
                attn_metadata.seq_start_loc, max_seq_len)

    elif attn_type == AttentionType.ENCODER_DECODER:
        # This is cross attention between the where the key
        # is the precomputed encoder attention and query
        # is the input sequence.
        # Choose query max length based on whether it is prompt
        # or not.
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_start_loc, max_seq_len,
                attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len)
    elif attn_type == AttentionType.ENCODER:
        # For encoder attention both the query and the key are same i.e the
        # encoder sequence.
        return (attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len,
                attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len)
    elif attn_type == AttentionType.ENCODER_ONLY:
        assert is_prompt, "Should not have decode for encoder only model."
        return (attn_metadata.seq_start_loc, attn_metadata.max_prefill_seq_len,
                attn_metadata.seq_start_loc, attn_metadata.max_prefill_seq_len)
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def _get_causal_option(attn_type: str) -> bool:
    """
    Determine whether the given attention type is suitable for causal
    attention mechanisms.

    Args:
        attn_type (AttentionType): The type of attention being evaluated

    Returns:
        bool: Returns `True` if the attention type is suitable for causal
        attention (i.e., not encoder, encoder-only, or encoder-decoder),
        otherwise returns `False`.
    """
    return not (attn_type == AttentionType.ENCODER
                or attn_type == AttentionType.ENCODER_ONLY
                or attn_type == AttentionType.ENCODER_DECODER)
