# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/opt/modeling_opt.py
# Copyright 2023 The vLLM team.
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights
# reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only OPT model compatible with HuggingFace weights."""
import os
from typing import Iterable, List, Optional, Set, Tuple, Union, Type

import torch
from torch import nn
from transformers import OPTConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)

from effide.vllm_plugin.model_executor.models.experimental import (NativeOPTAttention,
                                                                 NaiveOPTAttention,
                                                                 ParallelOPTAttention,
                                                                 NativeOPTFC,
                                                                 NaiveSVDOPTFC,
                                                                 ParallelSVDOPTFC,
                                                                 PREFIX_SET_OPT)


kMLPBehavior = os.environ.get("MLP_BEHAVIOR", "INVALID")
if kMLPBehavior == "INVALID":
    raise ValueError("MLP behavior not set")
if kMLPBehavior not in ["NAIVE", "OURS"]:
    raise ValueError(f"Invalid MLP behavior: {kMLPBehavior}")

kATTNBehavior = os.environ.get("ATTN_BEHAVIOR", "INVALID")
if kATTNBehavior == "INVALID":
    raise ValueError("ATTN behavior not set")
if kATTNBehavior not in ["NAIVE", "OURS"]:
    raise ValueError(f"Invalid ATTN behavior: {kATTNBehavior}")


class OPTLearnedPositionalEmbedding(nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the
        # embedding ids by 2 and adjust num_embeddings appropriately. Other
        # models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, positions: torch.Tensor):
        return super().forward(positions + self.offset)


def choose_opt_fc(prefix: str) -> Tuple[Type[nn.Module], Type[nn.Module]]:
    cls = NativeOPTFC
    if prefix in PREFIX_SET_OPT:
        if kMLPBehavior == "OURS":
            cls = ParallelSVDOPTFC
        elif kMLPBehavior == "NAIVE" or get_tensor_model_parallel_world_size() == 1:
            cls = NaiveSVDOPTFC  # fallback
        else:  # should be unreachable
            raise RuntimeError(f"Unreachable code path: choose_llama_mlp")
    
    # prefix: It is [model.decoder.layers.1] instead of 
    # [model.decoder.layers.1.fc1/fc2]
    prefix = '.'.join(prefix.split('.')[:-1])
    
    class OPTFC(cls):
        def __init__(
            self,
            config: OPTConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ) -> None:
            super().__init__(
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            )

    return OPTFC, cls


def choose_opt_attention(prefix: str) -> Tuple[Type[nn.Module], Type[nn.Module]]:
    cls = NativeOPTAttention
    if prefix in PREFIX_SET_OPT:
        if kATTNBehavior == "OURS":
            cls = ParallelOPTAttention
        elif kATTNBehavior == "NAIVE" or get_tensor_model_parallel_world_size() == 1:
            cls = NaiveOPTAttention  # fallback
        else:  # should be unreachable
            raise RuntimeError(f"Unreachable code path: choose_llama_mlp")

    class OPTAttention(cls):
        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            bias: bool = True,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
        ) -> None:
            super().__init__(
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=bias,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )

    return OPTAttention, cls


class OPTDecoderLayer(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        opt_attn_class, _ = choose_opt_attention(prefix=f"{prefix}.self_attn")
        self.self_attn = opt_attn_class(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.enable_bias,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.do_layer_norm_before = config.do_layer_norm_before

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)
        
        # .fc1 is to hit PREFIX_SET_OPT
        opt_fc_class, _ = choose_opt_fc(prefix=f"{prefix}.fc1")
        self.fc = opt_fc_class(config, quant_config, prefix)
        
        self.final_layer_norm = nn.LayerNorm(
            self.embed_dim,
            elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc(hidden_states)
        hidden_states = residual + hidden_states
        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class OPTDecoder(nn.Module):

    def __init__(
        self,
        config: OPTConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.word_embed_proj_dim,
        )
        # Positional embeddings are replicated (not sharded).
        self.embed_positions = OPTLearnedPositionalEmbedding(
            config.max_position_embeddings, config.hidden_size)

        # Project out & in will be replicated if they exist.
        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = ReplicatedLinear(config.hidden_size,
                                                config.word_embed_proj_dim,
                                                bias=False,
                                                quant_config=quant_config,
                                                prefix=f"{prefix}.project_out")
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = ReplicatedLinear(config.word_embed_proj_dim,
                                               config.hidden_size,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.project_in")
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to
        # keep backward compatibility with checkpoints that have been fine-tuned
        # before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size,
                elementwise_affine=config.layer_norm_elementwise_affine)
        else:
            self.final_layer_norm = None

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: OPTDecoderLayer(
                config, cache_config, quant_config, prefix=prefix),
            prefix=f"{prefix}.layers")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            pos_embeds = self.embed_positions(positions)
            if self.project_in is not None:
                inputs_embeds, _ = self.project_in(inputs_embeds)
            hidden_states = inputs_embeds + pos_embeds
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states,
                                  kv_caches[i - self.start_layer],
                                  attn_metadata)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)
        if self.project_out is not None:
            hidden_states, _ = self.project_out(hidden_states)
        return hidden_states


@support_torch_compile
class OPTModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.decoder = OPTDecoder(config,
                                  cache_config,
                                  quant_config,
                                  prefix=f"{prefix}.decoder")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(["hidden_states"],
                                                    config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        return self.decoder(input_ids,
                            positions,
                            kv_caches,
                            attn_metadata,
                            intermediate_tensors,
                            inputs_embeds=inputs_embeds)


class SVDOPTForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = OPTModel(vllm_config=vllm_config,
                              prefix=maybe_prefix(prefix, "model"))
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.decoder.embed_tokens
        else:
            self.lm_head = ParallelLMHead(config.vocab_size,
                                          config.word_embed_proj_dim)
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = get_sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        DEFAULT_SELF_ATTN_STACKED_PARAMS = [# (param_name, shard_name, shard_id)
                                            (".qkv_proj", ".q_proj", "q"), 
                                            (".qkv_proj", ".k_proj", "k"), 
                                            (".qkv_proj", ".v_proj", "v")]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "lm_head.weight" in name and self.config.tie_word_embeddings:
                continue
            label = '.'.join(name.split('.')[:5])
            if name.startswith("decoder."):
                name = "model." + name
            
            # TODO: support layer-wise behavior controlling for both self_attn and mlp
            if label in PREFIX_SET_OPT:
                if name.split('.')[4] == 'self_attn':
                    _, template_class = choose_opt_attention(label)
                    stacked_params_mapping = template_class.stacked_params_mapping
                    decomposed_params_mapping = template_class.decomposed_params_mapping
                elif 'fc' in name.split('.')[4]:
                    _, template_class = choose_opt_fc(label)
                    stacked_params_mapping = DEFAULT_SELF_ATTN_STACKED_PARAMS
                    decomposed_params_mapping = template_class.decomposed_params_mapping
            else:
                # handle the un-decomposed case
                stacked_params_mapping = DEFAULT_SELF_ATTN_STACKED_PARAMS
                decomposed_params_mapping = []
            
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                if name.split('.')[-2] in ["q_proj_u", "k_proj_u", "v_proj_u"]:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                # handle special case
                for param_name, weight_name in decomposed_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                
                if 'fc1' in name or 'fc2' in name:
                    # insert 'fc': model.decoder.layers.0.fc1/2xxx to 
                    # model.decoder.layers.0.fc.fc1/2xxx
                    name_list = name.split('.')
                    name_list.insert(4, 'fc')
                    name_ = '.'.join(name_list)
                else:
                    name_ = name
                
                # Skip loading extra bias for GPTQ models.
                if name_.endswith(".bias") and name_ not in params_dict:
                    continue
                if is_pp_missing_parameter(name_, self):
                    continue
                param = params_dict[name_]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name_)
        return loaded_params
