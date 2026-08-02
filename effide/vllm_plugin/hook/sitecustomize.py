from typing import Optional
# add customized flash_attn backends
import enum
import os
import importlib

# register our models
from effide.vllm_plugin.model_executor.models.llama import SVDLlamaForCausalLM
from effide.vllm_plugin.model_executor.models.opt import SVDOPTForCausalLM
from vllm import ModelRegistry
ModelRegistry.register_model("SVDLlamaForCausalLM", SVDLlamaForCausalLM)
ModelRegistry.register_model("SVDOPTForCausalLM", SVDOPTForCausalLM)

if os.environ.get("ENABLE_LOW_RANK_CACHE", "OFF") == "ON":

    vllm_interface = importlib.import_module("vllm.platforms.interface")

    class CustomizedBackend(enum.Enum):
        FLASH_ATTN = enum.auto()
        FLASH_ATTN_VLLM_V1 = enum.auto()
        XFORMERS = enum.auto()
        ROCM_FLASH = enum.auto()
        TORCH_SDPA = enum.auto()
        OPENVINO = enum.auto()
        FLASHINFER = enum.auto()
        TRITON_MLA = enum.auto()
        HPU_ATTN = enum.auto()
        PALLAS = enum.auto()
        IPEX = enum.auto()
        BLOCK_SPARSE_FLASH_ATTN = enum.auto()
        NO_ATTENTION = enum.auto()
        CUSTOMIZED_FLASH_ATTN = enum.auto()

    vllm_interface.__dict__["_Backend"] = CustomizedBackend

    # register customized flash_attn backend
    import vllm.attention.selector
    import vllm.platforms.cuda
    original_get_attn_backend_cls = vllm.platforms.cuda.CudaPlatformBase.get_attn_backend_cls


    def backend_name_to_enum(backend_name: str) -> Optional[CustomizedBackend]:
        """
        Convert a string backend name to a _Backend enum value.

        Returns:
        * _Backend: enum value if backend_name is a valid in-tree type
        * None: otherwise it's an invalid in-tree type or an out-of-tree platform is
                loaded.
        """
        assert backend_name is not None
        return CustomizedBackend[backend_name] if backend_name in CustomizedBackend.__members__ else \
              None


    def customized_get_attn_backend_cls(cls, selected_backend, head_size, dtype,
                                        kv_cache_dtype, block_size, use_v1,
                                        use_mla) -> str:
        if selected_backend == CustomizedBackend.CUSTOMIZED_FLASH_ATTN:
            from vllm.logger import init_logger
            logger = init_logger(__name__)
            logger.info("Using customized flash_attn backend.")
            return "effide.vllm_plugin.attention.backends.cu_flash_attn.CustomizedFlashAttentionBackend"

        # forward to original method
        return original_get_attn_backend_cls(selected_backend, head_size, dtype, kv_cache_dtype, block_size, use_v1, use_mla)

    vllm.platforms.cuda.CudaPlatformBase.get_attn_backend_cls = customized_get_attn_backend_cls
    vllm.attention.selector.backend_name_to_enum = backend_name_to_enum

    from effide.vllm_plugin.variadic_cache_manager import VariadicCacheEngine
    import vllm.worker.cache_engine
    vllm.worker.cache_engine.CacheEngine = VariadicCacheEngine
