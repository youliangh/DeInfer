# DeInfer: Efficient Parallel Inferencing for Decomposed Large Language Models

[Paper](https://arxiv.org/abs/2604.17709) | [DAC 2026](https://63dac.conference-program.com/presentation/?id=RESEARCH2620&sess=sess126) | [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3770743.3804360)

**You-Liang Huang, Xinhao Huang, Chengxi Liao, and Zeyi Wen**

DeInfer accelerates tensor-parallel inference for large language models compressed through low-rank decomposition. Built on **vLLM 0.7.2**, it combines communication in the low-rank space, partitioned attention computation, and paged low-rank KV cache reconstruction compatible with CUDA Graph. The Python source package is named `effide`.

This repository contains the inference implementation and experiment drivers. Decomposed model checkpoints and rank configurations must be supplied separately; benchmark results are not bundled. The external CUDA headers and vLLM benchmark scripts also need to be populated as described below.

## How it works

Replacing a dense projection with two low-rank factors reduces parameter storage, but a straightforward parallel implementation introduces extra collective communication and can repeat attention computation across GPUs. Reconstructing a growing low-rank KV cache also complicates CUDA Graph replay.

| Component | Implementation |
| --- | --- |
| Low-rank communication | Rearranges projection partitions so that all-gather and reduce-sum operate on compressed activations. See [model layers](effide/vllm_plugin/model_executor/models/experimental.py) and [parallel linear layers](effide/vllm_plugin/model_executor/layers/linear.py). |
| Partitioned attention | Reconstructs each GPU's local attention heads from low-rank activations, avoiding duplicated full attention computation. |
| Paged low-rank KV cache | Stores compressed K/V with per-layer cache widths, then reconstructs them into preallocated buffers for decoding. See [cache management](effide/vllm_plugin/variadic_cache_manager.py) and the [attention backend](effide/vllm_plugin/attention/backends/cu_flash_attn.py). |
| CUDA Graph support | Uses remapped block tables and custom CUDA copy/rotary-embedding operators during cache reconstruction. See [CUDA operators](effide/ops/cuda_ops.cu). |

The implementation registers `SVDLlamaForCausalLM` and `SVDOPTForCausalLM`. It provides `NAIVE` and `OURS` execution modes for comparing a straightforward decomposed baseline with DeInfer's optimized layers.


## Repository layout

```text
DeInfer/
|-- effide/
|   |-- ops/                          # CUDA operators, buffer management, PyTorch reference copy
|   `-- vllm_plugin/
|       |-- attention/               # Custom attention layer and FlashAttention backend
|       |-- hook/sitecustomize.py     # Model registration and vLLM runtime patches
|       |-- model_executor/
|       |   |-- layers/linear.py      # Tensor-parallel low-rank linear layers
|       |   `-- models/              # LLaMA, OPT, and decomposition implementations
|       `-- variadic_cache_manager.py
|-- scripts/launch_benchmark.sh       # Benchmark setup, execution, and output collection
|-- tools/benchmark_patching.py       # Registers custom models in copied benchmark scripts
|-- third_party/wmma_extension/       # External CUDA headers; populate before building
|-- vllm_benchmarks/                  # External benchmark scripts; populate before running
|-- environment.yaml                 # Recorded Linux Conda environment
`-- run_experiments.py               # Workloads and ablation configurations
```

## Requirements

Use a **Linux environment with NVIDIA GPUs**, Bash, a CUDA toolkit containing `nvcc`, and a compatible C++ compiler. The supplied environment is Linux-specific. Plan for multiple GPUs with enough memory for the selected checkpoint; the documented tensor-parallel configurations use 2, 4, or 8 GPUs.

The versions recorded in [environment.yaml](environment.yaml) include:

| Dependency | Version |
| --- | --- |
| Python | 3.9.20 |
| PyTorch | 2.5.1+cu121 |
| vLLM | 0.7.2 |
| Transformers | 4.49.0 |
| Triton | 3.1.0 |
| GCC / G++ | 12.4.0 |

You also need **CMake 3.18 or newer**, Git, and `curl`. The serving launcher uses `netstat` and standard GNU utilities. The vLLM integration patches internal APIs, so keep the pinned version when setting up this code.

## Setup

Run the following commands from a Linux shell.

### 1. Create the environment

```bash
git clone https://github.com/youliangh/DeInfer.git
cd DeInfer
conda env create -f environment.yaml
conda activate deinfer
```

The environment file is a complete pinned snapshot. It does not install the system NVIDIA driver, CUDA compiler, or CMake. Check that `nvcc --version` and `cmake --version` work before building.

### 2. Fetch the CUDA headers

[.gitmodules](.gitmodules) declares `wmma_extension`, but this checkout does not contain a tracked submodule revision. If `third_party/wmma_extension/` is absent or empty, populate it explicitly:

```bash
git clone https://github.com/youliangh/wmma_extension.git third_party/wmma_extension
```

The build expects `third_party/wmma_extension/include/wmma_extension/`. Record the dependency commit used for your experiments, since this repository does not pin it.

### 3. Build the custom operators

[CMakeLists.txt](effide/ops/CMakeLists.txt) currently sets `COMPILE_CC` to `80`. Adjust that value for your GPU architecture before building. It also contains a historical Torch search path using an environment named `effide`; the explicit `Torch_DIR` below selects Torch from the active `deinfer` environment instead.

```bash
cmake -S effide/ops -B effide/ops/build \
  -DPATH_TO_DEINFER="$PWD" \
  -DTorch_DIR="$(python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")')" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CUDA_STANDARD=17

cmake --build effide/ops/build --parallel
python -c "from effide.ops.build import effide_ops; print(effide_ops.__file__)"
```

The legacy `PATH_TO_YOUR_CONDA` warning can appear during configuration; `Torch_DIR` supplies the package location directly. Keep the output in `effide/ops/build`, which is where the Python code imports `effide_ops`. The extension is required even when selecting the PyTorch reference copy implementation.

### 4. Populate the benchmark scripts

The launcher expects four files under `vllm_benchmarks/`. Retrieve the scripts from the matching [vLLM v0.7.2 benchmark suite](https://github.com/vllm-project/vllm/tree/v0.7.2/benchmarks):

```bash
mkdir -p vllm_benchmarks
for file in benchmark_latency.py benchmark_throughput.py benchmark_serving.py backend_request_func.py; do
  curl --fail --location \
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.7.2/benchmarks/$file" \
    --output "vllm_benchmarks/$file" || break
done
```

The launcher copies these files into each run directory and patches the copies to register DeInfer's models. No package installation command is provided for `effide`; the launcher adds the repository and its `sitecustomize.py` hook to `PYTHONPATH`.

## Prepare a decomposed checkpoint

Supply a local Hugging Face-style model directory containing model configuration, tokenizer files, decomposed weights, and `desired_r.pkl`. This repository does not include a checkpoint decomposition or conversion pipeline. Renaming a dense model's architecture is insufficient: the weights and rank configuration must already match the decomposed model.

1. Set `architectures` in `config.json` to `["SVDLlamaForCausalLM"]` or `["SVDOPTForCausalLM"]`.
2. Provide factor weights using the names expected by the loaders, such as `model.layers.0.self_attn.q_proj.v.weight` and `model.layers.0.self_attn.q_proj.u.weight`. For an original weight of shape `[out_features, in_features]` and retained rank `r`, these tensors have shapes `[r, in_features]` and `[out_features, r]`, respectively.
3. Prepare the compression-rank metadata with reference to the [SoLA implementation](https://github.com/xinhaoH/SoLA), and save the DeInfer rank dictionary as `desired_r.pkl` in the model directory. The launcher sets `DECOMPOSE_CONFIG` to this file.

SoLA's [optimizer export](https://github.com/xinhaoH/SoLA/blob/main/sola/optim.py#L173-L183) records `desired_rank` together with matrix shape and performance score. Its [decomposition driver](https://github.com/xinhaoH/SoLA/blob/main/sola/playground_sola.py#L222-L229) organizes ranks as `desired_ranks[layer][projection] = (rank, score)`, with `None` used when no score is supplied. DeInfer uses a **flat, rank-only dictionary**: extract `desired_rank` from an optimizer record, or the first element of a driver tuple, and associate it with the full projection name.

Serialize that dictionary with Python's `pickle.dump` as `desired_r.pkl`, consistent with SoLA's [pickle serialization helper](https://github.com/xinhaoH/SoLA/blob/main/sola/utils.py#L517-L524). Each key names the original projection without a `.weight` suffix; each value is the retained integer rank matching the factor tensors. For example:

```python
# Illustrative entries for one LLaMA block; use the actual checkpoint ranks
# and include every required projection in every decomposed block.
desired_r = {
    "model.layers.0.self_attn.q_proj": 1024,
    "model.layers.0.self_attn.k_proj": 512,
    "model.layers.0.self_attn.v_proj": 512,
    "model.layers.0.self_attn.o_proj": 1024,
    "model.layers.0.mlp.gate_proj": 2048,
    "model.layers.0.mlp.up_proj": 2048,
    "model.layers.0.mlp.down_proj": 2048,
}
```

The resulting file contains a pickled `dict[str, int]`; compression ratios, scores, and nested SoLA records must be resolved to integer ranks before loading. DeInfer selects decomposition at attention/MLP-block granularity, so a selected block needs all of its listed projections and corresponding factor weights:

| Model component | Rank-key pattern |
| --- | --- |
| LLaMA attention | `model.layers.<i>.self_attn.{q_proj,k_proj,v_proj,o_proj}` |
| LLaMA MLP | `model.layers.<i>.mlp.{gate_proj,up_proj,down_proj}` |
| OPT attention | `model.decoder.layers.<i>.self_attn.{q_proj,k_proj,v_proj,out_proj}` |
| OPT feed-forward | `model.decoder.layers.<i>.{fc1,fc2}` |

SoLA's compression outputs therefore serve as a preparation reference. Adapt their selected projections, tensor names, and tensor shapes to the DeInfer checkpoint requirements above when exporting the model.

For **low-rank KV caching**, also merge the following fields into the model's existing `config.json`, using the actual absolute rank-file path:

```json
{
  "enable_variadic_kv_cache": true,
  "variadic_kv_cache_config": "/absolute/path/to/decomposed-llama/desired_r.pkl"
}
```

The `OURS` attention layers require equal K and V ranks within each layer. When low-rank caching is enabled, that rank must also be divisible by 16, and the rank map must cover every attention layer. The optimized LLaMA MLP also expects equal gate/up ranks. Weight dimensions and attention heads must be compatible with the chosen tensor-parallel size.

Use a model-directory path containing `llama` or `opt` as appropriate: the launcher's cache bootstrap currently infers the family from that path. The configuration path and checkpoint must be accessible to every worker.

## Run benchmarks

### A single throughput run

After completing setup and checkpoint preparation, run this from the repository root. Replace the model path and GPU IDs for your machine. The tensor-parallel size is inferred from the number of comma-separated GPU IDs.

```bash
# Clear runtime overrides that may remain from an earlier interactive run.
unset NCCL_P2P_DISABLE ENABLE_LOW_RANK_CACHE VLLM_ATTENTION_BACKEND MODEL_TYPE

export EFFIDE_VAR_LOCAL_MODEL_PATH=/absolute/path/to/decomposed-llama
export EFFIDE_VAR_CUDA_DEVICES=0,1
export EFFIDE_VAR_BENCHMARK_RESULT_PATH=results/throughput
export EFFIDE_VAR_BEHAVIOR=OURS
export EFFIDE_VAR_ENABLE_LOW_RANK_CACHE=true
export EFFIDE_VAR_ENABLE_CUDA_GRAPH=true
export EFFIDE_VAR_ENABLE_NCCL_P2P=true
export EFFIDE_VAR_BATCH_SIZE=8
export EFFIDE_VAR_PREFILL_TOKENS=32
export EFFIDE_VAR_OUTPUT_TOKENS=128
export EFFIDE_VAR_EVAL_THROUGHPUT=true
export EFFIDE_VAR_EVAL_SERVING=false
export ENABLE_CONTIGUOUS_COPY=OFF

bash scripts/launch_benchmark.sh
```

This is a small workload example, not a paper reproduction configuration. `EFFIDE_VAR_*` Boolean options use lowercase `true`/`false`; `ENABLE_CONTIGUOUS_COPY` uses `ON`/`OFF`. Keep the result path relative to the repository root. The launcher uses GPU memory utilization `0.97`; adjust it in [launch_benchmark.sh](scripts/launch_benchmark.sh) if needed.

To run serving with the same checkpoint and execution mode, keep the variables above and change:

```bash
export EFFIDE_VAR_BENCHMARK_RESULT_PATH=results/serving
export EFFIDE_VAR_EVAL_THROUGHPUT=false
export EFFIDE_VAR_EVAL_SERVING=true
export EFFIDE_VAR_BATCH_SIZE=128
export EFFIDE_VAR_PREFILL_TOKENS=32
export EFFIDE_VAR_OUTPUT_TOKENS=256
export ENABLE_CONTIGUOUS_COPY=OFF

bash scripts/launch_benchmark.sh
```

Serving starts a local vLLM OpenAI-compatible server on an available port, runs random-prompt requests, and stops the server after a successful benchmark. The launcher requests one extra prompt for its serving setup.

### Experiment matrix

[run_experiments.py](run_experiments.py) defines four configurations. Each is run with NCCL peer-to-peer communication disabled and enabled.

| Label | Execution mode | Low-rank KV cache | CUDA Graph |
| --- | --- | --- | --- |
| `c1` | `NAIVE` | Off | On |
| `c2` | `OURS` | Off | On |
| `c3` | `OURS` | On | On |
| `c4` | `OURS` | On | Off |

The driver supplies these workload settings:

| Workload function | Batch size / prompt count | Input tokens | Output tokens |
| --- | --- | --- | --- |
| `run_throughput_benchmark_s1` | 512 | 32 | 128 |
| `run_throughput_benchmark_s2` | 64 | 64 | 448 |
| `run_serving_benchmark` | 128 | 32 | 256 |

Edit `cuda_visible_devices`, `local_model_path`, and the result-directory setting near the top of `run_experiments.py`. Adjust workload sizes for the target model and hardware. Then invoke each workload in a fresh process:

```bash
python -c "import run_experiments as r; r.run_throughput_benchmark_s1()"
python -c "import run_experiments as r; r.run_throughput_benchmark_s2()"
python -c "import run_experiments as r; r.run_serving_benchmark()"
```

The current `python run_experiments.py` entry point leaves throughput enabled when it reaches the serving phase. Separate invocations above isolate the intended workloads. The driver enables contiguous copying for throughput and disables it for serving; contiguous copying assumes one contiguous cache-block range starting at block zero.

### Outputs

Each launcher invocation creates a numbered directory such as `results/throughput/00001/`, containing the patched benchmark scripts and outputs for the enabled workload:

- `benchmark_results.md`: timestamp and benchmark summaries.
- `benchmark_throughput.txt` and `throughput_results.json`: offline throughput output.
- `benchmark_serving.txt`: serving output, including latency and throughput metrics.

Serving JSON uses the upstream benchmark's default output location, which is the repository working directory. The launcher currently overwrites its initial configuration section in `benchmark_results.md`, so retain the environment settings separately when recording experiments. A latency-only path exists in the launcher but is disabled by its hardcoded `eval_latency=false` setting.

## Implementation scope

- The custom low-rank cache backend does not support prefix caching, speculative decoding, or MLA. Quantized checkpoints are not generally supported by the custom low-rank weight loaders.
- Low-rank caching requires `OURS`; the launcher rejects `NAIVE` with low-rank caching enabled. Use tensor-parallel sizes 2, 4, or 8 for the documented LLaMA workflow; its optimized merged loader is not suitable for TP=1.
- Retain the default cache block size of 16. Cache reconstruction buffers must cover the maximum batch size and sequence length; the launcher derives their sizes from the workload and adds padding for serving.
- The PyTorch reference copy path cannot be used with CUDA Graph. The launcher selects the optimized CUDA operators by default.
- For custom entry points, set `MLP_BEHAVIOR`, `ATTN_BEHAVIOR`, and `DECOMPOSE_CONFIG` before Python loads the hook. Low-rank caching additionally requires the custom backend and cache environment settings shown in [the launcher](scripts/launch_benchmark.sh).
- Start comparisons with clean runtime environment settings. Enabling NCCL P2P in the launcher does not clear an inherited `NCCL_P2P_DISABLE=1`; disabling low-rank caching likewise does not clear inherited backend overrides. NCCL P2P availability also depends on the machine's interconnect topology.

## Citation

If you use DeInfer in your research, please cite:

```bibtex
@inproceedings{10.1145/3770743.3804360,
  author = {Huang, You-Liang and Huang, Xinhao and Liao, Chengxi and Wen, Zeyi},
  title = {DeInfer: Efficient Parallel Inferencing for Decomposed Large Language Models},
  year = {2026},
  isbn = {9798400722547},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3770743.3804360},
  doi = {10.1145/3770743.3804360},
  booktitle = {Proceedings of the 63rd ACM/IEEE Design Automation Conference},
  articleno = {518},
  numpages = {7},
  location = {Long Beach, CA, USA},
  series = {DAC '26}
}
```

## License

The project license is [MIT](LICENSE). Third-party and adapted source files retain their own license notices.
