import os

from typing import Dict, List

cuda_visible_devices = "0,1,2,3,4,5,6,7"
local_model_path = "PATH_TO_YOUR_LOCAL_MODEL"  # Replace with the actual path to your local model

os.environ["ENABLE_CONTIGUOUS_COPY"] = "ON"
os.environ["EFFIDE_VAR_CUDA_DEVICES"] = cuda_visible_devices
os.environ["EFFIDE_VAR_LOCAL_MODEL_PATH"] = local_model_path
os.environ["EFFIDE_VAR_BENCHMARK_RESULT_PATH"] = "c40-tp4-llama-65b-throughput"
os.environ["EFFIDE_VAR_EVAL_THROUGHPUT"] = "false"
os.environ["EFFIDE_VAR_EVAL_SERVING"] = "false"

config_list = [
    {
        "label": "c1",
        "EFFIDE_VAR_BEHAVIOR": "NAIVE",
        "EFFIDE_VAR_ENABLE_LOW_RANK_CACHE": "false",
        "EFFIDE_VAR_ENABLE_CUDA_GRAPH": "true"
    },
    {
        "label": "c2",
        "EFFIDE_VAR_BEHAVIOR": "OURS",
        "EFFIDE_VAR_ENABLE_LOW_RANK_CACHE": "false",
        "EFFIDE_VAR_ENABLE_CUDA_GRAPH": "true"
    },
    {
        "label": "c3",
        "EFFIDE_VAR_BEHAVIOR": "OURS",
        "EFFIDE_VAR_ENABLE_LOW_RANK_CACHE": "true",
        "EFFIDE_VAR_ENABLE_CUDA_GRAPH": "true"
    },
    {
        "label": "c4",
        "EFFIDE_VAR_BEHAVIOR": "OURS",
        "EFFIDE_VAR_ENABLE_LOW_RANK_CACHE": "true",
        "EFFIDE_VAR_ENABLE_CUDA_GRAPH": "false"
    }
]


def set_environ_variables(environ_config: Dict):
    os.environ["EFFIDE_VAR_BEHAVIOR"] = environ_config["EFFIDE_VAR_BEHAVIOR"]
    os.environ["EFFIDE_VAR_ENABLE_LOW_RANK_CACHE"] = environ_config["EFFIDE_VAR_ENABLE_LOW_RANK_CACHE"]
    os.environ["EFFIDE_VAR_ENABLE_CUDA_GRAPH"] = environ_config["EFFIDE_VAR_ENABLE_CUDA_GRAPH"]


def execute_with_permute_variables(command, var_key: str, var_val: List):
    for val in var_val:
        os.environ[var_key] = val
        ret = os.system(command)
        ret >>= 8
        if ret != 0:
            print("Encounter error during executing, abort.")
            exit(1)


def run_throughput_benchmark_s1():
    # batch size: 512
    # prefill tokens: 32
    # output tokens: 128
    os.environ["EFFIDE_VAR_BATCH_SIZE"] = "512"
    os.environ["EFFIDE_VAR_PREFILL_TOKENS"] = "32"
    os.environ["EFFIDE_VAR_OUTPUT_TOKENS"] = "128"
    os.environ["EFFIDE_VAR_EVAL_THROUGHPUT"] = "true"

    for idx, config in enumerate(config_list):
        set_environ_variables(config)
        print("=" * 80)
        print(f"Executing config {idx}")
        print("=" * 80)
        execute_with_permute_variables("bash scripts/launch_benchmark.sh", "EFFIDE_VAR_ENABLE_NCCL_P2P",
                                       ["false", "true"])


def run_throughput_benchmark_s2():
    # batch size: 64
    # prefill tokens: 64
    # output tokens: 448
    os.environ["EFFIDE_VAR_BATCH_SIZE"] = "64"
    os.environ["EFFIDE_VAR_PREFILL_TOKENS"] = "64"
    os.environ["EFFIDE_VAR_OUTPUT_TOKENS"] = "448"
    os.environ["EFFIDE_VAR_EVAL_THROUGHPUT"] = "true"

    for idx, config in enumerate(config_list):
        set_environ_variables(config)
        print("=" * 80)
        print(f"Executing config {idx}")
        print("=" * 80)
        execute_with_permute_variables("bash scripts/launch_benchmark.sh", "EFFIDE_VAR_ENABLE_NCCL_P2P",
                                       ["false", "true"])


def run_serving_benchmark():
    os.environ["EFFIDE_VAR_BATCH_SIZE"] = "128"
    os.environ["EFFIDE_VAR_PREFILL_TOKENS"] = "32"
    os.environ["EFFIDE_VAR_OUTPUT_TOKENS"] = "256"
    os.environ["EFFIDE_VAR_EVAL_SERVING"] = "true"
    os.environ["ENABLE_CONTIGUOUS_COPY"] = "OFF"
    for idx, config in enumerate(config_list):
        set_environ_variables(config)
        execute_with_permute_variables("bash scripts/launch_benchmark.sh", "EFFIDE_VAR_ENABLE_NCCL_P2P",
                                       ["false", "true"])


if __name__ == "__main__":
    run_throughput_benchmark_s1()
    run_throughput_benchmark_s2()
    run_serving_benchmark()
