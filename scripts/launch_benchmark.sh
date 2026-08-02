#!/bin/bash
set -o pipefail
# This script is modified on the basis of https://github.com/vllm-project/vllm/blob/main/.buildkite/run-benchmarks.sh

### Helper
function Listening {
   TCPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "tcp" && $NF == "LISTEN" {print $0}' | wc -l`
   UDPListeningnum=`netstat -an | grep ":$1 " | awk '$1 == "udp" && $NF == "0.0.0.0:*" {print $0}' | wc -l`
   (( Listeningnum = TCPListeningnum + UDPListeningnum ))
   if [ $Listeningnum == 0 ]; then
       echo "0"
   else
       echo "1"
   fi
}

function random_range {
   shuf -i 8192-65535 -n 1
}

echo "====================="

##############Setup variables###############
using_decomposition=true
enable_nccl_p2p=$EFFIDE_VAR_ENABLE_NCCL_P2P
enable_low_rank_cache=$EFFIDE_VAR_ENABLE_LOW_RANK_CACHE
enable_cuda_graph=$EFFIDE_VAR_ENABLE_CUDA_GRAPH
enable_optimized_op=true  # only come into effects when enable_low_rank_cache, pytorch impl doesn't support CUDA Graph
local_model_path=$EFFIDE_VAR_LOCAL_MODEL_PATH
CUDA_DEVICES=$EFFIDE_VAR_CUDA_DEVICES
gpu_memory_utilization=0.97

# mlp behavior [NAIVE, OURS] (only when using decomposed models)
BASH_MLP_BEHAVIOR=$EFFIDE_VAR_BEHAVIOR

# attn behavior [NAIVE, OURS] (only when using decomposed models)
BASH_ATTN_BEHAVIOR=$EFFIDE_VAR_BEHAVIOR

# latency only
enable_torch_profiler=false

# throughput only
enable_nsight_sys=false

# disable variadic cudagraph_capture_sizes
disable_cudagraph_capture_sizes=true

# throughput settings
max_num_seq=$EFFIDE_VAR_BATCH_SIZE
num_prompts=$EFFIDE_VAR_BATCH_SIZE
input_len=$EFFIDE_VAR_PREFILL_TOKENS
output_len=$EFFIDE_VAR_OUTPUT_TOKENS

eval_latency=false
eval_throughput=$EFFIDE_VAR_EVAL_THROUGHPUT
eval_serving=$EFFIDE_VAR_EVAL_SERVING

##################General###################
# disable tokenizers parallelism to avoid issues with torch.multiprocessing
export TOKENIZERS_PARALLELISM=false
# set the path to the desired rank file
if [ $using_decomposition = true ]; then
    export DECOMPOSE_CONFIG="${local_model_path}/desired_r.pkl"
fi
# layer selection
export MLP_BEHAVIOR=$BASH_MLP_BEHAVIOR
export ATTN_BEHAVIOR=$BASH_ATTN_BEHAVIOR
echo "MLP behavior selection: $MLP_BEHAVIOR"
echo "Attention behavior selection: $ATTN_BEHAVIOR"

if [ $enable_nccl_p2p = false ]; then
    export NCCL_P2P_DISABLE=1
    echo "NVLINK HAS BEEN DISABLED"
fi

# reset working directory
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# add sitecustomize.py to PYTHONPATH
export PYTHONPATH=$(pwd)/effide/vllm_plugin/hook:$(pwd)

############################################
if [ $enable_low_rank_cache = true ]; then
    # VLLM_ATTENTION_BACKEND is used in sitecustomize.py to choose attn backend
    export VLLM_ATTENTION_BACKEND=CUSTOMIZED_FLASH_ATTN
    # ENABLE_LOW_RANK_CACHE is used for controlling layer initialization behavior
    export ENABLE_LOW_RANK_CACHE=ON
    # MODEL_TYPE is used in variadic_cache_manager.py
    export MODEL_TYPE=$local_model_path

    # there is one more token in the actual output_len, so add pad to avoid illegal memory access
    pad=0
    if [ $eval_serving = true ]; then
        pad=4
    fi

    # set buffer size
    export BUFFER_MAX_BATCH_SIZE=$max_num_seq
    export BUFFER_MAX_SEQ_LEN=$(($input_len + $output_len + $pad))
    if [ $ATTN_BEHAVIOR = NAIVE ]; then
        echo "Warning: 'NAIVE' attn behavior is not compatible with low rank cache compression"
        exit 1
    fi

    echo "BUFFER_MAX_BATCH_SIZE: $BUFFER_MAX_BATCH_SIZE, BUFFER_MAX_SEQ_LEN: $BUFFER_MAX_SEQ_LEN"

    if [ $enable_optimized_op = true ]; then
        export ENABLE_OPTIMIZED_OP=ON
    else
        export ENABLE_OPTIMIZED_OP=OFF
        if [ $enable_cuda_graph = true ]; then
            echo "Warning: CUDA Graph is not supported in PyTorch implementation of operators"
            exit 1
        fi
    fi
fi

# if not use cuda graph
cuda_graph_opts="--enforce-eager"
if [ $enable_cuda_graph = true ]; then
    cuda_graph_opts=""
fi
############################################

# process CUDA_DEVICES
CUDA_DEVICES=$(echo $CUDA_DEVICES | tr -d " ")
DEVICES=$(echo $CUDA_DEVICES | tr "," "\n")
tp_size=$(echo $DEVICES | wc -w)
echo "CUDA_VISIBLE_DEVICES: '$CUDA_DEVICES', tensor parallelism size: $tp_size"

# make root output directory
benchmark_result_path="$(pwd)/${EFFIDE_VAR_BENCHMARK_RESULT_PATH}"
echo "Benchmark results will be saved to $benchmark_result_path"
mkdir -p $benchmark_result_path

dir_count=1
printf -v dname "%05d" "$dir_count"
while [ -d "$benchmark_result_path/$dname" ]
do
  ((dir_count++))
  printf -v dname "%05d" "$dir_count"
done

printf "Creating output directory %s...\n" "$benchmark_result_path/$dname"
mkdir -p "$benchmark_result_path/$dname"
output_dir=$benchmark_result_path/$dname
workspace=$output_dir

# make benchmark .py file
cp vllm_benchmarks/benchmark_latency.py $workspace/benchmark_latency.py
cp vllm_benchmarks/benchmark_throughput.py $workspace/benchmark_throughput.py
cp vllm_benchmarks/benchmark_serving.py $workspace/benchmark_serving.py
cp vllm_benchmarks/backend_request_func.py $workspace/backend_request_func.py
if [ $enable_torch_profiler = true ]; then
    echo "Enabling Torch Profiler for latency benchmark, results will be saved to $output_dir/profile_results"
    torch_profiler_opts="--profile --profile-result-dir $output_dir/profile_results"
else
    torch_profiler_opts=""
fi
if [ $enable_nsight_sys = true ]; then
    nsight_outs="$output_dir/nsight_results"
    mkdir -p $nsight_outs
    echo "Enabling Nsight Systems profiling for throughput benchmark, results will be saved to $nsight_outs"
    nsight_stat="nsys profile --cuda-graph-trace node -o $nsight_outs/profile"
else
    nsight_stat=""
fi

###################Print####################
echo "=====================" | tee -a $output_dir/benchmark_results.md
echo "Configurations" | tee -a $output_dir/benchmark_results.md
echo "=====================" | tee -a $output_dir/benchmark_results.md
echo "IS_DECOMPOSED_MODEL=${using_decomposition}" | tee -a $output_dir/benchmark_results.md
echo "ENABLE_NCCL_P2P=${enable_nccl_p2p}" | tee -a $output_dir/benchmark_results.md
echo "ENABLE_LOW_RANK_CACHE=${enable_low_rank_cache}" | tee -a $output_dir/benchmark_results.md
echo "ENABLE_CUDA_GRAPH=${enable_cuda_graph}" | tee -a $output_dir/benchmark_results.md
echo "MLP_BEHAVIOR=${BASH_MLP_BEHAVIOR}" | tee -a $output_dir/benchmark_results.md
echo "ATTN_BEHAVIOR=${BASH_ATTN_BEHAVIOR}" | tee -a $output_dir/benchmark_results.md
echo "DISABLE_VARIADIC_BATCH_SIZE=${disable_cudagraph_capture_sizes}" | tee -a $output_dir/benchmark_results.md
echo "OUTPUT_DESTINATION=${output_dir}" | tee -a $output_dir/benchmark_results.md
echo "THROUGHPUT_LOG=${output_dir}/benchmark_throughput.txt" | tee -a $output_dir/benchmark_results.md
echo "LATENCY_LOG=${output_dir}/benchmark_latency.txt" | tee -a $output_dir/benchmark_results.md
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}" | tee -a $output_dir/benchmark_results.md
echo "TP_SIZE=${tp_size}" | tee -a $output_dir/benchmark_results.md
echo "=====================" | tee -a $output_dir/benchmark_results.md
############################################

# apply patch
python3 tools/benchmark_patching.py --dir=$workspace

echo "====================="
echo "Running benchmarks..."
echo "====================="
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo "=========================" > $output_dir/benchmark_results.md
echo $timestamp >> $output_dir/benchmark_results.md
echo "=========================" >> $output_dir/benchmark_results.md
set -ex

export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# run python-based benchmarks
if [ $eval_latency = true ]; then
    python3 $workspace/benchmark_latency.py $cuda_graph_opts $torch_profiler_opts --num-iters 30 --tensor_parallel_size $tp_size --model $local_model_path --output-json $output_dir/latency_results.json 2>&1 | tee -a $output_dir/benchmark_latency.txt
    bench_latency_exit_code=$?
    echo "### Latency Benchmarks" >> $output_dir/benchmark_results.md
    sed -n '1p' $output_dir/benchmark_latency.txt >> $output_dir/benchmark_results.md # first line
    echo "" >> $output_dir/benchmark_results.md
    sed -n '$p' $output_dir/benchmark_latency.txt >> $output_dir/benchmark_results.md # last line
fi

if [ $eval_throughput = true ]; then
    if [ $disable_cudagraph_capture_sizes = true ]; then
        if [ $enable_cuda_graph = false ]; then
            echo "Warning: CUDA Graph must be enabled while setting disable_cudagraph_capture_sizes to true"
            echo "disable_cudagraph_capture_sizes=true has been ignored."
            $nsight_stat python3 $workspace/benchmark_throughput.py $cuda_graph_opts --gpu-memory-utilization=$gpu_memory_utilization --num-prompts $num_prompts --max-num-seq $max_num_seq --tensor_parallel_size $tp_size --model $local_model_path --input-len $input_len --output-len $output_len --output-json $output_dir/throughput_results.json 2>&1 | tee -a $output_dir/benchmark_throughput.txt
        else
            $nsight_stat python3 $workspace/benchmark_throughput.py --compilation_config="{'cudagraph_capture_sizes': [$max_num_seq]}" $cuda_graph_opts --gpu-memory-utilization=$gpu_memory_utilization --num-prompts $num_prompts --max-num-seq $max_num_seq --tensor_parallel_size $tp_size --model $local_model_path --input-len $input_len --output-len $output_len --output-json $output_dir/throughput_results.json 2>&1 | tee -a $output_dir/benchmark_throughput.txt
        fi
    else
        $nsight_stat python3 $workspace/benchmark_throughput.py $cuda_graph_opts --gpu-memory-utilization=$gpu_memory_utilization --num-prompts $num_prompts --max-num-seq $max_num_seq --tensor_parallel_size $tp_size --model $local_model_path --input-len $input_len --output-len $output_len --output-json $output_dir/throughput_results.json 2>&1 | tee -a $output_dir/benchmark_throughput.txt
    fi
    bench_throughput_exit_code=$?
    set +ex
    echo "### Throughput Benchmarks" | tee -a $output_dir/benchmark_results.md
    echo "=====================" | tee -a $output_dir/benchmark_results.md
    echo $(cat $output_dir/benchmark_throughput.txt | grep "Throughput:") | tee -a $output_dir/benchmark_results.md
    echo "=====================" | tee -a $output_dir/benchmark_results.md
fi

if [ $eval_serving = true ]; then
    # Get a random port
    set +ex
    avail_port=0
    while [ $avail_port == 0 ]; do
        temp1=`random_range`
        if [ `Listening $temp1` == 0 ] ; then
            avail_port=$temp1
            break
       fi
    done
    set -ex

    if [ $disable_cudagraph_capture_sizes = true ]; then
        if [ $enable_cuda_graph = false ]; then
            echo "Warning: CUDA Graph must be enabled while setting disable_cudagraph_capture_sizes to true"
            echo "disable_cudagraph_capture_sizes=true has been ignored."
            python3 -m vllm.entrypoints.openai.api_server --port $avail_port $cuda_graph_opts --tensor_parallel_size $tp_size --model $local_model_path --gpu-memory-utilization $gpu_memory_utilization &
        else
            python3 -m vllm.entrypoints.openai.api_server --port $avail_port --compilation_config="{'cudagraph_capture_sizes': [$max_num_seq]}" $cuda_graph_opts --tensor_parallel_size $tp_size --model $local_model_path --gpu-memory-utilization $gpu_memory_utilization &
        fi
    else
        python3 -m vllm.entrypoints.openai.api_server --port $avail_port $cuda_graph_opts --tensor_parallel_size $tp_size --model $local_model_path --gpu-memory-utilization $gpu_memory_utilization &
    fi
    server_pid=$!
    timeout 180 bash -c "until curl localhost:$avail_port/v1/models; do sleep 1; done" || exit 1
    # The first prompt is used for testing and then might be ignored subsequently.
    python3 $workspace/benchmark_serving.py \
        --port $avail_port \
        --backend vllm \
        --dataset-name random \
        --num-prompts $((num_prompts + 1)) \
        --random-input-len $input_len \
        --random-output-len $output_len \
        --model $local_model_path \
        --endpoint /v1/completions \
        --tokenizer $local_model_path \
        --save-result \
        2>&1 | tee $output_dir/benchmark_serving.txt
    bench_serving_exit_code=$?
    kill $server_pid

    echo "### Serving Benchmarks" >> $output_dir/benchmark_results.md
    echo '```' >> $output_dir/benchmark_results.md
    tail -n 24 $output_dir/benchmark_serving.txt >> $output_dir/benchmark_results.md # last 24 lines
    echo '```' >> $output_dir/benchmark_results.md
fi