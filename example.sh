#!/bin/bash
set -x
SIZE=536870912 #1024*1024*512
export HSA_NO_SCRATCH_RECLAIM=1
LD_LIBRARY_PATH=./build/release/:$LD_LIBRARY_PATH \
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
NCCL_P2P_PCI_CHUNKSIZE=$SIZE NCCL_P2P_NVL_CHUNKSIZE=$SIZE NCCL_BUFFSIZE=$SIZE \
NCCL_P2P_NET_CHUNKSIZE=$SIZE NCCL_CHUNK_SIZE=$SIZE \
NCCL_SOCKET_IFNAME=lo NCCL_PROTO="SIMPLE" NCCL_DEBUG=TRACE \
NCCL_P2P_USE_CUDA_MEMCPY=1 GPU_MAX_HW_QUEUES=8 NCCL_MAX_NCHANNELS=8 NCCL_MAX_P2P_NCHANNELS=8  \
NCCL_NCHANNELS_PER_PEER=1 python ./example.py

exit

# git clone https://github.com/AMD-AIG-AIMA/pytorch-training-benchmark.git
SIZE=536870912 #1024*1024*512
export HSA_NO_SCRATCH_RECLAIM=1
LD_LIBRARY_PATH=/workspace/rccl-sDMA/build/release/:$LD_LIBRARY_PATH \
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
NCCL_P2P_PCI_CHUNKSIZE=$SIZE NCCL_P2P_NVL_CHUNKSIZE=$SIZE NCCL_BUFFSIZE=$SIZE \
NCCL_P2P_NET_CHUNKSIZE=$SIZE NCCL_CHUNK_SIZE=$SIZE \
NCCL_SOCKET_IFNAME=lo NCCL_PROTO="SIMPLE" \
NCCL_P2P_USE_CUDA_MEMCPY=1 GPU_MAX_HW_QUEUES=20 NCCL_MAX_NCHANNELS=8 NCCL_MAX_P2P_NCHANNELS=8  \
NCCL_NCHANNELS_PER_PEER=1 torchrun --nnodes=1  --node_rank=0 --nproc_per_node=8  --master_addr="0.0.0.0"     --master_port="12234"  ./train_fsdp.py \
    configs/llama-3.1-8b-4k.json llama --batch_size 1
# 6817792
NCCL_DEBUG=TRACE \

# torchtitan
python scripts/download_tokenizer.py \
      --repo_id meta-llama/Meta-Llama-3.1-70B \
      --tokenizer_path "original" \
      --hf_token=xx
# normal one
LD_LIBRARY_PATH=/workspace/rccl-sDMA/build/release/:$LD_LIBRARY_PATH NCCL_SOCKET_IFNAME=lo \
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0"  --role rank --tee 3 \
-m torchtitan.train --job.config_file ./torchtitan/models/llama3/train_configs/llama3_70b.toml \
--profiling.enable-profiling --profiling.profile_freq=5 --lr_scheduler.warmup_steps=0 --training.batch_size=3

# with sdma

SIZE=136870912 #1024*1024*512
export HSA_NO_SCRATCH_RECLAIM=1
LD_LIBRARY_PATH=/workspace/rccl-sDMA/build/release/:$LD_LIBRARY_PATH \
TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=true \
NCCL_P2P_PCI_CHUNKSIZE=$SIZE NCCL_P2P_NVL_CHUNKSIZE=$SIZE NCCL_BUFFSIZE=$SIZE \
NCCL_P2P_NET_CHUNKSIZE=$SIZE NCCL_CHUNK_SIZE=$SIZE \
NCCL_SOCKET_IFNAME=lo NCCL_PROTO="SIMPLE" \
NCCL_P2P_USE_CUDA_MEMCPY=1 GPU_MAX_HW_QUEUES=20 NCCL_MAX_NCHANNELS=8 NCCL_MAX_P2P_NCHANNELS=8  \
NCCL_NCHANNELS_PER_PEER=1 torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0"  --role rank --tee 3 \
-m torchtitan.train --job.config_file ./torchtitan/models/llama3/train_configs/llama3_70b.toml \
--profiling.enable-profiling --profiling.profile_freq=5 --lr_scheduler.warmup_steps=0 --training.batch_size=3