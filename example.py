import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
from datetime import datetime

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_all_gather(rank, world_size):
    """
    Gathers tensors from all processes and prints the result.
    Includes PyTorch profiling for performance analysis.
    """
    setup(rank, world_size)
    
    enable_profiling=True
    # Configure profiler
    if enable_profiling:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_file = f"./trace_rank_{rank}_{timestamp}.json"
        # Main profiled section
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            # Each process creates a tensor with its rank
            tensor = torch.full([12, 1024], rank).to(device)
            # Prepare a list of tensors to receive the gathered data
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        
            # Perform the all_gather operation
            dist.all_gather(tensor_list, tensor)
            print(f"Rank {rank} gathered tensors: {tensor_list}")
        prof.export_chrome_trace(trace_file)
        cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs.")
    mp.spawn(run_all_gather,
             args=(world_size,),
             nprocs=world_size,
             join=True)
