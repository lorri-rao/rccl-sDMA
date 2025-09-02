import os
from time import sleep
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import profile, record_function, ProfilerActivity
import datetime

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, timeout=datetime.timedelta(seconds=20), world_size=world_size)

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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trace_file = f"./trace_{rank}.json"
        # Each process creates a tensor with its rank
        device=torch.device(f"cuda:{rank}")
        tensor = torch.full([10240, 1024], rank, dtype=torch.bfloat16).to(device)
        tensor_out = torch.zeros([10240*world_size*1024], dtype=torch.bfloat16).to(device)
        # Prepare a list of tensors to receive the gathered data
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        
        # Main profiled section
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            # Perform the all_gather operation
            # dist.all_gather(tensor_list, tensor)
            dist.all_gather_into_tensor(tensor_out, tensor)
            # print(f"Rank {rank} gathered tensors: {tensor_list}")
            torch.cuda.synchronize()
            sleep(1)
        prof.export_chrome_trace(trace_file)
        cleanup()

if __name__ == "__main__":
    world_size = 8
    mp.spawn(run_all_gather,
             args=(world_size,),
             nprocs=world_size,
             join=True)
