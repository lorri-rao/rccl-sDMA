import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run_all_gather(rank, world_size):
    """
    Gathers tensors from all processes and prints the result.
    """
    setup(rank, world_size)
    
    # Each process creates a tensor with its rank
    tensor = torch.tensor([rank]).cuda(rank)
    
    # Prepare a list of tensors to receive the gathered data
    tensor_list = [torch.zeros(1, dtype=torch.int64).cuda(rank) for _ in range(world_size)]
    
    # Perform the all_gather operation
    dist.all_gather(tensor_list, tensor)
    
    if rank == 0:
        print(f"Rank {rank} gathered tensors: {tensor_list}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs.")
    mp.spawn(run_all_gather,
             args=(world_size,),
             nprocs=world_size,
             join=True)
