import os

import torch
import mini_nccl

import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("cuda:mini_nccl", rank=0, world_size=1)

x = torch.ones(6, device='cuda')
dist.all_reduce(x)
print(f"cuda allreduce: {x}")
