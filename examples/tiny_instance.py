import os
import re
import time
import logging
from functools import lru_cache

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor.api import DTensor
from torch.distributed.device_mesh import DeviceMesh

from torch.distributed._composable.fsdp import fully_shard

@lru_cache(maxsize=None)
def _find_first_number(s: str) -> int:
    match = re.search(r"\d+", s)
    if match:
        return int(match.group())
    else:
        return -1

# # ========== Logger ==========
# def get_logger(rank: int):
#     logger = logging.getLogger(f"rank{rank}")
#     handler = logging.StreamHandler()
#     formatter = logging.Formatter("%(asctime)s [Rank %(rank)d][%(funcName)s] %(message)s", "%Y-%m-%d %H:%M:%S")
#     handler.setFormatter(formatter)

#     # 自定义 filter 加 rank 信息
#     class RankFilter(logging.Filter):
#         def filter(self, record):
#             record.rank = rank
#             return True

#     logger.addFilter(RankFilter())
#     logger.setLevel(logging.INFO)
#     if not logger.handlers:
#         logger.addHandler(handler)
#     return logger

# import logging
# import os

def get_logger(rank: int, log_dir: str = "./logs_4process"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(f"rank{rank}")
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger  

    # 屏幕输出
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s [Rank %(rank)d][%(funcName)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # 文件输出 (每个 rank 一个文件)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"log_rank{rank}.log"))
    file_formatter = logging.Formatter(
        "%(asctime)s [Rank %(rank)d][%(funcName)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # 让 record 知道 rank
    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = rank
            return True

    logger.addFilter(RankFilter())
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# ========== Tiny 模型 ==========
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10, bias=False)
        self.layer2 = nn.Linear(10, 10, bias=False)

    def forward(self, x):
        return self.layer2(self.layer1(x))


# ========== 初始化分布式 ==========
def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


# ========== 打印 shard 后的参数 ==========
def shard_model(model, mesh, rank, logger):
    for _, module in model.named_children():
        fully_shard(module, mesh=mesh)
    fully_shard(model, mesh=mesh)

    # for name, param in model.named_parameters():
    #     logger.info(f"Param {name}, "
    #                 f"local_shape={param.to_local().shape}, "
    #                 f"placements={param.placements}")


# ========== 带日志的 offload 方法 ==========
@torch.no_grad()
def get_offloaded_param(model: nn.Module, cpu_mesh: DeviceMesh, logger):
    
    logger.info(f"Offloading model parameters..., model info: {model}")
    # logger.info(f"Model parameters: {model.parameters()}")
    # logger.info(f"Model named parameters: {model.named_parameters()}")
    for name, param in model.named_parameters():
        logger.info(f"<named> >> {name}, "
                    f"<param> >>\n {param},\n "
                    f"<local_shape> >> {param.to_local().shape}, "
                    f"<placements> >> {param.placements}")
    
    param_items = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    logger.info(f"Model param items: {param_items}")
    
    numels = sum(param.to_local().numel() for _, param in param_items)

    offloaded_data_flat_tensor = torch.empty((numels,), device="cpu", dtype=torch.float32)
    offloaded_grad_flat_tensor = torch.zeros((numels,), device="cpu", dtype=torch.float32)
    current_offset = 0
    offloaded_params = []
    param_group_cutoff = []

    prev_id = None
    for name, param in param_items:
        
        if _find_first_number(name) != prev_id:
            param_group_cutoff.append(current_offset)
            prev_id = _find_first_number(name)
        
        target = param.data.to_local().detach()
        data_tensor = offloaded_data_flat_tensor.as_strided(target.size(), target.stride(), current_offset)
        grad_tensor = offloaded_grad_flat_tensor.as_strided(target.size(), target.stride(), current_offset)

        logger.info(f"<named> >> {name}, "
                    f"<param> >>\n {param},\n "
                    f"<param_shape> >> {param.shape}, "
                    f"<local_shape> >> {param.to_local().shape}, "
                    f"<placements> >> {param.placements}, "
                    f"<target_shape> >> {target.shape}, "
                    f"copy {target.numel()} elements from GPU → CPU at offset {current_offset}")

        current_offset += data_tensor.numel()
        data_tensor.copy_(target)

        offloaded_param = nn.Parameter(
            DTensor.from_local(
                data_tensor,
                device_mesh=cpu_mesh,
                placements=param.data.placements,
            )
        )
        offloaded_param.grad = DTensor.from_local(
            grad_tensor,
            device_mesh=cpu_mesh,
            placements=param.data.placements,
        )
        offloaded_param.requires_grad = True
        offloaded_params.append(offloaded_param)
    
    param_group_cutoff.append(current_offset)
    logger.info(f"Offloaded params: {offloaded_params}")
    
    
    logger.info(f"Param group cutoff: {param_group_cutoff}")
    
    offloaded_grad_grouped_tensor= [
            offloaded_grad_flat_tensor.as_strided((j - i,), (1,), i)
            for i, j in zip(param_group_cutoff, param_group_cutoff[1:])
        ]
    logger.info(f"Offloaded grad grouped tensor: {offloaded_grad_grouped_tensor}")
    
    

    return offloaded_params


# ========== 主逻辑 ==========
def main():
    rank, world_size = setup()
    if rank == 0:
        print(f"World size: {world_size}")
    
    logger = get_logger(rank)

    model = TinyModel()  # 初始化在 CPU
    
    logger.info(f"Model initialized on CPU, model info: {model}")
    for name, param in model.named_parameters():
        logger.info(f"<named> >> {name}, "
                    f"<param> >>\n {param},\n "
                    f"<param_shape> >> {param.shape}, ")
    
    cuda_mesh = DeviceMesh("cuda", list(range(world_size)))
    shard_model(model, cuda_mesh, rank, logger)
    
    logger.info(f"Model sharded on CUDA, model info: {model}")
    for name, param in model.named_parameters():
        logger.info(f"<named> >> {name}, "
                    f"<param> >>\n {param},\n "
                    f"<param_shape> >> {param.shape}, "
                    f"<local_shape> >> {param.to_local().shape}, "
                    f"<placements> >> {param.placements}")

    cpu_mesh = DeviceMesh("cpu", list(range(world_size)))
    _ = get_offloaded_param(model, cpu_mesh, logger)

    if rank == 0:
        logger.info("=== Training setup done ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
