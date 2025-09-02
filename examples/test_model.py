import os
import logging

import torch
import torch.nn as nn

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


def get_logger(rank: int, log_dir: str = "./logs_for_test_model"):
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

def print_model_params(model,logger):
    for name, param in model.named_parameters():
        logger.info(f"-----------{name}---------------\n "
              f"shape={tuple(param.shape)}  dtype={param.dtype}  device={param.device}\n"
              f" [Params]:\n{param}\n"
              f"------------------\n")

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 10)       
        self.layers = nn.ModuleDict({
            "0": nn.Linear(10, 10),
            "1": nn.Linear(10, 10),
        })
        self.head = nn.Linear(10, 2)              

    def forward(self, x):
        x = self.embed(x)
        for _, blk in self.layers.items():
            x = blk(x)
        return self.head(x)
    

def main():
    dist.init_process_group("gloo")

    # 获取 rank/world_size
    rank = dist.get_rank()
    logger = get_logger(rank)
    world_size = dist.get_world_size()
    logger.info(f"Hello from {rank}/{world_size}")

    # 初始化 device mesh
    mesh = init_device_mesh("cpu", mesh_shape=(world_size,))
    
    mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16
        )
    
    model = TinyModel()
    logger.info(">>>>>>>>>>>before shard, the original model info<<<<<<<<<<<<<")
    print_model_params(model,logger)
    
    for layer_id, transformer_block in model.layers.items():
        reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard(
            transformer_block,
            mp_policy=mp_policy,
            mesh=mesh,
            reshard_after_forward=reshard_after_forward,
        )
    logger.info(">>>>>>>>>>>>>>>after shard layers<<<<<<<<<<<<<")
    print_model_params(model,logger)
    
    fully_shard(
        model,
        mp_policy=mp_policy,
        mesh=mesh,
        reshard_after_forward=reshard_after_forward,
    )
    
    logger.info(">>>>>>>>>>>>>>>after fully shard<<<<<<<<<<<<<")
    
    print_model_params(model,logger)
    
    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
    
    