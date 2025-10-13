import os
from typing import List

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from zeroband.comms import ElasticDeviceMesh
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logger import get_logger


class HierarchicalDeviceMesh(ElasticDeviceMesh):
    """扩展的设备网格，支持多节点作为内层 FSDP 分组，同时保留外层 Diloco 通信进程组。

    通过环境变量 ZERO_BAND_INNER_NODES_PER_GROUP 控制每个 FSDP 分组包含的节点数。
    - 内层 FSDP 使用 `self.local_pg` 和 `self.cuda_local_mesh`/`self.cpu_local_mesh`
    - 外层 Diloco 使用 `self.global_pg`（与基类一致）
    """

    def __init__(
        self,
        backend: str = "cpu:gloo,cuda:nccl",
        enable: bool = True,
        live_recovery_rank_src: int | None = None,
    ):
        self._logger = get_logger()
        self.world_info = get_world_info()

        # 初始化外层 PG、商店、心跳等（以及默认 intranode/local_pg）
        super().__init__(backend=backend, enable=enable, live_recovery_rank_src=live_recovery_rank_src)

        # 读取每个 FSDP 组包含的节点数，默认 1（即仅单节点）
        inner_nodes_per_group = int(os.getenv("ZERO_BAND_INNER_NODES_PER_GROUP", "1"))

        # 若为 1，沿用基类的单节点 FSDP
        if inner_nodes_per_group <= 1:
            self._logger.info("HierarchicalDeviceMesh: inner_nodes_per_group=1，保持单节点 FSDP 设置")
            return

        # 重新构建多节点 FSDP 分组
        group_ranks = self._build_fsdp_group(inner_nodes_per_group)

        # 创建多节点本地进程组（用于 FSDP/损失 allreduce 等）
        self.local_pg = dist.new_group(ranks=group_ranks)

        # 使用 DeviceMesh 明确指定多节点 FSDP 设备网格
        # 注意：将现有代码中的 cuda_local_mesh/cpu_local_mesh 指向新的多节点网格，
        # 以保持调用方无需改动。
        self.cuda_local_mesh = DeviceMesh("cuda", group_ranks)
        self.cpu_local_mesh = DeviceMesh("cpu", group_ranks)

        self._logger.info(
            "HierarchicalDeviceMesh: 启用多节点 FSDP 分组，节点数/组=%d，组内总进程=%d，组内ranks=%s",
            inner_nodes_per_group,
            len(group_ranks),
            group_ranks,
        )

    def _build_fsdp_group(self, inner_nodes_per_group: int) -> List[int]:
        """根据全局 rank 拓扑，按节点划分构建多节点 FSDP 分组。

        假设全局 rank 按节点连续排列：
        [node0: 0..(G-1)], [node1: G..(2G-1)], ...
        其中 G=local_world_size。
        """
        local_world_size = self.world_info.local_world_size
        total_nodes = self.world_info.nnodes
        assert local_world_size > 0 and total_nodes > 0

        # 当前进程所在节点 id
        node_id = self.world_info.global_rank

        group_index = node_id // inner_nodes_per_group
        start_node = group_index * inner_nodes_per_group
        end_node = min(start_node + inner_nodes_per_group, total_nodes)

        group_ranks: List[int] = []
        for n in range(start_node, end_node):
            base = n * local_world_size
            group_ranks.extend(base + i for i in range(local_world_size))

        if self.world_info.global_rank not in group_ranks:
            raise RuntimeError(
                f"构建的 FSDP 分组不包含本 rank: rank={self.world_info.global_rank}, group={group_ranks}"
            )

        return group_ranks


