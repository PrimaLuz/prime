import os

world_info = None


class WorldInfo:
    """This class parse env var about torch world into class variables."""

    world_size: int
    rank: int
    local_rank: int
    local_world_size: int

    def __init__(self):
        # GPU总数
        self.world_size = int(os.environ["WORLD_SIZE"])
        # 进程的全局 ID，范围是 0 到 world_size - 1。
        self.rank = int(os.environ["RANK"])
        # 进程的本地 ID，范围是 0 到 local_world_size - 1。
        self.local_rank = int(os.environ["LOCAL_RANK"])
        # 单个节点上的 GPU 数（进程数）
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        # 节点数
        self.nnodes = self.world_size // self.local_world_size

        self.global_addr = os.environ.get("GLOBAL_ADDR", None)
        self.global_port = int(os.environ.get("GLOBAL_PORT")) if "GLOBAL_PORT" in os.environ else None
        # 节点总数
        self.global_world_size = int(os.environ.get("GLOBAL_WORLD_SIZE", 1))
        # 节点排名归属
        self.global_rank = int(os.environ.get("GLOBAL_RANK", 0))
        # 节点的唯一ID
        self.global_unique_id = f"{self.global_rank}"
    def __repr__(self):
        return f"WorldInfo(world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}, local_world_size={self.local_world_size}, nnodes={self.nnodes}, global_unique_id={self.global_unique_id}, global_addr={self.global_addr}, global_port={self.global_port}, global_world_size={self.global_world_size}, global_rank={self.global_rank})"

    @property
    def diloco_rank(self):
        return self.global_rank

    def json(self) -> dict[str, int | str]:
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "nnodes": self.nnodes,
            "global_unique_id": self.global_unique_id,
            "global_addr": self.global_addr,
            "global_port": self.global_port,
            "global_world_size": self.global_world_size,
            "global_rank": self.global_rank,
        }


def get_world_info() -> WorldInfo:
    """
    Return a WorldInfo singleton.
    """
    global world_info
    if world_info is None:
        world_info = WorldInfo()
    return world_info
