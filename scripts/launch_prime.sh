export PCCL_LOG_LEVEL=DEBUG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=1234

# this fixes crashes on some NCCL versions that attempt to create a multi-node topology on some single-node cloud instances
export NCCL_MNNVL_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

RANK=0 GPU_ORDINAL=0 WORLD_SIZE=8 ZERO_BAND_LOG_LEVEL=DEBUG python src/zeroband/train.py @configs/7B/H100.toml &
RANK=1 GPU_ORDINAL=1 WORLD_SIZE=8 ZERO_BAND_LOG_LEVEL=DEBUG python src/zeroband/train.py @configs/7B/H100.toml &
RANK=2 GPU_ORDINAL=2 WORLD_SIZE=8 ZERO_BAND_LOG_LEVEL=DEBUG python src/zeroband/train.py @configs/7B/H100.toml &
RANK=3 GPU_ORDINAL=3 WORLD_SIZE=8 ZERO_BAND_LOG_LEVEL=DEBUG python src/zeroband/train.py @configs/7B/H100.toml &
RANK=4 GPU_ORDINAL=4 WORLD_SIZE=8 ZERO_BAND_LOG_LEVEL=DEBUG python src/zeroband/train.py @configs/7B/H100.toml &
RANK=5 GPU_ORDINAL=5 WORLD_SIZE=8 ZERO_BAND_LOG_LEVEL=DEBUG python src/zeroband/train.py @configs/7B/H100.toml &
RANK=6 GPU_ORDINAL=6 WORLD_SIZE=8 ZERO_BAND_LOG_LEVEL=DEBUG python src/zeroband/train.py @configs/7B/H100.toml &
RANK=7 GPU_ORDINAL=7 WORLD_SIZE=8 ZERO_BAND_LOG_LEVEL=DEBUG python src/zeroband/train.py @configs/7B/H100.toml &
wait