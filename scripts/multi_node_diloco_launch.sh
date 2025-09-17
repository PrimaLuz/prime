#!/bin/bash

# 双节点7B diloco训练启动脚本
# 节点间diloco + 节点内FSDP分片

set -e

# 配置参数
NODE_RANK=${1:-0}  # 当前节点rank
MASTER_ADDR=${2:-localhost}  # 主节点地址
MASTER_PORT=${3:-29500}  # 主节点端口
WORLD_SIZE=${4:-2}  # 总节点数
GPUS_PER_NODE=${5:-8}  # 每节点GPU数

export GLOBAL_RANK=$NODE_RANK
export GLOBAL_WORLD_SIZE=$WORLD_SIZE
export GLOBAL_ADDR=$MASTER_ADDR
export GLOBAL_PORT=${6:-26969}
export GLOBAL_UNIQUE_ID=$NODE_RANK

# 验证参数
if [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ]; then
    echo "Usage: $0 <node_rank> <master_addr> <master_port> [world_size] [gpus_per_node]"
    echo "Example: $0 0 node1 29500 2 8"
    exit 1
fi

# 环境变量配置
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  

export PYTHONPATH=/userdata/workspace/Prime

# 数据路径配置
DATA_DIR="/userdata"
OUTPUT_DIR="/userdata/workspace/output/outputs_7b_diloco_multi"
mkdir -p "$OUTPUT_DIR"

# 检查数据
if [ ! -d "$DATA_DIR/datasets/fineweb-edu" ]; then
    echo "警告: 数据集未找到，请先下载数据集"
    echo "运行: python scripts/subset_data.py --dataset_name PrimeIntellect/fineweb-edu --data_world_size $WORLD_SIZE --data_rank $NODE_RANK"
    exit 1
fi

# 日志配置
LOG_DIR="/userdata/workspace/train_logs/node_${NODE_RANK}"
mkdir -p "$LOG_DIR"

# 启动命令
echo "启动节点 $NODE_RANK/$WORLD_SIZE..."
echo "主节点: $MASTER_ADDR:$MASTER_PORT"
echo "GPU配置: $GPUS_PER_NODE GPUs per node"
echo "日志目录: $LOG_DIR"

# 使用torchrun启动
uv run torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$WORLD_SIZE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    src/zeroband/train.py \
    @configs/7B_diloco/multi_node.toml \
    --data.data_world_size $WORLD_SIZE \
    --data.data_rank $NODE_RANK \
    --ckpt.path $OUTPUT_DIR \
    --log_level INFO \
    --log_all_rank true \
    2>&1 | tee "$LOG_DIR/train.log"

echo "节点 $NODE_RANK 训练完成"