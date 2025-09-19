#!/bin/bash

# 多节点训练测试脚本
# 包含连接测试和训练启动

set -e

# 配置参数
NODE_RANK=${1:-0}  # 当前节点rank
MASTER_ADDR=${2:-localhost}  # 主节点地址
MASTER_PORT=${3:-29500}  # 主节点端口
WORLD_SIZE=${4:-2}  # 总节点数
GPUS_PER_NODE=${5:-8}  # 每节点GPU数
TEST_ONLY=${6:-false}  # 是否只运行测试

# 验证参数
if [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ]; then
    echo "Usage: $0 <node_rank> <master_addr> <master_port> [world_size] [gpus_per_node] [test_only]"
    echo "Example: $0 0 localhost 29500 2 8 false"
    echo "Example: $0 1 192.168.1.100 29500 2 8 true"
    echo ""
    echo "参数说明:"
    echo "  node_rank: 当前节点的排名 (0为主节点)"
    echo "  master_addr: 主节点IP地址"
    echo "  master_port: 主节点端口"
    echo "  world_size: 总节点数"
    echo "  gpus_per_node: 每节点GPU数"
    echo "  test_only: 是否只运行连接测试 (true/false)"
    exit 1
fi

echo "=========================================="
echo "多节点训练测试脚本"
echo "=========================================="
echo "节点排名: $NODE_RANK/$WORLD_SIZE"
echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"
echo "每节点GPU数: $GPUS_PER_NODE"
echo "仅测试模式: $TEST_ONLY"
echo "=========================================="

# 运行连接测试
echo "运行连接测试..."
if ./scripts/test_multi_node_connection.sh $NODE_RANK $MASTER_ADDR $MASTER_PORT $WORLD_SIZE $GPUS_PER_NODE; then
    echo "✓ 连接测试通过"
else
    echo "✗ 连接测试失败"
    exit 1
fi

# 如果只是测试模式，则退出
if [ "$TEST_ONLY" = "true" ]; then
    echo "测试完成，退出。"
    exit 0
fi

echo ""
echo "开始训练..."
echo "=========================================="

# 运行训练
./scripts/multi_node_diloco_launch.sh $NODE_RANK $MASTER_ADDR $MASTER_PORT $WORLD_SIZE $GPUS_PER_NODE
