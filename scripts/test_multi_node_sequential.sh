#!/bin/bash

# 多节点顺序测试脚本
# 主节点先启动TCPStore服务，然后从节点连接测试

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
export BASE_PORT=${BASE_PORT:-10001}
export GLOO_SOCKET_IFNAME=eth0

# 验证参数
if [ -z "$NODE_RANK" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ]; then
    echo "Usage: $0 <node_rank> <master_addr> <master_port> [world_size] [gpus_per_node] [global_port]"
    echo "Example: $0 0 localhost 29500 2 8 26969"
    echo "Example: $0 1 192.168.1.100 29500 2 8 26969"
    exit 1
fi

echo "=========================================="
echo "多节点顺序连接测试"
echo "=========================================="
echo "节点排名: $NODE_RANK/$WORLD_SIZE"
echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"
echo "全局端口: $GLOBAL_PORT"
echo "每节点GPU数: $GPUS_PER_NODE"
echo "=========================================="

# 环境变量配置
export PYTHONPATH=/userdata/workspace/Prime

if [ "$NODE_RANK" = "0" ]; then
    echo "主节点模式：启动TCPStore服务并等待从节点连接..."
    echo ""
    echo "请在其他节点上运行以下命令："
    echo "./scripts/test_multi_node_connection.sh 1 $MASTER_ADDR $MASTER_PORT $WORLD_SIZE $GPUS_PER_NODE $GLOBAL_PORT"
    echo ""
    echo "等待从节点连接..."
    
    # 主节点启动TCPStore服务
    uv run python scripts/test_tcpstore_connection.py &
    MASTER_PID=$!
    
    echo "TCPStore服务已启动 (PID: $MASTER_PID)"
    echo "等待从节点连接测试完成..."
    
    # 等待从节点完成测试
    wait $MASTER_PID
    echo "主节点测试完成"
    
else
    echo "从节点模式：等待主节点启动TCPStore服务..."
    echo "请确保主节点已启动TCPStore服务"
    echo ""
    
    # 等待一段时间让主节点启动
    echo "等待5秒让主节点启动服务..."
    sleep 5
    
    # 从节点连接测试
    echo "开始连接测试..."
    if uv run python scripts/test_tcpstore_connection.py; then
        echo "✓ 从节点连接测试成功"
    else
        echo "✗ 从节点连接测试失败"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
