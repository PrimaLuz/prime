#!/bin/bash

# 多节点TCPStore连接测试脚本
# 用于测试节点间的网络连接和TCPStore通信

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
echo "TCPStore连接测试"
echo "=========================================="
echo "节点排名: $NODE_RANK/$WORLD_SIZE"
echo "主节点地址: $MASTER_ADDR:$MASTER_PORT"
echo "全局端口: $GLOBAL_PORT"
echo "每节点GPU数: $GPUS_PER_NODE"
echo "=========================================="

# 环境变量配置
export PYTHONPATH=/userdata/workspace/Prime

# 网络连接测试
echo "1. 测试基础网络连接..."
if [ "$NODE_RANK" != "0" ]; then
    echo "   测试到主节点的连接: $MASTER_ADDR:$GLOBAL_PORT"
    if timeout 10 bash -c "echo > /dev/tcp/$MASTER_ADDR/$GLOBAL_PORT" 2>/dev/null; then
        echo "   ✓ 主节点连接成功"
    else
        echo "   ⚠ 主节点端口 $GLOBAL_PORT 不可达"
        echo "   这是正常的，因为TCPStore master还没有启动"
        echo "   继续测试其他端口..."
    fi
    
    # 测试所有将使用的端口
    echo "   测试所有TCPStore端口..."
    failed_ports=()
    for port in $(seq $GLOBAL_PORT $((GLOBAL_PORT + GPUS_PER_NODE - 1))); do
        if timeout 5 bash -c "echo > /dev/tcp/$MASTER_ADDR/$port" 2>/dev/null; then
            echo "   ✓ 端口 $port 可达"
        else
            echo "   ⚠ 端口 $port 不可达 (TCPStore master未启动)"
            failed_ports+=($port)
        fi
    done
    
    if [ ${#failed_ports[@]} -gt 0 ]; then
        echo "   注意: 以下端口不可达: ${failed_ports[*]}"
        echo "   这是正常的，因为TCPStore master还没有启动"
        echo "   将在TCPStore测试中验证实际连接"
    fi
else
    echo "   主节点跳过网络连接测试"
fi

echo ""
echo "2. 测试TCPStore连接..."

# 运行TCPStore连接测试
if uv run python scripts/test_tcpstore_connection.py; then
    echo "   ✓ TCPStore连接测试成功"
    echo ""
    echo "=========================================="
    echo "所有测试通过！可以开始训练。"
    echo "=========================================="
else
    echo "   ✗ TCPStore连接测试失败"
    echo ""
    echo "=========================================="
    echo "测试失败！请检查以下问题："
    echo "1. 网络连接是否正常"
    echo "2. 端口是否被占用"
    echo "3. 防火墙设置"
    echo "4. 主节点是否已启动"
    echo "=========================================="
    exit 1
fi
