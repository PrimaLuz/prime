# 多节点训练脚本使用说明

本目录包含用于多节点训练的各种脚本，用于解决TCPStore连接问题。

## 脚本说明

### 1. `multi_node_diloco_launch.sh` - 训练启动脚本
**用途**: 启动多节点训练
**用法**:
```bash
./scripts/multi_node_diloco_launch.sh <node_rank> <master_addr> <master_port> [world_size] [gpus_per_node] [global_port]
```

**示例**:
```bash
# 主节点 (node_rank=0)
./scripts/multi_node_diloco_launch.sh 0 localhost 29500 2 8 26969

# 从节点 (node_rank=1)
./scripts/multi_node_diloco_launch.sh 1 192.168.1.100 29500 2 8 26969
```

### 2. `test_multi_node_connection.sh` - 连接测试脚本
**用途**: 测试节点间的网络连接和TCPStore通信
**用法**:
```bash
./scripts/test_multi_node_connection.sh <node_rank> <master_addr> <master_port> [world_size] [gpus_per_node] [global_port]
```

**示例**:
```bash
# 测试主节点连接
./scripts/test_multi_node_connection.sh 0 localhost 29500 2 8 26969

# 测试从节点连接
./scripts/test_multi_node_connection.sh 1 192.168.1.100 29500 2 8 26969
```

### 3. `run_multi_node_test.sh` - 综合测试脚本
**用途**: 先运行连接测试，然后启动训练
**用法**:
```bash
./scripts/run_multi_node_test.sh <node_rank> <master_addr> <master_port> [world_size] [gpus_per_node] [test_only]
```

**示例**:
```bash
# 运行测试并启动训练
./scripts/run_multi_node_test.sh 0 localhost 29500 2 8 false

# 只运行连接测试
./scripts/run_multi_node_test.sh 1 192.168.1.100 29500 2 8 true
```

### 4. `test_tcpstore_connection.py` - Python连接测试
**用途**: 底层的TCPStore连接测试
**用法**: 通常由其他脚本调用，也可以单独运行

## 使用流程

### 推荐的使用流程:

1. **首先运行连接测试**:
   ```bash
   # 在主节点上
   ./scripts/test_multi_node_connection.sh 0 localhost 29500 2 8 26969
   
   # 在从节点上
   ./scripts/test_multi_node_connection.sh 1 <master_ip> 29500 2 8 26969
   ```

2. **如果测试通过，启动训练**:
   ```bash
   # 在主节点上
   ./scripts/multi_node_diloco_launch.sh 0 localhost 29500 2 8 26969
   
   # 在从节点上
   ./scripts/multi_node_diloco_launch.sh 1 <master_ip> 29500 2 8 26969
   ```

3. **或者使用综合脚本**:
   ```bash
   # 在主节点上
   ./scripts/run_multi_node_test.sh 0 localhost 29500 2 8 false
   
   # 在从节点上
   ./scripts/run_multi_node_test.sh 1 <master_ip> 29500 2 8 false
   ```

## 故障排除

### 常见问题:

1. **连接超时**: 检查网络连接和防火墙设置
2. **端口不可达**: 确保所有需要的端口都开放
3. **TCPStore初始化失败**: 检查主节点是否已启动

### 调试步骤:

1. 运行连接测试脚本查看具体错误
2. 检查网络连接: `ping <master_ip>`
3. 检查端口开放: `telnet <master_ip> <port>`
4. 查看详细日志输出

## 环境变量

脚本会自动设置以下环境变量:
- `GLOBAL_RANK`: 节点排名
- `GLOBAL_WORLD_SIZE`: 总节点数
- `GLOBAL_ADDR`: 主节点地址
- `GLOBAL_PORT`: 全局端口
- `GLOO_SOCKET_IFNAME`: 网络接口名称

## 注意事项

1. 确保所有节点都能访问主节点
2. 确保所有需要的端口都开放
3. 主节点应该先启动
4. 如果遇到问题，先运行连接测试脚本诊断
