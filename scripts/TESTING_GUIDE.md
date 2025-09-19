# 多节点连接测试指南

## 问题说明

TCPStore需要主节点先启动并监听端口，然后从节点才能连接。因此测试需要按特定顺序进行。

## 正确的测试顺序

### 方法1：顺序测试（推荐）

**步骤1：在主节点上启动TCPStore服务**
```bash
# 在主节点 (node_rank=0) 上运行
./scripts/test_multi_node_connection.sh 0 localhost 29500 2 4 26969
```

**步骤2：在从节点上连接测试**
```bash
# 在从节点 (node_rank=1) 上运行
./scripts/test_multi_node_connection.sh 1 172.16.92.214 29500 2 4 26969
```

### 方法2：使用顺序测试脚本

**步骤1：在主节点上启动**
```bash
./scripts/test_multi_node_sequential.sh 0 localhost 29500 2 4 26969
```

**步骤2：在从节点上连接**
```bash
./scripts/test_multi_node_sequential.sh 1 172.16.92.214 29500 2 4 26969
```

## 测试结果解释

### 主节点测试结果
```
✓ TCPStore连接测试成功
所有测试通过！可以开始训练。
```

### 从节点测试结果（正常情况）
```
⚠ 主节点端口 26969 不可达
这是正常的，因为TCPStore master还没有启动
继续测试其他端口...

⚠ 端口 26969 不可达 (TCPStore master未启动)
⚠ 端口 26970 不可达 (TCPStore master未启动)
⚠ 端口 26971 不可达 (TCPStore master未启动)
⚠ 端口 26972 不可达 (TCPStore master未启动)

注意: 以下端口不可达: 26969 26970 26971 26972
这是正常的，因为TCPStore master还没有启动
将在TCPStore测试中验证实际连接

✓ TCPStore连接测试成功
所有测试通过！可以开始训练。
```

## 故障排除

### 如果从节点连接失败

1. **检查主节点是否已启动**
   ```bash
   # 在主节点上检查端口是否在监听
   netstat -tlnp | grep 26969
   ```

2. **检查网络连接**
   ```bash
   # 在从节点上测试网络连接
   ping 172.16.92.214
   telnet 172.16.92.214 26969
   ```

3. **检查防火墙设置**
   ```bash
   # 在主节点上检查防火墙
   sudo ufw status
   sudo iptables -L
   ```

### 如果主节点启动失败

1. **检查端口是否被占用**
   ```bash
   lsof -i :26969
   ```

2. **检查Python环境**
   ```bash
   uv run python --version
   uv run python -c "import torch; print(torch.__version__)"
   ```

## 训练启动

测试通过后，可以启动训练：

**主节点**
```bash
./scripts/multi_node_diloco_launch.sh 0 localhost 29500 2 4 26969
```

**从节点**
```bash
./scripts/multi_node_diloco_launch.sh 1 172.16.92.214 29500 2 4 26969
```

## 注意事项

1. **必须按顺序启动**：主节点必须先启动，然后从节点才能连接
2. **网络配置**：确保所有节点都能相互访问
3. **端口开放**：确保所有需要的端口都开放
4. **时间同步**：确保所有节点时间同步
