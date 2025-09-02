# 7B Diloco 双节点训练完整指南

## 架构设计

### 训练策略
- **节点间**: DiLoCo 分布式优化
- **节点内**: FSDP 参数分片
- **数据并行**: 跨节点数据分片
- **通信优化**: Int8量化 + 多连接带宽利用

### 配置参数
- **模型**: 7B Llama2
- **节点数**: 2
- **每节点GPU**: 8 (H100)
- **总GPU数**: 16
- **微批量**: 2 per GPU
- **全局批量**: 2048 tokens
- **DiLoCo步**: 50 inner steps

## 环境准备

### 1. 系统要求
```bash
# 每节点配置
- GPU: 8x H100 80GB
- CPU: 32+ cores
- RAM: 512GB+
- 网络: 10Gbps+ 互联
- 存储: 2TB+ NVMe
```

### 2. 软件安装
```bash
# 在所有节点执行
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime/main/scripts/install/install.sh | bash
```

### 3. 网络配置
```bash
# 节点1 (主节点)
export MASTER_ADDR=node1_ip  # 替换为实际IP
export MASTER_PORT=29500

# 节点2
export MASTER_ADDR=node1_ip
export MASTER_PORT=29500
```

## 数据准备

### 1. 数据集下载
```bash
# 在主节点执行
./scripts/download_models_and_data.sh 2

# 或手动下载
python scripts/subset_data.py \
    --dataset_name PrimeIntellect/fineweb-edu \
    --data_world_size 2 \
    --data_rank 0 \
    --max_shards 100 \
    --output_dir /data/datasets/fineweb-edu
```

### 2. 数据分布
```bash
# 数据分片策略
Rank 0: /data/datasets/fineweb-edu-rank0
Rank 1: /data/datasets/fineweb-edu-rank1
```

## 模型准备

### 1. 7B Llama2下载
```bash
# 需要Hugging Face权限
huggingface-cli login

# 下载模型
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
model.save_pretrained('/data/models/llama2-7b')
"
```

### 2. 替代方案
```bash
# 使用开源模型
MODEL_NAME="microsoft/DialoGPT-medium"
```

## 训练启动

### 1. 主节点启动
```bash
# 节点1 (主节点)
./scripts/multi_node_diloco_launch.sh 0 node1_ip 29500 2 8
```

### 2. 工作节点启动
```bash
# 节点2 (在第二台机器执行)
./scripts/multi_node_diloco_launch.sh 1 node1_ip 29500 2 8
```

### 3. 监控启动
```bash
# 实时日志监控
tail -f logs/node_0/train.log
tail -f logs/node_1/train.log
```

## 配置文件详解

### 7B_diloco/multi_node.toml
```toml
name_model = "7B"
type_model = "llama2"

[train]
micro_bs = 2          # 每GPU微批量
ac_ckpt = true        # 激活检查点

[optim]
batch_size = 2048     # 全局批量大小
warmup_steps = 1000
total_steps = 88_000

[diloco]
inner_steps = 50      # 每50步同步一次
compression = "uint8" # 4倍通信压缩
```

## 性能优化

### 1. 网络优化
```bash
# 启用RDMA
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# 多连接优化
export ZERO_BAND_MAX_CONNECTIONS=8
```

### 2. 内存优化
```bash
# CPU Offload
export ZERO_BAND_OFFLOAD=1
export ZERO_BAND_OFFLOAD_RATIO=0.8
```

### 3. 检查点优化
```bash
# 异步检查点
export ZERO_BAND_ASYNC_CKPT=1
export ZERO_BAND_LIVE_RECO_PORT=8888
```

## 故障排除

### 1. 网络问题
```bash
# 检查连通性
iperf3 -s  # 节点1
iperf3 -c node1_ip  # 节点2

# 检查NCCL
export NCCL_DEBUG=INFO
```

### 2. 内存问题
```bash
# 监控内存
watch -n 1 nvidia-smi
htop
```

### 3. 数据问题
```bash
# 验证数据集
python -c "
from datasets import load_dataset
ds = load_dataset('/data/datasets/fineweb-edu', split='train')
print(len(ds))
"
```

## 监控指标

### 1. 训练指标
- **Loss**: 每step的training loss
- **Throughput**: tokens/second
- **Memory**: GPU/CPU使用率
- **Network**: 通信带宽利用率

### 2. DiLoCo特定
- **Outer step time**: 全局同步时间
- **Pseudo-gradient norm**: 伪梯度范数
- **Compression ratio**: Int8压缩效果

### 3. 日志查看
```bash
# 训练日志
grep "loss" logs/node_0/train.log

# 性能日志
grep "tokens" logs/node_0/train.log

# 通信日志
grep "all-reduce" logs/node_0/train.log
```

## 扩展配置

### 1. 多节点扩展
```bash
# 4节点配置
./scripts/multi_node_diloco_launch.sh 0 node1_ip 29500 4 8
./scripts/multi_node_diloco_launch.sh 1 node1_ip 29500 4 8
./scripts/multi_node_diloco_launch.sh 2 node1_ip 29500 4 8
./scripts/multi_node_diloco_launch.sh 3 node1_ip 29500 4 8
```

### 2. 模型扩展
```bash
# 10B模型配置
name_model = "10B"
batch_size = 1024  # 调整批量大小
```

## 预期性能

### 7B模型在2节点16GPU上
- **训练速度**: ~2,000 tokens/sec/GPU
- **全局吞吐**: ~32,000 tokens/sec
- **内存使用**: ~60GB per GPU
- **网络通信**: ~4Gbps per connection
- **检查点时间**: ~30 seconds

## 检查点管理

### 1. 本地检查点
```bash
# 检查点路径
/data/outputs_7b_diloco_multi/
├── step_1000/
├── step_2000/
└── latest/
```

### 2. 远程备份
```bash
# 上传到S3
aws s3 sync /data/outputs_7b_diloco_multi s3://your-bucket/7b-diloco/
```

### 3. 模型导出
```bash
# 转换为HuggingFace格式
python scripts/export_dcp.py @configs/7B_diloco/multi_node.toml \
    --ckpt.path /path/to/exported \
    --ckpt.resume /data/outputs_7b_diloco_multi/step_XXXXX \
    --torch_dtype bfloat16
```