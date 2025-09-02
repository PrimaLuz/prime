#!/bin/bash

# 7B diloco训练前准备脚本
# 下载模型和数据集

set -e

# 配置参数
DATA_WORLD_SIZE=${1:-2}  # 数据分片数量
MODEL_SIZE="7B"
DATA_DIR="/data"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${GREEN}开始准备7B diloco训练资源...${NC}"

# 检查必要工具
command -v uv >/dev/null 2>&1 || { echo -e "${RED}uv未安装，请先安装uv${NC}"; exit 1; }
command -v huggingface-cli >/dev/null 2>&1 || { echo -e "${RED}huggingface-cli未安装${NC}"; exit 1; }

# 创建目录
mkdir -p "$DATA_DIR/datasets"
mkdir -p "$DATA_DIR/models"

# 登录Hugging Face
echo -e "${YELLOW}请确保已登录Hugging Face: huggingface-cli login${NC}"

# 下载数据集
echo -e "${GREEN}下载FineWeb-Edu数据集...${NC}"
cd "$DATA_DIR"

# 下载完整数据集到本地
if [ ! -d "datasets/fineweb-edu" ]; then
    echo "下载FineWeb-Edu数据集..."
    uv run python3 /root/github/prime/scripts/subset_data.py \
        --dataset_name PrimeIntellect/fineweb-edu \
        --data_world_size $DATA_WORLD_SIZE \
        --data_rank 0 \
        --max_shards 100 \
        --output_dir datasets/fineweb-edu
    
    # 为其他rank创建符号链接
    for rank in $(seq 1 $((DATA_WORLD_SIZE-1))); do
        echo "为rank $rank 创建数据链接..."
        ln -sf "$(pwd)/datasets/fineweb-edu" "$(pwd)/datasets/fineweb-edu-rank$rank"
    done
else
    echo "数据集已存在，跳过下载"
fi

# 下载7B模型权重
echo -e "${GREEN}下载7B Llama2模型...${NC}"
MODEL_NAME="meta-llama/Llama-2-7b-hf"

# 检查模型访问权限
if huggingface-cli whoami | grep -q "meta-llama"; then
    echo "已授权访问Llama2模型"
    
    if [ ! -d "models/llama2-7b" ]; then
        echo "下载Llama2-7B模型..."
        uv run python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = '$MODEL_NAME'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='cpu'
)

# 保存到本地
tokenizer.save_pretrained('models/llama2-7b')
model.save_pretrained('models/llama2-7b')
print('模型下载完成')
"
    else
        echo "模型已存在，跳过下载"
    fi
else
    echo -e "${RED}警告: 需要申请Llama2模型访问权限${NC}"
    echo "请访问: https://huggingface.co/meta-llama/Llama-2-7b-hf"
    echo "或使用开源替代:"
    echo "MODEL_NAME='microsoft/DialoGPT-medium' 或其他开源模型"
fi

# 验证下载
echo -e "${GREEN}验证下载结果...${NC}"
if [ -d "$DATA_DIR/datasets/fineweb-edu" ]; then
    echo "✓ 数据集已准备就绪"
    du -sh "$DATA_DIR/datasets/fineweb-edu"
else
    echo -e "${RED}✗ 数据集下载失败${NC}"
fi

if [ -d "$DATA_DIR/models/llama2-7b" ]; then
    echo "✓ 模型已准备就绪"
    du -sh "$DATA_DIR/models/llama2-7b"
else
    echo -e "${YELLOW}⚠ 模型未完全下载，将使用随机初始化${NC}"
fi

# 创建启动脚本模板
echo -e "${GREEN}创建启动脚本模板...${NC}"
cat > /root/github/prime/scripts/start_7b_diloco.sh << 'EOF'
#!/bin/bash
# 7B diloco双节点训练启动模板

# 节点1 (主节点)
# ./scripts/multi_node_diloco_launch.sh 0 node1_ip 29500 2 8

# 节点2 (工作节点)
# ./scripts/multi_node_diloco_launch.sh 1 node1_ip 29500 2 8

echo "请根据实际IP地址替换node1_ip"
echo "确保所有节点网络连通"
EOF

chmod +x /root/github/prime/scripts/start_7b_diloco.sh

echo -e "${GREEN}准备完成！${NC}"
echo "数据集路径: $DATA_DIR/datasets/fineweb-edu"
echo "模型路径: $DATA_DIR/models/llama2-7b"
echo "启动脚本: /root/github/prime/scripts/start_7b_diloco.sh"