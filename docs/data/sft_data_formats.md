# SFT数据格式支持文档

## 支持的数据格式

`sft_data.py` 现在支持以下数据格式：

### 1. Parquet格式 (原有)
- **文件扩展名**: `.parquet`
- **数据结构**: 
  ```json
  {
    "messages": "[{\"role\": \"user\", \"content\": \"Hello\"}, {\"role\": \"assistant\", \"content\": \"Hi there!\"}]"
  }
  ```
- **使用示例**:
  ```toml
  dataset_name_or_paths = "datasets/my_sft_data"
  ```

### 2. JSON Lines格式
- **文件扩展名**: `.jsonl`
- **数据结构**: 每行一个JSON对象
  ```json
  {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
  {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm good!"}]}
  ```
- **使用示例**:
  ```toml
  dataset_name_or_paths = "datasets/my_data.jsonl"
  ```

### 3. CSV格式
- **文件扩展名**: `.csv`
- **支持两种结构**:
  
  **结构A**: 完整对话在单列中
  ```csv
  messages
  "[{\"role\": \"user\", \"content\": \"Hello\"}, {\"role\": \"assistant\", \"content\": \"Hi!\"}]"
  ```
  
  **结构B**: 每行一条消息
  ```csv
  role,content
  user,Hello
  assistant,Hi there!
  user,How are you?
  assistant,I'm doing well!
  ```
- **使用示例**:
  ```toml
  dataset_name_or_paths = "datasets/my_data.csv"
  ```

### 4. Hugging Face Hub格式
- **格式**: `username/dataset_name`
- **使用示例**:
  ```toml
  dataset_name_or_paths = "microsoft/DialoGPT"
  ```

### 5. JSON格式
- **文件扩展名**: `.json`
- **数据结构**: 单个JSON文件包含对话数组
  ```json
  [
    {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]},
    {"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "Good!"}]}
  ]
  ```

## 配置示例

### 配置文件 (TOML)
```toml
[data]
dataset_name_or_paths = "datasets/my_sft_data.jsonl"  # 可以是任何支持的格式
seq_length = 2048
num_workers = 4
fake = false
```

### 代码使用示例
```python
from zeroband.sft_data import get_sft_dataloader
from transformers import AutoTokenizer

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 获取数据加载器
dataloader = get_sft_dataloader(
    tokenizer=tokenizer,
    world_size=1,
    rank=0,
    batch_size=8,
    data_config=data_config,  # 包含dataset_name_or_paths配置
    train_on_inputs=False  # 只计算assistant回复的loss
)
```

## 数据格式验证

### 对话格式要求
所有格式都必须遵循以下对话结构：
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help you?"},
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "I don't have access to real-time weather data."}
  ]
}
```

### 角色支持
- `user`: 用户消息
- `assistant`: 助手回复
- `system`: 系统提示

## 错误处理

如果文件格式不被识别，系统会：
1. 根据文件扩展名自动选择合适的数据集类
2. 如果扩展名未知，默认使用ParquetSFTDataset
3. 记录警告信息到日志

## 性能建议

1. **大文件处理**: 对于大文件，建议使用Parquet格式，性能最佳
2. **小文件处理**: 对于小文件，JSON Lines格式更易编辑和调试
3. **Hugging Face Hub**: 自动支持流式加载，适合大数据集
4. **并行处理**: 所有格式都支持多进程数据加载