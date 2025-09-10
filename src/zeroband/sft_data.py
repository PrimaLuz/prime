from dataclasses import dataclass, asdict
import random
import os
from typing import Any, Generator, Optional, List, Dict, TypedDict, Union
import functools
import threading
import json
import csv

from zeroband.models.llama.model import create_block_mask_from_seqlens
from zeroband.utils.logger import get_logger
from zeroband.utils.world_info import get_world_info
from zeroband.config import DataConfig

import torch
from torch.utils.data import IterableDataset, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed.checkpoint.stateful import Stateful

from datasets import load_dataset, Dataset as HFDataset
from pyarrow import parquet as pq
from transformers import PreTrainedTokenizer


class SFTConversation(TypedDict):
    """SFT对话数据结构"""
    messages: List[Dict[str, str]]  # [{"role": "user", "content": "..."}, ...]


class SFTBatchOutput(TypedDict):
    """SFT批次输出格式"""
    input_ids: torch.IntTensor
    labels: torch.IntTensor
    attention_mask: torch.BoolTensor
    conversation_lengths: List[int]  # 每个对话的长度


@dataclass
class SFTDatasetState:
    """SFT数据集状态"""
    current_conversation_idx: int
    processed_conversations: int
    epoch: int


class SFTDataset(IterableDataset, Stateful):
    """
    SFT数据集处理类
    
    功能:
    1. 处理对话格式的SFT数据
    2. 支持instruction-response结构
    3. 只计算assistant回复部分的loss
    4. 支持多轮对话
    """
    
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int,
        train_on_inputs: bool = False,  # 是否对用户输入也计算loss
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_on_inputs = train_on_inputs
        
        # 设置特殊token
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"
        self.system_token = "<|system|>"
        self.eos_token = tokenizer.eos_token
        
        # 添加特殊token到tokenizer
        special_tokens = {
            "additional_special_tokens": [self.user_token, self.assistant_token, self.system_token]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        self.state = SFTDatasetState(
            current_conversation_idx=0,
            processed_conversations=0,
            epoch=0
        )
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """格式化对话为训练格式"""
        formatted = ""
        
        for message in conversation:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                formatted += f"{self.user_token}\n{content}\n{self.assistant_token}\n"
            elif role == "assistant":
                formatted += f"{content}{self.eos_token}\n"
            elif role == "system":
                formatted = f"{self.system_token}\n{content}\n" + formatted
        
        return formatted.strip()
    
    def _create_sft_sample(self, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
        """创建单个SFT训练样本"""
        full_text = self._format_conversation(conversation)
        
        # 编码完整对话
        full_tokens = self.tokenizer.encode(full_text)
        
        # 如果超过最大长度，截断
        if len(full_tokens) > self.max_seq_length:
            full_tokens = full_tokens[:self.max_seq_length]
        
        # 创建input_ids和labels
        input_ids = full_tokens[:-1]
        labels = full_tokens[1:]
        
        # 创建attention mask
        attention_mask = [True] * len(input_ids)
        
        # 标记哪些位置应该计算loss
        if not self.train_on_inputs:
            # 找到assistant回复的位置，只对这些位置计算loss
            assistant_start = False
            for i, (token_id, next_token_id) in enumerate(zip(input_ids, labels)):
                token = self.tokenizer.decode([token_id])
                next_token = self.tokenizer.decode([next_token_id])
                
                if self.assistant_token in token:
                    assistant_start = True
                    continue
                
                if assistant_start:
                    # 检查是否是assistant回复的结束
                    if self.user_token in next_token or self.eos_token in next_token:
                        assistant_start = False
                else:
                    # 用户输入或系统消息，不计算loss
                    labels[i] = -100
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "conversation_length": len(input_ids)
        }
    
    def __iter__(self) -> Generator[SFTBatchOutput, Any, None]:
        """生成SFT训练数据"""
        data_iter = iter(self.dataset)
        
        while True:
            try:
                # 获取下一个对话
                sample = next(data_iter)
                conversation = sample["messages"]  # 期望格式: [{"role": "user", "content": "..."}, ...]
                
                sft_sample = self._create_sft_sample(conversation)
                
                yield {
                    "input_ids": torch.tensor(sft_sample["input_ids"], dtype=torch.long),
                    "labels": torch.tensor(sft_sample["labels"], dtype=torch.long),
                    "attention_mask": torch.tensor(sft_sample["attention_mask"], dtype=torch.bool),
                    "conversation_lengths": [sft_sample["conversation_length"]]
                }
                
                self.state.processed_conversations += 1
                
            except StopIteration:
                # 数据集遍历完成，重新开始
                data_iter = iter(self.dataset)
                self.state.epoch += 1
                self.state.current_conversation_idx = 0
    
    def state_dict(self):
        """保存状态"""
        return {
            "dataset": self.dataset.state_dict() if hasattr(self.dataset, 'state_dict') else {},
            "sft_state": asdict(self.state)
        }
    
    def load_state_dict(self, state_dict):
        """恢复状态"""
        if hasattr(self.dataset, 'load_state_dict'):
            self.dataset.load_state_dict(state_dict["dataset"])
        self.state = SFTDatasetState(**state_dict["sft_state"])


class JsonlSFTDataset(IterableDataset, Stateful):
    """
    从JSON Lines文件加载SFT数据
    
    期望的JSONL格式:
    每行一个JSON对象: {"messages": [{"role": "user", "content": "..."}, ...]}
    """
    
    def __init__(self, files: List[str], tokenizer: PreTrainedTokenizer):
        self.arg_files = files
        self.tokenizer = tokenizer
        self.state = None

    def _lazy_init(self):
        """延迟初始化，支持多进程"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if worker_info.num_workers > len(self.arg_files):
                get_logger().warning(
                    f"dataloader rank {worker_info.id} Number of workers {worker_info.num_workers} "
                    f"is greater than the number of files {len(self.arg_files)}"
                )
                self.state = PQDatasetState(
                    files=self.arg_files,
                    file_index=0,
                    row_index=worker_info.id,
                    increment=worker_info.num_workers,
                    init_row_index=worker_info.id,
                )
                return
            
            files = self.arg_files[worker_info.id :: worker_info.num_workers]
        else:
            files = self.arg_files
        
        self.state = PQDatasetState(files=files, file_index=0, row_index=0, increment=1, init_row_index=0)

    def __iter__(self):
        """生成对话数据"""
        if self.state is None:
            self._lazy_init()
        
        while True:
            file = self.state.files[self.state.file_index]
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                    while self.state.row_index < len(lines):
                        line = lines[self.state.row_index].strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if "messages" in data and isinstance(data["messages"], list):
                                    yield {"messages": data["messages"]}
                            except json.JSONDecodeError:
                                pass
                        
                        self.state.row_index += self.state.increment
                        
                        if self.state.row_index >= len(lines):
                            self.state.row_index = self.state.init_row_index
                            self.state.file_index += 1
                            if self.state.file_index >= len(self.state.files):
                                self.state.file_index = 0
                            break
                            
            except FileNotFoundError:
                get_logger().warning(f"File not found: {file}")
                self.state.file_index += 1
                if self.state.file_index >= len(self.state.files):
                    self.state.file_index = 0
                break

    @property
    def is_empty(self):
        return len(self.arg_files) == 0

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state) if self.state is not None else {}

    def load_state_dict(self, state_dict):
        self.state = PQDatasetState(**state_dict)


class CSVSFTDataset(IterableDataset, Stateful):
    """
    从CSV文件加载SFT数据
    
    期望的CSV格式:
    - 列: role, content (每行一条消息)
    - 或: messages (JSON字符串格式的完整对话)
    """
    
    def __init__(self, files: List[str], tokenizer: PreTrainedTokenizer, messages_column: str = "messages"):
        self.arg_files = files
        self.tokenizer = tokenizer
        self.messages_column = messages_column
        self.state = None

    def _lazy_init(self):
        """延迟初始化，支持多进程"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if worker_info.num_workers > len(self.arg_files):
                get_logger().warning(
                    f"dataloader rank {worker_info.id} Number of workers {worker_info.num_workers} "
                    f"is greater than the number of files {len(self.arg_files)}"
                )
                self.state = PQDatasetState(
                    files=self.arg_files,
                    file_index=0,
                    row_index=worker_info.id,
                    increment=worker_info.num_workers,
                    init_row_index=worker_info.id,
                )
                return
            
            files = self.arg_files[worker_info.id :: worker_info.num_workers]
        else:
            files = self.arg_files
        
        self.state = PQDatasetState(files=files, file_index=0, row_index=0, increment=1, init_row_index=0)

    def __iter__(self):
        """生成对话数据"""
        if self.state is None:
            self._lazy_init()
        
        while True:
            file = self.state.files[self.state.file_index]
            
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in list(reader)[self.state.row_index::self.state.increment]:
                        try:
                            if self.messages_column in row and row[self.messages_column]:
                                # 完整对话在单个列中
                                messages = json.loads(row[self.messages_column])
                                if isinstance(messages, list):
                                    yield {"messages": messages}
                            elif "role" in row and "content" in row:
                                # 每行一条消息，需要组合成对话
                                messages = [{"role": row["role"], "content": row["content"]}]
                                yield {"messages": messages}
                        
                        except (json.JSONDecodeError, KeyError):
                            continue
            
            except FileNotFoundError:
                get_logger().warning(f"File not found: {file}")
            
            self.state.file_index += 1
            if self.state.file_index >= len(self.state.files):
                self.state.file_index = 0
                break

    @property
    def is_empty(self):
        return len(self.arg_files) == 0

    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state) if self.state is not None else {}

    def load_state_dict(self, state_dict):
        self.state = PQDatasetState(**state_dict)


class HFSFTDataset(IterableDataset, Stateful):
    """
    从Hugging Face Hub加载SFT数据
    
    支持Hugging Face数据集库中的任何对话格式数据集
    """
    
    def __init__(self, dataset_name: str, split: str = "train", messages_field: str = "messages"):
        self.dataset_name = dataset_name
        self.split = split
        self.messages_field = messages_field
        self.dataset = None
        self.current_index = 0

    def _load_dataset(self):
        """延迟加载Hugging Face数据集"""
        if self.dataset is None:
            try:
                self.dataset = load_dataset(self.dataset_name, split=self.split)
                get_logger().info(f"Loaded Hugging Face dataset: {self.dataset_name}")
            except Exception as e:
                get_logger().error(f"Failed to load dataset {self.dataset_name}: {e}")
                raise

    def __iter__(self):
        """生成对话数据"""
        self._load_dataset()
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # 在多进程环境中分配数据
            start_idx = worker_info.id
            step = worker_info.num_workers
        else:
            start_idx = 0
            step = 1
        
        while True:
            if self.current_index >= len(self.dataset):
                self.current_index = 0
            
            for idx in range(self.current_index + start_idx, len(self.dataset), step):
                if idx >= len(self.dataset):
                    continue
                
                item = self.dataset[idx]
                if self.messages_field in item:
                    messages = item[self.messages_field]
                    if isinstance(messages, list):
                        yield {"messages": messages}
                
                self.current_index = idx + 1
            
            # 重新开始下一轮
            self.current_index = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "current_index": self.current_index,
            "dataset_name": self.dataset_name,
            "split": self.split
        }

    def load_state_dict(self, state_dict):
        self.current_index = state_dict["current_index"]
        self.dataset_name = state_dict["dataset_name"]
        self.split = state_dict["split"]
        self.dataset = None  # 重新加载数据集


class ParquetSFTDataset(IterableDataset, Stateful):
    """
    从Parquet文件加载SFT数据
    
    期望的Parquet格式:
    - messages: JSON字符串，格式为 [{"role": "user", "content": "..."}, ...]
    """
    
    def __init__(self, files: List[str], tokenizer: PreTrainedTokenizer):
        self.arg_files = files
        self.tokenizer = tokenizer
        self.state = None
    
    def _lazy_init(self):
        """延迟初始化，支持多进程"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            if worker_info.num_workers > len(self.arg_files):
                get_logger().warning(
                    f"dataloader rank {worker_info.id} Number of workers {worker_info.num_workers} "
                    f"is greater than the number of files {len(self.arg_files)}"
                )
                self.state = PQDatasetState(
                    files=self.arg_files,
                    file_index=0,
                    row_index=worker_info.id,
                    increment=worker_info.num_workers,
                    init_row_index=worker_info.id,
                )
                return
            
            files = self.arg_files[worker_info.id :: worker_info.num_workers]
        else:
            files = self.arg_files
        
        self.state = PQDatasetState(files=files, file_index=0, row_index=0, increment=1, init_row_index=0)
    
    def __iter__(self):
        """生成对话数据"""
        if self.state is None:
            self._lazy_init()
        
        while True:
            file = self.state.files[self.state.file_index]
            
            parquet_file = pq.ParquetFile(file)
            table = parquet_file.read()
            
            # 检查是否有messages列
            if "messages" not in table.column_names:
                raise ValueError(f"File {file} does not contain 'messages' column")
            
            messages_column = table["messages"]
            
            while True:
                try:
                    # 获取对话数据
                    messages_str = str(messages_column[self.state.row_index])
                    messages = json.loads(messages_str)
                    
                    # 验证格式
                    if not isinstance(messages, list):
                        continue
                    
                    valid = all(
                        isinstance(msg, dict) and "role" in msg and "content" in msg
                        for msg in messages
                    )
                    
                    if valid:
                        yield {"messages": messages}
                    
                    self.state.row_index += self.state.increment
                    
                    if self.state.row_index >= len(messages_column):
                        self.state.row_index = self.state.init_row_index
                        self.state.file_index += 1
                        if self.state.file_index >= len(self.state.files):
                            self.state.file_index = 0
                        break
                        
                except (json.JSONDecodeError, IndexError):
                    self.state.row_index += self.state.increment
                    continue
    
    @property
    def is_empty(self):
        return len(self.arg_files) == 0
    
    def state_dict(self) -> dict[str, Any]:
        return asdict(self.state) if self.state is not None else {}
    
    def load_state_dict(self, state_dict):
        self.state = PQDatasetState(**state_dict)


def sft_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """SFT数据的批处理函数"""
    
    input_ids = []
    labels = []
    attention_masks = []
    
    max_len = max(len(sample["input_ids"]) for sample in samples)
    
    for sample in samples:
        curr_len = len(sample["input_ids"])
        
        # 填充到最大长度
        padding_length = max_len - curr_len
        
        input_ids.append(
            torch.cat([
                sample["input_ids"],
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
        )
        
        labels.append(
            torch.cat([
                sample["labels"],
                torch.full((padding_length,), -100, dtype=torch.long)
            ])
        )
        
        attention_masks.append(
            torch.cat([
                sample["attention_mask"],
                torch.zeros(padding_length, dtype=torch.bool)
            ])
        )
    
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0),
        "attention_mask": torch.stack(attention_masks, dim=0),
    }


class SFTDataLoader(StatefulDataLoader):
    """SFT专用的数据加载器包装器"""
    
    def __init__(self, original_dataloader: StatefulDataLoader):
        self.original_dataloader = original_dataloader
        self._prefetch_iterator = None
    
    def __iter__(self):
        if self._prefetch_iterator is not None:
            return self._prefetch_iterator
        
        self._prefetch_iterator = self._SFTIterator(self.original_dataloader)
        return self._prefetch_iterator
    
    def state_dict(self):
        if self._prefetch_iterator is not None:
            self._prefetch_iterator._await_prefetch()
        
        return {
            'dataloader_state': None if self._prefetch_iterator else self.original_dataloader.state_dict(),
            '_prefetch_iterator': None if self._prefetch_iterator is None else self._prefetch_iterator.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        if state_dict['dataloader_state'] is not None:
            self.original_dataloader.load_state_dict(state_dict['dataloader_state'])
        if state_dict['_prefetch_iterator'] is not None:
            self._prefetch_iterator = self._SFTIterator(self.original_dataloader)
    
    class _SFTIterator(Stateful):
        def __init__(self, original_dataloader: StatefulDataLoader):
            self.dataloader_iter = iter(original_dataloader)
            self.ready_batch = None
            self.thread = None
            self._prefetch_next()
        
        def state_dict(self) -> Dict[str, Any]:
            self._await_prefetch()
            return {
                'dataloader_iter': self.dataloader_iter.state_dict(),
                'ready_batch': self.ready_batch
            }
        
        def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
            self.dataloader_iter = state_dict['dataloader_iter']
            self.ready_batch = state_dict['ready_batch']
        
        def _prefetch_next(self):
            def _task() -> None:
                local_rank = get_world_info().local_rank
                torch.cuda.set_device(local_rank)
                
                try:
                    batch = next(self.dataloader_iter)
                except StopIteration:
                    self.ready_batch = StopIteration
                    return None
                
                # 异步传输到GPU
                newstream = torch.cuda.Stream(local_rank)
                with torch.cuda.stream(newstream):
                    input_ids = batch["input_ids"].to("cuda", non_blocking=True)
                    labels = batch["labels"].to("cuda", non_blocking=True)
                    attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
                
                self.ready_batch = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask
                }
                return None
            
            self.thread = threading.Thread(target=_task)
            self.thread.start()
        
        def __next__(self):
            self._await_prefetch()
            if self.ready_batch is StopIteration:
                raise StopIteration
            
            batch = self.ready_batch
            self.ready_batch = None
            self._prefetch_next()
            return batch
        
        def _await_prefetch(self):
            if self.thread is not None and self.thread.is_alive():
                self.thread.join()
                self.thread = None
        
        def __del__(self):
            self._await_prefetch()


def get_sft_dataloader(
    tokenizer: PreTrainedTokenizer,
    world_size: int,
    rank: int,
    batch_size: int,
    data_config: DataConfig,
    train_on_inputs: bool = False,
) -> SFTDataLoader:
    """获取SFT数据加载器"""
    
    if data_config.fake:
        # 创建假的SFT数据集
        from datasets import Dataset
        fake_data = [
            {
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"},
                    {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
                ]
            }
        ] * 1000
        train_dataset = Dataset.from_list(fake_data)
    else:
        # 从Parquet文件加载SFT数据
        train_dataset = load_sft_datasets(
            data_config=data_config,
            split="train",
            tokenizer=tokenizer,
            rank=rank,
            world_size=world_size
        )
    
    # 创建SFT数据集
    dataset = SFTDataset(
        train_dataset,
        tokenizer=tokenizer,
        max_seq_length=data_config.seq_length,
        train_on_inputs=train_on_inputs
    )
    
    # 创建数据加载器
    mp_batch_dataloader = StatefulDataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=sft_collate_fn,
        num_workers=data_config.num_workers,
    )
    
    return SFTDataLoader(mp_batch_dataloader)


def load_sft_datasets(
    data_config: DataConfig,
    split: str,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    world_size: int,
) -> Union[ParquetSFTDataset, JsonlSFTDataset, CSVSFTDataset, HFSFTDataset]:
    """加载SFT数据集，根据文件类型自动选择合适的数据集类"""
    
    dataset_path = data_config.dataset_name_or_paths
    
    # 检查是否为Hugging Face Hub数据集
    if not os.path.exists(dataset_path) and "/" in dataset_path:
        # 可能是Hugging Face Hub数据集，格式为 "username/dataset_name"
        try:
            return HFSFTDataset(dataset_name=dataset_path, split=split)
        except Exception as e:
            get_logger().warning(f"Failed to load from Hugging Face Hub: {e}")
    
    # 本地文件处理
    from .data import _get_datafiles
    
    # 获取所有数据文件
    if "," in dataset_path:
        # 多个路径，取第一个
        dataset_path = dataset_path.split(",")[0]
    
    _ds_name, _, _ds_config = dataset_path.partition(":")
    data_files = _get_datafiles(_ds_name, _ds_config, split)
    
    if data_config.data_rank is not None and data_config.data_world_size is not None:
        data_files = data_files[rank::world_size]
    
    if not data_files:
        raise ValueError(f"No data files found for {dataset_path}")
    
    # 根据文件扩展名选择数据集类
    first_file = data_files[0]
    file_extension = os.path.splitext(first_file)[1].lower()
    
    if file_extension == '.parquet':
        return ParquetSFTDataset(files=data_files, tokenizer=tokenizer)
    elif file_extension == '.jsonl':
        return JsonlSFTDataset(files=data_files, tokenizer=tokenizer)
    elif file_extension == '.csv':
        return CSVSFTDataset(files=data_files, tokenizer=tokenizer)
    elif file_extension == '.json':
        # 单个JSON文件
        return JsonlSFTDataset(files=data_files, tokenizer=tokenizer)
    else:
        # 默认使用Parquet
        get_logger().warning(f"Unknown file extension {file_extension}, defaulting to ParquetSFTDataset")
        return ParquetSFTDataset(files=data_files, tokenizer=tokenizer)