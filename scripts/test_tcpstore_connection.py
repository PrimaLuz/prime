#!/usr/bin/env python3
"""
测试TCPStore连接的脚本
用于诊断多节点训练中的连接问题
"""

import os
import sys
import time
import torch.distributed as dist
from datetime import timedelta

def test_tcpstore_connection():
    """测试TCPStore连接"""
    
    # 获取环境变量
    global_rank = int(os.environ.get("GLOBAL_RANK", 0))
    global_world_size = int(os.environ.get("GLOBAL_WORLD_SIZE", 1))
    global_addr = os.environ.get("GLOBAL_ADDR", "localhost")
    global_port = int(os.environ.get("GLOBAL_PORT", 26969))
    rank = int(os.environ.get("RANK", 0))
    
    print(f"Testing TCPStore connection:")
    print(f"  Global Rank: {global_rank}")
    print(f"  Global World Size: {global_world_size}")
    print(f"  Global Addr: {global_addr}")
    print(f"  Global Port: {global_port}")
    print(f"  Local Rank: {rank}")
    
    is_leader = global_rank == 0
    port = global_port + rank
    
    print(f"  Is Leader: {is_leader}")
    print(f"  Connecting to port: {port}")
    
    try:
        # 创建TCPStore
        print(f"Creating TCPStore...")
        store = dist.TCPStore(
            host_name=global_addr,
            port=port,
            timeout=timedelta(seconds=30),
            is_master=is_leader,
        )
        print(f"TCPStore created successfully!")
        
        if is_leader:
            print("Setting test value...")
            store.set("test_key", "test_value")
            print("Test value set successfully!")
            print("TCPStore master is ready, waiting for connections...")
            # 主节点等待一段时间让从节点连接
            time.sleep(50)
        else:
            print("Waiting for test value...")
            start_time = time.time()
            while True:
                try:
                    value = store.get("test_key").decode("utf-8")
                    print(f"Received test value: {value}")
                    break
                except dist.DistStoreError as e:
                    elapsed = time.time() - start_time
                    if elapsed > 30:
                        print(f"Timeout after {elapsed:.2f}s")
                        print("This might be because the master node hasn't started yet.")
                        print("Please ensure the master node is running the test first.")
                        return False
                    print(f"Waiting... ({elapsed:.2f}s)")
                    time.sleep(0.1)
        
        print("TCPStore test completed successfully!")
        return True
        
    except Exception as e:
        print(f"TCPStore test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_tcpstore_connection()
    sys.exit(0 if success else 1)
