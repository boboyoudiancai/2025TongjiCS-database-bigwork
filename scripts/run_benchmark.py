#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import argparse
import json
from datetime import datetime

# 设置数据目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Milvus连接参数
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# 测试参数
COLLECTION_NAME = "sift_benchmark"
DEFAULT_DIM = 128  # SIFT向量维度
TOP_K = 100  # 查询返回的最近邻数量

# 索引配置
INDEX_CONFIGS = {
    "FLAT": {
        "index_type": "FLAT",
        "metric_type": "L2",
        "params": {}
    },
    "IVF_FLAT": {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    },
    "IVF_SQ8": {
        "index_type": "IVF_SQ8",
        "metric_type": "L2",
        "params": {"nlist": 1024}
    },
    "HNSW": {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 16, "efConstruction": 500}
    }
}

# 搜索参数
SEARCH_PARAMS = {
    "FLAT": {"metric_type": "L2", "params": {}},
    "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 16}},
    "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 16}},
    "HNSW": {"metric_type": "L2", "params": {"ef": 64}}
}

def connect_to_milvus():
    """连接到Milvus服务器"""
    print(f"连接到Milvus服务器 {MILVUS_HOST}:{MILVUS_PORT}...")
    
    # 添加重试机制
    max_retries = 5
    for attempt in range(max_retries):
        try:
            connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT, timeout=30.0)
            
            # 验证连接是否成功建立
            if connections.has_connection("default"):
                print("连接成功")
                return True
            else:
                print(f"连接验证失败，尝试重新连接 ({attempt+1}/{max_retries})...")
                time.sleep(3)  # 等待3秒后重试
        except Exception as e:
            print(f"连接失败 ({attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print("等待5秒后重试...")
                time.sleep(5)
    
    print(f"经过{max_retries}次尝试后，无法连接到Milvus服务器")
    return False

def load_data():
    """加载SIFT数据集"""
    try:
        base_vectors = np.load(os.path.join(DATA_DIR, "sift_base.npy"))
        query_vectors = np.load(os.path.join(DATA_DIR, "sift_query.npy"))
        groundtruth = np.load(os.path.join(DATA_DIR, "sift_groundtruth.npy"))
        
        print(f"加载的基础向量: {base_vectors.shape}")
        print(f"加载的查询向量: {query_vectors.shape}")
        print(f"加载的真实结果: {groundtruth.shape}")
        
        return base_vectors, query_vectors, groundtruth
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None, None, None

def create_collection(dim=DEFAULT_DIM):
    """创建Milvus集合"""
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"已删除现有集合: {COLLECTION_NAME}")
    
    # 定义集合字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    
    # 创建集合
    schema = CollectionSchema(fields=fields, description="SIFT向量基准测试集合")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    print(f"已创建集合: {COLLECTION_NAME}")
    return collection

def insert_data(collection, base_vectors):
    """向集合中插入数据"""
    # 准备数据
    ids = np.arange(len(base_vectors))
    entities = [ids, base_vectors]
    
    # 分批插入数据
    batch_size = 50000
    num_batches = (len(base_vectors) + batch_size - 1) // batch_size
    
    print(f"开始插入数据，共{len(base_vectors)}条，分{num_batches}批处理...")
    
    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(base_vectors))
        
        batch_ids = ids[start:end]
        batch_vectors = base_vectors[start:end]
        
        collection.insert([batch_ids, batch_vectors])
    
    # 确保数据写入
    collection.flush()
    print(f"数据插入完成，共插入{collection.num_entities}条记录")

def build_index(collection, index_type):
    """创建索引"""
    if index_type not in INDEX_CONFIGS:
        print(f"未知的索引类型: {index_type}")
        return False
    
    config = INDEX_CONFIGS[index_type]
    print(f"正在创建索引: {index_type}, 参数: {config['params']}")
    
    start_time = time.time()
    collection.create_index(
        field_name="vector",
        index_params=config
    )
    build_time = time.time() - start_time
    
    print(f"索引创建完成，耗时: {build_time:.2f}秒")
    collection.load()
    return build_time

def evaluate_search(collection, query_vectors, groundtruth, index_type, runs=5):
    """评估搜索性能"""
    if index_type not in SEARCH_PARAMS:
        print(f"未知的索引类型: {index_type}")
        return None
    
    search_params = SEARCH_PARAMS[index_type]
    print(f"开始评估索引: {index_type}, 搜索参数: {search_params['params']}")
    
    # 执行搜索
    latencies = []
    recalls = []
    
    # 随机选择部分查询向量进行测试
    num_queries = min(100, len(query_vectors))
    query_indices = np.random.choice(len(query_vectors), num_queries, replace=False)
    test_queries = query_vectors[query_indices]
    test_ground_truth = groundtruth[query_indices]
    
    # 多次运行以获得稳定结果
    for run in range(runs):
        print(f"运行 {run+1}/{runs}")
        
        # 计算批量查询的延迟
        start_time = time.time()
        results = collection.search(
            data=test_queries, 
            anns_field="vector", 
            param=search_params,
            limit=TOP_K,
            output_fields=[]
        )
        end_time = time.time()
        
        # 计算平均延迟（毫秒）
        latency = (end_time - start_time) * 1000 / num_queries
        latencies.append(latency)
        
        # 计算召回率
        recall_sum = 0
        for i, result in enumerate(results):
            # 获取Milvus返回的ID
            milvus_ids = [hit.id for hit in result]
            # 获取真实的最近邻ID（取TOP_K个）
            true_ids = test_ground_truth[i][:TOP_K]
            # 计算交集大小
            intersection = len(set(milvus_ids) & set(true_ids))
            # 计算召回率
            recall = intersection / len(true_ids)
            recall_sum += recall
        
        # 计算平均召回率
        avg_recall = recall_sum / num_queries
        recalls.append(avg_recall)
    
    # 计算平均值和标准差
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    avg_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    
    print(f"平均延迟: {avg_latency:.2f}毫秒 (±{std_latency:.2f})")
    print(f"平均召回率: {avg_recall:.4f} (±{std_recall:.4f})")
    
    return {
        "index_type": index_type,
        "avg_latency": float(avg_latency),
        "std_latency": float(std_latency),
        "avg_recall": float(avg_recall),
        "std_recall": float(std_recall),
        "search_params": search_params["params"]
    }

def run_benchmark(index_types=None):
    """运行基准测试"""
    if index_types is None:
        index_types = list(INDEX_CONFIGS.keys())
    
    # 连接到Milvus
    if not connect_to_milvus():
        return
    
    # 加载数据
    base_vectors, query_vectors, groundtruth = load_data()
    if base_vectors is None:
        return
    
    # 记录结果
    results = []
    build_times = {}
    
    # 为每种索引类型运行测试
    for index_type in index_types:
        print(f"\n========== 测试索引: {index_type} ==========")
        
        try:
            # 为每种索引类型创建新的集合
            collection_name = f"{COLLECTION_NAME}_{index_type.lower()}"
            
            # 创建集合
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                print(f"已删除现有集合: {collection_name}")
                
            # 定义集合字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=base_vectors.shape[1])
            ]
            
            # 创建集合
            schema = CollectionSchema(fields=fields, description=f"SIFT向量基准测试集合-{index_type}")
            collection = Collection(name=collection_name, schema=schema)
            print(f"已创建集合: {collection_name}")
            
            # 插入数据
            insert_data(collection, base_vectors)
            
            # 构建索引
            build_time = build_index(collection, index_type)
            build_times[index_type] = build_time
            
            # 评估搜索性能
            result = evaluate_search(collection, query_vectors, groundtruth, index_type)
            if result:
                result["build_time"] = build_time
                results.append(result)
        except Exception as e:
            print(f"测试索引 {index_type} 时出错: {e}")
            print("跳过此索引，继续测试下一个...")
            continue
    
    # 保存结果
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.json")
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n基准测试结果已保存到: {result_file}")
        
        # 生成报告
        try:
            generate_report(results, timestamp)
        except Exception as e:
            print(f"生成报告时出错: {e}")
            print("跳过报告生成，但结果已保存为JSON")
    else:
        print("\n没有成功完成的测试结果")
    
    return results

def generate_report(results, timestamp):
    """生成测试报告"""
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 保存为CSV
    csv_file = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.csv")
    df.to_csv(csv_file, index=False)
    
    # 生成延迟和召回率图表
    plt.figure(figsize=(14, 10))
    
    # 延迟条形图
    plt.subplot(2, 1, 1)
    ax = sns.barplot(x="index_type", y="avg_latency", data=df)
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f} ms",
            ha='center'
        )
    plt.title("Average Query Latency by Index Type")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Index Type")
    
    # 召回率条形图
    plt.subplot(2, 1, 2)
    ax = sns.barplot(x="index_type", y="avg_recall", data=df)
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha='center'
        )
    plt.title("Average Recall Rate by Index Type")
    plt.ylabel("Recall Rate")
    plt.xlabel("Index Type")
    
    plt.tight_layout()
    plot_file = os.path.join(RESULTS_DIR, f"benchmark_plot_{timestamp}.png")
    plt.savefig(plot_file)
    
    # 生成索引构建时间图表
    plt.figure(figsize=(10, 6))
    build_times = {result["index_type"]: result["build_time"] for result in results}
    index_types = list(build_times.keys())
    times = list(build_times.values())
    
    ax = sns.barplot(x=index_types, y=times)
    for i, bar in enumerate(ax.patches):
        ax.text(
            bar.get_x() + bar.get_width()/2.,
            bar.get_height() + 0.3,
            f"{bar.get_height():.1f} s",
            ha='center'
        )
    
    plt.title("Index Build Time by Index Type")
    plt.ylabel("Time (s)")
    plt.xlabel("Index Type")
    plt.tight_layout()
    
    build_time_plot = os.path.join(RESULTS_DIR, f"build_time_plot_{timestamp}.png")
    plt.savefig(build_time_plot)
    
    print(f"Generated reports:\n1. CSV: {csv_file}\n2. Performance chart: {plot_file}\n3. Build time chart: {build_time_plot}")

def main():
    parser = argparse.ArgumentParser(description="Milvus性能基准测试工具")
    parser.add_argument('--indices', type=str, nargs='+', 
                        choices=list(INDEX_CONFIGS.keys()),
                        help='要测试的索引类型，如不指定则测试所有索引')
    
    args = parser.parse_args()
    run_benchmark(args.indices)

if __name__ == "__main__":
    main()