#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import requests
import numpy as np
from tqdm import tqdm
import gzip
import shutil
import time
import sys
import random
from urllib.parse import urlparse

# 创建数据目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 最大重试次数
MAX_RETRIES = 3

def download_file(url, filename, retries=0):
    """下载文件并显示进度条"""
    try:
        print(f"正在从 {url} 下载文件...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        
        # 检查响应状态
        if response.status_code != 200:
            print(f"下载失败，状态码: {response.status_code}")
            if retries < MAX_RETRIES:
                print(f"尝试重新下载 (尝试 {retries+1}/{MAX_RETRIES})...")
                time.sleep(2)  # 等待2秒后重试
                return download_file(url, filename, retries+1)
            else:
                print("达到最大重试次数，下载失败")
                return None
        
        # 检查内容类型，如果是HTML可能是错误页面
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type or 'text/plain' in content_type:
            # 检查前100个字节看是否为HTML
            first_bytes = next(response.iter_content(100))
            if b'<!' in first_bytes or b'<html' in first_bytes:
                print(f"警告: 下载的内容可能是HTML页面而不是数据文件: {content_type}")
                if retries < MAX_RETRIES:
                    print(f"尝试重新下载 (尝试 {retries+1}/{MAX_RETRIES})...")
                    time.sleep(2)  # 等待2秒后重试
                    return download_file(url, filename, retries+1)
                else:
                    print("达到最大重试次数，下载失败")
                    return None
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            desc=f"下载 {filename}"
        )
        
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        
        if total_size != 0 and progress_bar.n != total_size:
            print("下载可能不完整")
            if retries < MAX_RETRIES:
                print(f"尝试重新下载 (尝试 {retries+1}/{MAX_RETRIES})...")
                time.sleep(2)  # 等待2秒后重试
                return download_file(url, filename, retries+1)
        
        return file_path
    except Exception as e:
        print(f"下载过程中出现错误: {e}")
        if retries < MAX_RETRIES:
            print(f"尝试重新下载 (尝试 {retries+1}/{MAX_RETRIES})...")
            time.sleep(2)  # 等待2秒后重试
            return download_file(url, filename, retries+1)
        else:
            print("达到最大重试次数，下载失败")
            return None

def extract_gzip(gzip_path, extract_path):
    """解压gzip文件"""
    try:
        print(f"正在解压 {gzip_path}...")
        with gzip.open(gzip_path, 'rb') as f_in:
            with open(extract_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"已解压: {extract_path}")
        return True
    except gzip.BadGzipFile:
        print(f"错误: {gzip_path} 不是有效的gzip文件")
        return False
    except Exception as e:
        print(f"解压过程中出现错误: {e}")
        return False

def download_sift():
    """下载并处理SIFT数据集"""
    # 尝试多个数据源
    data_sources = [
        {
            "base_url": "https://ann-benchmarks.com/sift",
            "files": {
                "sift-128-euclidean.hdf5": "SIFT欧几里得数据集"
            }
        },
        {
            "base_url": "http://corpus-texmex.irisa.fr/sift",
            "files": {
                "sift_base.fvecs.gz": "基础向量集",
                "sift_query.fvecs.gz": "查询向量集",
                "sift_groundtruth.ivecs.gz": "查询的真实结果"
            }
        }
    ]
    
    # 随机打乱数据源顺序
    random.shuffle(data_sources)
    
    # 尝试每个数据源
    for source in data_sources:
        base_url = source["base_url"]
        files = source["files"]
        hostname = urlparse(base_url).netloc
        
        print(f"\n尝试从 {hostname} 下载数据集...")
        download_success = True
        downloaded_files = []
        
        for filename, desc in files.items():
            print(f"开始下载{desc}...")
            file_url = f"{base_url}/{filename}"
            
            # 下载压缩文件
            gz_path = download_file(file_url, filename)
            if not gz_path:
                download_success = False
                break
            
            downloaded_files.append(gz_path)
            
            # 如果是.gz文件，尝试解压
            if filename.endswith('.gz'):
                extract_path = os.path.join(DATA_DIR, filename[:-3])  # 去掉.gz后缀
                if not extract_gzip(gz_path, extract_path):
                    download_success = False
                    break
                
                # 删除压缩文件
                os.remove(gz_path)
            
        if download_success:
            print(f"从 {hostname} 下载数据集成功")
            # 如果下载了HDF5格式，需要转换
            if any(f.endswith('.hdf5') for f in files.keys()):
                print("检测到HDF5格式数据，正在转换...")
                return convert_hdf5_to_numpy()
            return True
        else:
            print(f"从 {hostname} 下载数据集失败，尝试下一个源")
            # 清理已下载的文件
            for file_path in downloaded_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    print("所有下载源均失败")
    return False

def convert_hdf5_to_numpy():
    """将HDF5格式的数据集转换为numpy格式"""
    try:
        import h5py
        hdf5_path = os.path.join(DATA_DIR, "sift-128-euclidean.hdf5")
        
        if not os.path.exists(hdf5_path):
            print(f"错误: HDF5文件 {hdf5_path} 不存在")
            return False
        
        print("正在读取HDF5文件...")
        with h5py.File(hdf5_path, 'r') as f:
            # 读取训练和测试数据
            print("数据集包含以下键:")
            for key in f.keys():
                print(f"  - {key}")
            
            # 读取向量数据
            print("处理基础向量集...")
            base_vectors = f['train'][:]
            print(f"基础向量集维度: {base_vectors.shape}")
            
            print("处理查询向量集...")
            query_vectors = f['test'][:]
            print(f"查询向量集维度: {query_vectors.shape}")
            
            # 读取距离信息作为真实结果
            print("处理真实结果集...")
            if 'distances' in f:
                distances = f['distances'][:]
                neighbors = f['neighbors'][:]
                print(f"真实结果维度: {neighbors.shape}")
            else:
                # 如果没有提供真实结果，计算欧几里得距离
                print("数据集中没有真实结果，使用暴力计算（取前100个结果）...")
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean').fit(base_vectors)
                distances, neighbors = nbrs.kneighbors(query_vectors)
        
        # 保存为numpy格式
        print("保存为numpy格式...")
        np.save(os.path.join(DATA_DIR, "sift_base.npy"), base_vectors)
        np.save(os.path.join(DATA_DIR, "sift_query.npy"), query_vectors)
        np.save(os.path.join(DATA_DIR, "sift_groundtruth.npy"), neighbors)
        
        print("HDF5转换完成")
        return True
    except ImportError:
        print("错误: 请安装h5py库以支持HDF5格式: pip install h5py")
        return False
    except Exception as e:
        print(f"转换过程中发生错误: {e}")
        return False

def load_fvecs(filename):
    """加载fvecs格式文件"""
    try:
        fv = np.fromfile(filename, dtype=np.float32)
        if fv.size == 0:
            raise ValueError(f"文件 {filename} 是空的")
            
        dim = fv.view(np.int32)[0]
        if dim <= 0 or dim > 10000:  # 检查维度是否合理
            raise ValueError(f"读取到不合理的维度值: {dim}")
            
        return fv.reshape(-1, dim + 1)[:, 1:].copy()
    except Exception as e:
        print(f"加载fvecs文件时出错: {e}")
        return None

def load_ivecs(filename):
    """加载ivecs格式文件"""
    try:
        iv = np.fromfile(filename, dtype=np.int32)
        if iv.size == 0:
            raise ValueError(f"文件 {filename} 是空的")
            
        dim = iv[0]
        if dim <= 0 or dim > 10000:  # 检查维度是否合理
            raise ValueError(f"读取到不合理的维度值: {dim}")
            
        return iv.reshape(-1, dim + 1)[:, 1:].copy()
    except Exception as e:
        print(f"加载ivecs文件时出错: {e}")
        return None

def process_data():
    """处理下载的数据集为numpy格式"""
    # 检查是否已经存在处理好的numpy文件
    np_files = [
        os.path.join(DATA_DIR, "sift_base.npy"),
        os.path.join(DATA_DIR, "sift_query.npy"),
        os.path.join(DATA_DIR, "sift_groundtruth.npy")
    ]
    
    # 如果numpy文件已存在，直接跳过处理
    if all(os.path.exists(f) for f in np_files):
        print("检测到处理好的numpy文件已存在，跳过处理步骤")
        return True
    
    # 检查是否存在fvecs/ivecs文件
    base_file = os.path.join(DATA_DIR, "sift_base.fvecs")
    query_file = os.path.join(DATA_DIR, "sift_query.fvecs")
    groundtruth_file = os.path.join(DATA_DIR, "sift_groundtruth.ivecs")
    
    if not all(os.path.exists(f) for f in [base_file, query_file, groundtruth_file]):
        print("缺少必要的原始数据文件，无法进行处理")
        return False
    
    print("处理基础向量集...")
    base_vectors = load_fvecs(base_file)
    if base_vectors is None:
        return False
    print(f"基础向量集维度: {base_vectors.shape}")
    
    print("处理查询向量集...")
    query_vectors = load_fvecs(query_file)
    if query_vectors is None:
        return False
    print(f"查询向量集维度: {query_vectors.shape}")
    
    print("处理真实结果集...")
    groundtruth = load_ivecs(groundtruth_file)
    if groundtruth is None:
        return False
    print(f"真实结果集维度: {groundtruth.shape}")
    
    # 保存为numpy格式
    np.save(os.path.join(DATA_DIR, "sift_base.npy"), base_vectors)
    np.save(os.path.join(DATA_DIR, "sift_query.npy"), query_vectors)
    np.save(os.path.join(DATA_DIR, "sift_groundtruth.npy"), groundtruth)
    
    print("数据已处理并保存为numpy格式")
    return True

def download_sample_dataset():
    """生成一个小型示例数据集，用于测试"""
    print("正在生成随机示例数据集...")
    
    # 生成随机向量
    np.random.seed(42)
    dim = 128  # SIFT向量维度
    
    # 生成10000个基础向量和100个查询向量
    print("生成基础向量集...")
    base_vectors = np.random.rand(10000, dim).astype(np.float32)
    
    print("生成查询向量集...")
    query_vectors = np.random.rand(100, dim).astype(np.float32)
    
    # 计算近邻作为真实结果
    print("计算真实结果集...")
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=100, algorithm='brute', metric='euclidean').fit(base_vectors)
    distances, neighbors = nbrs.kneighbors(query_vectors)
    
    # 保存为numpy格式
    print("保存为numpy格式...")
    np.save(os.path.join(DATA_DIR, "sift_base.npy"), base_vectors)
    np.save(os.path.join(DATA_DIR, "sift_query.npy"), query_vectors)
    np.save(os.path.join(DATA_DIR, "sift_groundtruth.npy"), neighbors)
    
    print("示例数据集已生成")
    return True

if __name__ == "__main__":
    print("开始下载SIFT数据集...")
    
    if download_sift():
        print("下载成功")
        print("开始处理数据...")
        if not process_data():
            print("处理数据失败，尝试生成示例数据集...")
            if download_sample_dataset():
                print("已生成示例数据集")
            else:
                print("无法生成示例数据集")
                sys.exit(1)
    else:
        print("下载数据集失败，尝试生成示例数据集...")
        if download_sample_dataset():
            print("已生成示例数据集")
        else:
            print("无法生成示例数据集")
            sys.exit(1)
    
    print("数据集准备完成！") 