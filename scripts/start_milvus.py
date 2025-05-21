#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import docker
import subprocess
from dotenv import load_dotenv
import requests
import re

# 加载环境变量
load_dotenv()

# Milvus配置
MILVUS_VERSION = os.getenv("MILVUS_VERSION", "2.3.0")
DOCKER_COMPOSE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "docker-compose.yml")

def check_docker():
    """检查Docker是否安装并运行"""
    try:
        client = docker.from_env()
        version = client.version()["Version"]
        print(f"Docker版本: {version}")
        return True
    except Exception as e:
        print(f"Docker检查失败: {e}")
        return False

def download_docker_compose():
    """下载Milvus的docker-compose配置文件"""
    compose_dir = os.path.dirname(DOCKER_COMPOSE_FILE)
    os.makedirs(compose_dir, exist_ok=True)
    
    # 下载docker-compose.yml
    url = f"https://github.com/milvus-io/milvus/releases/download/v{MILVUS_VERSION}/milvus-standalone-docker-compose.yml"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            compose_content = response.text
            
            # 直接使用原始内容，不替换镜像源
            with open(DOCKER_COMPOSE_FILE, 'w') as f:
                f.write(compose_content)
            print(f"Docker Compose配置文件已下载到: {DOCKER_COMPOSE_FILE}")
            return True
        else:
            print(f"下载失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"下载Docker Compose配置失败: {e}")
        return False

def create_simple_compose_file():
    """创建一个简单的Milvus docker-compose配置文件"""
    compose_dir = os.path.dirname(DOCKER_COMPOSE_FILE)
    os.makedirs(compose_dir, exist_ok=True)
    
    compose_content = """version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.3.0
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus
"""
    
    with open(DOCKER_COMPOSE_FILE, 'w') as f:
        f.write(compose_content)
    print(f"已创建简化版Docker Compose配置文件: {DOCKER_COMPOSE_FILE}")
    return True

def start_milvus():
    """启动Milvus服务"""
    if not os.path.exists(DOCKER_COMPOSE_FILE):
        print("Docker Compose配置文件不存在，正在下载...")
        if not download_docker_compose():
            print("无法下载Docker Compose配置，尝试创建简化版配置...")
            if not create_simple_compose_file():
                print("无法创建Docker Compose配置文件")
                return False
    else:
        # 如果文件已存在，检查是否使用了国内镜像源，需要替换回官方镜像
        with open(DOCKER_COMPOSE_FILE, 'r') as f:
            compose_content = f.read()
        
        if 'registry.cn-hangzhou.aliyuncs.com' in compose_content:
            print("检测到Docker Compose配置使用国内镜像源，替换为官方镜像源...")
            if not create_simple_compose_file():
                print("无法更新Docker Compose配置文件")
                return False
    
    try:
        print("正在启动Milvus服务...")
        subprocess.run(
            f"docker compose -f {DOCKER_COMPOSE_FILE} up -d",
            shell=True,
            check=True
        )
        
        # 等待Milvus启动
        print("等待Milvus服务启动...")
        retries = 0
        max_retries = 30
        while retries < max_retries:
            try:
                response = requests.get("http://localhost:9091/api/v1/health")
                if response.status_code == 200:
                    print("Milvus服务已成功启动！")
                    # 等待额外的5秒，确保服务完全可用
                    time.sleep(10)
                    print("额外等待完成，服务应该已完全就绪")
                    return True
            except:
                pass
            
            retries += 1
            time.sleep(2)
            print(f"正在等待Milvus启动... ({retries}/{max_retries})")
        
        print("Milvus服务似乎未能正常启动，请检查Docker日志")
        return False
    
    except Exception as e:
        print(f"启动Milvus服务失败: {e}")
        return False

def stop_milvus():
    """停止Milvus服务"""
    try:
        print("正在停止Milvus服务...")
        subprocess.run(
            f"docker compose -f {DOCKER_COMPOSE_FILE} down",
            shell=True,
            check=True
        )
        print("Milvus服务已停止")
        return True
    except Exception as e:
        print(f"停止Milvus服务失败: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Milvus服务管理工具")
    parser.add_argument('action', choices=['start', 'stop', 'restart'], help='执行的操作: start, stop, 或 restart')
    
    args = parser.parse_args()
    
    if not check_docker():
        print("请确保Docker已安装并正在运行")
        exit(1)
    
    if args.action == 'start':
        start_milvus()
    elif args.action == 'stop':
        stop_milvus()
    elif args.action == 'restart':
        stop_milvus()
        start_milvus()