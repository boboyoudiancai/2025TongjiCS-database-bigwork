#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import platform
import subprocess
import psutil
import importlib.util
import socket
import time
import json
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import setup_logger

# 设置日志
logger = setup_logger('env_checker')

def check_python_version():
    """检查Python版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    logger.info(f"Python版本: {version_str}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.warning("警告: 推荐使用Python 3.8或更高版本")
        return False, version_str
    return True, version_str

def check_dependencies():
    """检查依赖项是否已安装"""
    required_packages = [
        "pymilvus",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "requests",
        "docker",
        "psutil"
    ]
    
    missing_packages = []
    installed_packages = {}
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            logger.warning(f"依赖项 {package} 未安装")
            missing_packages.append(package)
        else:
            try:
                # 尝试获取版本号
                module = importlib.import_module(package)
                version = getattr(module, "__version__", "未知")
                installed_packages[package] = version
                logger.info(f"依赖项 {package} 已安装，版本: {version}")
            except Exception as e:
                logger.warning(f"无法获取 {package} 的版本信息: {e}")
                installed_packages[package] = "已安装，版本未知"
    
    return len(missing_packages) == 0, {
        "missing_packages": missing_packages,
        "installed_packages": installed_packages
    }

def check_docker():
    """检查Docker版本和状态"""
    docker_info = {
        "installed": False,
        "version": "未知",
        "api_version": "未知",
        "running": False
    }
    
    # 检查docker命令是否可用
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            docker_info["installed"] = True
            version_output = result.stdout.strip()
            logger.info(f"Docker版本信息: {version_output}")
            
            # 提取版本号
            import re
            match = re.search(r"Docker version ([0-9.]+)", version_output)
            if match:
                docker_info["version"] = match.group(1)
        else:
            logger.warning("Docker命令不可用")
            return False, docker_info
    except Exception as e:
        logger.error(f"检查Docker版本时出错: {e}")
        return False, docker_info
    
    # 检查Docker是否正在运行
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True)
        if result.returncode == 0:
            docker_info["running"] = True
            logger.info("Docker服务正在运行")
            
            # 尝试提取API版本
            api_match = re.search(r"API version:\s+([0-9.]+)", result.stdout)
            if api_match:
                docker_info["api_version"] = api_match.group(1)
        else:
            logger.warning("Docker服务未运行")
            return False, docker_info
    except Exception as e:
        logger.error(f"检查Docker状态时出错: {e}")
        return False, docker_info
    
    # 检查docker-compose命令
    try:
        # 先尝试新版命令格式
        result = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
        if result.returncode == 0:
            docker_info["compose_version"] = "新版格式可用"
            logger.info("Docker Compose (新版格式) 可用")
        else:
            # 尝试旧版命令格式
            result = subprocess.run(["docker-compose", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                docker_info["compose_version"] = "旧版格式可用"
                logger.info("Docker Compose (旧版格式) 可用")
            else:
                docker_info["compose_version"] = "不可用"
                logger.warning("Docker Compose 不可用")
    except Exception as e:
        docker_info["compose_version"] = "检查失败"
        logger.error(f"检查Docker Compose时出错: {e}")
    
    return docker_info["installed"] and docker_info["running"], docker_info

def check_system_resources():
    """检查系统资源"""
    resources = {
        "cpu": {
            "count": psutil.cpu_count(logical=True),
            "usage": psutil.cpu_percent(interval=1)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "percent_used": psutil.virtual_memory().percent
        },
        "disk": {
            "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2),
            "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
            "percent_used": psutil.disk_usage('/').percent
        }
    }
    
    # 记录资源信息
    logger.info(f"CPU核心数: {resources['cpu']['count']}")
    logger.info(f"CPU使用率: {resources['cpu']['usage']}%")
    logger.info(f"总内存: {resources['memory']['total_gb']} GB")
    logger.info(f"可用内存: {resources['memory']['available_gb']} GB")
    logger.info(f"内存使用率: {resources['memory']['percent_used']}%")
    logger.info(f"磁盘总空间: {resources['disk']['total_gb']} GB")
    logger.info(f"磁盘可用空间: {resources['disk']['free_gb']} GB")
    logger.info(f"磁盘使用率: {resources['disk']['percent_used']}%")
    
    # 检查资源是否满足要求
    meets_requirements = True
    
    # 检查CPU
    if resources["cpu"]["count"] < 2:
        logger.warning("警告: CPU核心数少于2，可能影响性能")
        meets_requirements = False
    
    # 检查内存
    if resources["memory"]["available_gb"] < 4:
        logger.warning("警告: 可用内存不足4GB，可能影响性能")
        meets_requirements = False
    
    # 检查磁盘
    if resources["disk"]["free_gb"] < 10:
        logger.warning("警告: 可用磁盘空间不足10GB，可能影响数据存储")
        meets_requirements = False
    
    return meets_requirements, resources

def check_milvus_connection(host="localhost", port="19530"):
    """检查Milvus连接"""
    connection_info = {
        "host": host,
        "port": port,
        "connected": False,
        "version": "未知"
    }
    
    # 首先检查端口是否开放
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, int(port)))
        sock.close()
        
        if result != 0:
            logger.warning(f"Milvus端口 {port} 未开放")
            return False, connection_info
    except Exception as e:
        logger.error(f"检查Milvus端口时出错: {e}")
        return False, connection_info
    
    # 尝试导入pymilvus
    try:
        from pymilvus import connections
        
        # 尝试连接
        connections.connect("default", host=host, port=port, timeout=5.0)
        connection_info["connected"] = True
        logger.info("Milvus连接成功")
        
        # 尝试获取版本信息
        try:
            from pymilvus import utility
            version_info = utility.get_server_version()
            connection_info["version"] = version_info
            logger.info(f"Milvus版本: {version_info}")
        except Exception as e:
            logger.warning(f"无法获取Milvus版本信息: {e}")
        
        # 断开连接
        connections.disconnect("default")
        
        return True, connection_info
    except ImportError:
        logger.error("未安装pymilvus库，无法连接Milvus")
        return False, connection_info
    except Exception as e:
        logger.error(f"连接Milvus时出错: {e}")
        return False, connection_info

def check_data_directory():
    """检查数据目录"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    data_info = {
        "exists": os.path.exists(data_dir),
        "files": [],
        "total_size_mb": 0
    }
    
    if data_info["exists"]:
        logger.info(f"数据目录存在: {data_dir}")
        
        # 列出数据文件
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                data_info["files"].append({
                    "name": filename,
                    "size_mb": round(size_mb, 2)
                })
                data_info["total_size_mb"] += size_mb
        
        data_info["total_size_mb"] = round(data_info["total_size_mb"], 2)
        
        if data_info["files"]:
            logger.info(f"数据目录中有 {len(data_info['files'])} 个文件，总大小: {data_info['total_size_mb']} MB")
            for file_info in data_info["files"]:
                logger.debug(f"文件: {file_info['name']}, 大小: {file_info['size_mb']} MB")
        else:
            logger.warning("数据目录为空")
    else:
        logger.warning(f"数据目录不存在: {data_dir}")
    
    return data_info["exists"] and len(data_info["files"]) > 0, data_info

def generate_report(results, output_dir=None):
    """生成环境检查报告"""
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system": {
            "platform": platform.platform(),
            "python": results["python"]["version"],
            "hostname": platform.node()
        },
        "checks": {
            "python_version": {
                "status": "通过" if results["python"]["success"] else "警告",
                "details": results["python"]["version"]
            },
            "dependencies": {
                "status": "通过" if results["dependencies"]["success"] else "警告",
                "missing": results["dependencies"]["details"]["missing_packages"],
                "installed": results["dependencies"]["details"]["installed_packages"]
            },
            "docker": {
                "status": "通过" if results["docker"]["success"] else "失败",
                "details": results["docker"]["details"]
            },
            "system_resources": {
                "status": "通过" if results["resources"]["success"] else "警告",
                "details": results["resources"]["details"]
            },
            "milvus_connection": {
                "status": "通过" if results["milvus"]["success"] else "失败",
                "details": results["milvus"]["details"]
            },
            "data_directory": {
                "status": "通过" if results["data"]["success"] else "警告",
                "details": results["data"]["details"]
            }
        },
        "overall": "通过" if all([
            results["python"]["success"],
            results["dependencies"]["success"],
            results["docker"]["success"],
            results["resources"]["success"],
            results["milvus"]["success"],
            results["data"]["success"]
        ]) else "失败"
    }
    
    # 打印报告摘要
    print("\n=== 环境检查报告 ===")
    print(f"检查时间: {report['timestamp']}")
    print(f"系统信息: {report['system']['platform']}")
    print(f"Python版本: {report['system']['python']}")
    print("\n检查结果:")
    
    for check_name, check_result in report["checks"].items():
        status_symbol = "✓" if check_result["status"] == "通过" else "✗" if check_result["status"] == "失败" else "⚠"
        print(f"{status_symbol} {check_name}: {check_result['status']}")
    
    print(f"\n总体结果: {report['overall']}")
    
    # 如果指定了输出目录，保存报告
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_file = os.path.join(output_dir, f"env_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"环境检查报告已保存到: {report_file}")
    
    return report

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="检查Milvus基准测试环境")
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录路径 (用于保存环境检查报告)')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细日志')
    parser.add_argument('--milvus-host', type=str, default='localhost',
                        help='Milvus服务器主机名')
    parser.add_argument('--milvus-port', type=str, default='19530',
                        help='Milvus服务器端口')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("开始环境检查")
    
    # 解析输出目录路径
    output_dir = args.output_dir
    if output_dir and not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, output_dir))
    
    # 执行各项检查
    results = {
        "python": {},
        "dependencies": {},
        "docker": {},
        "resources": {},
        "milvus": {},
        "data": {}
    }
    
    # 检查Python版本
    results["python"]["success"], results["python"]["version"] = check_python_version()
    
    # 检查依赖项
    results["dependencies"]["success"], results["dependencies"]["details"] = check_dependencies()
    
    # 检查Docker
    results["docker"]["success"], results["docker"]["details"] = check_docker()
    
    # 检查系统资源
    results["resources"]["success"], results["resources"]["details"] = check_system_resources()
    
    # 检查Milvus连接
    results["milvus"]["success"], results["milvus"]["details"] = check_milvus_connection(
        host=args.milvus_host, 
        port=args.milvus_port
    )
    
    # 检查数据目录
    results["data"]["success"], results["data"]["details"] = check_data_directory()
    
    # 生成报告
    report = generate_report(results, output_dir)
    
    # 返回状态码
    return 0 if report["overall"] == "通过" else 1

if __name__ == "__main__":
    import argparse
    sys.exit(main()) 