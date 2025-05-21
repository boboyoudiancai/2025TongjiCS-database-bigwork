#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import subprocess
import sys
import time
import logging
from scripts.start_milvus import start_milvus, stop_milvus, check_docker
from scripts.logger import setup_logger
from scripts.check_environment import check_system_resources, check_python_version, check_dependencies

# 设置日志记录器
logger = setup_logger('milvus_benchmark')

def run_command(command, description=None):
    """运行命令并显示状态"""
    if description:
        logger.info(f"{description}")
    
    try:
        process = subprocess.run(command, shell=True, check=True, text=True)
        if process.returncode == 0:
            return True
        else:
            logger.error(f"命令执行失败，返回代码: {process.returncode}")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行错误: {e}")
        return False
    except Exception as e:
        logger.error(f"发生异常: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Milvus向量数据库性能评估实验",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--skip-download', action='store_true', help='跳过数据下载步骤')
    parser.add_argument('--skip-milvus', action='store_true', help='跳过Milvus启动步骤')
    parser.add_argument('--indices', type=str, nargs='+', 
                        choices=['FLAT', 'IVF_FLAT', 'IVF_SQ8', 'HNSW'],
                        help='要测试的索引类型，如不指定则测试所有索引')
    parser.add_argument('--fast-test', action='store_true', help='仅测试FLAT索引，用于快速验证')
    parser.add_argument('--check-env', action='store_true', help='仅检查环境，不运行测试')
    parser.add_argument('--analyze-only', action='store_true', help='仅分析已有结果，不运行测试')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 如果仅检查环境
    if args.check_env:
        logger.info("执行环境检查...")
        cmd = f"python scripts/check_environment.py --output-dir {results_dir}"
        if args.verbose:
            cmd += " --verbose"
        return run_command(cmd, "检查环境")
    
    # 如果仅分析已有结果
    if args.analyze_only:
        logger.info("分析已有测试结果...")
        cmd = f"python scripts/explain_results.py --results-dir {results_dir} --output-dir {results_dir}"
        if args.verbose:
            cmd += " --verbose"
        return run_command(cmd, "分析结果")
    
    # 检查项目依赖
    logger.info("检查项目依赖...")
    if not run_command("pip install -r requirements.txt", "安装项目依赖"):
        return
    
    # 简单环境检查
    logger.info("执行基本环境检查...")
    python_ok, _ = check_python_version()
    if not python_ok:
        logger.warning("Python版本较低，可能影响性能")
    
    deps_ok, deps_details = check_dependencies()
    if not deps_ok:
        logger.warning(f"缺少依赖项: {', '.join(deps_details['missing_packages'])}")
        if not run_command("pip install -r requirements.txt", "尝试安装缺失的依赖"):
            return
    
    # 检查系统资源
    resources_ok, resources = check_system_resources()
    if not resources_ok:
        logger.warning("系统资源不满足最低要求，可能影响性能")
        logger.info(f"可用内存: {resources['memory']['available_gb']} GB (建议至少4GB)")
        logger.info(f"可用磁盘空间: {resources['disk']['free_gb']} GB (建议至少10GB)")
    
    # 检查Docker可用性
    if not args.skip_milvus and not check_docker():
        logger.error("Docker不可用，无法启动Milvus服务")
        return
    
    # 步骤1：下载数据集
    if not args.skip_download:
        if not run_command("python scripts/download_data.py", "下载SIFT数据集"):
            return
    else:
        logger.info("跳过数据下载步骤")
    
    # 步骤2：启动Milvus服务
    if not args.skip_milvus:
        logger.info("启动Milvus服务")
        if not start_milvus():
            logger.error("Milvus服务启动失败")
            return
    else:
        logger.info("跳过Milvus启动步骤")
    
    # 步骤3：运行性能基准测试
    benchmark_cmd = "python scripts/run_benchmark.py"
    
    # 如果指定了快速测试，只测试FLAT索引
    if args.fast_test:
        benchmark_cmd += " --indices FLAT"
    # 否则使用命令行参数指定的索引
    elif args.indices:
        indices_str = " ".join(args.indices)
        benchmark_cmd += f" --indices {indices_str}"
    
    # 添加详细日志选项
    if args.verbose:
        benchmark_cmd += " --verbose"
    
    try:
        if not run_command(benchmark_cmd, "运行性能基准测试"):
            logger.error("性能基准测试失败")
    finally:
        # 停止Milvus服务
        if not args.skip_milvus:
            logger.info("停止Milvus服务")
            stop_milvus()
    
    # 步骤4：分析结果
    logger.info("分析测试结果...")
    analyze_cmd = f"python scripts/explain_results.py --results-dir {results_dir} --output-dir {results_dir}"
    if args.verbose:
        analyze_cmd += " --verbose"
    run_command(analyze_cmd, "分析结果")
    
    logger.info("实验完成！")
    logger.info(f"结果保存在 {results_dir} 目录中")

if __name__ == "__main__":
    main() 