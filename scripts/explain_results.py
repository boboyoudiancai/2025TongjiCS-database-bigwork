#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import sys
import numpy as np
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import setup_logger

# 设置日志
logger = setup_logger('results_explainer')

def load_results(results_dir):
    """加载结果文件"""
    results = []
    logger.info(f"从目录加载结果: {results_dir}")
    
    if not os.path.exists(results_dir):
        logger.error(f"结果目录不存在: {results_dir}")
        return results
        
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(results_dir, filename)
            logger.debug(f"处理文件: {file_path}")
            
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    results.append(data)
                    logger.debug(f"成功加载文件: {filename}")
                except json.JSONDecodeError as e:
                    logger.error(f"无法解析文件 {filename}: {e}")
    
    logger.info(f"共加载了 {len(results)} 个结果文件")
    return results

def explain_results(results, output_dir=None):
    """解释结果"""
    if not results:
        logger.warning("没有结果可供分析")
        return
        
    # 提取关键指标
    metrics = []
    for result in results:
        for index_name, data in result.items():
            if isinstance(data, dict) and 'metrics' in data:
                metrics.append({
                    'index_type': index_name,
                    'build_time': data.get('build_time', 0),
                    'avg_latency': data['metrics'].get('avg_latency', 0),
                    'avg_recall': data['metrics'].get('avg_recall', 0),
                    'qps': data['metrics'].get('qps', 0),
                    'index_size': data.get('index_size', 0)
                })
    
    if not metrics:
        logger.warning("没有找到有效的指标数据")
        return
        
    # 转换为DataFrame
    df = pd.DataFrame(metrics)
    logger.info(f"分析了 {len(df)} 种索引类型的性能")
    
    # 排序
    df_latency = df.sort_values('avg_latency')
    df_recall = df.sort_values('avg_recall', ascending=False)
    df_qps = df.sort_values('qps', ascending=False)
    
    # 打印结果
    print("\n=== Milvus索引性能分析报告 ===\n")
    
    print("\n最佳延迟性能 (毫秒):")
    print(tabulate(df_latency[['index_type', 'avg_latency']].head(3), headers=['索引类型', '平均延迟(ms)'], tablefmt='pretty'))
    
    print("\n最佳召回率:")
    print(tabulate(df_recall[['index_type', 'avg_recall']].head(3), headers=['索引类型', '平均召回率'], tablefmt='pretty'))
    
    print("\n最佳吞吐量 (QPS):")
    print(tabulate(df_qps[['index_type', 'qps']].head(3), headers=['索引类型', '每秒查询数'], tablefmt='pretty'))
    
    print("\n索引构建时间 (秒):")
    df_build = df.sort_values('build_time')
    print(tabulate(df_build[['index_type', 'build_time']].head(3), headers=['索引类型', '构建时间(s)'], tablefmt='pretty'))
    
    # 计算综合评分
    # 对每个指标进行归一化处理
    df['norm_recall'] = df['avg_recall'] / df['avg_recall'].max() if df['avg_recall'].max() > 0 else 0
    df['norm_latency'] = 1 - (df['avg_latency'] / df['avg_latency'].max() if df['avg_latency'].max() > 0 else 0)
    df['norm_qps'] = df['qps'] / df['qps'].max() if df['qps'].max() > 0 else 0
    df['norm_build'] = 1 - (df['build_time'] / df['build_time'].max() if df['build_time'].max() > 0 else 0)
    
    # 综合评分 (可根据需求调整权重)
    df['score'] = (
        df['norm_recall'] * 0.4 +  # 召回率权重40%
        df['norm_latency'] * 0.3 +  # 延迟权重30%
        df['norm_qps'] * 0.2 +      # 吞吐量权重20%
        df['norm_build'] * 0.1      # 构建时间权重10%
    )
    
    df_score = df.sort_values('score', ascending=False)
    
    print("\n综合评分 (综合考虑召回率、延迟、吞吐量和构建时间):")
    print(tabulate(df_score[['index_type', 'score']].head(3), headers=['索引类型', '综合评分'], tablefmt='pretty'))
    
    # 提供建议
    print("\n=== 索引选择建议 ===")
    print(f"1. 对于追求准确性的场景，推荐使用: {df_recall.iloc[0]['index_type']}")
    print(f"2. 对于追求低延迟的场景，推荐使用: {df_latency.iloc[0]['index_type']}")
    print(f"3. 对于高吞吐量场景，推荐使用: {df_qps.iloc[0]['index_type']}")
    print(f"4. 对于快速构建索引场景，推荐使用: {df_build.iloc[0]['index_type']}")
    print(f"5. 综合性能最佳的索引: {df_score.iloc[0]['index_type']}")
    
    # 生成可视化图表
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置绘图风格
        sns.set(style="whitegrid")
        
        # 1. 延迟对比图
        plt.figure(figsize=(10, 6))
        chart = sns.barplot(x='index_type', y='avg_latency', data=df.sort_values('avg_latency'))
        chart.set_title('不同索引类型的查询延迟对比')
        chart.set_xlabel('索引类型')
        chart.set_ylabel('平均延迟 (毫秒)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'latency_comparison.png'))
        logger.info(f"已保存延迟对比图到 {output_dir}")
        
        # 2. 召回率对比图
        plt.figure(figsize=(10, 6))
        chart = sns.barplot(x='index_type', y='avg_recall', data=df.sort_values('avg_recall', ascending=False))
        chart.set_title('不同索引类型的召回率对比')
        chart.set_xlabel('索引类型')
        chart.set_ylabel('平均召回率')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'recall_comparison.png'))
        logger.info(f"已保存召回率对比图到 {output_dir}")
        
        # 3. 综合评分图
        plt.figure(figsize=(10, 6))
        chart = sns.barplot(x='index_type', y='score', data=df.sort_values('score', ascending=False))
        chart.set_title('不同索引类型的综合评分对比')
        chart.set_xlabel('索引类型')
        chart.set_ylabel('综合评分')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_comparison.png'))
        logger.info(f"已保存综合评分对比图到 {output_dir}")
        
        # 4. 雷达图 (综合性能)
        categories = ['召回率', '延迟性能', '吞吐量', '构建速度']
        
        # 选择前3个索引进行对比
        top_indices = df_score.head(3)['index_type'].tolist()
        
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        
        # 设置雷达图的角度
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories)
        
        for idx, index_type in enumerate(top_indices):
            row = df[df['index_type'] == index_type].iloc[0]
            values = [
                row['norm_recall'],
                row['norm_latency'],
                row['norm_qps'],
                row['norm_build']
            ]
            values += values[:1]  # 闭合图形
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=index_type)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_title('索引性能雷达图')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radar_comparison.png'))
        logger.info(f"已保存性能雷达图到 {output_dir}")
    
    # 导出分析结果到CSV
    if output_dir:
        csv_file = os.path.join(output_dir, 'index_analysis.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"已导出分析结果到 {csv_file}")
    
    logger.info("结果解释完成")
    return df

def main():
    parser = argparse.ArgumentParser(description="解释Milvus基准测试结果")
    parser.add_argument('--results-dir', type=str, default='../results',
                        help='结果目录路径')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录路径 (用于保存图表和分析结果)')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细日志')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 解析结果目录路径
    results_dir = args.results_dir
    if not os.path.isabs(results_dir):
        # 如果是相对路径，则相对于脚本目录解析
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.abspath(os.path.join(script_dir, results_dir))
    
    # 解析输出目录路径
    output_dir = args.output_dir
    if output_dir and not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(script_dir, output_dir))
    
    results = load_results(results_dir)
    if results:
        explain_results(results, output_dir)
    else:
        print("未找到结果文件或结果文件为空")

if __name__ == "__main__":
    main() 