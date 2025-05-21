# Milvus向量数据库基准测试

这个项目用于评估Milvus向量数据库在不同索引类型下的性能表现。

## 环境要求

- Python 3.8或更高版本
- Docker 19.03或更高版本 (已测试兼容Docker 26.1.0)
- 至少4GB内存

## 快速开始

1. 克隆仓库并进入目录

```bash
git clone https://github.com/your-username/milvus-benchmark.git
cd milvus-benchmark
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 运行基准测试

```bash
python run_milvus_benchmark.py
```

这将自动:
- 下载SIFT数据集
- 启动Milvus服务
- 运行性能测试
- 生成测试报告和可视化图表
- 关闭Milvus服务

## 命令行选项

```
usage: run_milvus_benchmark.py [-h] [--skip-download] [--skip-milvus]
                              [--indices {FLAT,IVF_FLAT,IVF_SQ8,HNSW} [{FLAT,IVF_FLAT,IVF_SQ8,HNSW} ...]]
                              [--fast-test] [--check-env] [--analyze-only] [--verbose]

Milvus向量数据库性能评估实验

options:
  -h, --help            显示帮助信息并退出
  --skip-download       跳过数据下载步骤
  --skip-milvus         跳过Milvus启动步骤
  --indices {FLAT,IVF_FLAT,IVF_SQ8,HNSW} [{FLAT,IVF_FLAT,IVF_SQ8,HNSW} ...]
                        要测试的索引类型，如不指定则测试所有索引
  --fast-test           仅测试FLAT索引，用于快速验证
  --check-env           仅检查环境，不运行测试
  --analyze-only        仅分析已有结果，不运行测试
  --verbose             显示详细日志
```

## 新增功能

### 环境检查

运行环境检查以确保系统满足运行基准测试的要求：

```bash
python run_milvus_benchmark.py --check-env
```

环境检查将验证：
- Python版本
- 必要依赖
- Docker可用性
- 系统资源（CPU、内存、磁盘空间）
- Milvus连接
- 数据目录

### 结果分析

如果您已经运行过测试，可以单独分析结果：

```bash
python run_milvus_benchmark.py --analyze-only
```

分析功能将生成：
- 详细的性能分析报告
- 不同索引类型的对比图表
- 综合性能评分
- 索引选择建议

## 测试的索引类型

- **FLAT**: 暴力搜索，100%准确但速度较慢
- **IVF_FLAT**: 基于聚类的搜索，平衡了准确性和速度
- **IVF_SQ8**: 基于聚类且使用标量量化的搜索，减少内存使用
- **HNSW**: 层次导航小世界图，提供高速搜索

## 生成的报告

测试完成后，结果将保存在`results/`目录中，包括:
- JSON格式的详细结果数据
- CSV格式的摘要数据
- 展示不同索引性能的图表
- 性能分析报告
- 索引性能雷达图

## 日志记录

系统会自动记录详细的运行日志，保存在`logs/`目录中。使用`--verbose`选项可以显示更详细的日志信息。

## 故障排除

### 环境问题

- **环境检查失败**: 运行`python run_milvus_benchmark.py --check-env --verbose`获取详细诊断信息。
- **依赖安装问题**: 尝试手动安装依赖`pip install -r requirements.txt`。

### 数据问题

- **无法下载数据集**: 如果数据集下载失败，脚本会自动生成一个小型示例数据集用于测试。
- **数据格式错误**: 检查`data/`目录中的文件格式是否符合要求。

### Docker相关问题

- **Docker连接问题**: 确保Docker服务正在运行，并且当前用户有权限访问Docker。
- **Docker版本兼容性**: 对于Docker 26.1.0版本，命令已更新为`docker compose`格式（无连字符）。
- **Milvus启动失败**: 检查是否有其他服务占用了19530端口。可以使用`docker compose -f config/docker-compose.yml down`手动停止现有的Milvus服务。

## 注意事项

1. 对于Docker 26.1.0版本，命令已更新为`docker compose`格式（无连字符）。
2. 本测试默认使用官方Docker镜像。
3. 测试结果可能因硬件配置不同而有所差异。

## 系统要求

- Python 3.8+
- Docker 26.1.0+
- 足够的内存和存储空间（建议至少8GB内存）

## 性能评估指标

- **准确率/召回率**：搜索结果的准确性，即正确返回的相关结果占总相关结果的比例
- **查询延迟**：单次查询所需的时间
- **吞吐量**：系统每秒能处理的查询数量
- **资源使用**：CPU、内存、磁盘空间的使用情况
- **扩展性**：随着数据规模增长的性能变化

## 项目结构

- `data/`: 存放实验数据集
- `scripts/`: 存放实验脚本
  - `download_data.py`: 下载和准备数据集
  - `start_milvus.py`: 管理Milvus服务
  - `run_benchmark.py`: 执行基准测试
  - `explain_results.py`: 分析和可视化结果
  - `check_environment.py`: 检查系统环境
  - `logger.py`: 日志管理模块
- `config/`: 存放配置文件
- `results/`: 存放实验结果
- `logs/`: 存放运行日志
- `requirements.txt`: 项目依赖

## 使用说明

1. 安装依赖：`pip install -r requirements.txt`
2. 检查环境：`python run_milvus_benchmark.py --check-env`
3. 下载数据集：运行`python scripts/download_data.py`
4. 启动Milvus服务：运行`python scripts/start_milvus.py start`
5. 运行性能测试：`python scripts/run_benchmark.py`
6. 分析结果：`python scripts/explain_results.py`
7. 一键运行所有步骤：`python run_milvus_benchmark.py`

## 贡献指南

欢迎提交问题和改进建议！请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个Pull Request 