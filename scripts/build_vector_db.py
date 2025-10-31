#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
向量数据库构建脚本
===================

功能:
- 加载原始游戏数据
- 数据归一化与特征工程
- 向量化并存入ChromaDB
- 性能监控与错误处理
- 结果验证

使用示例:
    # 全量构建
    python scripts/build_vector_db.py --input data/raw/93182_steam_games.csv
    
    # 自定义批次大小
    python scripts/build_vector_db.py --input data/raw/93182_steam_games.csv --batch-size 512
    
    # 重置向量库
    python scripts/build_vector_db.py --input data/raw/93182_steam_games.csv --reset
    
    # 增量更新(未来支持)
    python scripts/build_vector_db.py --input data/raw/new_games.csv --incremental
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import time
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_processing.data_normalizer import DataNormalizer
from src.data_processing.feature_engineer import FeatureEngineer
from src.vectordb.chroma_store import ChromaVectorStore
from src.indexing.vectorizer import GameVectorizer


# ==================== 日志配置 ====================
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger("build_vector_db")
    logger.setLevel(log_level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # 文件处理器
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_dir / f"build_vector_db_{timestamp}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# ==================== 数据加载 ====================
def load_data(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    加载原始游戏数据
    
    Args:
        input_path: CSV或Parquet文件路径
        logger: 日志对象
        
    Returns:
        原始DataFrame
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不支持
    """
    if not input_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {input_path}")
    
    logger.info(f"开始加载数据: {input_path}")
    start_time = time.time()
    
    suffix = input_path.suffix.lower()
    if suffix == '.csv':
        df = pd.read_csv(input_path)
    elif suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}, 仅支持 .csv 或 .parquet")
    
    elapsed = time.time() - start_time
    logger.info(f"数据加载完成: {len(df)} 行, {len(df.columns)} 列, 耗时 {elapsed:.2f}秒")
    logger.info(f"数据列: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}")
    
    return df


# ==================== 数据预处理 ====================
def preprocess_data(
    df: pd.DataFrame,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    数据归一化与特征工程
    
    Args:
        df: 原始DataFrame
        logger: 日志对象
        
    Returns:
        处理后的DataFrame
    """
    logger.info("=" * 60)
    logger.info("开始数据预处理")
    logger.info("=" * 60)
    
    # 步骤1: 数据归一化
    logger.info("步骤 1/2: 数据归一化")
    start_time = time.time()
    
    normalizer = DataNormalizer()
    df_normalized = normalizer.normalize(df)
    
    elapsed = time.time() - start_time
    logger.info(f"归一化完成: {len(df_normalized)} 行保留, 耗时 {elapsed:.2f}秒")
    
    # 步骤2: 特征工程
    logger.info("步骤 2/2: 特征工程")
    start_time = time.time()
    
    engineer = FeatureEngineer()
    df_processed = engineer.add_derived_features(df_normalized)
    
    elapsed = time.time() - start_time
    logger.info(f"特征工程完成: 新增 {len(df_processed.columns) - len(df_normalized.columns)} 个特征, 耗时 {elapsed:.2f}秒")
    
    # 新增特征列表
    new_features = set(df_processed.columns) - set(df_normalized.columns)
    if new_features:
        logger.info(f"新增特征: {', '.join(sorted(new_features))}")
    
    logger.info(f"预处理后数据: {len(df_processed)} 行, {len(df_processed.columns)} 列")
    
    return df_processed


# ==================== 向量化与入库 ====================
def vectorize_and_store(
    df: pd.DataFrame,
    output_dir: Path,
    collection_name: str,
    batch_size: int,
    reset: bool,
    logger: logging.Logger
) -> dict:
    """
    向量化游戏数据并存入ChromaDB
    
    Args:
        df: 预处理后的DataFrame
        output_dir: 向量库输出目录
        collection_name: 集合名称
        batch_size: 批处理大小
        reset: 是否重置向量库
        logger: 日志对象
        
    Returns:
        统计信息字典
    """
    logger.info("=" * 60)
    logger.info("开始向量化与入库")
    logger.info("=" * 60)
    
    # 初始化向量库
    logger.info(f"初始化ChromaDB: {output_dir}")
    vectorstore = ChromaVectorStore(
        persist_directory=str(output_dir),
        collection_name=collection_name
    )
    
    # 重置向量库
    if reset:
        logger.warning("重置模式: 将清空现有向量库")
        vectorstore.reset()
        logger.info("向量库已重置")
    else:
        existing_count = vectorstore.get_count()
        if existing_count > 0:
            logger.info(f"向量库已存在 {existing_count} 条记录")
    
    # 初始化向量化器
    logger.info("初始化GameVectorizer")
    vectorizer = GameVectorizer(vectorstore=vectorstore)
    
    # 执行向量化
    logger.info(f"开始向量化: {len(df)} 个游戏, 批次大小={batch_size}")
    start_time = time.time()
    
    stats = vectorizer.vectorize_batch(
        df=df,
        batch_size=batch_size
    )
    
    elapsed = time.time() - start_time
    
    # 输出统计信息
    logger.info("=" * 60)
    logger.info("向量化完成统计")
    logger.info("=" * 60)
    logger.info(f"总耗时: {elapsed:.2f}秒")
    logger.info(f"总游戏数: {stats['total_games']}")
    logger.info(f"成功向量化: {stats['successful']}")
    logger.info(f"失败数: {stats['failed']}")
    logger.info(f"成功率: {stats['successful'] / stats['total_games'] * 100:.2f}%")
    logger.info(f"平均速度: {stats['total_games'] / elapsed:.2f} 游戏/秒")
    
    if stats['failed'] > 0:
        logger.warning(f"有 {stats['failed']} 个游戏向量化失败，请检查日志")
    
    # 持久化
    logger.info("持久化向量库到磁盘...")
    vectorstore.persist()
    
    # 最终统计
    final_count = vectorstore.get_count()
    logger.info(f"向量库最终记录数: {final_count}")
    
    return {
        **stats,
        'elapsed_seconds': elapsed,
        'games_per_second': stats['total_games'] / elapsed if elapsed > 0 else 0,
        'final_count': final_count
    }


# ==================== 结果验证 ====================
def validate_results(
    output_dir: Path,
    collection_name: str,
    logger: logging.Logger
) -> bool:
    """
    验证向量库质量
    
    Args:
        output_dir: 向量库目录
        collection_name: 集合名称
        logger: 日志对象
        
    Returns:
        是否验证通过
    """
    logger.info("=" * 60)
    logger.info("开始结果验证")
    logger.info("=" * 60)
    
    try:
        # 重新连接向量库
        vectorstore = ChromaVectorStore(
            persist_directory=str(output_dir),
            collection_name=collection_name
        )
        
        total_count = vectorstore.get_count()
        logger.info(f"向量库总记录数: {total_count}")
        
        if total_count == 0:
            logger.error("向量库为空，验证失败")
            return False
        
        # 测试查询1: 通用查询
        logger.info("\n测试查询 1: 'RPG adventure games'")
        results = vectorstore.query(
            query_texts=["RPG adventure games"],
            n_results=5
        )
        
        if results:
            logger.info(f"返回 {len(results)} 个结果:")
            for i, game in enumerate(results[:3], 1):
                name = game.get('metadata', {}).get('Name', 'Unknown')
                genres = game.get('metadata', {}).get('Genres', 'Unknown')
                distance = game.get('distance', 0)
                logger.info(f"  {i}. {name} (Genres: {genres}, Distance: {distance:.4f})")
        else:
            logger.warning("查询未返回结果")
        
        # 测试查询2: 特定类型
        logger.info("\n测试查询 2: 'multiplayer shooter games'")
        results = vectorstore.query(
            query_texts=["multiplayer shooter games"],
            n_results=5
        )
        
        if results:
            logger.info(f"返回 {len(results)} 个结果:")
            for i, game in enumerate(results[:3], 1):
                name = game.get('metadata', {}).get('Name', 'Unknown')
                tags = game.get('metadata', {}).get('Tags', 'Unknown')
                distance = game.get('distance', 0)
                logger.info(f"  {i}. {name} (Tags: {tags}, Distance: {distance:.4f})")
        else:
            logger.warning("查询未返回结果")
        
        # 测试查询3: 元数据过滤
        logger.info("\n测试查询 3: 带元数据过滤 (Windows平台)")
        results = vectorstore.query(
            query_texts=["strategy games"],
            n_results=5,
            where={"Windows": True}
        )
        
        if results:
            logger.info(f"返回 {len(results)} 个结果 (仅Windows游戏):")
            for i, game in enumerate(results[:3], 1):
                name = game.get('metadata', {}).get('Name', 'Unknown')
                platforms = []
                if game.get('metadata', {}).get('Windows'): platforms.append('Win')
                if game.get('metadata', {}).get('Mac'): platforms.append('Mac')
                if game.get('metadata', {}).get('Linux'): platforms.append('Linux')
                logger.info(f"  {i}. {name} (Platforms: {','.join(platforms)})")
        else:
            logger.warning("查询未返回结果")
        
        logger.info("\n✅ 向量库验证通过")
        return True
        
    except Exception as e:
        logger.error(f"验证过程出错: {e}", exc_info=True)
        return False


# ==================== 主函数 ====================
def main():
    """主执行流程"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="构建游戏推荐向量数据库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/raw/93182_steam_games.csv',
        help='输入数据文件路径 (CSV或Parquet)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/vector_db',
        help='向量库输出目录 (默认: data/vector_db)'
    )
    
    parser.add_argument(
        '--collection', '-c',
        type=str,
        default='steam_games',
        help='ChromaDB集合名称 (默认: steam_games)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=256,
        help='批处理大小 (默认: 256)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='重置向量库 (清空现有数据)'
    )
    
    parser.add_argument(
        '--incremental',
        action='store_true',
        help='增量更新模式 (跳过已存在的游戏) [未来支持]'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别 (默认: INFO)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='跳过结果验证'
    )
    
    args = parser.parse_args()
    
    # 路径处理
    input_path = project_root / args.input
    output_dir = project_root / args.output
    
    # 初始化日志
    logger = setup_logging(args.log_level)
    
    # 打印配置
    logger.info("=" * 60)
    logger.info("向量数据库构建脚本")
    logger.info("=" * 60)
    logger.info(f"输入文件: {input_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"集合名称: {args.collection}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"重置模式: {args.reset}")
    logger.info(f"增量模式: {args.incremental}")
    logger.info(f"日志级别: {args.log_level}")
    logger.info("=" * 60)
    
    # 增量更新提示
    if args.incremental:
        logger.warning("增量更新模式暂未实现，将按全量模式处理")
    
    try:
        # 总计时开始
        total_start = time.time()
        
        # 步骤1: 加载数据
        df = load_data(input_path, logger)
        
        # 步骤2: 数据预处理
        df_processed = preprocess_data(df, logger)
        
        # 步骤3: 向量化与入库
        stats = vectorize_and_store(
            df=df_processed,
            output_dir=output_dir,
            collection_name=args.collection,
            batch_size=args.batch_size,
            reset=args.reset,
            logger=logger
        )
        
        # 步骤4: 结果验证
        if not args.skip_validation:
            validation_passed = validate_results(
                output_dir=output_dir,
                collection_name=args.collection,
                logger=logger
            )
            
            if not validation_passed:
                logger.error("验证失败，但向量库已构建")
        else:
            logger.info("跳过结果验证")
        
        # 总结
        total_elapsed = time.time() - total_start
        logger.info("=" * 60)
        logger.info("构建完成")
        logger.info("=" * 60)
        logger.info(f"总耗时: {total_elapsed:.2f}秒 ({total_elapsed/60:.2f}分钟)")
        logger.info(f"向量库路径: {output_dir}")
        logger.info(f"成功向量化: {stats['successful']} / {stats['total_games']}")
        logger.info(f"平均速度: {stats['games_per_second']:.2f} 游戏/秒")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"构建过程出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
