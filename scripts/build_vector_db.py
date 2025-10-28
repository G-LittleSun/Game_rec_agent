"""
向量数据库构建脚本
一键构建游戏推荐系统的向量索引
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from src.indexing import IndexingPipeline
from src.utils.logger import setup_logger


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="构建Steam游戏推荐系统向量数据库"
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='清空现有索引重新构建'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='vectorization.yaml配置文件路径(默认: config/vectorization.yaml)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger("BuildVectorDB", level=args.log_level)
    
    logger.info("=" * 60)
    logger.info("🎮 Steam游戏推荐系统 - 向量数据库构建")
    logger.info("=" * 60)
    
    try:
        # 初始化管线
        logger.info("\n📦 初始化索引构建管线...")
        pipeline = IndexingPipeline(config_path=args.config)
        
        # 构建索引
        logger.info("\n🚀 开始构建索引...")
        if args.reset:
            logger.warning("⚠️  将清空现有索引!")
            confirm = input("确认继续? (y/N): ")
            if confirm.lower() != 'y':
                logger.info("❌ 用户取消操作")
                return
        
        pipeline.build_index(reset=args.reset)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 向量数据库构建完成!")
        logger.info("=" * 60)
        
        # 显示统计信息
        count = pipeline.vector_store.get_count()
        logger.info(f"\n📊 数据库统计:")
        logger.info(f"   总文档数: {count}")
        logger.info(f"   集合名称: {pipeline.vector_store.collection_name}")
        logger.info(f"   持久化路径: {pipeline.vector_store.persist_directory}")
        
    except Exception as e:
        logger.error(f"\n❌ 构建失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
