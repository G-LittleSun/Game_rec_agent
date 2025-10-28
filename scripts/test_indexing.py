"""
测试索引构建流程
处理少量数据验证管线是否正常工作
"""
import sys
from pathlib import Path
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from src.indexing import IndexingPipeline
from src.utils.logger import setup_logger


def test_small_batch():
    """测试小批量数据"""
    logger = setup_logger("TestIndexing", level="INFO")
    
    logger.info("=" * 60)
    logger.info("🧪 测试索引构建流程(小批量)")
    logger.info("=" * 60)
    
    try:
        # 1. 加载少量数据进行测试
        logger.info("\n📂 加载测试数据...")
        data_path = project_root / "data" / "processed" / "93182_steam_games_cleaned.parquet"
        
        if not data_path.exists():
            data_path = project_root / "data" / "processed" / "93182_steam_games_cleaned.csv"
        
        df = pd.read_parquet(data_path) if data_path.suffix == '.parquet' else pd.read_csv(data_path)
        
        # 只取前10条数据
        df_test = df.head(10).copy()
        logger.info(f"   测试数据: {len(df_test)} 行")
        
        # 2. 临时保存测试数据
        test_parquet = project_root / "data" / "processed" / "test_sample.parquet"
        df_test.to_parquet(test_parquet, index=False)
        logger.info(f"   保存测试数据到: {test_parquet}")
        
        # 3. 初始化管线
        logger.info("\n📦 初始化索引构建管线...")
        pipeline = IndexingPipeline()
        
        # 临时修改配置指向测试数据
        pipeline.config['data_source']['input_parquet'] = str(test_parquet)
        pipeline.config['batch_processing']['chunk_size'] = 5  # 小批量
        
        # 4. 构建索引
        logger.info("\n🚀 开始构建测试索引...")
        pipeline.build_index(reset=True)
        
        # 5. 验证结果
        logger.info("\n✅ 索引构建完成!")
        count = pipeline.vector_store.get_count()
        logger.info(f"   向量数量: {count}")
        
        # 6. 测试查询
        logger.info("\n🔍 测试查询功能...")
        results = pipeline.vector_store.query(
            query_texts=["action game"],
            n_results=3
        )
        
        logger.info(f"   查询结果数: {len(results['ids'][0])}")
        if results['ids'][0]:
            logger.info(f"   前3个游戏ID: {results['ids'][0][:3]}")
            if 'metadatas' in results and results['metadatas'][0]:
                logger.info(f"   第一个游戏: {results['metadatas'][0][0].get('Name', 'Unknown')}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 测试通过!")
        logger.info("=" * 60)
        
        # 清理测试文件
        test_parquet.unlink()
        
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    test_small_batch()
