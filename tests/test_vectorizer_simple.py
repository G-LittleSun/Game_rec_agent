"""
向量化管线测试脚本

测试完整的向量化流程
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.indexing.vectorizer import GameVectorizer
from src.utils.logger import setup_logger


def test_vectorizer():
    """测试向量化器基本功能"""
    logger = setup_logger("VectorizerTest")
    
    logger.info("=" * 60)
    logger.info("🧪 测试向量化管线")
    logger.info("=" * 60)
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'AppID': ['1', '2', '3'],
        'Name': ['塞尔达传说', '艾尔登法环', '我的世界'],
        'About the game': [
            '开放世界动作冒险游戏',
            '黑暗奇幻动作RPG游戏',
            '沙盒建造游戏'
        ],
        'Genres': ['Adventure,RPG', 'RPG,Action', 'Sandbox'],
        'Tags': ['Open World,Adventure', 'RPG,Dark Fantasy', 'Creative,Building'],
        'Categories': ['Single-player', 'Single-player', 'Multi-player'],
        'Platforms': ['Windows', 'Windows', 'Windows, Mac, Linux'],
        'Developers': ['Nintendo', 'FromSoftware', 'Mojang'],
        'Publishers': ['Nintendo', 'Bandai Namco', 'Microsoft'],
        'Price': [60.0, 50.0, 30.0],
        'Positive': [5000, 10000, 50000],
        'Negative': [500, 1000, 2000],
        'Recommendations': [3000, 8000, 40000],
        'Release_year': ['2017', '2022', '2011'],
        'rating_score': [0.89, 0.91, 0.95],
        'popularity_score': [0.6, 0.8, 0.95],
        'quality_score': [0.85, 0.88, 0.92],
        'Metacritic score': [95, 96, 93],
        'Peak CCU': [5000, 50000, 100000],
        'Average playtime forever': [300, 500, 1000]
    })
    
    logger.info(f"📊 测试数据: {len(test_data)} 条")
    
    # 初始化向量化器
    try:
        vectorizer = GameVectorizer(
            vectorization_config_path="config/vectorization.yaml",
            models_config_path="config/models.yaml"
        )
        logger.info("✅ 向量化器初始化成功")
    except Exception as e:
        logger.error(f"❌ 向量化器初始化失败: {e}")
        return
    
    # 测试文本融合
    logger.info("\n📝 测试文本融合...")
    try:
        fused_text = vectorizer.fuse_text(test_data.iloc[0])
        logger.info(f"融合结果示例:\n{fused_text[:200]}...")
        logger.info("✅ 文本融合测试通过")
    except Exception as e:
        logger.error(f"❌ 文本融合失败: {e}")
        return
    
    # 测试metadata准备
    logger.info("\n🗂️ 测试Metadata准备...")
    try:
        metadata = vectorizer.prepare_metadata(test_data.iloc[0])
        logger.info(f"Metadata字段数: {len(metadata)}")
        logger.info(f"示例字段: {list(metadata.keys())[:5]}")
        logger.info("✅ Metadata准备测试通过")
    except Exception as e:
        logger.error(f"❌ Metadata准备失败: {e}")
        return
    
    # 测试批量向量化
    logger.info("\n🚀 测试批量向量化...")
    try:
        vectorizer.vectorize_batch(test_data, batch_size=2)
        logger.info("✅ 批量向量化测试通过")
    except Exception as e:
        logger.error(f"❌ 批量向量化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试查询
    logger.info("\n🔍 测试向量检索...")
    try:
        results = vectorizer.test_query("开放世界冒险游戏", top_k=3)
        logger.info("✅ 向量检索测试通过")
    except Exception as e:
        logger.error(f"❌ 向量检索失败: {e}")
        return
    
    # 获取统计信息
    logger.info("\n📊 向量库统计信息...")
    try:
        stats = vectorizer.get_statistics()
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")
        logger.info("✅ 统计信息获取成功")
    except Exception as e:
        logger.error(f"❌ 获取统计信息失败: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 所有测试完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_vectorizer()
