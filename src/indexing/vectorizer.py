"""
向量化管线模块

负责将游戏数据转换为向量并存入向量数据库
核心流程：文本融合 → 向量生成 → 批量入库
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import yaml
from tqdm import tqdm

from src.models.model_manager import ModelManager
from src.vectordb.chroma_store import ChromaVectorStore
from src.data_processing.feature_engineer import FeatureEngineer
from src.utils.logger import setup_logger


class GameVectorizer:
    """游戏数据向量化器
    
    将标准化后的游戏数据转换为向量并存储到向量数据库
    """
    
    def __init__(
        self,
        vectorization_config_path: str = "config/vectorization.yaml",
        models_config_path: str = "config/models.yaml"
    ):
        """
        初始化向量化器
        
        Args:
            vectorization_config_path: 向量化配置文件路径
            models_config_path: 模型配置文件路径
        """
        self.logger = setup_logger("GameVectorizer")
        
        # 保存配置路径
        self.models_config_path = models_config_path
        
        # 加载配置
        self.vec_config = self._load_config(vectorization_config_path)
        self.model_config = self._load_config(models_config_path)
        
        # 初始化模型管理器（传递配置文件路径）
        self.logger.info("🤖 初始化模型管理器...")
        self.model_manager = ModelManager(config_path=self.models_config_path)
        
        # 初始化特征工程器
        self.logger.info("⚙️ 初始化特征工程器...")
        self.feature_engineer = FeatureEngineer(
            config=self.vec_config.get("feature_engineering", {})
        )
        
        # 初始化向量数据库
        self.logger.info("💾 初始化向量数据库...")
        vectordb_config = self.vec_config.get("vectordb", {})
        self.vector_store = ChromaVectorStore(
            persist_directory=self.model_config.get("vectordb", {}).get("persist_directory", "./data/vector_db"),
            collection_name=vectordb_config.get("collection_name", "steam_games"),
            distance_metric=vectordb_config.get("distance_metric", "cosine")
        )
        
        self.logger.info("✅ 向量化器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def fuse_text(self, row: pd.Series) -> str:
        """
        根据模板融合文本
        
        Args:
            row: DataFrame的一行数据
            
        Returns:
            融合后的文本字符串
        """
        template = self.vec_config["text_fusion"]["template"]
        fields = self.vec_config["text_fusion"]["fields"] if "fields" in self.vec_config["text_fusion"] else []
        
        # 准备填充数据（处理缺失值）
        fill_data = {}
        for field in fields:
            value = row.get(field, '')
            # 处理NaN和None
            if pd.isna(value) or value is None:
                value = ''
            fill_data[field] = str(value)
        
        # 如果模板中有其他字段（如衍生特征），也添加进去
        for key in row.index:
            if key not in fill_data:
                value = row[key]
                if pd.isna(value) or value is None:
                    value = ''
                fill_data[key] = str(value)
        
        try:
            # 填充模板
            fused_text = template.format(**fill_data)
            return fused_text.strip()
        except KeyError as e:
            self.logger.warning(f"文本融合缺少字段 {e}, AppID: {row.get('AppID', 'Unknown')}")
            # 降级方案：只使用名称和描述
            name = row.get('Name', '')
            desc = row.get('About the game', '')
            return f"Game: {name}\nDescription: {desc}".strip()
        except Exception as e:
            self.logger.error(f"文本融合失败: {e}, AppID: {row.get('AppID', 'Unknown')}")
            return row.get('Name', 'Unknown Game')
    
    def prepare_metadata(self, row: pd.Series) -> Dict[str, Any]:
        """
        准备metadata字典
        
        Args:
            row: DataFrame的一行数据
            
        Returns:
            清洗后的metadata字典（符合ChromaDB要求）
        """
        metadata = {}
        
        # 获取metadata字段配置
        metadata_config = self.vec_config.get("metadata_fields", {})
        
        # 合并所有字段类型
        all_fields = []
        for category in ["required", "text", "numeric", "boolean", "temporal", "other"]:
            if category in metadata_config:
                all_fields.extend(metadata_config[category])
        
        # 如果没有配置，使用所有列
        if not all_fields:
            all_fields = row.index.tolist()
        
        # 填充metadata
        for field in all_fields:
            if field not in row.index:
                continue
            
            value = row[field]
            
            # 跳过NaN值
            if pd.isna(value):
                continue
            
            # 类型转换（ChromaDB要求metadata值必须是基本类型）
            if isinstance(value, (np.integer, np.int64, np.int32)):
                metadata[field] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                # 将NaN和inf转为0
                if np.isnan(value) or np.isinf(value):
                    metadata[field] = 0.0
                else:
                    metadata[field] = float(value)
            elif isinstance(value, (bool, np.bool_)):
                metadata[field] = bool(value)
            elif isinstance(value, str):
                metadata[field] = value
            else:
                # 其他类型转为字符串
                metadata[field] = str(value)
        
        return metadata
    
    def vectorize_batch(
        self,
        df: pd.DataFrame,
        batch_size: Optional[int] = None
    ) -> None:
        """
        批量向量化并存入向量数据库
        
        Args:
            df: 已标准化且完成特征工程的DataFrame
            batch_size: 批处理大小，None则使用配置文件的值
        """
        if batch_size is None:
            batch_size = self.vec_config.get("batch_processing", {}).get("chunk_size", 32)
        
        total = len(df)
        self.logger.info(f"🚀 开始向量化 {total} 条游戏数据...")
        
        # 进度条配置
        show_progress = self.vec_config.get("batch_processing", {}).get("show_progress", True)
        
        # 统计信息
        success_count = 0
        error_count = 0
        
        # 分批处理
        iterator = range(0, total, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="向量化进度", unit="batch")
        
        for i in iterator:
            batch_df = df.iloc[i:i+batch_size]
            
            try:
                # 1. 文本融合
                documents = batch_df.apply(self.fuse_text, axis=1).tolist()
                
                # 2. 生成向量
                self.logger.debug(f"正在生成向量 (batch {i//batch_size + 1})...")
                embeddings = self.model_manager.encode_text(
                    texts=documents,
                    is_query=False
                )
                
                # 3. 准备metadata
                metadatas = batch_df.apply(self.prepare_metadata, axis=1).tolist()
                
                # 4. 准备IDs（使用AppID）
                if 'AppID' in batch_df.columns:
                    ids = batch_df['AppID'].astype(str).tolist()
                else:
                    # 如果没有AppID，使用索引
                    ids = [f"game_{idx}" for idx in batch_df.index]
                
                # 5. 入库
                self.vector_store.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings
                )
                
                success_count += len(batch_df)
                
                # 更新进度信息
                if show_progress and isinstance(iterator, tqdm):
                    iterator.set_postfix({
                        'success': success_count,
                        'errors': error_count
                    })
                
            except Exception as e:
                error_count += len(batch_df)
                self.logger.error(f"❌ 批次 {i//batch_size + 1} 入库失败: {e}")
                
                # 根据配置决定是否继续
                continue_on_error = self.vec_config.get("batch_processing", {}).get("continue_on_error", True)
                if not continue_on_error:
                    raise
        
        # 持久化
        self.logger.info("💾 持久化向量数据库...")
        self.vector_store.persist()
        
        # 最终统计
        final_count = self.vector_store.get_count()
        self.logger.info(f"✅ 向量化完成！")
        self.logger.info(f"   - 成功: {success_count} 条")
        self.logger.info(f"   - 失败: {error_count} 条")
        self.logger.info(f"   - 数据库总数: {final_count} 条")
    
    def test_query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        测试查询功能
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            
        Returns:
            查询结果列表
        """
        self.logger.info(f"🔍 测试查询: '{query_text}'")
        
        # 1. 将查询文本转为向量
        query_embedding = self.model_manager.encode_text(
            texts=query_text,
            is_query=True
        )
        
        # 2. 使用向量查询
        results = self.vector_store.query(
            query_embeddings=query_embedding,
            top_k=top_k
        )
        
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            distance = result.get('distance', 0)
            self.logger.info(f"  {i}. {metadata.get('Name', 'Unknown')} (距离: {distance:.4f})")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取向量库统计信息
        
        Returns:
            统计信息字典
        """
        count = self.vector_store.get_count()
        
        stats = {
            'total_documents': count,
            'collection_name': self.vector_store.collection.name,
            'distance_metric': self.vec_config.get("vectordb", {}).get("distance_metric", "cosine"),
            'embedding_dimension': self.vec_config.get("embedding", {}).get("dimension", 384)
        }
        
        return stats


class VectorizationPipeline:
    """完整的向量化管线
    
    整合数据加载、标准化、特征工程、向量化的完整流程
    """
    
    def __init__(
        self,
        vectorization_config_path: str = "config/vectorization.yaml",
        models_config_path: str = "config/models.yaml"
    ):
        """
        初始化向量化管线
        
        Args:
            vectorization_config_path: 向量化配置文件路径
            models_config_path: 模型配置文件路径
        """
        self.logger = setup_logger("VectorizationPipeline")
        
        # 加载配置
        self.vec_config = self._load_config(vectorization_config_path)
        
        # 初始化向量化器
        self.vectorizer = GameVectorizer(
            vectorization_config_path=vectorization_config_path,
            models_config_path=models_config_path
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def run(
        self,
        input_data: pd.DataFrame,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        运行完整的向量化管线
        
        Args:
            input_data: 已标准化且完成特征工程的数据
            batch_size: 批处理大小
            
        Returns:
            执行结果统计
        """
        self.logger.info("=" * 60)
        self.logger.info("🚀 开始向量化管线")
        self.logger.info("=" * 60)
        
        # 数据验证
        self.logger.info(f"📊 输入数据: {len(input_data)} 条")
        
        # 向量化
        self.vectorizer.vectorize_batch(input_data, batch_size=batch_size)
        
        # 获取统计信息
        stats = self.vectorizer.get_statistics()
        
        self.logger.info("=" * 60)
        self.logger.info("✅ 向量化管线完成")
        self.logger.info(f"📊 统计信息:")
        for key, value in stats.items():
            self.logger.info(f"   - {key}: {value}")
        self.logger.info("=" * 60)
        
        return stats


if __name__ == "__main__":
    # 测试代码
    import sys
    from pathlib import Path
    
    # 添加项目根目录到路径
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # 创建向量化器
    vectorizer = GameVectorizer()
    
    # 测试查询
    vectorizer.test_query("开放世界角色扮演游戏", top_k=3)
    
    # 显示统计信息
    stats = vectorizer.get_statistics()
    print("\n统计信息:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
