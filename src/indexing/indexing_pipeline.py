"""
索引构建管线
数据加载 → 清洗 → 特征工程 → 文本融合 → Embedding → 入库
"""
import os
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from src.data_processing import TextCleaner, FeatureEngineer, DataNormalizer
from src.models.model_manager import get_model_manager
from src.vectordb import ChromaVectorStore
from src.utils.logger import setup_logger


class IndexingPipeline:
    """索引构建管线"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化索引构建管线
        
        Args:
            config_path: vectorization.yaml配置文件路径
        """
        self.logger = setup_logger("IndexingPipeline")
        
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parents[2] / "config" / "vectorization.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化组件
        self.text_cleaner = TextCleaner(
            **self.config['text_fusion']['about_cleaning']
        )
        
        self.feature_engineer = FeatureEngineer(
            config=self.config.get('feature_engineering', {})
        )
        
        self.normalizer = DataNormalizer(
            default_values=self.config['text_fusion']['default_values']
        )
        
        # 初始化模型管理器
        self.model_manager = get_model_manager()
        
        # 初始化向量库
        from src.config.model_config import get_config
        model_cfg = get_config()
        vdb_cfg = model_cfg.vectordb
        
        self.vector_store = ChromaVectorStore(
            persist_directory=vdb_cfg['persist_directory'],
            collection_name=self.config['vectordb']['collection_name'],
            distance_metric=self.config['vectordb']['distance_metric']
        )
        
        self.logger.info("✅ 索引构建管线初始化完成")
    
    def build_index(self, reset: bool = False) -> None:
        """
        构建完整索引
        
        Args:
            reset: 是否清空已有索引重新构建
        """
        if reset:
            self.logger.warning("⚠️ 清空现有索引...")
            self.vector_store.reset()
            # 重新创建集合
            from src.config.model_config import get_config
            model_cfg = get_config()
            vdb_cfg = model_cfg.vectordb
            self.vector_store = ChromaVectorStore(
                persist_directory=vdb_cfg['persist_directory'],
                collection_name=self.config['vectordb']['collection_name'],
                distance_metric=self.config['vectordb']['distance_metric']
            )
        
        # 1. 加载数据
        self.logger.info("📂 加载数据...")
        df = self._load_data()
        self.logger.info(f"   数据行数: {len(df)}")
        
        # 2. 数据预处理
        self.logger.info("🔧 数据预处理...")
        df = self._preprocess_data(df)
        
        # 3. 特征工程
        self.logger.info("⚙️ 特征工程...")
        df = self._engineer_features(df)
        
        # 4. 文本融合
        self.logger.info("📝 文本融合...")
        df = self._fuse_text(df)
        
        # 5. 准备metadata
        self.logger.info("📦 准备metadata...")
        metadatas = self._prepare_metadatas(df)
        
        # 6. Embedding + 入库
        self.logger.info("🔢 Embedding并入库...")
        self._embed_and_index(df, metadatas)
        
        # 7. 持久化
        self.vector_store.persist()
        
        self.logger.info(f"✅ 索引构建完成! 总文档数: {self.vector_store.get_count()}")
    
    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        data_cfg = self.config['data_source']
        
        # 优先加载parquet(更快)
        parquet_path = data_cfg['input_parquet']
        csv_path = data_cfg['input_csv']
        
        if os.path.exists(parquet_path):
            self.logger.info(f"   从Parquet加载: {parquet_path}")
            return pd.read_parquet(parquet_path)
        elif os.path.exists(csv_path):
            self.logger.info(f"   从CSV加载: {csv_path}")
            return pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"数据文件不存在: {parquet_path} 或 {csv_path}"
            )
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        # 1. 标准化
        df = self.normalizer.normalize_dataframe(df)
        
        # 2. 清洗About the game
        if 'About the game' in df.columns:
            self.logger.info("   清洗游戏描述文本...")
            df['About_cleaned'] = df['About the game'].apply(
                self.text_cleaner.clean
            )
        else:
            df['About_cleaned'] = ''
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征工程"""
        rating_cfg = self.config['feature_engineering']['rating']
        pop_cfg = self.config['feature_engineering']['popularity']
        quality_cfg = self.config['feature_engineering']['quality']
        
        # 1. 计算评分
        if 'Positive' in df.columns and 'Negative' in df.columns:
            self.logger.info("   计算用户评分...")
            df['final_rating'] = self.feature_engineer.compute_rating(
                df['Positive'],
                df['Negative'],
                method=rating_cfg['method'],
                confidence=rating_cfg['confidence']
            )
        else:
            df['final_rating'] = 0.5
        
        # 2. 计算热度评分
        self.logger.info("   计算热度评分...")
        df['popularity_score'] = self.feature_engineer.compute_popularity_score(
            df,
            weights=pop_cfg['weights']
        )
        
        # 3. 计算质量评分
        self.logger.info("   计算质量评分...")
        df['quality_score'] = self.feature_engineer.compute_quality_score(
            df,
            weights=quality_cfg['weights']
        )
        
        return df
    
    def _fuse_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """文本融合 - 按模板拼接"""
        template = self.config['text_fusion']['template']
        
        self.logger.info("   应用文本融合模板...")
        
        def fuse_single(row):
            try:
                return template.format(
                    Name=str(row.get('Name', 'Unknown')),
                    About_cleaned=str(row.get('About_cleaned', ''))[:600],  # 限制长度
                    Genres=str(row.get('Genres', 'Unknown')),
                    Tags=str(row.get('Tags', 'Unknown')),
                    Categories=str(row.get('Categories', 'Unknown')),
                    Release_year=str(row.get('Release_year', 'Unknown')),
                    Platforms=str(row.get('Platforms', 'Unknown')),
                    popularity_score=float(row.get('popularity_score', 0)),
                    quality_score=float(row.get('quality_score', 0))
                )
            except Exception as e:
                self.logger.warning(f"   文本融合失败: {e}")
                return f"Name: {row.get('Name', 'Unknown')}"
        
        df['text_combined'] = df.apply(fuse_single, axis=1)
        
        return df
    
    def _prepare_metadatas(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """准备metadata字典列表"""
        meta_cfg = self.config['metadata_fields']
        
        # 收集所有要保存的字段
        all_fields = set()
        for field_list in meta_cfg.values():
            if isinstance(field_list, list):
                all_fields.update(field_list)
        
        metadatas = []
        
        for _, row in df.iterrows():
            meta = {}
            
            for field in all_fields:
                if field in row:
                    value = row[field]
                    
                    # 处理NaN和None
                    if pd.isna(value):
                        continue
                    
                    # 转换为JSON可序列化类型
                    if isinstance(value, (int, float, bool, str)):
                        meta[field] = value
                    else:
                        meta[field] = str(value)
            
            metadatas.append(meta)
        
        return metadatas
    
    def _embed_and_index(
        self,
        df: pd.DataFrame,
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Embedding并批量入库"""
        batch_cfg = self.config['batch_processing']
        chunk_size = batch_cfg['chunk_size']
        show_progress = batch_cfg['show_progress']
        
        # 准备数据
        ids = df['AppID'].astype(str).tolist()
        documents = df['text_combined'].tolist()
        
        # 批量处理
        total_batches = (len(documents) + chunk_size - 1) // chunk_size
        
        iterator = range(0, len(documents), chunk_size)
        if show_progress:
            iterator = tqdm(iterator, total=total_batches, desc="Embedding")
        
        for i in iterator:
            batch_ids = ids[i:i + chunk_size]
            batch_docs = documents[i:i + chunk_size]
            batch_metas = metadatas[i:i + chunk_size]
            
            # Embedding
            embeddings = self.model_manager.embedding.encode_corpus(
                batch_docs,
                show_progress=False
            )
            
            # 入库
            self.vector_store.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=embeddings
            )
            
            # 保存检查点(可选)
            if batch_cfg.get('save_checkpoint', False):
                if (i + chunk_size) % batch_cfg.get('checkpoint_interval', 1000) == 0:
                    self.logger.info(f"   检查点: 已处理 {i + chunk_size} 条")


# 示例用法
if __name__ == "__main__":
    pipeline = IndexingPipeline()
    
    # 构建索引(清空重建)
    pipeline.build_index(reset=True)
