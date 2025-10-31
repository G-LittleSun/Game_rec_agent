"""
ChromaDB向量存储实现
"""
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings

# 兼容直接运行和模块导入
try:
    from src.vectordb.base import VectorStore
    from src.utils.logger import setup_logger
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.vectordb.base import VectorStore
    from src.utils.logger import setup_logger


class ChromaVectorStore(VectorStore):
    """ChromaDB向量存储"""
    
    def __init__(
        self,
        persist_directory: str,
        collection_name: str = "steam_games",
        distance_metric: str = "cosine",
        embedding_function: Optional[Any] = None
    ):
        """
        初始化ChromaDB向量存储
        
        Args:
            persist_directory: 持久化目录
            collection_name: 集合名称
            distance_metric: 距离度量 (cosine, l2, ip)
            embedding_function: 嵌入函数(可选,如果提供embeddings则不需要)
        """
        self.logger = setup_logger("ChromaVectorStore")
        
        # 创建客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,  # 关闭遥测（保护隐私）
                allow_reset=True   # 允许重置（测试时有用）
            )
        )
        
        # 距离度量映射
        distance_map = {
            "cosine": "cosine",  # 余弦相似度（推荐，归一化后等价于点积）
            "l2": "l2",   # 欧氏距离（对向量长度敏感）
            "ip": "ip"  # inner product 内积（适合已归一化的向量）
        }
        
        # 创建或获取集合
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": distance_map.get(distance_metric, "cosine")
                },
                embedding_function=embedding_function
            )
            self.logger.info(
                f"✅ ChromaDB集合已加载: {collection_name}, "
                f"文档数: {self.get_count()}"
            )
        except Exception as e:
            self.logger.error(f"❌ 创建/加载集合失败: {e}")
            raise
    
    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """添加文档到向量库"""
        try:
            # 准备数据
            add_kwargs = {
                "ids": ids,
                "documents": documents,
            }
            
            if metadatas is not None:
                # 清洗metadata(移除None值,确保JSON可序列化)。ChromaDB 对 metadata 有严格要求
                cleaned_metadatas = [
                    self._clean_metadata(m) for m in metadatas
                ]
                add_kwargs["metadatas"] = cleaned_metadatas
            
            if embeddings is not None:
                add_kwargs["embeddings"] = embeddings.tolist()
            
            self.collection.add(**add_kwargs)
            
        except Exception as e:
            self.logger.error(f"❌ 添加文档失败: {e}")
            raise
    
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[np.ndarray] = None,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """查询向量库"""
        try:
            query_kwargs = {
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if query_texts is not None:
                query_kwargs["query_texts"] = query_texts
            elif query_embeddings is not None:
                query_kwargs["query_embeddings"] = query_embeddings.tolist()
            else:
                raise ValueError("必须提供 query_texts 或 query_embeddings")
            
            if filter_dict is not None:
                query_kwargs["where"] = filter_dict
            
            results = self.collection.query(**query_kwargs)
            
            # 格式化结果
            formatted_results = []
            
            # Chroma返回的结果按批次组织(支持多个查询)
            # 这里假设单个查询,取第一个批次
            if results["ids"]:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else None,
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"❌ 查询失败: {e}")
            raise
    
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"✅ 已删除 {len(ids)} 个文档")
        except Exception as e:
            self.logger.error(f"❌ 删除文档失败: {e}")
            raise
    
    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """更新文档"""
        try:
            update_kwargs = {"ids": ids}
            
            if documents is not None:
                update_kwargs["documents"] = documents
            
            if metadatas is not None:
                cleaned_metadatas = [
                    self._clean_metadata(m) for m in metadatas
                ]
                update_kwargs["metadatas"] = cleaned_metadatas
            
            if embeddings is not None:
                update_kwargs["embeddings"] = embeddings.tolist()
            
            self.collection.update(**update_kwargs)
            
        except Exception as e:
            self.logger.error(f"❌ 更新文档失败: {e}")
            raise
    
    def get_count(self) -> int:
        """获取文档数量"""
        try:
            return self.collection.count()
        except Exception as e:
            self.logger.error(f"❌ 获取文档数量失败: {e}")
            return 0
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """根据ID或条件获取文档"""
        try:
            get_kwargs = {
                "include": ["documents", "metadatas"]
            }
            
            if ids is not None:
                get_kwargs["ids"] = ids
            
            if filter_dict is not None:
                get_kwargs["where"] = filter_dict
            
            if limit is not None:
                get_kwargs["limit"] = limit
            
            results = self.collection.get(**get_kwargs)
            
            # 格式化结果
            formatted_results = []
            if results["ids"]:
                for i in range(len(results["ids"])):
                    formatted_results.append({
                        "id": results["ids"][i],
                        "document": results["documents"][i] if results["documents"] else None,
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"❌ 获取文档失败: {e}")
            raise
    
    def persist(self) -> None:
        """持久化(PersistentClient自动持久化)"""
        self.logger.info("✅ ChromaDB自动持久化")
    
    def reset(self) -> None:
        """清空向量库并重新创建collection"""
        try:
            collection_name = self.collection.name
            collection_metadata = self.collection.metadata
            
            # 删除旧集合
            self.client.delete_collection(collection_name)
            self.logger.info(f"✅ 集合 {collection_name} 已删除")
            
            # 重新创建集合
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            self.logger.info(f"✅ 集合 {collection_name} 已重新创建")
        except Exception as e:
            self.logger.error(f"❌ 重置集合失败: {e}")
            raise
    
    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        清洗metadata,确保可JSON序列化
        
        - 移除None值
        - 转换numpy类型为Python原生类型
        - 处理特殊类型
        """
        cleaned = {}
        
        for key, value in metadata.items():
            # 跳过None值
            if value is None:
                continue
            
            # 转换numpy类型
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, (list, tuple)):
                # 递归清洗列表
                value = [v.item() if isinstance(v, (np.integer, np.floating)) else v for v in value]
            
            # 转换pandas类型
            if hasattr(value, 'item'):  # pandas scalar
                value = value.item()
            
            cleaned[key] = value
        
        return cleaned


# 示例用法
if __name__ == "__main__":
    import os
    
    # 创建测试向量库
    test_dir = "./test_chroma_db"
    os.makedirs(test_dir, exist_ok=True)
    
    store = ChromaVectorStore(
        persist_directory=test_dir,
        collection_name="test_games"
    )
    
    # 添加测试数据
    test_ids = ["game1", "game2", "game3"]
    test_docs = [
        "Open world RPG with great story",
        "Fast-paced FPS shooter game",
        "Relaxing puzzle adventure game"
    ]
    test_metas = [
        {"name": "Game A", "price": 29.99},
        {"name": "Game B", "price": 19.99},
        {"name": "Game C", "price": 9.99}
    ]
    
    # 创建简单的测试向量(实际应该用embedding模型)
    test_embs = np.random.rand(3, 384)
    
    store.add(test_ids, test_docs, test_metas, test_embs)
    
    print(f"向量库文档数: {store.get_count()}")
    
    # 查询测试
    query_emb = np.random.rand(1, 384)
    results = store.query(query_embeddings=query_emb, top_k=2)
    
    print("\n查询结果:")
    for r in results:
        print(f"- {r['metadata']['name']}: {r['document'][:50]}...")
