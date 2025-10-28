"""
向量存储抽象基类
定义统一的向量数据库接口
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class VectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """
        添加文档到向量库
        
        Args:
            ids: 文档ID列表
            documents: 文档文本列表
            metadatas: 元数据列表
            embeddings: 预计算的向量(可选,如果提供则不会重新编码)
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[np.ndarray] = None,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        查询向量库
        
        Args:
            query_texts: 查询文本列表
            query_embeddings: 查询向量(可选)
            top_k: 返回前K个结果
            filter_dict: 元数据过滤条件
        
        Returns:
            查询结果列表,每个结果包含id, document, metadata, distance/score
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        删除指定ID的文档
        
        Args:
            ids: 要删除的文档ID列表
        """
        pass
    
    @abstractmethod
    def update(
        self,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """
        更新文档
        
        Args:
            ids: 文档ID列表
            documents: 新文档文本(可选)
            metadatas: 新元数据(可选)
            embeddings: 新向量(可选)
        """
        pass
    
    @abstractmethod
    def get(
        self,
        ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        根据ID或条件获取文档
        
        Args:
            ids: 文档ID列表(可选)
            filter_dict: 元数据过滤条件(可选)
            limit: 返回数量限制(可选)
        
        Returns:
            文档列表,每个文档包含id, document, metadata
        """
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """
        获取向量库中的文档数量
        
        Returns:
            文档总数
        """
        pass
    
    @abstractmethod
    def persist(self) -> None:
        """持久化到磁盘"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """清空向量库"""
        pass
