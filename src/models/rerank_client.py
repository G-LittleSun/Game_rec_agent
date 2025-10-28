"""
Rerank 模型客户端
使用 CrossEncoder 对检索结果进行重排序
"""
import torch
import numpy as np
from typing import List, Tuple
from sentence_transformers import CrossEncoder
from src.utils.logger import setup_logger


class RerankClient:
    """Rerank模型客户端"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512,
        cache_dir: str = "./data/model_cache"
    ):
        """
        初始化Rerank客户端
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备 (cuda/cpu/mps)
            batch_size: 批处理大小
            max_length: 最大序列长度
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 设置设备
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("⚠️ CUDA不可用,使用CPU")
        elif device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
            print("⚠️ MPS不可用,使用CPU")
        
        self.device = device

        
        self.logger = setup_logger("RerankClient")
        
        # 加载模型
        self.logger.info(f"🔄 加载Rerank模型: {model_name}")
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
            default_activation_function=torch.nn.Sigmoid()  # 输出0-1之间的分数
        )
        self.logger.info(f"✅ 模型加载完成,设备: {device}")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
        return_scores: bool = True,
        batch_size: int = None
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前k个结果,为None则返回全部
            return_scores: 是否返回分数
            batch_size: 批处理大小
        
        Returns:
            重排序后的结果列表 [(doc_index, score), ...]
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # 构造query-document对
        pairs = [[query, doc] for doc in documents]
        
        # 预测相关性分数
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # 排序
        ranked_results = [
            (idx, float(score))
            for idx, score in enumerate(scores)
        ]
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # 截断top_k
        if top_k is not None:
            ranked_results = ranked_results[:top_k]
        
        if return_scores:
            return ranked_results
        else:
            return [idx for idx, _ in ranked_results]
    
    def rerank_with_docs(
        self,
        query: str,
        documents: List[str],
        top_k: int = None,
        batch_size: int = None
    ) -> List[Tuple[str, float]]:
        """
        对文档进行重排序并返回文档内容
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前k个结果
            batch_size: 批处理大小
        
        Returns:
            重排序后的结果列表 [(document, score), ...]
        """
        ranked_indices = self.rerank(
            query,
            documents,
            top_k=top_k,
            return_scores=True,
            batch_size=batch_size
        )
        
        return [
            (documents[idx], score)
            for idx, score in ranked_indices
        ]
    
    def score(
        self,
        query_doc_pairs: List[Tuple[str, str]],
        batch_size: int = None
    ) -> np.ndarray:
        """
        直接计算query-document对的相关性分数
        
        Args:
            query_doc_pairs: (query, document)对列表
            batch_size: 批处理大小
        
        Returns:
            相关性分数数组
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        pairs = [[q, d] for q, d in query_doc_pairs]
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        return scores
    
    def switch_model(self, model_name: str):
        """切换模型"""
        self.logger.info(f"🔄 切换Rerank模型: {model_name}")
        
        self.model_name = model_name
        self.model = CrossEncoder(
            model_name,
            max_length=self.max_length,
            device=self.device,
            default_activation_function=torch.nn.Sigmoid()
        )
        self.logger.info(f"✅ 模型切换完成")


# 示例用法
if __name__ == "__main__":
    # 创建客户端
    client = RerankClient(
        model_name="BAAI/bge-reranker-base",
        device="cpu"  # 测试用CPU
    )
    
    # 准备数据
    query = "推荐一款开放世界RPG游戏"
    documents = [
        "《塞尔达传说:旷野之息》是一款开放世界冒险游戏,拥有广阔的探索空间",
        "《艾尔登法环》是FromSoftware开发的开放世界动作RPG游戏",
        "《我的世界》是一款沙盒建造游戏",
        "《超级马力欧:奥德赛》是一款3D平台跳跃游戏",
        "《巫师3:狂猎》是一款中世纪奇幻开放世界RPG"
    ]
    
    # 重排序
    results = client.rerank_with_docs(query, documents, top_k=3)
    
    print(f"查询: {query}\n")
    print("重排序结果:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"{i}. [分数: {score:.4f}] {doc}")