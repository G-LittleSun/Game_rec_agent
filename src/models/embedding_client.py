"""
Embedding 模型客户端
使用 sentence-transformers 进行文本向量化
"""
import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from src.utils.logger import setup_logger


class EmbeddingClient:
    """Embedding模型客户端"""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str = "cuda",
        batch_size: int = 32,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        cache_dir: str = "./data/model_cache"
    ):
        """
        初始化Embedding客户端
        
        Args:
            model_name: 模型名称或路径
            device: 运行设备 (cuda/cpu/mps)
            batch_size: 批处理大小
            max_length: 最大序列长度
            normalize_embeddings: 是否归一化向量
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        
        # 设置设备
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
            print("⚠️ CUDA不可用,使用CPU")
        elif device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"
            print("⚠️ MPS不可用,使用CPU")
        
        self.device = device
        

        
        self.logger = setup_logger("EmbeddingClient")
        
        # 加载模型
        self.logger.info(f"🔄 加载Embedding模型: {model_name}")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir
        )
        self.logger.info(f"✅ 模型加载完成,设备: {device}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = False,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        将文本编码为向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小,默认使用初始化时的值
            show_progress: 是否显示进度条
            convert_to_numpy: 是否转换为numpy数组
        
        Returns:
            文本向量,形状为 (n_texts, embedding_dim)
        """
        # 处理单个文本
        if isinstance(texts, str):
            texts = [texts]
        
        if batch_size is None:
            batch_size = self.batch_size
        
        # 编码
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=convert_to_numpy,
            device=self.device
        )
        
        return embeddings
    
    def encode_queries(self, queries: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        编码查询文本(为兼容不同模型,可添加特殊处理)
        
        Args:
            queries: 查询文本
            **kwargs: 其他参数
        
        Returns:
            查询向量
        """
        # BGE模型建议为查询添加前缀
        if "bge" in self.model_name.lower():
            if isinstance(queries, str):
                queries = f"为这个句子生成表示以用于检索相关文章:{queries}"
            else:
                queries = [f"为这个句子生成表示以用于检索相关文章:{q}" for q in queries]
        
        return self.encode(queries, **kwargs)
    
    def encode_corpus(self, corpus: List[str], **kwargs) -> np.ndarray:
        """
        编码语料库文本
        
        Args:
            corpus: 文档列表
            **kwargs: 其他参数
        
        Returns:
            文档向量
        """
        # 语料库不需要添加前缀
        return self.encode(corpus, **kwargs)
    
    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        计算向量相似度
        
        Args:
            embeddings1: 第一组向量
            embeddings2: 第二组向量
            metric: 相似度度量 (cosine/dot/euclidean)
        
        Returns:
            相似度矩阵
        """
        if metric == "cosine":
            # 余弦相似度
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "dot":
            # 点积
            return np.dot(embeddings1, embeddings2.T)
        elif metric == "euclidean":
            # 欧氏距离(转换为相似度)
            distances = np.linalg.norm(
                embeddings1[:, np.newaxis] - embeddings2[np.newaxis, :],
                axis=2
            )
            return 1 / (1 + distances)
        else:
            raise ValueError(f"不支持的相似度度量: {metric}")
    
    def get_embedding_dim(self) -> int:
        """获取向量维度"""
        return self.model.get_sentence_embedding_dimension()
    
    def switch_model(self, model_name: str, cache_dir: str = None):
        """切换模型"""
        self.logger.info(f"🔄 切换Embedding模型: {model_name}")
        if cache_dir is None:
            cache_dir = self.model.cache_folder
        
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=self.device,
            cache_folder=cache_dir
        )
        self.logger.info(f"✅ 模型切换完成")


# 示例用法
if __name__ == "__main__":
    # 创建客户端
    client = EmbeddingClient(
        model_name="BAAI/bge-base-zh-v1.5",
        device="cuda"  
    )
    
    # 编码文本
    texts = [
        "《塞尔达传说:旷野之息》是一款开放世界冒险游戏",
        "《艾尔登法环》是一款黑暗奇幻开放世界RPG",
        "《我的世界》是一款沙盒建造游戏"
    ]
    
    embeddings = client.encode(texts)
    print(f"向量形状: {embeddings.shape}")
    print(f"向量维度: {client.get_embedding_dim()}")
    
    # 计算相似度
    query = "推荐一款开放世界游戏"
    query_emb = client.encode_queries(query)
    
    similarities = client.similarity(query_emb, embeddings)
    print(f"\n查询: {query}")
    for i, (text, sim) in enumerate(zip(texts, similarities[0])):
        print(f"{i+1}. [{sim:.4f}] {text}")
    print("/n")