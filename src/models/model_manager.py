"""
模型管理器
统一管理LLM、Embedding、Rerank模型,提供简洁的调用接口
"""
from typing import List, Union, Optional, Dict, Any
from src.config.model_config import get_config
from src.models.ollama_client import OllamaClient
from src.models.embedding_client import EmbeddingClient
from src.models.rerank_client import RerankClient
from src.utils.logger import setup_logger


class ModelManager:
    """模型管理器 - 统一入口"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模型管理器
        
        Args:
            config_path: 配置文件路径,为None则使用默认配置
        """
        # 加载配置
        self.config = get_config(config_path)
        self.logger = setup_logger(
            "ModelManager",
            level=self.config.logging.get("level", "INFO"),
            log_file=self.config.logging.get("log_file")
        )
        
        # 初始化各模型客户端
        self._llm_client = None
        self._embedding_client = None
        self._rerank_client = None
        
        self.logger.info("✅ 模型管理器初始化完成")
    
    @property
    def llm(self) -> OllamaClient:
        """获取LLM客户端(懒加载)"""
        if self._llm_client is None:
            llm_config = self.config.llm
            provider = llm_config.get("provider", "ollama")
            
            if provider == "ollama":
                ollama_cfg = llm_config["ollama"]
                self._llm_client = OllamaClient(
                    base_url=ollama_cfg["base_url"],
                    model=ollama_cfg["model"],
                    temperature=ollama_cfg.get("temperature", 0.7),
                    top_p=ollama_cfg.get("top_p", 0.9),
                    max_tokens=ollama_cfg.get("max_tokens", 2048),
                    timeout=ollama_cfg.get("timeout", 120)
                )
            else:
                raise ValueError(f"不支持的LLM provider: {provider}")
        
        return self._llm_client
    
    @property
    def embedding(self) -> EmbeddingClient:
        """获取Embedding客户端(懒加载)"""
        if self._embedding_client is None:
            emb_config = self.config.embedding
            self._embedding_client = EmbeddingClient(
                model_name=emb_config["model_name"],
                device=emb_config.get("device", "cuda"),
                batch_size=emb_config.get("batch_size", 32),
                max_length=emb_config.get("max_length", 512),
                normalize_embeddings=emb_config.get("normalize_embeddings", True),
                cache_dir=emb_config.get("cache_dir", "./data/model_cache")
            )
        
        return self._embedding_client
    
    @property
    def rerank(self) -> RerankClient:
        """获取Rerank客户端(懒加载)"""
        if self._rerank_client is None:
            rerank_config = self.config.rerank
            self._rerank_client = RerankClient(
                model_name=rerank_config["model_name"],
                device=rerank_config.get("device", "cuda"),
                batch_size=rerank_config.get("batch_size", 16),
                max_length=rerank_config.get("max_length", 512),
                cache_dir=rerank_config.get("cache_dir", "./data/model_cache")
            )
        
        return self._rerank_client
    
    # ==================== 便捷方法 ====================
    
    def generate_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        生成文本(LLM)
        
        Args:
            prompt: 用户输入
            system: 系统提示词
            stream: 是否流式返回
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        return self.llm.generate(prompt, system=system, stream=stream, **kwargs)
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        is_query: bool = False,
        **kwargs
    ) -> Any:
        """
        文本编码为向量
        
        Args:
            texts: 文本或文本列表
            is_query: 是否为查询文本(会添加特殊前缀)
            **kwargs: 其他参数
        
        Returns:
            文本向量
        """
        if is_query:
            return self.embedding.encode_queries(texts, **kwargs)
        else:
            return self.embedding.encode_corpus(texts if isinstance(texts, list) else [texts], **kwargs)
    
    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        **kwargs
    ) -> List[tuple]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回前k个结果
            **kwargs: 其他参数
        
        Returns:
            重排序后的结果 [(document, score), ...]
        """
        return self.rerank.rerank_with_docs(query, documents, top_k=top_k, **kwargs)
    
    def reload_config(self):
        """重新加载配置"""
        self.config.reload()
        # 重置客户端,下次使用时会根据新配置重新初始化
        self._llm_client = None
        self._embedding_client = None
        self._rerank_client = None
        self.logger.info("✅ 配置已重新加载")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        info = {
            "llm": {
                "provider": self.config.llm.get("provider"),
                "model": self.config.llm.get("ollama", {}).get("model"),
            },
            "embedding": {
                "model": self.config.embedding.get("model_name"),
                "device": self.config.embedding.get("device"),
                "dimension": self.embedding.get_embedding_dim() if self._embedding_client else None
            },
            "rerank": {
                "model": self.config.rerank.get("model_name"),
                "device": self.config.rerank.get("device"),
            }
        }
        return info


# 全局单例
_global_manager = None


def get_model_manager(config_path: Optional[str] = None) -> ModelManager:
    """
    获取全局模型管理器实例(单例)
    
    Args:
        config_path: 配置文件路径,仅首次调用时有效
    
    Returns:
        ModelManager实例
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelManager(config_path)
    return _global_manager


# 示例用法
if __name__ == "__main__":
    # 创建管理器
    manager = ModelManager()
    
    # 打印模型信息
    print("📊 当前模型配置:")
    import json
    print(json.dumps(manager.get_model_info(), indent=2, ensure_ascii=False))
    
    # 测试LLM
    print("\n🤖 测试LLM:")
    response = manager.generate_text(
        prompt="简单介绍一下《塞尔达传说:旷野之息》",
        system="你是一个游戏推荐专家"
    )
    print(response)
    
    # 测试Embedding
    print("\n🔢 测试Embedding:")
    texts = ["开放世界游戏", "RPG游戏", "射击游戏"]
    embeddings = manager.encode_text(texts)
    print(f"向量形状: {embeddings.shape}")
    
    # 测试Rerank
    print("\n🔄 测试Rerank:")
    query = "推荐开放世界游戏"
    docs = [
        "《塞尔达传说》是一款开放世界冒险游戏",
        "《反恐精英》是一款第一人称射击游戏",
        "《艾尔登法环》是开放世界RPG"
    ]
    results = manager.rerank_documents(query, docs, top_k=2)
    for doc, score in results:
        print(f"[{score:.4f}] {doc}")