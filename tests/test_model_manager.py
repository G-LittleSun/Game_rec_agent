"""
模型管理器单元测试
"""
import pytest
from src.models.model_manager import ModelManager


class TestModelManager:
    """测试模型管理器"""
    
    @pytest.fixture
    def manager(self):
        """创建管理器实例"""
        return ModelManager()
    
    def test_manager_init(self, manager):
        """测试管理器初始化"""
        assert manager is not None
        assert manager.config is not None
    
    def test_get_model_info(self, manager):
        """测试获取模型信息"""
        info = manager.get_model_info()
        assert "llm" in info
        assert "embedding" in info
        assert "rerank" in info
    
    def test_embedding_encode(self, manager):
        """测试Embedding编码"""
        texts = ["测试文本1", "测试文本2"]
        embeddings = manager.encode_text(texts)
        
        assert embeddings is not None
        assert len(embeddings) == 2
    
    def test_rerank(self, manager):
        """测试Rerank"""
        query = "测试查询"
        docs = ["文档1", "文档2", "文档3"]
        
        results = manager.rerank_documents(query, docs, top_k=2)
        
        assert len(results) == 2
        assert all(isinstance(item, tuple) for item in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])