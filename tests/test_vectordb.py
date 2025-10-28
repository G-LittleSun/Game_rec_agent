"""
向量数据库模块单元测试
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.vectordb.base import VectorStore
from src.vectordb.chroma_store import ChromaVectorStore


class TestChromaVectorStore:
    """测试 ChromaDB 向量存储"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        # 测试结束后清理
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def vector_store(self, temp_dir):
        """创建测试用的向量存储实例"""
        store = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="test_games",
            distance_metric="cosine"  # 参数名修正
        )
        return store
    
    def test_init(self, temp_dir):
        """测试向量存储初始化"""
        store = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="test_collection"
        )
        
        assert store is not None
        assert store.collection.name == "test_collection"  # 修正属性访问
        assert store.get_count() == 0  # 使用 get_count() 方法
    
    def test_add_documents(self, vector_store):
        """测试添加文档"""
        ids = ["game_1", "game_2", "game_3"]
        documents = [
            "Game: Zelda. Genres: Adventure, RPG. Tags: open-world, fantasy.",
            "Game: Elden Ring. Genres: RPG, Action. Tags: souls-like, difficult.",
            "Game: Minecraft. Genres: Sandbox, Survival. Tags: building, creative."
        ]
        metadatas = [
            {"name": "Zelda", "price": 60, "genres": "Adventure,RPG"},
            {"name": "Elden Ring", "price": 50, "genres": "RPG,Action"},
            {"name": "Minecraft", "price": 30, "genres": "Sandbox,Survival"}
        ]
        
        # 创建随机向量
        embeddings = np.random.rand(3, 384).astype(np.float32)
        
        # 添加文档
        vector_store.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )
        
        # 验证添加成功
        count = vector_store.get_count()  # 使用 get_count() 方法
        assert count == 3
    
    def test_query_by_embeddings(self, vector_store):
        """测试向量查询"""
        # 先添加一些文档
        ids = ["game_1", "game_2", "game_3"]
        documents = [
            "Open world RPG with great story",
            "First person shooter game",
            "Adventure game with exploration"
        ]
        metadatas = [
            {"name": "Game1", "price": 60},
            {"name": "Game2", "price": 40},
            {"name": "Game3", "price": 50}
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)
        
        vector_store.add(ids, documents, metadatas, embeddings)
        
        # 查询 - 使用 query() 方法
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        results = vector_store.query(
            query_embeddings=query_embedding,
            top_k=2
        )
        
        # 验证结果
        assert len(results) == 2
        assert "id" in results[0]
        assert "document" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]
    
    def test_query_with_filter(self, vector_store):
        """测试带过滤条件的查询"""
        # 添加测试数据
        ids = ["game_1", "game_2", "game_3"]
        documents = ["Game 1", "Game 2", "Game 3"]
        metadatas = [
            {"name": "Game1", "price": 60, "platform": "Windows"},
            {"name": "Game2", "price": 30, "platform": "Mac"},
            {"name": "Game3", "price": 50, "platform": "Windows"}
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)
        
        vector_store.add(ids, documents, metadatas, embeddings)
        
        # 查询时过滤价格 - 使用 query() 和 filter_dict 参数
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        results = vector_store.query(
            query_embeddings=query_embedding,
            top_k=5,
            filter_dict={"price": {"$lte": 50}}  # 价格 <= 50
        )
        
        # 验证过滤结果
        assert len(results) <= 2  # 只有 game_2 和 game_3 符合
        for result in results:
            assert result["metadata"]["price"] <= 50
    
    def test_delete(self, vector_store):
        """测试删除文档"""
        # 添加测试数据
        ids = ["game_1", "game_2"]
        documents = ["Game 1", "Game 2"]
        metadatas = [{"name": "Game1"}, {"name": "Game2"}]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        
        vector_store.add(ids, documents, metadatas, embeddings)
        assert vector_store.get_count() == 2
        
        # 删除一个文档
        vector_store.delete(ids=["game_1"])
        
        # 验证删除成功
        assert vector_store.get_count() == 1
    
    def test_update(self, vector_store):
        """测试更新文档"""
        # 添加初始数据
        ids = ["game_1"]
        documents = ["Original document"]
        metadatas = [{"name": "Game1", "price": 60}]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        
        vector_store.add(ids, documents, metadatas, embeddings)
        
        # 更新文档
        new_documents = ["Updated document"]
        new_metadatas = [{"name": "Game1", "price": 50}]
        new_embeddings = np.random.rand(1, 384).astype(np.float32)
        
        vector_store.update(
            ids=ids,
            documents=new_documents,
            metadatas=new_metadatas,
            embeddings=new_embeddings
        )
        
        # 验证更新成功
        query_embedding = new_embeddings
        results = vector_store.query(query_embeddings=query_embedding, top_k=1)
        assert results[0]["document"] == "Updated document"
        assert results[0]["metadata"]["price"] == 50
    
    def test_persist_and_load(self, temp_dir):
        """测试持久化和加载"""
        # 创建第一个实例并添加数据
        store1 = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="persist_test"
        )
        
        ids = ["game_1", "game_2"]
        documents = ["Game 1", "Game 2"]
        metadatas = [{"name": "Game1"}, {"name": "Game2"}]
        embeddings = np.random.rand(2, 384).astype(np.float32)
        
        store1.add(ids, documents, metadatas, embeddings)
        store1.persist()
        
        # 创建新实例,加载持久化的数据
        store2 = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="persist_test"
        )
        
        # 验证数据被正确加载
        assert store2.get_count() == 2
    
    def test_count(self, vector_store):
        """测试计数功能"""
        assert vector_store.get_count() == 0
        
        # 添加文档
        ids = ["game_1", "game_2", "game_3"]
        documents = ["Doc1", "Doc2", "Doc3"]
        embeddings = np.random.rand(3, 384).astype(np.float32)
        
        vector_store.add(ids, documents, embeddings=embeddings)
        
        assert vector_store.get_count() == 3
    
    def test_get_by_ids(self, vector_store):
        """测试通过ID获取文档"""
        # 添加数据
        ids = ["game_1", "game_2", "game_3"]
        documents = ["Game 1", "Game 2", "Game 3"]
        metadatas = [
            {"name": "Game1"},
            {"name": "Game2"},
            {"name": "Game3"}
        ]
        embeddings = np.random.rand(3, 384).astype(np.float32)
        
        vector_store.add(ids, documents, metadatas, embeddings)
        
        # 获取指定ID的文档
        results = vector_store.get(ids=["game_1", "game_3"])
        
        assert len(results) == 2
        result_ids = [r["id"] for r in results]
        assert "game_1" in result_ids
        assert "game_3" in result_ids
    
    def test_empty_query(self, vector_store):
        """测试空库查询"""
        query_embedding = np.random.rand(1, 384).astype(np.float32)
        results = vector_store.query(query_embeddings=query_embedding, top_k=5)
        
        assert len(results) == 0
    
    def test_large_batch_add(self, vector_store):
        """测试大批量添加"""
        n_docs = 100
        ids = [f"game_{i}" for i in range(n_docs)]
        documents = [f"This is game {i}" for i in range(n_docs)]
        metadatas = [{"name": f"Game{i}", "index": i} for i in range(n_docs)]
        embeddings = np.random.rand(n_docs, 384).astype(np.float32)
        
        vector_store.add(ids, documents, metadatas, embeddings)
        
        assert vector_store.get_count() == n_docs
    
    def test_metadata_types(self, vector_store):
        """测试不同类型的metadata"""
        ids = ["game_1"]
        documents = ["Test game"]
        metadatas = [{
            "name": "TestGame",
            "price": 59.99,  # float
            "rating": 95,     # int
            "is_free": False, # bool
            "genres": "RPG,Action",  # string with comma
            "release_year": 2023  # int
        }]
        embeddings = np.random.rand(1, 384).astype(np.float32)
        
        vector_store.add(ids, documents, metadatas, embeddings)
        
        # 获取并验证metadata
        results = vector_store.get(ids=["game_1"])
        assert len(results) == 1
        meta = results[0]["metadata"]
        assert meta["name"] == "TestGame"
        assert meta["price"] == 59.99
        assert meta["rating"] == 95
        assert meta["is_free"] is False


class TestVectorStoreInterface:
    """测试向量存储接口"""
    
    def test_interface_methods(self):
        """验证接口定义了所有必需的方法"""
        required_methods = [
            'add',
            'query',
            'delete',
            'update',
            'get',
            'get_count',
            'persist'
        ]
        
        for method in required_methods:
            assert hasattr(VectorStore, method), f"VectorStore 缺少方法: {method}"


# 集成测试
class TestVectorStoreIntegration:
    """向量存储集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_end_to_end_workflow(self, temp_dir):
        """测试完整的工作流程"""
        # 1. 创建向量存储
        store = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="e2e_test"
        )
        
        # 2. 准备测试数据
        game_data = [
            {
                "id": "1",
                "doc": "Zelda: Open world adventure RPG",
                "meta": {"name": "Zelda", "price": 60, "genres": "Adventure,RPG"}
            },
            {
                "id": "2",
                "doc": "Elden Ring: Dark fantasy action RPG",
                "meta": {"name": "Elden Ring", "price": 50, "genres": "RPG,Action"}
            },
            {
                "id": "3",
                "doc": "Minecraft: Creative sandbox building",
                "meta": {"name": "Minecraft", "price": 30, "genres": "Sandbox"}
            }
        ]
        
        ids = [g["id"] for g in game_data]
        docs = [g["doc"] for g in game_data]
        metas = [g["meta"] for g in game_data]
        embeddings = np.random.rand(3, 384).astype(np.float32)
        
        # 3. 添加数据
        store.add(ids, docs, metas, embeddings)
        assert store.get_count() == 3
        
        # 4. 查询
        query_emb = np.random.rand(1, 384).astype(np.float32)
        results = store.query(query_embeddings=query_emb, top_k=2)
        assert len(results) == 2
        
        # 5. 过滤查询
        results = store.query(
            query_embeddings=query_emb,
            top_k=5,
            filter_dict={"price": {"$lte": 50}}
        )
        assert all(r["metadata"]["price"] <= 50 for r in results)
        
        # 6. 更新
        store.update(
            ids=["1"],
            documents=["Zelda: Updated description"],
            metadatas=[{"name": "Zelda", "price": 55, "genres": "Adventure,RPG"}],
            embeddings=np.random.rand(1, 384).astype(np.float32)
        )
        
        # 7. 删除
        store.delete(ids=["3"])
        assert store.get_count() == 2
        
        # 8. 持久化
        store.persist()
        
        # 9. 重新加载
        store2 = ChromaVectorStore(
            persist_directory=temp_dir,
            collection_name="e2e_test"
        )
        assert store2.get_count() == 2


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
