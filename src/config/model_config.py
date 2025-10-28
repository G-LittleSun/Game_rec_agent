"""
模型配置加载器
从 YAML 文件加载配置,支持环境变量替换
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class ModelConfig:
    """模型配置管理类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径,默认为项目根目录的 config/models.yaml
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "models.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 替换环境变量
        config = self._replace_env_vars(config)
        return config
    
    def _replace_env_vars(self, config: Any) -> Any:
        """递归替换配置中的环境变量"""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            # 提取环境变量名
            env_var = config[2:-1]
            return os.getenv(env_var, "")
        else:
            return config
    
    @property
    def llm(self) -> Dict[str, Any]:
        """获取LLM配置"""
        return self._config.get("llm", {})
    
    @property
    def embedding(self) -> Dict[str, Any]:
        """获取Embedding配置"""
        return self._config.get("embedding", {})
    
    @property
    def rerank(self) -> Dict[str, Any]:
        """获取Rerank配置"""
        return self._config.get("rerank", {})
    
    @property
    def vectordb(self) -> Dict[str, Any]:
        """获取向量数据库配置"""
        return self._config.get("vectordb", {})
    
    @property
    def logging(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self._config.get("logging", {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取指定配置项"""
        return self._config.get(key, default)
    
    def reload(self):
        """重新加载配置"""
        self._config = self._load_config()


# 全局配置实例
_global_config = None


def get_config(config_path: str = None) -> ModelConfig:
    """
    获取全局配置实例(单例模式)
    
    Args:
        config_path: 配置文件路径,仅首次调用时有效
    
    Returns:
        ModelConfig实例
    """
    global _global_config
    if _global_config is None:
        _global_config = ModelConfig(config_path)
    return _global_config