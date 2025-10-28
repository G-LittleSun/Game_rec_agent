"""
Ollama LLM 客户端
支持本地部署的大语言模型调用
"""
import requests
import json
from typing import Optional, Dict, Any, List, Generator
from src.utils.logger import setup_logger


class OllamaClient:
    """Ollama客户端类"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "deepseek-r1:latest",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        timeout: int = 120
    ):
        """
        初始化Ollama客户端
        
        Args:
            base_url: Ollama服务地址
            model: 模型名称
            temperature: 温度参数(0-1),控制随机性
            top_p: nucleus sampling参数
            max_tokens: 最大生成token数
            timeout: 请求超时时间(秒)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.logger = setup_logger("OllamaClient")
        
        # 检查服务是否可用
        self._check_connection()
    
    def _check_connection(self):
        """检查Ollama服务连接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            self.logger.info(f"✅ Ollama服务连接成功: {self.base_url}")
        except Exception as e:
            self.logger.warning(f"⚠️ Ollama服务连接失败: {e}")
            self.logger.warning("请确保Ollama服务已启动: ollama serve")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        对话接口
        
        Args:
            messages: 消息列表,格式: [{"role": "user", "content": "..."}]
            stream: 是否流式返回
            **kwargs: 其他参数(temperature, max_tokens等)
        
        Returns:
            模型回复文本
        """
        url = f"{self.base_url}/api/chat"
        
        # 合并参数
        params = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        try:
            if stream:
                return self._chat_stream(url, params)
            else:
                return self._chat_normal(url, params)
        except Exception as e:
            self.logger.error(f"❌ Ollama对话失败: {e}")
            raise
    
    def _chat_normal(self, url: str, params: Dict[str, Any]) -> str:
        """普通模式(非流式)"""
        response = requests.post(
            url,
            json=params,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        return result["message"]["content"]
    
    def _chat_stream(self, url: str, params: Dict[str, Any]) -> Generator[str, None, None]:
        """流式模式"""
        response = requests.post(
            url,
            json=params,
            stream=True,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if not data.get("done"):
                    yield data["message"]["content"]
    
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        文本生成接口(简化版)
        
        Args:
            prompt: 用户输入
            system: 系统提示词
            stream: 是否流式返回
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(messages, stream=stream, **kwargs)
    
    def list_models(self) -> List[str]:
        """列出可用的模型"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except Exception as e:
            self.logger.error(f"❌ 获取模型列表失败: {e}")
            return []
    
    def switch_model(self, model: str):
        """切换模型"""
        available_models = self.list_models()
        if model not in available_models:
            raise ValueError(f"模型 {model} 不可用,可用模型: {available_models}")
        
        self.model = model
        self.logger.info(f"✅ 已切换到模型: {model}")


# 示例用法
if __name__ == "__main__":
    # 创建客户端
    client = OllamaClient(model="deepseek-r1:latest")
    
    # 列出可用模型
    print("可用模型:", client.list_models())
    
    # 简单对话
    response = client.generate(
        prompt="请推荐3款开放世界RPG游戏",
        system="你是一个专业的游戏推荐助手"
    )
    print("回复:", response)
    
    # 流式对话
    print("\n流式回复:")
    for chunk in client.generate(
        prompt="介绍一下《塞尔达传说:旷野之息》",
        stream=True
    ):
        print(chunk, end="", flush=True)