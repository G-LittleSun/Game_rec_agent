"""
文本清洗模块
处理游戏描述文本的HTML、URL、特殊字符清洗
"""
import re
from typing import Optional
from bs4 import BeautifulSoup


class TextCleaner:
    """文本清洗器"""
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_extra_spaces: bool = True,
        max_tokens: Optional[int] = None
    ):
        """
        初始化文本清洗器
        
        Args:
            remove_html: 是否移除HTML标签
            remove_urls: 是否移除URL链接
            remove_extra_spaces: 是否移除多余空格
            max_tokens: 最大token数(粗略按空格分词)
        """
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_extra_spaces = remove_extra_spaces
        self.max_tokens = max_tokens
    
    def clean(self, text: str) -> str:
        """
        清洗单个文本
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # 1. 移除HTML标签
        if self.remove_html:
            text = self._remove_html_tags(text)
        
        # 2. 移除URL
        if self.remove_urls:
            text = self._remove_urls(text)
        
        # 3. 移除特殊字符和控制字符
        text = self._remove_special_chars(text)
        
        # 4. 移除多余空格
        if self.remove_extra_spaces:
            text = self._remove_extra_spaces(text)
        
        # 5. 截断到最大token数
        if self.max_tokens:
            text = self._truncate_tokens(text, self.max_tokens)
        
        return text.strip()
    
    @staticmethod
    def _remove_html_tags(text: str) -> str:
        """移除HTML标签"""
        try:
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(text, "html.parser")
            # 提取纯文本
            text = soup.get_text(separator=" ")
        except Exception:
            # 如果BeautifulSoup失败,使用正则表达式
            text = re.sub(r'<[^>]+>', ' ', text)
        
        return text
    
    @staticmethod
    def _remove_urls(text: str) -> str:
        """移除URL链接"""
        # 移除http/https链接
        text = re.sub(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            '',
            text
        )
        # 移除www开头的链接
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', '', text)
        return text
    
    @staticmethod
    def _remove_special_chars(text: str) -> str:
        """移除特殊字符和控制字符"""
        # 移除控制字符
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
        # 保留基本标点,移除其他特殊符号
        text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)
        return text
    
    @staticmethod
    def _remove_extra_spaces(text: str) -> str:
        """移除多余空格"""
        # 将多个空格替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def _truncate_tokens(text: str, max_tokens: int) -> str:
        """截断到最大token数(粗略按空格分词)"""
        tokens = text.split()
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return ' '.join(tokens)


# 示例用法
if __name__ == "__main__":
    cleaner = TextCleaner(max_tokens=50)
    
    sample_text = """
    <h1>Game Title</h1>
    <p>This is a <b>great</b> game! Visit our website at http://example.com for more info.</p>
    <br><br>
    Features:
    - Open World
    - Multiplayer    Support
    """
    
    cleaned = cleaner.clean(sample_text)
    print("Original:", repr(sample_text))
    print("\nCleaned:", cleaned)
