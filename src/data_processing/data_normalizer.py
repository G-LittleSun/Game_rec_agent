"""
数据标准化模块
处理缺失值填充、数据类型转换等
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime


class DataNormalizer:
    """数据标准化器"""
    
    def __init__(self, default_values: Optional[Dict[str, Any]] = None):
        """
        初始化数据标准化器
        
        Args:
            default_values: 缺失值的默认填充值
        """
        self.default_values = default_values or {
            'Name': 'Unknown Game',
            'About the game': 'No description available.',
            'Genres': 'Unknown',
            'Tags': 'Unknown',
            'Categories': 'Unknown',
            'Release date': 'Unknown'
        }
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化整个数据框
        
        Args:
            df: 输入数据框
        
        Returns:
            标准化后的数据框
        """
        df = df.copy()
        
        # 1. 填充缺失值
        df = self._fill_missing_values(df)
        
        # 2. 提取年份
        df = self._extract_year(df)
        
        # 3. 格式化平台信息
        df = self._format_platforms(df)
        
        # 4. 确保数值字段为数值类型
        df = self._ensure_numeric_types(df)
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """填充缺失值"""
        for col, default_val in self.default_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
        
        return df
    
    def _extract_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """从Release date提取年份"""
        if 'Release date' not in df.columns:
            df['Release_year'] = 'Unknown'
            return df
        
        def extract_year(date_str):
            if pd.isna(date_str) or str(date_str) == 'Unknown':
                return 'Unknown'
            
            try:
                # 尝试解析日期
                if isinstance(date_str, str):
                    # 常见格式: "Jan 1, 2020", "2020-01-01", "2020"
                    date_str = str(date_str).strip()
                    
                    # 如果已经是年份格式
                    if date_str.isdigit() and len(date_str) == 4:
                        return date_str
                    
                    # 尝试pandas解析
                    try:
                        dt = pd.to_datetime(date_str)
                        return str(dt.year)
                    except:
                        # 尝试提取4位数字(年份)
                        import re
                        match = re.search(r'\b(19|20)\d{2}\b', date_str)
                        if match:
                            return match.group(0)
                
                return 'Unknown'
            except:
                return 'Unknown'
        
        df['Release_year'] = df['Release date'].apply(extract_year)
        return df
    
    def _format_platforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化平台信息为字符串"""
        platforms_list = []
        
        for platform in ['Windows', 'Mac', 'Linux']:
            if platform not in df.columns:
                continue
            
            # 确保为布尔型
            df[platform] = df[platform].fillna(False).astype(bool)
        
        # 创建平台字符串
        def get_platforms(row):
            platforms = []
            for p in ['Windows', 'Mac', 'Linux']:
                if p in row and row[p]:
                    platforms.append(p)
            
            return ', '.join(platforms) if platforms else 'Unknown'
        
        df['Platforms'] = df.apply(get_platforms, axis=1)
        return df
    
    def _ensure_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保数值字段为正确类型"""
        numeric_fields = [
            'Price', 'Positive', 'Negative', 'Recommendations',
            'Peak CCU', 'Average playtime forever',
            'Median playtime forever', 'Metacritic score'
        ]
        
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
        
        return df


# 示例用法
if __name__ == "__main__":
    # 创建测试数据
    test_df = pd.DataFrame({
        'Name': ['Game A', None, 'Game C'],
        'About the game': ['Description A', None, 'Description C'],
        'Release date': ['Jan 15, 2020', '2019', None],
        'Windows': [True, False, None],
        'Mac': [False, True, True],
        'Linux': [False, False, None],
        'Price': ['19.99', '29.99', None]
    })
    
    normalizer = DataNormalizer()
    normalized = normalizer.normalize_dataframe(test_df)
    
    print("原始数据:")
    print(test_df)
    print("\n标准化后:")
    print(normalized[['Name', 'Release_year', 'Platforms', 'Price']])
