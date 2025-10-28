"""
特征工程模块
计算评分、热度、质量等衍生特征
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats


class FeatureEngineer:
    """特征工程器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化特征工程器
        
        Args:
            config: 特征工程配置(来自vectorization.yaml)
        """
        self.config = config or {}
    
    def compute_rating(
        self,
        positive: pd.Series,
        negative: pd.Series,
        method: str = "wilson_score",
        confidence: float = 0.95
    ) -> pd.Series:
        """
        计算评分
        
        Args:
            positive: 正面评价数
            negative: 负面评价数
            method: 计算方法 (simple, wilson_score, bayesian)
            confidence: 置信度(用于wilson_score)
        
        Returns:
            评分Series (0-1之间)
        """
        if method == "simple":
            # 简单比例
            total = positive + negative
            return (positive / total).fillna(0)
        
        elif method == "wilson_score":
            # Wilson Score置信区间下界
            # 考虑了评价数量,评价少的游戏分数会被拉低
            return self._wilson_score(positive, negative, confidence)
        
        elif method == "bayesian":
            # 贝叶斯平均
            # 假设先验平均分为0.7,先验样本数为10
            prior_rating = 0.7
            prior_count = 10
            total = positive + negative
            return ((positive + prior_rating * prior_count) / 
                   (total + prior_count))
        
        else:
            raise ValueError(f"Unknown rating method: {method}")
    
    @staticmethod
    def _wilson_score(
        positive: pd.Series,
        negative: pd.Series,
        confidence: float = 0.95
    ) -> pd.Series:
        """
        Wilson Score置信区间下界
        
        参考: https://www.evanmiller.org/how-not-to-sort-by-average-rating.html
        """
        n = positive + negative
        
        # 避免除零
        n = n.replace(0, 1)
        
        p_hat = positive / n  # 样本比例
        z = stats.norm.ppf(1 - (1 - confidence) / 2)  # 置信度对应的z值
        
        # Wilson Score公式
        denominator = 1 + z**2 / n
        center = p_hat + z**2 / (2 * n)
        spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
        
        lower_bound = (center - spread) / denominator
        
        return lower_bound.fillna(0).clip(0, 1)
    
    def compute_popularity_score(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        计算热度评分
        
        Args:
            df: 数据框
            weights: 各字段权重
        
        Returns:
            热度评分Series (0-1之间)
        """
        if weights is None:
            weights = {
                'Estimated owners': 0.4,
                'Recommendations': 0.3,
                'Peak CCU': 0.2,
                'Average playtime forever': 0.1
            }
        
        # 提取并归一化各项指标
        scores = []
        
        # 处理Estimated owners(字符串格式: "20000 - 50000")
        if 'Estimated owners' in df.columns and weights.get('Estimated owners', 0) > 0:
            owners = self._parse_owners(df['Estimated owners'])
            owners_norm = self._normalize(owners, method='log')
            scores.append(owners_norm * weights.get('Estimated owners', 0))
        
        # 处理其他数值字段
        for field, weight in weights.items():
            if field == 'Estimated owners':
                continue  # 已处理
            
            if field in df.columns and weight > 0:
                values = pd.to_numeric(df[field], errors='coerce').fillna(0)
                values_norm = self._normalize(values, method='log')
                scores.append(values_norm * weight)
        
        # 加权求和
        if scores:
            popularity = sum(scores)
            # 再次归一化到0-1
            return self._normalize(popularity, method='minmax')
        else:
            return pd.Series(0, index=df.index)
    
    @staticmethod
    def _parse_owners(owners_series: pd.Series) -> pd.Series:
        """
        解析Estimated owners字符串
        
        "20000 - 50000" -> 取中间值35000
        """
        def parse_single(s):
            if pd.isna(s) or s == '0 - 0':
                return 0
            try:
                # 分割并取均值
                parts = str(s).split('-')
                if len(parts) == 2:
                    low = float(parts[0].strip().replace(',', ''))
                    high = float(parts[1].strip().replace(',', ''))
                    return (low + high) / 2
                return 0
            except:
                return 0
        
        return owners_series.apply(parse_single)
    
    @staticmethod
    def _normalize(
        series: pd.Series,
        method: str = 'minmax'
    ) -> pd.Series:
        """
        归一化到0-1
        
        Args:
            series: 输入序列
            method: 归一化方法 (minmax, zscore, log)
        """
        if method == 'minmax':
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
            return (series - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            # Z-score标准化后映射到0-1
            mean = series.mean()
            std = series.std()
            if std == 0:
                return pd.Series(0.5, index=series.index)
            zscore = (series - mean) / std
            # 3-sigma规则,映射到0-1
            return ((zscore + 3) / 6).clip(0, 1)
        
        elif method == 'log':
            # 对数归一化(适合长尾分布)
            log_series = np.log1p(series)  # log(1+x)避免log(0)
            return FeatureEngineer._normalize(log_series, method='minmax')
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def compute_quality_score(
        self,
        df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        计算质量评分
        
        Args:
            df: 数据框
            weights: 各字段权重
        
        Returns:
            质量评分Series (0-1之间)
        """
        if weights is None:
            weights = {
                'user_rating': 0.5,
                'metacritic': 0.3,
                'playtime': 0.2
            }
        
        scores = []
        
        # 用户评分(需要先计算)
        if 'final_rating' in df.columns:
            scores.append(df['final_rating'] * weights.get('user_rating', 0))
        
        # Metacritic评分(0-100,归一化到0-1)
        if 'Metacritic score' in df.columns:
            meta_norm = pd.to_numeric(df['Metacritic score'], errors='coerce').fillna(0) / 100
            scores.append(meta_norm * weights.get('metacritic', 0))
        
        # 游戏时长(反映可玩性)
        if 'Average playtime forever' in df.columns:
            playtime = pd.to_numeric(df['Average playtime forever'], errors='coerce').fillna(0)
            playtime_norm = self._normalize(playtime, method='log')
            scores.append(playtime_norm * weights.get('playtime', 0))
        
        if scores:
            quality = sum(scores)
            return quality.clip(0, 1)
        else:
            return pd.Series(0, index=df.index)


# 示例用法
if __name__ == "__main__":
    # 创建测试数据
    test_data = pd.DataFrame({
        'Positive': [1000, 500, 100, 10],
        'Negative': [100, 500, 900, 90],
        'Estimated owners': ['10000 - 20000', '50000 - 100000', '1000 - 5000', '0 - 0'],
        'Recommendations': [500, 200, 50, 5],
        'Peak CCU': [10000, 5000, 1000, 100],
        'Average playtime forever': [3000, 1500, 500, 100],
        'Metacritic score': [85, 70, 60, 0]
    })
    
    engineer = FeatureEngineer()
    
    # 计算评分
    test_data['rating'] = engineer.compute_rating(
        test_data['Positive'],
        test_data['Negative'],
        method='wilson_score'
    )
    
    # 计算热度
    test_data['popularity'] = engineer.compute_popularity_score(test_data)
    
    print(test_data[['Positive', 'Negative', 'rating', 'popularity']])
