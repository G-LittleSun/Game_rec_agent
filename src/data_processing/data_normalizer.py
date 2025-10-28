"""
数据标准化模块

按字段契约校验、处理缺失值、类型转换和字段派生。
确保后续向量化模块输入的数据格式一致。

字段契约：
- 核心字段（必需）: AppID, Name, About the game, Genres, Tags, Categories
- 重要字段（强烈推荐）: Price, Positive, Negative, Recommendations, Release date
- 可选字段: Developers, Publishers, Estimated owners, Average playtime forever, 
  Metacritic score, Windows, Mac, Linux, Required age, Peak CCU, Supported languages等
- 派生字段: Release_year（从Release date提取）, Platforms（从Windows/Mac/Linux汇总）
- 忽略字段: Support url, Header image, Screenshots等对推荐无价值的字段
"""
import pandas as pd
import re
import warnings
from typing import Dict, Any, Optional, List, Tuple

# ========= 字段契约（数据源与标准化约定） =========
# 🔴 核心特征 - 用于语义检索和基本信息
CORE_FEATURES: List[str] = [
    'AppID',           # 唯一标识
    'Name',            # 游戏名称
    'About the game',  # 游戏描述（语义搜索核心）
    'Genres',          # 类型（主要玩法）
    'Tags',            # 标签（用户定义，维度丰富）
    'Categories',      # 分类（官方功能特性）
]

# 🟡 重要特征 - 用于排序和筛选
IMPORTANT_FEATURES: List[str] = [
    'Price',           # 价格筛选
    'Positive',        # 好评数
    'Negative',        # 差评数
    'Recommendations', # 推荐数
    'Release date',    # 发布时间
]

# 🟢 辅助特征 - 提供额外信息
OPTIONAL_FEATURES: List[str] = [
    'Developers',      # 开发商
    'Publishers',      # 发行商
    'Estimated owners',# 热度指标
    'Average playtime forever',  # 所有时间平均时长
    'Metacritic score',# 专业评分
    'Windows', 'Mac', 'Linux',  # 平台
    'Required age',    # 年龄限制
    'Peak CCU',        # 历史玩家在线峰值
    'Supported languages', # 支持语言
    'Average playtime two weeks', # 近两周平均游戏时长
    'Median playtime forever',    # 所有时间游戏时长中位数
    'Median playtime two weeks',  # 近两周游戏时长中位数
]

# 🔵 忽略特征 - 对推荐系统无价值
IGNORE_FEATURES: List[str] = [
    'Support url', 'Support email', 'Website',
    'Header image', 'Screenshots', 'Movies',
    'Metacritic url', 'Notes', 'Reviews',
    'User score', 'Score rank', 'DLC count',
    'Full audio languages', 'Achievements'
]

# 📊 派生字段（本模块自动生成）
DERIVED_FEATURES: List[str] = [
    'Release_year',  # 从 Release date 解析年份（YYYY 或 "Unknown"）
    'Platforms',     # 从 Windows/Mac/Linux 汇总（如 "Windows, Mac"）
]

# 🔢 数值字段（会转为数值类型，无法解析时置0）
NUMERIC_FIELDS: List[str] = [
    'Price', 'Positive', 'Negative', 'Recommendations',
    'Estimated owners', 'Required age', 'Peak CCU',
    'Average playtime forever', 'Average playtime two weeks',
    'Median playtime forever', 'Median playtime two weeks',
    'Metacritic score',
]

# 🎮 平台布尔字段
PLATFORM_BOOL_FIELDS: List[str] = ['Windows', 'Mac', 'Linux']


class DataNormalizer:
    """数据标准化器
    
    标准化后保证：
    - 核心/重要文本字段缺失有默认值
    - AppID 统一为字符串类型（用作向量库ID）
    - 平台字段为 bool，并派生 Platforms 汇总字符串
    - 从 Release date 派生 Release_year（YYYY 或 'Unknown'）
    - 所有数值字段转为数值类型，无法解析时置 0
    - 删除 IGNORE_FEATURES 中的无用字段
    
    使用示例：
        normalizer = DataNormalizer()
        df_clean = normalizer.normalize_dataframe(df_raw, strict=False)
        schema = normalizer.schema_info()  # 查看字段契约
    """
    
    def __init__(self, default_values: Optional[Dict[str, Any]] = None):
        """
        初始化数据标准化器
        
        Args:
            default_values: 自定义缺失值填充规则（默认使用标准规则）
        """
        self.default_values = default_values or {
            'Name': 'Unknown Game',
            'About the game': 'No description available.',
            'Genres': 'Unknown',
            'Tags': 'Unknown',
            'Categories': 'Unknown',
            'Release date': 'Unknown',
            'Developers': 'Unknown',
            'Publishers': 'Unknown',
        }
    
    def schema_info(self) -> Dict[str, Any]:
        """返回字段契约说明，便于文档化和调试
        
        Returns:
            包含所有字段分类的字典
        """
        return {
            "core_features": CORE_FEATURES,
            "important_features": IMPORTANT_FEATURES,
            "optional_features": OPTIONAL_FEATURES,
            "derived_features": DERIVED_FEATURES,
            "ignore_features": IGNORE_FEATURES,
            "normalized_numeric_fields": NUMERIC_FIELDS,
            "normalized_boolean_fields": PLATFORM_BOOL_FIELDS,
        }
    
    def normalize_dataframe(self, df: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
        """
        标准化整个数据框
        
        Args:
            df: 输入数据框
            strict: 是否严格校验（True时缺少核心字段会报错）
        
        Returns:
            标准化后的数据框
        """
        df = df.copy()
        
        # 0. 契约校验（strict=True 时缺少核心字段直接报错）
        self._validate_schema(df, strict=strict)
        
        # 1. 删除无用字段
        df = self._drop_ignored_features(df)
        
        # 2. 填充缺失值
        df = self._fill_missing_values(df)
        
        # 3. AppID 规范化为字符串（用作向量库ID）
        if 'AppID' in df.columns:
            df['AppID'] = df['AppID'].astype(str)
        
        # 4. 提取年份
        df = self._extract_year(df)
        
        # 5. 格式化平台信息（布尔化 + 派生 Platforms）
        df = self._format_platforms(df)
        
        # 6. 确保数值字段为数值类型
        df = self._ensure_numeric_types(df)
        
        return df
    
    
    # ========== 内部处理方法 ==========
    
    def _validate_schema(self, df: pd.DataFrame, strict: bool = False) -> Tuple[List[str], List[str]]:
        """检查字段是否符合契约
        
        Args:
            df: 输入数据框
            strict: 严格模式（缺少核心字段时抛出异常）
        
        Returns:
            (缺失的核心字段列表, 缺失的重要字段列表)
        """
        cols = set(df.columns)
        missing_core = [c for c in CORE_FEATURES if c not in cols]
        missing_important = [c for c in IMPORTANT_FEATURES if c not in cols]
        
        if missing_core:
            msg = f"❌ 缺少核心字段: {missing_core}"
            if strict:
                raise ValueError(msg)
            warnings.warn(msg)
        
        if missing_important:
            warnings.warn(f"⚠️  缺少重要字段（建议补齐以提升推荐质量）: {missing_important}")
        
        return missing_core, missing_important
    
    def _drop_ignored_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除对推荐无价值的字段"""
        drop_cols = [c for c in IGNORE_FEATURES if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """填充文本/分类字段的缺失值（包括空字符串）"""
        for col, default_val in self.default_values.items():
            if col in df.columns:
                # 先填充 NaN
                df[col] = df[col].fillna(default_val)
                # 再替换空字符串和纯空白字符串
                df[col] = df[col].replace('', default_val)
                df[col] = df[col].apply(
                    lambda x: default_val if isinstance(x, str) and x.strip() == '' else x
                )
        
        return df
    
    
    def _extract_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 Release date 提取年份
        
        支持格式：
        - "2020-01-15"
        - "Jan 15, 2020"
        - "2020"
        - 其他包含四位年份的字符串
        
        Returns:
            添加 Release_year 列的数据框（YYYY 或 "Unknown"）
        """
        if 'Release date' not in df.columns:
            df['Release_year'] = 'Unknown'
            return df
        
        def extract_year(date_str):
            if pd.isna(date_str):
                return 'Unknown'
            
            s = str(date_str).strip()
            if s.lower() == 'unknown':
                return 'Unknown'
            
            # 直接是4位年份
            if s.isdigit() and len(s) == 4:
                return s
            
            # 尝试 pandas 日期解析
            try:
                dt = pd.to_datetime(s, errors='raise')
                return str(dt.year)
            except Exception:
                # 正则提取年份
                match = re.search(r'\b(19|20)\d{2}\b', s)
                return match.group(0) if match else 'Unknown'
        
        df['Release_year'] = df['Release date'].apply(extract_year)
        return df
    
    
    def _coerce_bool_series(self, s: pd.Series) -> pd.Series:
        """将多种真值/假值表示统一转为 bool
        
        支持：True/False, 1/0, "yes"/"no", "true"/"false" 等
        缺失值视为 False
        """
        true_vals = {"1", "true", "t", "y", "yes"}
        false_vals = {"0", "false", "f", "n", "no", ""}
        
        def to_bool(v):
            if pd.isna(v):
                return False
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))
            
            v_str = str(v).strip().lower()
            if v_str in true_vals:
                return True
            if v_str in false_vals:
                return False
            # 其他未知值保守处理为 False
            return False
        
        return s.map(to_bool)
    
    def _parse_platforms_field(self, v) -> set:
        """从字符串/列表解析平台集合
        
        Args:
            v: 平台字段值（如 "Windows, Mac" 或 ['Windows', 'Mac']）
        
        Returns:
            平台集合（如 {'Windows', 'Mac'}）
        """
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return set()
        
        if isinstance(v, (list, tuple, set)):
            items = [str(x).strip() for x in v]
        else:
            # 去除可能的引号和方括号
            v_str = str(v).strip().strip("[]")
            # 按常见分隔符拆分
            items = re.split(r"[,\|;/]+", v_str)
            items = [x.strip(" '\"\t") for x in items]
        
        normalized = set()
        for x in items:
            xl = x.lower()
            if not xl:
                continue
            if "windows" in xl or xl == "win":
                normalized.add("Windows")
            elif "mac" in xl or "osx" in xl or "macos" in xl:
                normalized.add("Mac")
            elif "linux" in xl or "steamos" in xl:
                normalized.add("Linux")
        
        return normalized
    
    def _format_platforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化平台信息
        
        处理逻辑：
        1. 若存在 Windows/Mac/Linux 列：规范为 bool 并派生 Platforms
        2. 若不存在但有 Platforms/Platform 字符串列：解析后派生三个 bool 列
        3. 两者都没有：三个 bool 列置 False，Platforms 为 "Unknown"
        
        Returns:
            添加 Windows(bool), Mac(bool), Linux(bool), Platforms(str) 的数据框
        """
        platform_cols = PLATFORM_BOOL_FIELDS
        has_bool_cols = any(c in df.columns for c in platform_cols)
        
        if has_bool_cols:
            # 场景1：已有 Windows/Mac/Linux 列 → 布尔化并派生 Platforms
            for col in platform_cols:
                if col in df.columns:
                    df[col] = self._coerce_bool_series(df[col])
                else:
                    df[col] = False
            
            df['Platforms'] = df.apply(
                lambda row: ', '.join([p for p in platform_cols if bool(row[p])]) or 'Unknown',
                axis=1
            )
            return df
        
        # 场景2：没有三列，尝试从 Platforms/Platform 字符串解析
        source_col = None
        for c in ['Platforms', 'Platform']:
            if c in df.columns:
                source_col = c
                break
        
        if source_col:
            parsed_sets = df[source_col].apply(self._parse_platforms_field)
            # 派生三个布尔列
            for col in platform_cols:
                df[col] = parsed_sets.apply(lambda s: col in s)
            # 生成规范化 Platforms 字符串
            df['Platforms'] = parsed_sets.apply(
                lambda s: ', '.join(sorted(s)) if s else 'Unknown'
            )
        else:
            # 场景3：两者都没有 → 降级处理
            for col in platform_cols:
                df[col] = False
            df['Platforms'] = 'Unknown'
        
        return df
    
    
    def _ensure_numeric_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保数值字段为正确类型
        
        将所有数值字段转为数值类型，无法解析的置 0
        """
        for field in NUMERIC_FIELDS:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(0)
        
        return df


"""# ========== 测试和使用示例 ==========
if __name__ == "__main__":
    # 创建测试数据（模拟真实 Steam 数据）
    test_df = pd.DataFrame({
        'AppID': [1424640, 402890, 123456],
        'Name': ['余烬', 'Nyctophilia', None],
        'About the game': ['Anti-war adventure game...', 'Psychological thriller...', None],
        'Genres': ['Adventure,Casual,Indie,RPG', 'Adventure,Free To Play,Indie', 'Unknown'],
        'Tags': ['Sokoban,RPG,Puzzle', 'Free to Play,Indie', None],
        'Categories': ['Single-player', 'Single-player', None],
        'Release date': ['2020-10-03', '2015-09-23', None],
        'Windows': ['True', 'True', 'False'],
        'Mac': ['False', 'False', 'True'],
        'Linux': ['False', 'False', None],
        'Price': ['3.99', '0.0', None],
        'Positive': [5, 196, None],
        'Negative': [7, 106, None],
        'Recommendations': [0, 0, 100],
    })
    
    print("=" * 80)
    print("📊 数据标准化模块测试")
    print("=" * 80)
    
    # 创建标准化器
    normalizer = DataNormalizer()
    
    # 查看字段契约
    print("\n📋 字段契约:")
    schema = normalizer.schema_info()
    for key, value in schema.items():
        print(f"\n  {key}:")
        print(f"    {value}")
    
    print("\n" + "=" * 80)
    print("🔧 开始标准化...")
    print("=" * 80)
    
    # 标准化数据
    normalized_df = normalizer.normalize_dataframe(test_df, strict=False)
    
    print("\n✅ 标准化完成！")
    print("\n原始数据形状:", test_df.shape)
    print("标准化后形状:", normalized_df.shape)
    
    print("\n📌 核心字段示例:")
    display_cols = ['AppID', 'Name', 'Genres', 'Release_year', 'Platforms', 'Price', 'Positive']
    available_cols = [c for c in display_cols if c in normalized_df.columns]
    print(normalized_df[available_cols].to_string(index=False))
    
    print("\n📌 平台字段详情:")
    platform_cols = ['Windows', 'Mac', 'Linux', 'Platforms']
    available_platform_cols = [c for c in platform_cols if c in normalized_df.columns]
    print(normalized_df[available_platform_cols].to_string(index=False))
    
    print("\n" + "=" * 80)
    print("✨ 测试完成！数据已准备好用于向量化")
    print("=" * 80)"""
