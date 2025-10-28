"""
数据标准化模块测试
测试 DataNormalizer 的各种场景和边界情况
"""
"""
# 运行所有测试
pytest tests/test_data_normalizer.py -v

# 运行特定测试类
pytest tests/test_data_normalizer.py::TestPlatformHandling -v

# 运行特定测试方法
pytest tests/test_data_normalizer.py::TestMissingValueHandling::test_fill_missing_name -v

# 显示测试覆盖率
pytest tests/test_data_normalizer.py --cov=src.data_processing.data_normalizer --cov-report=html
"""


import pytest
import pandas as pd
import numpy as np
from src.data_processing.data_normalizer import DataNormalizer


class TestDataNormalizerBasic:
    """基础功能测试"""

    @pytest.fixture
    def normalizer(self):
        """创建标准化器实例"""
        return DataNormalizer()

    def test_init(self, normalizer):
        """测试初始化"""
        assert normalizer is not None
        assert isinstance(normalizer.default_values, dict)
        assert 'Name' in normalizer.default_values

    def test_schema_info(self, normalizer):
        """测试字段契约信息获取"""
        info = normalizer.schema_info()
        
        assert 'core_features' in info
        assert 'important_features' in info
        assert 'optional_features' in info
        assert 'derived_features' in info
        assert 'ignore_features' in info
        
        # 验证核心字段
        assert 'AppID' in info['core_features']
        assert 'Name' in info['core_features']
        assert 'About the game' in info['core_features']
        
        # 验证派生字段
        assert 'Release_year' in info['derived_features']
        assert 'Platforms' in info['derived_features']


class TestMissingValueHandling:
    """缺失值处理测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_fill_missing_name(self, normalizer):
        """测试缺失游戏名称填充"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3'],
            'Name': ['Game A', None, np.nan],
            'About the game': ['Desc A', 'Desc B', 'Desc C']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Name'][0] == 'Game A'
        assert result['Name'][1] == 'Unknown Game'
        assert result['Name'][2] == 'Unknown Game'

    def test_fill_missing_description(self, normalizer):
        """测试缺失描述填充"""
        df = pd.DataFrame({
            'AppID': ['1', '2'],
            'Name': ['Game A', 'Game B'],
            'About the game': [None, '']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['About the game'][0] == 'No description available.'
        assert result['About the game'][1] != ''  # 空字符串应被填充

    def test_fill_missing_genres_tags(self, normalizer):
        """测试缺失类型和标签填充"""
        df = pd.DataFrame({
            'AppID': ['1'],
            'Name': ['Game A'],
            'About the game': ['Description'],
            'Genres': [None],
            'Tags': [np.nan],
            'Categories': ['']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Genres'][0] == 'Unknown'
        assert result['Tags'][0] == 'Unknown'


class TestNumericNormalization:
    """数值字段标准化测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_numeric_fields_conversion(self, normalizer):
        """测试数值字段转换"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3'],
            'Name': ['A', 'B', 'C'],
            'About the game': ['D1', 'D2', 'D3'],
            'Price': ['9.99', 'invalid', None],
            'Positive': [100, '200', 'abc'],
            'Negative': ['50', None, 75],
            'Recommendations': [10, 20, 30]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        # Price: 9.99, 0, 0
        assert result['Price'][0] == 9.99
        assert result['Price'][1] == 0.0  # 无法解析
        assert result['Price'][2] == 0.0  # 缺失
        
        # Positive: 100, 200, 0
        assert result['Positive'][0] == 100
        assert result['Positive'][1] == 200
        assert result['Positive'][2] == 0.0  # 无法解析
        
        # 验证类型都是数值
        assert pd.api.types.is_numeric_dtype(result['Price'])
        assert pd.api.types.is_numeric_dtype(result['Positive'])

    def test_large_numbers(self, normalizer):
        """测试大数值处理"""
        df = pd.DataFrame({
            'AppID': ['1'],
            'Name': ['Game'],
            'About the game': ['Desc'],
            'Peak CCU': ['1234567'],
            'Positive': [9999999],
            'Estimated owners': ['1000000 - 2000000']  # 带范围的字符串
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Peak CCU'][0] == 1234567
        assert result['Positive'][0] == 9999999
        # Estimated owners 可能无法直接解析
        assert pd.api.types.is_numeric_dtype(result['Estimated owners'])

    def test_negative_numbers(self, normalizer):
        """测试负数处理"""
        df = pd.DataFrame({
            'AppID': ['1'],
            'Name': ['Game'],
            'About the game': ['Desc'],
            'Price': [-5.99],  # 理论上不应该有负价格
            'Required age': [-1]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        # 数值化不应改变负数（除非有额外逻辑）
        assert result['Price'][0] == -5.99


class TestPlatformHandling:
    """平台字段处理测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_boolean_platform_fields(self, normalizer):
        """测试布尔型平台字段"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3'],
            'Name': ['A', 'B', 'C'],
            'About the game': ['D1', 'D2', 'D3'],
            'Windows': [True, 'True', 1],
            'Mac': [False, 'False', 0],
            'Linux': [True, 'yes', None]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        # 验证布尔类型
        assert result['Windows'].dtype == bool
        assert result['Mac'].dtype == bool
        assert result['Linux'].dtype == bool
        
        # 验证值
        assert result['Windows'][0] == True
        assert result['Windows'][1] == True
        assert result['Windows'][2] == True
        
        assert result['Mac'][0] == False
        assert result['Mac'][1] == False
        
        assert result['Linux'][0] == True
        assert result['Linux'][1] == True
        assert result['Linux'][2] == False  # None -> False

    def test_platforms_string_generation(self, normalizer):
        """测试 Platforms 字符串生成"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3', '4'],
            'Name': ['A', 'B', 'C', 'D'],
            'About the game': ['D1', 'D2', 'D3', 'D4'],
            'Windows': [True, True, False, False],
            'Mac': [True, False, True, False],
            'Linux': [True, False, False, False]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Platforms'][0] == 'Windows, Mac, Linux'
        assert result['Platforms'][1] == 'Windows'
        assert result['Platforms'][2] == 'Mac'
        assert result['Platforms'][3] == 'Unknown'

    def test_parse_platforms_from_string(self, normalizer):
        """测试从字符串解析平台（无布尔列的情况）"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3'],
            'Name': ['A', 'B', 'C'],
            'About the game': ['D1', 'D2', 'D3'],
            'Platforms': ['Windows, Mac', 'Linux', 'Windows']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        # 应派生出布尔列
        assert 'Windows' in result.columns
        assert 'Mac' in result.columns
        assert 'Linux' in result.columns
        
        # 验证解析结果
        assert result['Windows'][0] == True
        assert result['Mac'][0] == True
        assert result['Linux'][0] == False
        
        assert result['Windows'][1] == False
        assert result['Linux'][1] == True
        
        # Platforms 应规范化
        assert 'Mac' in result['Platforms'][0]
        assert 'Windows' in result['Platforms'][0]

    def test_no_platform_info(self, normalizer):
        """测试无平台信息的情况"""
        df = pd.DataFrame({
            'AppID': ['1'],
            'Name': ['Game'],
            'About the game': ['Desc']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        # 应创建默认平台字段
        assert result['Windows'][0] == False
        assert result['Mac'][0] == False
        assert result['Linux'][0] == False
        assert result['Platforms'][0] == 'Unknown'


class TestReleaseYearExtraction:
    """发布年份提取测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_standard_date_formats(self, normalizer):
        """测试标准日期格式"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3', '4'],
            'Name': ['A', 'B', 'C', 'D'],
            'About the game': ['D1', 'D2', 'D3', 'D4'],
            'Release date': [
                '2020-10-03',
                '2015-09-23',
                'Oct 3, 2020',
                '2019/12/25'
            ]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Release_year'][0] == '2020'
        assert result['Release_year'][1] == '2015'
        assert result['Release_year'][2] == '2020'
        assert result['Release_year'][3] == '2019'

    def test_year_only(self, normalizer):
        """测试只有年份的情况"""
        df = pd.DataFrame({
            'AppID': ['1', '2'],
            'Name': ['A', 'B'],
            'About the game': ['D1', 'D2'],
            'Release date': ['2020', '2015']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Release_year'][0] == '2020'
        assert result['Release_year'][1] == '2015'

    def test_complex_date_strings(self, normalizer):
        """测试复杂日期字符串"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3'],
            'Name': ['A', 'B', 'C'],
            'About the game': ['D1', 'D2', 'D3'],
            'Release date': [
                'Early Access: 2020',
                'Coming Soon 2023',
                'Released in 2019'
            ]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Release_year'][0] == '2020'
        assert result['Release_year'][1] == '2023'
        assert result['Release_year'][2] == '2019'

    def test_missing_or_invalid_dates(self, normalizer):
        """测试缺失或无效日期"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3', '4'],
            'Name': ['A', 'B', 'C', 'D'],
            'About the game': ['D1', 'D2', 'D3', 'D4'],
            'Release date': [None, 'Unknown', 'TBA', 'Invalid']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Release_year'][0] == 'Unknown'
        assert result['Release_year'][1] == 'Unknown'
        assert result['Release_year'][2] == 'Unknown'
        assert result['Release_year'][3] == 'Unknown'

    def test_no_release_date_column(self, normalizer):
        """测试无发布日期列的情况"""
        df = pd.DataFrame({
            'AppID': ['1'],
            'Name': ['Game'],
            'About the game': ['Desc']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert 'Release_year' in result.columns
        assert result['Release_year'][0] == 'Unknown'


class TestAppIDHandling:
    """AppID 处理测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_appid_string_conversion(self, normalizer):
        """测试 AppID 转换为字符串"""
        df = pd.DataFrame({
            'AppID': [1424640, 402890, 123456],
            'Name': ['A', 'B', 'C'],
            'About the game': ['D1', 'D2', 'D3']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['AppID'].dtype == object  # 字符串类型
        assert result['AppID'][0] == '1424640'
        assert result['AppID'][1] == '402890'

    def test_appid_already_string(self, normalizer):
        """测试 AppID 已经是字符串"""
        df = pd.DataFrame({
            'AppID': ['1424640', '402890'],
            'Name': ['A', 'B'],
            'About the game': ['D1', 'D2']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['AppID'][0] == '1424640'
        assert result['AppID'][1] == '402890'


class TestIgnoredFields:
    """忽略字段测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_drop_ignored_fields(self, normalizer):
        """测试删除忽略字段"""
        df = pd.DataFrame({
            'AppID': ['1'],
            'Name': ['Game'],
            'About the game': ['Desc'],
            'Support url': ['http://support.example.com'],
            'Screenshots': ['image1.jpg,image2.jpg'],
            'Achievements': [50],
            'Notes': ['Some notes']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        # 应删除忽略字段
        assert 'Support url' not in result.columns
        assert 'Screenshots' not in result.columns
        assert 'Achievements' not in result.columns
        assert 'Notes' not in result.columns
        
        # 核心字段应保留
        assert 'AppID' in result.columns
        assert 'Name' in result.columns


class TestCompleteWorkflow:
    """完整工作流测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_real_world_example(self, normalizer):
        """测试真实世界数据示例"""
        df = pd.DataFrame({
            'AppID': [1424640, 402890],
            'Name': ['余烬', 'Nyctophilia'],
            'About the game': [
                "'Ashes of war' is an anti war theme adventure...",
                "NYCTOPHILIA is a 2D psychological thriller..."
            ],
            'Genres': ['Adventure,Casual,Indie,RPG', 'Adventure,Free To Play,Indie'],
            'Tags': [
                'Sokoban,RPG,Puzzle-Platformer,Exploration',
                'Free to Play,Indie,Adventure,Horror'
            ],
            'Categories': ['Single-player,Family Sharing', 'Single-player'],
            'Price': [3.99, 0.0],
            'Positive': [5, 196],
            'Negative': [7, 106],
            'Recommendations': [0, 0],
            'Release date': ['2020-10-03', '2015-09-23'],
            'Developers': ['宁夏华夏西部影视城有限公司', 'Cat In A Jar Games'],
            'Publishers': ['宁夏华夏西部影视城有限公司', 'Cat In A Jar Games'],
            'Estimated owners': ['20000 - 50000', '50000 - 100000'],
            'Average playtime forever': [0, 0],
            'Metacritic score': [0, 0],
            'Windows': [True, True],
            'Mac': [False, False],
            'Linux': [False, False],
            'Required age': [0, 0],
            'Peak CCU': [0, 0],
            'Supported languages': [
                "['Simplified Chinese']",
                "['English', 'Russian']"
            ]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        # 验证基础字段
        assert len(result) == 2
        assert result['AppID'][0] == '1424640'
        assert result['Name'][0] == '余烬'
        
        # 验证数值字段
        assert result['Price'][0] == 3.99
        assert result['Positive'][1] == 196
        
        # 验证平台
        assert result['Windows'][0] == True
        assert result['Platforms'][0] == 'Windows'
        
        # 验证年份提取
        assert result['Release_year'][0] == '2020'
        assert result['Release_year'][1] == '2015'

    def test_empty_dataframe(self, normalizer):
        """测试空数据框"""
        df = pd.DataFrame()
        
        result = normalizer.normalize_dataframe(df)
        
        assert len(result) == 0

    def test_single_row(self, normalizer):
        """测试单行数据"""
        df = pd.DataFrame({
            'AppID': ['123'],
            'Name': ['Test Game'],
            'About the game': ['A test game description']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert len(result) == 1
        assert result['Name'][0] == 'Test Game'


class TestValidation:
    """数据验证测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_missing_core_fields_non_strict(self, normalizer):
        """测试缺少核心字段（非严格模式）"""
        df = pd.DataFrame({
            'AppID': ['1'],
            'Price': [9.99]
        })
        
        # 应该发出警告但不报错
        with pytest.warns(UserWarning):
            result = normalizer.normalize_dataframe(df, strict=False)
        
        assert len(result) == 1

    def test_missing_core_fields_strict(self, normalizer):
        """测试缺少核心字段（严格模式）"""
        df = pd.DataFrame({
            'AppID': ['1'],
            'Price': [9.99]
        })
        
        # 应该抛出 ValueError
        with pytest.raises(ValueError, match="缺少核心字段"):
            normalizer.normalize_dataframe(df, strict=True)


class TestEdgeCases:
    """边界情况测试"""

    @pytest.fixture
    def normalizer(self):
        return DataNormalizer()

    def test_all_missing_values(self, normalizer):
        """测试全部缺失值"""
        df = pd.DataFrame({
            'AppID': ['1', '2'],
            'Name': [None, None],
            'About the game': [None, None],
            'Price': [None, None]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Name'][0] == 'Unknown Game'
        assert result['About the game'][0] == 'No description available.'
        assert result['Price'][0] == 0.0

    def test_unicode_and_special_characters(self, normalizer):
        """测试 Unicode 和特殊字符"""
        df = pd.DataFrame({
            'AppID': ['1', '2'],
            'Name': ['游戏名称 🎮', 'Jeu français'],
            'About the game': [
                '这是一个包含特殊字符的描述: @#$%^&*()',
                'Description avec des caractères spéciaux: éàü'
            ],
            'Genres': ['动作,冒险', 'Action,Aventure']
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert result['Name'][0] == '游戏名称 🎮'
        assert '特殊字符' in result['About the game'][0]

    def test_very_long_strings(self, normalizer):
        """测试超长字符串"""
        long_desc = 'A' * 10000
        df = pd.DataFrame({
            'AppID': ['1'],
            'Name': ['Game'],
            'About the game': [long_desc]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        assert len(result['About the game'][0]) == 10000

    def test_mixed_data_types(self, normalizer):
        """测试混合数据类型"""
        df = pd.DataFrame({
            'AppID': ['1', '2', '3'],
            'Name': ['Game A', 123, None],  # 混合字符串、数字、None
            'About the game': ['Desc', 456, np.nan],
            'Price': ['9.99', 10, None]
        })
        
        result = normalizer.normalize_dataframe(df)
        
        # 应该能处理混合类型
        assert len(result) == 3
        assert result['Price'][1] == 10.0


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
