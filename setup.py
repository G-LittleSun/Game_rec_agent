"""
游戏推荐系统安装配置
"""
from setuptools import setup, find_packages

setup(
    name="game-rec-agent",
    version="0.1.0",
    description="Steam游戏推荐智能Agent系统",
    author="Your Name",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "sentence-transformers>=2.2.0",
        "transformers>=4.30.0",
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "chromadb>=0.4.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ]
    },
)
