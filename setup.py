"""
RAG-MCP Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="rag-mcp",
    version="0.1.0",
    description="Retrieval-Augmented Generation for Model Context Protocol tool selection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.9.0",
        "redis>=4.0.0",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
)