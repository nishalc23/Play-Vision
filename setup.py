#!/usr/bin/env python3
"""
Setup configuration for Sports Video Highlight Generator
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sports-highlight-generator",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@domain.com",
    description="AI-powered sports video highlight generator using computer vision",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sports-highlight-generator",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Video :: Display",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "pytest-cov>=4.0.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "torchvision[cuda]>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sports-highlights=src.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src": ["config/*.py"],
    },
    keywords="sports video highlights ai computer-vision yolo basketball soccer",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/sports-highlight-generator/issues",
        "Source": "https://github.com/yourusername/sports-highlight-generator",
        "Documentation": "https://github.com/yourusername/sports-highlight-generator/docs",
    },
)
