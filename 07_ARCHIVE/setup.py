#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
إعداد مشروع الأعداد الأولية المتقدم
Advanced Prime Numbers Project Setup

ملف الإعداد والتثبيت لأفكار الباحث العلمي باسل يحيى عبدالله
Setup and installation file for researcher Basel Yahya Abdullah's ideas

الباحث العلمي: باسل يحيى عبدالله (Basel Yahya Abdullah)
المطور: مبتكر (Mubtakir)
التاريخ: 2025
"""

from setuptools import setup, find_packages
import os

# قراءة ملف README
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# قراءة متطلبات المشروع
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh.readlines() 
                          if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="advanced-prime-numbers",
    version="1.0.0",
    author="Basel Yahya Abdullah (Researcher), Mubtakir (Developer)",
    author_email="mubtakir@example.com",
    description="تطبيق لأفكار الباحث العلمي باسل يحيى عبدالله في الأعداد الأولية - Implementation of Basel Yahya Abdullah's Prime Numbers Research",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mubtakir/advanced-prime-numbers",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Natural Language :: Arabic",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "performance": [
            "numba>=0.54.0",
            "cython>=0.29.0",
        ],
        "visualization": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prime-demo=run_demo:main",
            "prime-gui=interactive_prime_explorer:main",
            "prime-analysis=mathematical_analysis:main",
            "prime-sieve=advanced_prime_algorithm:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.pkl"],
    },
    keywords=[
        "prime numbers",
        "mathematics",
        "number theory",
        "riemann hypothesis",
        "sieve of eratosthenes",
        "hilbert polya",
        "machine learning",
        "الأعداد الأولية",
        "نظرية الأعداد",
        "فرضية ريمان",
    ],
    project_urls={
        "Bug Reports": "https://github.com/mubtakir/advanced-prime-numbers/issues",
        "Source": "https://github.com/mubtakir/advanced-prime-numbers",
        "Documentation": "https://advanced-prime-numbers.readthedocs.io/",
    },
)
