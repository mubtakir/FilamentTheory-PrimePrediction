#!/usr/bin/env python3
"""
إعداد مشروع FilamentPrime
=========================

ملف الإعداد لتثبيت مكتبة FilamentPrime

تطوير: د. باسل يحيى عبدالله
"""

from setuptools import setup, find_packages
import os

# قراءة ملف README
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "FilamentPrime: نظام التنبؤ المتكامل للأعداد الأولية"

# قراءة المتطلبات
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "matplotlib>=3.4.0",
            "scikit-learn>=1.0.0",
            "sympy>=1.8.0",
            "joblib>=1.0.0"
        ]

setup(
    name="FilamentPrime",
    version="1.0.0",
    author="د. باسل يحيى عبدالله",
    author_email="basel.yahya@example.com",
    description="نظام التنبؤ المتكامل للأعداد الأولية باستخدام نظرية الفتائل",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/FilamentPrime",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
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
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "plotting": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "performance": [
            "numba>=0.54.0",
            "cython>=0.29.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "filamentprime-demo=FilamentPrime.run_demo:main",
            "filamentprime-test=FilamentPrime.test_simple:main",
        ],
    },
    include_package_data=True,
    package_data={
        "FilamentPrime": [
            "data/*.txt",
            "data/trained_models/*.pkl",
            "examples/*.py",
            "docs/*.md",
        ],
    },
    keywords=[
        "prime numbers",
        "riemann zeta",
        "number theory",
        "quantum chaos",
        "filament theory",
        "mathematical physics",
        "prediction",
        "machine learning"
    ],
    project_urls={
        "Documentation": "https://github.com/username/FilamentPrime/docs",
        "Source": "https://github.com/username/FilamentPrime",
        "Tracker": "https://github.com/username/FilamentPrime/issues",
    },
)
