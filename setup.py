"""
Setup configuration for DBPS (Decentralised Bottleneck Prioritization Simulation).
"""

from setuptools import setup, find_packages

setup(
    name="dbps",
    version="1.0.0",
    description="Decentralised Bottleneck Prioritization Simulation - Traffic simulation with intelligent signal control",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "shapely>=2.0",
        "geopandas>=0.12",
        "networkx>=3.0",
        "sumolib>=1.16.0",
        "traci>=1.16.0",
        "xmltodict>=0.13.0",
        "alive-progress>=2.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pandas>=1.4.0",
        "scipy>=1.9.0",
        "streamlit>=1.28.0",
    ],
    entry_points={
        "console_scripts": [
            "dbps=src.gui:main",
        ],
    },
    python_requires=">=3.8",
    author="DBPS Development Team",
    author_email="dbps@example.com",
    url="https://github.com/your-org/dbps",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)