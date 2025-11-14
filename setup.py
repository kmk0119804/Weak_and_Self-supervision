"""
Weak and Self-Supervision Model Setup
"""

from pathlib import Path
from setuptools import find_packages, setup

# Read requirements
HERE = Path(__file__).parent
with open(HERE / "requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip() 
        for line in f 
        if line.strip() and not line.startswith("#")
    ]

# Read README
with open(HERE / "README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="weak-self-supervision",
    version="0.1.0",
    author="Manguy",
    description="Weak and Self-Supervision Model using YOLOv8 and SAM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kmk0119804/Minkyu-KOO",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)