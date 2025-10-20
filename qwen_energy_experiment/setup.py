"""Setup script for the Qwen energy consumption experiment."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="qwen-energy-experiment",
    version="0.1.0",
    description="Energy consumption measurement experiment for Qwen LLM generations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Green Lab Research Team",
    author_email="hidde.n.g.makimei@gmail.com",
    url="https://github.com/Rawat-OpenSource/Green-Lab",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    python_requires=">=3.8",
    install_requires=requirements,
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    entry_points={
        "console_scripts": [
            "run-qwen-experiment=scripts.run_experiment:main",
        ],
    },
    
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
    
    include_package_data=True,
    zip_safe=False,
)