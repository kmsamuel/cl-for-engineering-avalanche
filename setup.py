from setuptools import setup, find_packages

setup(
    name="avalanche_custom",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "tqdm"
    ],
    description="Customized Avalanche library for continual learning in engineering benchmarks.",
    author="Kaira Samuel",
    url="https://github.com/kmsamuel/cl-for-engineering",
)
