from setuptools import setup


setup(
    name="lcapt",
    version="0.0.1",
    description="LCA Sparse Coding in PyTorch.",
    author="Michael Teti",
    author_email="mteti@lanl.gov",
    packages=["lcapt"],
    install_requires=[
        "black>=22.1.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.2",
        "pandas>=1.3.4",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.2",
        "seaborn>=0.11.2",
        "torch>=1.10.1",
        "torchaudio>=0.10.1",
        "torchvision>=0.11.2",
        "typing_extensions>=4.1.1",
        "wheel>=0.37.0",
    ],
)
