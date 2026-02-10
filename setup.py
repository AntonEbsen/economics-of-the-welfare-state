from setuptools import setup, find_packages

setup(
    name="econ-clean",
    version="0.3.0",
    author="Anton Ebsen",
    description="Comprehensive toolkit for economics panel data analysis",
    long_description=open("docs/README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AntonEbsen/economics-of-the-welfare-state",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "statsmodels>=0.13.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "advanced": [
            "linearmodels>=4.25",
            "stargazer>=0.0.5",
        ],
    },
)
