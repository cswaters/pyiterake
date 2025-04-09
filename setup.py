from setuptools import setup, find_packages

setup(
    name="pyiterake",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.1.0",
        "seaborn>=0.11.0",
    ],
    author="PyIterake Contributors",
    author_email="corywaters@gmail.com",
    description="Create weights with iterative raking - port of Iterake R package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cswaters/pyiterake",
    project_urls={
        "Bug Tracker": "https://github.com/cswaters/pyiterake/issues",
        "Source Code": "https://github.com/cswaters/pyiterake",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.12",
)
