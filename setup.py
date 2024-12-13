from setuptools import setup, find_packages

def read_requirements(filename="requirements.txt"):
    with open(filename, "r") as f:
        return [line.strip() for line in f.readlines() if line and not line.startswith("#")]

setup(
    name="sr-metric",
    version="0.1.0",
    description="A benchmarking framework for super-resolution microscopy techniques",
    author="Jacky Ka Long Ko",
    author_email="ka.ko@kennedy.ox.ac.uk",
    url="https://github.com/yourusername/sr-microscopy-benchmark",
    packages=find_packages(where="sr-metric"),
    package_dir={"": "sr-metric"},
    # include_package_data=True,
    install_requires=read_requirements(), 
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "benchmark-sr=benchmark:main",
        ],
    },
)
