from setuptools import setup, find_packages

with open("README.md", encoding='utf-8') as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

setup(
    name="TorchFly",
    version="0.0.1",
    author="Qingyang Wu",
    author_email="wilwu@ucdavis.edu",
    description="Pytorch Fast Development Kit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qywu/TorchFly",
    packages=find_packages(),
    install_requires=required_packages,
    classifiers=[
        'Programming Language :: Python :: 3',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
