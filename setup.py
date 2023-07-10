from setuptools import setup, find_packages


def readme():
    with open("README.md", encoding="utf-8") as f:
        return f.read()


def required():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name = 'ot_mmd',
    packages = find_packages(),
    version = "0.0.0",
    description = "PyTorch based library for conditional optimal transport.",
    license = "MIT",
)