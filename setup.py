from setuptools import setup, find_packages

setup(
    name="cv_course",
    version="0.0",
    packages=find_packages(where="./", exclude=["tests"]),
)
