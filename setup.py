from distutils.core import setup
from pathlib import Path

from setuptools import find_packages

import versioneer

CODE_DIRECTORY = Path(__file__).parent


def read_file(file_path: Path):
    """Source the contents of a file"""
    with open(str(file_path), encoding="utf-8") as file:
        return file.read()


setup(
    name="sql_to_ibis",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    long_description=read_file(CODE_DIRECTORY / "README.rst"),
    maintainer="Zach Brookler",
    maintainer_email="zachb1996@yahoo.com",
    description="A package for converting sql into ibis expressions",
    python_requires=">=3.7.0",
    install_requires=["lark-parser==0.8.5", "ibis-framework==1.4.0"],
    project_urls={
        "Source Code": "https://github.com/zbrookle/sql_to_ibis",
        "Documentation": "https://github.com/zbrookle/sql_to_ibis",
        "Bug Tracker": "https://github.com/zbrookle/sql_to_ibis/issues",
    },
    url="https://github.com/zbrookle/sql_to_ibis",
    download_url="https://github.com/zbrookle/sql_to_ibis/archive/master.zip",
    keywords=["pandas", "data", "dataframe", "sql", "ibis", "database", "query", "etl"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Database :: Front-Ends",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Typing :: Typed",
        "Operating System :: OS Independent",
    ],
    long_description_content_type="text/x-rst",
    include_package_data=True,
)
