#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2023, Miquel Massot
All rights reserved.
Licensed under the GPLv3 License.
See LICENSE.md file in the project root for full license information.
"""

from pathlib import Path

from setuptools import find_packages, setup

GITHUB_URL = "https://git.soton.ac.uk/feeg6043/uos_feeg6043_build"


classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python :: 3",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Hardware :: Hardware Drivers",
]

# read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="uos_feeg6043_build",
    version="1.0.6",
    description="Build repository for the module FEEG6043 - Intelligent Mobile Robot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Miquel Massot",
    author_email="miquel.massot@gmail.com",
    maintainer="Blair Thornton",
    maintainer_email="b.thornton@soton.ac.uk",
    url=GITHUB_URL,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=classifiers,
    license="GPLv3",
    install_requires=[
        "zeroros>=1.0.8",
        "pygame",
        "pbr",
        "tornado>=6.1",
        "pyyaml",
        "numpy==1.26.4",
        "matplotlib",
        "importlib-metadata==4.13.0",
        "setuptools==69.0.3",
        "pyserial==3.5",
        "pyzmq==25.1.1",
        "pglive==0.7.6",
        "pyqt5",
        "scipy",
        "scikit-learn",
    ],
    project_urls={"Bug Reports": GITHUB_URL + "/issues", "Source": GITHUB_URL},
)
