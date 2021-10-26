#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import setuptools

setuptools.setup(
    name="pterotactyl",
    version="0.1.0",
    author="Facebook AI Research",
    description="",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence :: Active Sensing",
    ],
    python_requires=">=3.6",
)