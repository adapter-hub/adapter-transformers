# Copyright 2020-2024 The AdapterHub Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

from setuptools import find_packages, setup


# NOTE: All setup logic is transferred & adapted from Transformer's setup.py.
# We try to follow their general layout wherever sensible.

_deps = [
    "accelerate>=0.26.0",
    "beautifulsoup4",
    "black~=24.4.0",
    "dataclasses",
    "datasets!=2.5.0",
    "dill<0.3.5",
    "docutils==0.16.0",
    "evaluate>=0.2.0",
    "flake8>=3.8.3",
    "GitPython<3.1.19",
    "isort>=5.5.4",
    "Jinja2==2.11.3",
    "nltk",
    "packaging",
    "parameterized",
    "pillow",
    "protobuf",
    "psutil",
    "pytest>=7.2.0,<8.0.0",
    "pytest-rich",
    "pytest-timeout",
    "pytest-xdist",
    "markupsafe==2.0.1",
    "myst-parser",
    "rjieba",
    "rouge-score!=0.0.7,!=0.0.8,!=0.1,!=0.1.1",
    "sacrebleu>=1.4.12,<2.0.0",
    "sacremoses",
    "scikit-learn",
    "sentencepiece>=0.1.91,!=0.1.92",
    "sphinx-copybutton==0.5.2",
    "sphinx-markdown-tables==0.0.17",
    "sphinx-rtd-theme==2.0.0",
    "sphinx==5.0.2",
    "sphinxext-opengraph==0.4.1",
    "sphinx-intl==2.1.0",
    "sphinx-multiversion==0.2.4",
    "timeout-decorator",
    "torch",
    "torchvision",
    "transformers~=4.48.0",
]


# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~ ]+)(?:[!=<>~ ].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}

extras["sklearn"] = deps_list("scikit-learn")

extras["torch"] = deps_list("torch", "accelerate")
extras["sentencepiece"] = deps_list("sentencepiece", "protobuf")
extras["torchvision"] = deps_list("torchvision")
extras["testing"] = deps_list(
    "pytest",
    "pytest-rich",
    "pytest-xdist",
    "timeout-decorator",
    "parameterized",
    "psutil",
    "datasets",
    "dill",
    "evaluate",
    "pytest-timeout",
    "black",
    "sacrebleu",
    "rouge-score",
    "nltk",
    "GitPython",
    "sacremoses",
    "rjieba",
    "beautifulsoup4",
    "pillow",
    "accelerate",
    "torchvision",
)

extras["quality"] = deps_list("black", "datasets", "isort", "flake8", "GitPython")

extras["docs"] = deps_list(
    "docutils",
    "Jinja2",
    "markupsafe",
    "myst-parser",
    "sphinx",
    "sphinx-markdown-tables",
    "sphinx-rtd-theme",
    "sphinx-copybutton",
    "sphinxext-opengraph",
    "sphinx-intl",
    "sphinx-multiversion",
)

extras["dev"] = (
    extras["testing"]
    + extras["torch"]
    + extras["sentencepiece"]
    + extras["quality"]
    + extras["docs"]
    + extras["sklearn"]
)

# when modifying the following list, make sure to update src/transformers/dependency_versions_check.py
install_requires = [
    deps["transformers"],
    deps["packaging"],
]

setup(
    name="adapters",
    version="1.1.0.dev0",
    author="The AdapterHub team and community contributors",
    author_email="calpt@mail.de",
    description="A Unified Library for Parameter-Efficient and Modular Transfer Learning",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch BERT adapters PEFT LoRA",
    license="Apache",
    url="https://github.com/adapter-hub/adapters",
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
    extras_require=extras,
    python_requires=">=3.9.0",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
