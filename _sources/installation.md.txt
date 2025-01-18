# Installation

The `adapters` package is designed as an add-on for Hugging Face's Transformers library.
It currently supports Python 3.9+ and PyTorch 2.0+. You will have to [install PyTorch](https://pytorch.org/get-started/locally/) first. 

```{eval-rst}
.. important::
    Each ``adapters`` version is built for one specific version of Transformers.
    While using a different version of Transformers with an ``adapters`` might work, it is highly recommended to use the intended version.
    ``adapters`` will automatically install the correct Transformers version if not installed.
```

## Using pip

### From PyPI

The simplest way of installation is by using pip to install the package from the Python Package Index:

```
pip install adapters
```

### From GitHub

You can also install the latest development version directly from our GitHub repository:

```
pip install git+https://github.com/adapter-hub/adapters.git
```

## From repository

Alternatively, you can clone the repository first and install the package from source.
This allows you to run the included example scripts directly:

```
git clone https://github.com/adapter-hub/adapters.git
cd adapters
pip install .
```
