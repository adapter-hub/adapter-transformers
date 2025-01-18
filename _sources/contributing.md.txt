# Contributing to AdapterHub

There are many ways in which you can contribute to AdapterHub and the `adapters` library.
This includes code contributions such as:
- implementing new adapter methods
- adding support for new Transformer
- fixing open issues

as well as non-code contributions such as:
- training and uploading adapters to the Hub
- writing documentation and blog posts
- helping others with their issues and questions

Whichever way you'd like to contribute, you're very welcome to do so!

## Contributing to the `adapters` codebase

### Setting up your dev environment

To get started with writing code for `adapters`, you'd want to set up the project on a local development environment.

`adapters` closely follows the original Hugging Face Transformers repository in many aspects.
This guide assumes that you want to set up your dev environment on a local machine and that you have basic knowledge of `git`.
Additionally, you require **Python 3.8** or above pre-installed to get started.

In the following, we go through the setup procedure step by step:

1. Fork [the `adapters` repository](https://github.com/adapter-hub/adapters) to get a local copy of the code under your user account.
2. Clone your fork to your local machine:
    ```
    git clone --recursive git@github.com:<YOUR_USERNAME>/adapters.git
    cd adapters
    ```
    **Note:** The `--recursive` flag is important to initialize git submodules.
3. Create a virtual environment, e.g. via `virtualenv` or `conda`.
4. Install PyTorch, following the installation command for your environment [on their website](https://pytorch.org/get-started/locally/).
5. Install Hugging Face Transformers from the local git submodule:
    ```
    pip install ./hf_transformers
    ```
6. Install `adapters` and required dev dependencies:
    ```
    pip install -e ".[dev]"
    ```

### Adding Adapter Methods

How to integrate new efficient fine-tuning/ adapter methods to `adapters` is described at [https://docs.adapterhub.ml/contributing/adding_adapter_methods.html](https://docs.adapterhub.ml/contributing/adding_adapter_methods.html).

### Adding Adapters to a Model

How to add adapter support to a model type already supported by Hugging Face Transformers is described at [https://docs.adapterhub.ml/contributing/adding_adapters_to_a_model.html](https://docs.adapterhub.ml/contributing/adding_adapters_to_a_model.html).

### Testing your changes to the codebase

`adapters` provides multiple Makefile targets for easily running tests and repo checks.
Make sure these checks run without errors to pass the CI pipeline tasks when you open a pull request.

To **run all tests** in the repository:
```
make test
```

To **auto format code and imports** in the whole codebase:
```
make style
```
This will run `black` and `isort`.

To **run all quality checks** ensuring code style and repo consistency:
```
make quality
```
This will run checks with `black`, `isort` and `flake8` as well as additional custom checks.

## Publishing Pre-Trained Adapters

How to make your own trained adapters accessible for the `adapters` library HuggingFace Model Hub is described at [https://docs.adapterhub.ml/huggingface_hub.html](https://docs.adapterhub.ml/huggingface_hub.html).
