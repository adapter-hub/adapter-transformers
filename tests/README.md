# Testing the Adapters Library

This README provides a comprehensive overview of the test directory organization and explains how to execute different types of tests within the adapters library.

## Test Directory Structure Overview

```
tests/
├── __init__.py
├── fixtures/                       # Datasets, test samples, ...
|   └── ...
├── test_methods/                   # Dynamic adapter method tests (all models)
│   ├── __init__.py
│   ├── method_test_impl/               # Implementation of tests
│   │   ├── __init__.py
│   │   ├── core/
│   │   ├── composition/
│   │   └── ...
│   ├── base.py                     # Base from which model test bases inherit
│   ├── generator.py                    # Testcase generation and registration
│   ├── test_on_albert.py               # Example model test base for testing adapter methods on albert adapter model
│   ├── test_on_beit.py 
│   └── ...
├── test_misc/                      # Miscellaneous adapter method tests (single model)
│   ├── test_adapter_config.py 
│   └── ...
├── test_models/                    # Adapter model tests with Hugging Face test suite
│   └── __init__.py
│   │   ├── base.py
│   │   ├── test_albert_model.py
│   │   └── ...
```

## Test Categories

The testing framework encompasses three distinct categories of tests:

1. Dynamic Adapter Method Tests: These tests cover core functionalities of the adapters library, including individual adapter methods (such as LoRA and prompt tuning) and head functionalities. These tests are executed across all supported models.

2. Miscellaneous Adapter Method Tests: These supplementary tests cover scenarios not included in the dynamic tests. To optimize resources, they are executed on a single model, as repeated execution across multiple models would not provide additional value.

3. Adapter Model Tests: These tests verify the implementation of the adapter models themselves using the Hugging Face model test suite.

## Test Generator and Pytest Markers

The test_methods directory contains the central component `generator.py`, which generates appropriate sets of adapter method tests. Each model test base registers these tests using the following pattern:

```python
method_tests = generate_method_tests(AlbertAdapterTestBase)

for test_class_name, test_class in method_tests.items():
    globals()[test_class_name] = test_class
```

Each generated test class is decorated with a specific marker type. For example:

```python
@require_torch
@pytest.mark.lora
class LoRA(
    AlbertAdapterTestBase,
    LoRATestMixin,
    unittest.TestCase,
):
    pass
```

These markers enable the execution of specific test types across all models. You can run these tests using either of these methods:

1. Using the make command:
```bash
make test-adapter-method-subset subset=lora
```

2. Directly executing from the test directory:
```bash
cd tests/test_methods
pytest -m lora
```

Both approaches will execute all LoRA tests across every model in the adapters library.

## Adding a New Adapter Method to the Test Suite

The modular design of the test base simplifies the process of adding tests for new adapter methods. To add tests for a new adapter method "X", follow these steps:

1. Create the Test Implementation:
   Create a new file `tests/test_methods/method_test_impl/peft/test_X.py` and implement the test mixin class:

   ```python
   @require_torch
   class XTestMixin(AdapterMethodBaseTestMixin):
       
       default_config = XConfig()

       def test_add_X(self):
           model = self.get_model()
           self.run_add_test(model, self.default_config, ["adapters.{name}."]) 
       
       def ...
   ```

2. Register the Test Mixin:
   Add the new test mixin class to `tests/test_methods/generator.py`:

   ```python
   from tests.test_methods.method_test_impl.peft.test_X import XTestMixin

   def generate_method_tests(model_test_base, ...):
       """ Generate method tests for the given model test base """
       test_classes = {}

       @require_torch
       @pytest.mark.core
       class Core(
           model_test_base,
           CompabilityTestMixin,
           AdapterFusionModelTestMixin,
           unittest.TestCase,
       ):
           pass

       if "Core" not in excluded_tests:
           test_classes["Core"] = Core

       @require_torch
       @pytest.mark.X
       class X(
           model_test_base,
           XTestMixin,
           unittest.TestCase,
       ):
           pass

       if "X" not in excluded_tests:
           test_classes["X"] = X   
   ```

    The pytest marker enables execution of the new method's tests across all adapter models using:
    ```bash
    make test-adapter-method-subset subset=X
    ```

    If the new method is incompatible with specific adapter models, you can exclude the tests in the respective `test_on_xyz.py` file:

    ```python
    method_tests = generate_method_tests(BartAdapterTestBase, excluded_tests=["PromptTuning", "X"])
    ```

    Note: It is recommended to design new methods to work with the complete library whenever possible. Only exclude tests when there are unavoidable compatibility issues and make them clear in the documenation.