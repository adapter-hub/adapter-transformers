# Adding Adapters to a Model
This document gives an overview of how new model architectures of Hugging Face Transformers can be supported by `adapters`.
Before delving into implementation details, you should familiarize yourself with the main design philosophies of `adapters`:

- _Adapters should integrate seamlessly with existing model classes_: If a model architecture supports adapters, it should be possible to use them with all model classes of this architecture.
- _Copied code should be minimal_: `adapters` extensively uses Python mixins to add adapter support to HF models. Functions that cannot be sufficiently modified by mixins are copied and then modified. Try to avoid copying functions as much as possible.

## Relevant Classes
Adding adapter support to an existing model architecture requires modifying some parts of the model forward pass logic. These modifications are realized by the four files in the `src/adapters/models/<model_type>/` directory. Let's examine the purpose of these files in the example of BERT. It's important to note that we are adapting the original Hugging Face model, implemented in [transformers/models/bert/modeling_bert.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py). The files in `src/adapters/models/bert/` are:

1. `src/adapters/models/bert/mixin_bert.py`:
This file contains mixins for each class we want to change. For example, in the `BertSelfAttention` class, we need to make changes for LoRA and Prefix Tuning. For this, we create a `BertSelfAttentionAdaptersMixin` to implement these changes. We will discuss how this works in detail below.
2. `src/adapters/models/bert/modeling_bert.py`:
For some classes of the BERT implementation (e.g. `BertModel` or `BertLayer`) the code can be sufficiently customized via mixins. For other classes (like `BertSelfAttention`), we need to edit the original code directly. These classes are copied into `src/adapters/models/bert/modeling_bert.py` and modified.
3. `src/adapters/models/bert/adapter_model.py`: 
In this file, the adapter model class is defined. This class allows flexible adding of and switching between multiple prediction heads of different types. This looks about the same for each model, except that each model has different heads and thus different `add_..._head()` functions.
4. `src/adapters/models/bert/__init__.py`: Defines Python's import structure.


## Implementation Steps 📝
Now that we have discussed the purpose of every file in `src/adapters/models/<model_type>/`, we go through the integration of adapters into an existing model architecture step by step. **The following steps might not be applicable to every model architecture.**

1. **Files:**
    - Create the `src/adapters/models/<model_type>/` directory and in it the 4 files: `mixin_<model_type>.py`, `modeling_<model_type>.py` `adapter_model.py` and `__init__.py`
2. **Mixins:**
    - In `src/adapters/models/<model_type>/mixin_<model_type>.py`, create mixins for any class you want to change and where you can't reuse an existing mixin from another class.
        - To figure out which classes to change, think about where to insert LoRA, Prefix Tuning, and bottleneck adapters.
        - You can use similar model implementations for guidance.
        - Often, existing mixins of another class can be reused. E.g. `BertLayer`, `RobertaLayer`, `XLMRobertaLayer`, `DebertaLayer`, `DebertaV2Layer` and `BertGenerationLayer` (all models derived from BERT) use the `BertLayerAdaptersMixin`.
    - To additionally support Prefix Tuning, it's necessary to apply the forward call to the `PrefixTuningLayer` module in the respective attention layer (see step 3 for how to modify the code of an Hugging Face class).
    - Make sure the calls to `bottleneck_layer_forward()` are added in the right places.
    - The mixin for the whole base model class (e.g., `BertModel`) should derive from `ModelBaseAdaptersMixin` and (if possible) `EmbeddingAdaptersMixin` and/or `InvertibleAdaptersMixin`. This mixin should at least implement the `iter_layers()` method but might require additional modifications depending on the architecture.
        - If the model is a combination of different models, such as the EncoderDecoderModel, use `ModelUsingSubmodelsAdaptersMixin` instead of `ModelBaseAdaptersMixin`.
3. **Copied functions:**
    - For those classes where the mixin is not enough to realize the wanted behavior, you must:
    - Create a new class in `src/adapters/models/<model_type>/modeling_<model_type>.py` with the name `<class>WithAdapters`. This class should derive from the corresponding mixin and HF class.
    - Copy the function you want to change into this class and modify it.
        - e.g., the `forward` method of the `BertSelfAttention` class must be adapted to support prefix tuning. We therefore create a class `BertSelfAttentionWithAdapters(BertSelfAttentionAdaptersMixin, BertSelfAttention)`, copy the forward method into it and modify it.
        - if the `forward` method of a module is copied and modified, make sure to call `adapters.utils.patch_forward()` in the module's `init_adapters()` method. This ensures adapters work correctly with the `accelerate` package.
4. **Modify MODEL_MIXIN_MAPPING**
    - For each mixin whose class was not copied into `modeling_<model_type>.py`, add the mixin/class combination into `MODEL_MIXIN_MAPPING` in the file `src/adapters/models/__init__.py`.
5. **Create the adapter model:**
    - Adapter-supporting architectures should provide a new model class `<model_type>AdapterModel`. This class allows flexible adding of and switching between multiple prediction heads of different types.
    - This is done in the `adapter_model.py` file:
        - This module should implement the `<model_type>AdapterModel` class, deriving from `ModelWithFlexibleHeadsAdaptersMixin` and `<model_type>PreTrainedModel`.
        - In the model class, add methods for those prediction heads that make sense for the new model architecture.
        - Again, have a look at existing implementations.
    - Add `<model_type>AdapterModel` to the `ADAPTER_MODEL_MAPPING_NAMES` mapping in `src/adapters/models/auto/adapter_model.py` and to `src/adapters/__init__.py`.
    - Define the classes to be added to Python's import structure in `src/adapters/models/<model_type>/__init__.py`. This will likely only be the `<model_type>AdapterModel`.
6. **Adapt the config classes:**
    - Adapt the config class to the requirements of adapters in `src/adapters/wrappers/configuration.py`.
    - There are some naming differences in the config attributes of different model architectures. The adapter implementation requires some additional attributes with a specific name to be available. These currently are `num_attention_heads`, `hidden_size`, `hidden_dropout_prob` and `attention_probs_dropout_prob` as in the `BertConfig` class.
    If your model config does not provide these, add corresponding mappings to `CONFIG_CLASS_KEYS_MAPPING`.


### Additional (optional) implementation steps 📝

- Parallel adapter inference via `Parallel` composition block (cf. [documentation](https://docs.adapterhub.ml/adapter_composition.html#parallel), [PR#150](https://github.com/Adapter-Hub/adapters/pull/150)).
- Provide mappings for an architecture's existing (static) prediction heads into `adapters` flex heads (cf. [implementation](https://github.com/adapter-hub/adapters/blob/main/src/adapters/head_utils.py#L11)).

## Testing

❓ In addition to the general Hugging Face model tests, there are adapter-specific test cases. All tests are executed from the `tests` folder. You need to add two different test classes.

**📝 Steps**
1. Add a new `test_<model_type>.py` module in `tests/`
    - This file is used to test that everything related to the usage of adapters (adding, removing, activating, ...) works.
    - This module typically holds 2 test classes and a test base class:
        - `<model_type>AdapterTestBase`: This class contains the `tokenizer_name`, `config_class` and `config`.
        - `<model_type>AdapterTest` derives from a collection of test mixins that hold various adapter tests (depending on the implementation).
        - (optionally) `<model_type>ClassConversionTest` runs tests for correct class conversion if conversion of prediction heads is implemented.
2. Add a new `test_<model_type>.py` module in `tests/models/`
    - This file is used to test the AdapterModel class.
    - This module typically holds 1 test class with the name `<model_type>AdapterModelTest`
        - `<model_type>AdapterModelTest` derives directly from Hugging Face's existing model test class `<model_type>ModelTest` and adds `<model_type>AdapterModel` as a class to test.

## Documentation

❓ The documentation for `adapters` lives in the `docs` folder.

**📝 Steps**

- Add `docs/classes/models/<model_type>.rst` (oriented at the doc file in the HF docs). Make sure to include `<model_type>AdapterModel` autodoc. Finally, list the file in `index.rst`.
- Add a new row for the model in the model table of the overview page at `docs/model_overview.md`, listing all the methods implemented by the new model.

## Training Example Adapters

❓ To make sure the new adapter implementation works properly, it is useful to train some example adapters and compare the training results to full model fine-tuning. Ideally, this would include training adapters on one (or more) tasks that are good for demonstrating the new model architecture (e.g. GLUE benchmark for BERT, summarization for BART) and uploading them to AdapterHub.

We provide training scripts for many tasks here: [https://github.com/Adapter-Hub/adapters/tree/main/examples/pytorch/](https://github.com/Adapter-Hub/adapters/tree/main/examples/pytorch/)
