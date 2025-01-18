# Adding Adapter Methods

This document describes how different efficient fine-tuning methods can be integrated into the codebase of `adapters`.
It can be used as a guide to add new efficient fine-tuning/ adapter methods.

Before we start to go into implementation details, first some important design philosophies of `adapters`:

- _Adapters should integrate seamlessly with existing model classes_: This means (a) if a model architecture supports adapters, it should be possible to use them with all model classes of this architecture and (b) adapters should be entirely opt-in, i.e. the model classes still must work without adapters.
- _Copying original should be minimal_: `adapters` tries to avoid copying of the original HF code as far as possible. We extensively use Python mixins to achieve this.

Now we highlight the most important components of integrating adapter methods into Transformer models.
Each integration is highly dependent on the specific details of the adapter methods.
Therefore, the described steps might not be applicable to each implementation.

## Implementation

❓ As adapter methods typically inject blocks of new parameters into an existing Transformer model, they mostly can be implemented using multiple blocks of classes deriving from `torch.nn.Module`.
These module classes then have to be inserted into the correct locations within the Transformer model implementation.
Thus, each adapter method implementation at least should provide two classes:

- a configuration class deriving from `AdapterConfig` that provides attributes for all configuration options of the method
- a module class deriving from the abstract `AdapterLayerBase` that provides the method parameters and a set of standard adapter management functions
    - modules supporting [adapter composition](https://docs.adapterhub.ml/adapter_composition.html) should instead derive from `ComposableAdapterLayerBase`

### Configuration

All configuration classes reside in `src/adapters/configuration/adapter_config.py`.
- To add a new configuration class for a new method, create a new subclass of [`AdapterConfig`](adapters.AdapterConfig).
    Make sure to set the `architecture` attribute in your class.
- Finally, also make sure the config class is added to the `__init__.py` files in `src/adapters`.

### Modeling

All adapter method implementations reside in `src/adapters/methods`.

#### For methods **without** composition support

The [`AdapterLayerBase`](adapters.AdapterLayerBase) class from which any new adapter modules should derive resides in `src/adapters/methods/adapter_layer_base.py`.
- This abstract base class defines a set of methods that should be implemented by each deriving class,
including methods for adding, enabling and deleting adapter weights. These methods are marked as abstract in the base class. See [`AdapterLayerBase`](adapters.AdapterLayerBase) for details.
- Most importantly however, the module classes deriving from this base class should implement the forward pass through an adaptation component.
- The concrete implementation of these classes heavily depends on the specifics of the adapter method.

#### For methods **with** composition support 

The [`ComposableAdapterLayerBase`](adapters.ComposableAdapterLayerBase) class (as subclass of [`AdapterLayerBase`](adapters.AdapterLayerBase)), which resides in `src/adapters/methods/adapter_layer_base.py` provides the basic skeleton for implementing adapter composition.
- Your deriving module class firstly should implement all methods required by [`AdapterLayerBase`](adapters.AdapterLayerBase). See section above for details.
- For adapter composition, the pre-implemented `compose()` method constitutes the main entry-point. This method should be called during the forward pass of your adapter module.
- `compose()` expects a `state` object, which is a generic named tuple object defined by your adapter method. This state object should hold all tensors (such as hidden states, attention masks etc.) and state attributes required for your adapter implementation. See `BottleneckState` for an example.
- Implementations for specific composition blocks are given in methods starting with `compose_`. Some composition blocks provide generic default implementations, some must be implemented by the deriving class if they should be supported. Make sure to list all supported composition blocks in the `supported_compositions` class attribute of your deriving module.
- In any case, a small set of helper methods should be implemented by any deriving module to support basic composition logic. These are marked as abstract methods in [`ComposableAdapterLayerBase`](adapters.ComposableAdapterLayerBase) and currently consist of the following: vslice(), pad_and_concat(), repeat(), mean(), compose_single(). See [`ComposableAdapterLayerBase`](adapters.ComposableAdapterLayerBase) for details.

For a reference implementation, have a look at `BottleneckLayer` for bottleneck adapters.

#### For all methods

To actually make use of the newly implemented classes, it's finally necessary to integrate the forward calls to the modules in the actual model implementations.
- This, again, is highly dependent on how the adapter method interacts with the base model classes. Typically, module classes can be integrated either via mixins (see modules starting with "mixin" in `src/adapters/models`) or directly as submodules of the respective model components.
- The model class integration has to be repeated for each supported Transformer model, as they typically don't share a codebase. At this point it is often important to consider where the adapters need to be added to the transformer model and whether there is an implementation that does not require more copying of classes than the current implementation.
Please try to integrate any new adapter method into every model class when it's reasonable.
You can find all currently supported model classes at https://docs.adapterhub.ml/model_overview.html.

**Additional things to consider**

- New adapter methods typically also require some changes in the `AdapterLoader` class in `src/adapters/loading.py` (also see [here](https://docs.adapterhub.ml/extending.html#loading-custom-module-weights)).
- Depending on the method to be integrated, further changes in other classes might be necessary.

## Testing

❓ `adapters` provides a framework for testing adapter methods on implementing models in `tests`.
Tests for each adapter method are provided via a mixin class.
All test mixins derive from the common `AdapterMethodBaseTestMixin` class and reside in `tests/methods`.

**📝 Steps**

- Add a new `test_<method>.py` module in `tests/methods`.
    - This module should contain a `<method>TestMixin` class deriving from `AdapterMethodBaseTestMixin` that implements typical methods of adding, loading and training modules of the new adapter method.
    - Have a look at existing test mixins for reference.
- Next, add the newly implemented test mixin to the tests of all model types that support the new adapter method.
    - Each model type has its own test class `tests/test_<model_type>.py` that contains a `<model_type>AdapterTest` class.
    Add the new test mixin to the mixins of this class.
    E.g., if the new method is supported by BERT, add the its test mixin to `BertAdapterTest`.

## Documentation

❓ The documentation for `adapters` lives in the `docs` folder.

**📝 Steps**

- Add the class documentation for the configuration class of the new method in `docs/classes/adapter_config.rst`.
- In `docs/overview.md`, add a new section for the new adapter method that describes the most important concepts. Please try to follow the general format of the existing methods.
- Add a new column in the table in `docs/model_overview.md` and check the models that support the new adapter method.

Finally, please add a row for the new method in the table of supported methods under _Implemented Methods_ in the main `README.md` of this repository.

## Training Example Adapters

❓ To make sure the new adapter implementation works properly, it is useful to train some example adapters and compare the training results to full model fine-tuning and/or reference implementations.
Ideally, this would include training adapters on one (or more) tasks that are good for demonstrating the new method and uploading them to AdapterHub.

Hugging Face already provides example training scripts for many tasks, some of them have already been modified to support adapter training (see https://github.com/Adapter-Hub/adapters/tree/main/examples).
