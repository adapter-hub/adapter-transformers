# Transitioning from `adapter-transformers`

```{eval-rst}
.. important::
    ``adapters`` is fully compatible to ``adapter-transformers`` in terms of model weights, meaning you can load any adapter trained with any version of ``adapter-transformers`` to the new library without degradation.
```

The new `adapters` library is the successor to the `adapter-transformers` library. It differs essentially in that `adapters` is now a stand-alone package, i.e., the package is disentangled from the `transformers` package from Hugging Face and is no longer a drop-in replacement.

This results in some breaking changes. To transition your code from `adapter-transformers` to `adapters` you need to consider the following changes:

## Package and Namespace
 To use the library you need to install 
`transformers` and `adapters` in the same environment (unlike `adapter-transformers` which contained `transformers` and could not be installed in the same environment). 

Run the following to install both (installing `adapters` will automatically trigger the installation of a compatible `transformers` version):

```
pip install adapters
```

This also changes the namespace to `adapters`. For all imports of adapter classes change the import from `transformers` to  `adapters`.
This mainly affects the following classes:
- AdapterModel classes, e.g. `AutoAdapterModel` (see [AdapterModels](https://docs.adapterhub.ml/model_overview.html) )
- Adapter configurations e.g. `PrefixTuningConfig` (see [Configurations](https://docs.adapterhub.ml/overview.html) )
- Adapter composition blocks, e.g. `Stack` (see [Composition Blocks](https://docs.adapterhub.ml/adapter_composition.html) )
- The `AdapterTrainer` class

## Model Initialisation

The Hugging Face model classes, such as `BertModel`, cannot be used directly with adapters. They must first be initialised for adding adapters:

```
from transformers import AutoModel
import adapters

model = AutoModel.from_pretrained("bert-base-uncased")
adapters.init(model) # prepare model for use with adapters
```

The necessary change is the call of the `adapters.init()` method. 
Note that no additional initialisation is required to use the AdapterModel classes such as the `BertAdapterModel`'. These classes are provided by the `adapters` library and are already prepared for using adapters in training and inference.

## Bottleneck Configuration Names

The `adapters` library supports the configuration of adapters using [config strings](https://docs.adapterhub.ml/overview.html#configuration-strings). Compared to the `adapter-transformers` library, we have changed some of the strings to make them more consistent and intuitive:
- `houlsby` -> `double_seq_bn`
- `pfeiffer` -> `seq_bn`
- `parallel`-> `par_seq_bn`
- `houlsby+inv` -> `double_seq_bn_inv`
- `pfeiffer+inv`-> `seq_bn_inv`


For a complete list of config strings and classes see [here](https://docs.adapterhub.ml/overview.html). We strongly recommend using the new config strings, but we will continue to support the old config strings for the time being to make the transition easier.
Note that with the config strings the corresponding adapter config classes have changed, e.g. `PfeifferConfig` -> `SeqBnConfig`.

Another consequence of this that the `AdapterConfig` class is now not only for the bottleneck adapters anymore, but the base class of all the configurations (previously `AdapterConfigBase`). Hence, the function this class serves has changed. However, you can still load adapter configs with:
```
adapter_config = AdapterConfig.load("lora")
```


## Features that are not supported by `adapters`

Compared to `adapter-transformers`, there are a few features that are no longer supported by the `adapters` library: 
- Using `transformers` pipelines with adapters.
- Using invertible adapters in the Hugging Face model classes. To use invertible adapters you must use the AdapterModel class.
- Loading model and adapter checkpoints saved with `save_pretrained` using Hugging Face classes. This is only supported by the AdapterModel classes.

## What has remained the same

- The new library is fully backwards compatible in terms of adapter weights, i.e. you can load all adapter modules trained with `adapter-transformers`.
- The functionality for adding, activating, and training adapters has __not__ changed, except for the renaming of some adapter configs. You still add and activate adapters as follows:
```
# add adapter to the model
model.add_adapter("adapter_name", config="lora")
# activate adapter
model.set_active_adapters("adapter_name")
# freeze model weights and activate adapter
model.train_adapter("adapter_name")
```

## Where can I still find `adapter-transformers`?

The codebase of `adapter-transformers` has moved to [https://github.com/adapter-hub/adapter-transformers-legacy](https://github.com/adapter-hub/adapter-transformers-legacy) for archival purposes.

The full documentation of the old library is now hosted at [https://docs-legacy.adapterhub.ml](https://docs-legacy.adapterhub.ml/).

