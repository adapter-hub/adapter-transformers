# Loading Pre-Trained Adapters

## Finding pre-trained adapters

**[AdapterHub.ml](https://adapterhub.ml/explore)** provides a central collection of all pre-trained adapters uploaded via Hugging Face's [Model Hub](https://huggingface.co/models).
You can easily find pre-trained adapters for your task of interest along with all relevant information and code snippets to get started.

```{eval-rst}
.. note::
    The original `Hub repository <https://github.com/adapter-hub/hub>`_ (via ``source="ah"``) has been archived and migrated to the HuggingFace Model Hub. The Adapters library supports automatic redirecting to the HF Model Hub when attempting to load adapters from the original Hub repository.
```

Alternatively, [`list_adapters()`](adapters.utils.list_adapters) provides a programmatical way of accessing all available pre-trained adapters.
This will return an [`AdapterInfo`](adapters.utils.AdapterInfo) object for each retrieved adapter.
E.g., we can use it to retrieve information for all adapters trained for a specific model:

```python
from adapters import list_adapters

adapter_infos = list_adapters(model_name="bert-base-uncased")

for adapter_info in adapter_infos:
    print("Id:", adapter_info.adapter_id)
    print("Model name:", adapter_info.model_name)
    print("Uploaded by:", adapter_info.username)
```

In case the adapter ID is known, information for a single adapter can also be retrieved via [`get_adapter_info()`](adapters.utils.get_adapter_info):

```python
adapter_info = get_adapter_info("AdapterHub/roberta-base-pf-imdb")

print("Id:", adapter_info.adapter_id)
print("Model name:", adapter_info.model_name)
print("Uploaded by:", adapter_info.username)
```

## Using pre-trained adapters in your code

Suppose we have loaded a pre-trained transformer model from Hugging Face, e.g. BERT, and initialized it for adding adapters:

```python
from transformers import BertModel
import adapters

model = BertModel.from_pretrained('bert-base-uncased')
adapters.init(model)
```

We can now easily load a pre-trained adapter module from Adapter Hub by its identifier using the [`load_adapter()`](adapters.ModelWithHeadsAdaptersMixin.load_adapter) method:

```python
adapter_name = model.load_adapter('sst-2')
```

In the minimal case, that's everything we need to specify to load a pre-trained task adapter for sentiment analysis, trained on the `sst-2` dataset using BERT base and a suitable adapter configuration.
The name of the adapter is returned by [`load_adapter()`](adapters.ModelWithHeadsAdaptersMixin.load_adapter), so we can [activate it](adapter_composition.md) in the next step:
```python
model.set_active_adapters(adapter_name)
```

As the second example, let's have a look at how to load an adapter based on the [`AdapterInfo`](adapters.utils.AdapterInfo) returned by the [`list_adapters()`](adapters.utils.list_adapters) method from [above](#finding-pre-trained-adapters):
```python
from adapters import AutoAdapterModel, list_adapters

adapter_infos = list_adapters()
# Take the first adapter info as an example
adapter_info = adapter_infos[0]

model = AutoAdapterModel.from_pretrained(adapter_info.model_name)
model.load_adapter(adapter_info.adapter_id)
```

### Advanced usage of `load_adapter()`

To examine what's happening underneath in a bit more detail, let's first write out the full method call with all relevant arguments explicitly stated:

```python
model.load_adapter(
    "AdapterHub/roberta-base-pf-imdb",
    version="main",
    load_as="sentiment_adapter",
    set_active=True,
)
```

We will go through the different arguments and their meaning one by one:

- The first argument passed to the method specifies the name or path from where to load the adapter. This can be the name of a repository on the [HuggingFace Model Hub](https://huggingface.co/models), a local path or a URL. To get an overview of all available adapters on the Hub, please refer to [the Adapter-Hub website](https://adapterhub.ml/explore).

- There could be multiple versions of the same adapter available as revisions in a Model Hub repository. To load a specific revision, use the `version` parameter.

- By default, the `load_adapter()` method will add the loaded adapter using the identifier string given as the first argument.
To load the adapter using a custom name, we can use the `load_as` parameter.

- Finally, `set_active` will directly activate the loaded adapter for usage in each model forward pass. Otherwise, you have to manually activate the adapter via `set_active_adapters()`.

## Saving and loading adapter compositions

In addition to saving and loading individual adapters, you can also save, load and share entire [compositions of adapters](adapter_composition.md) with a single line of code.
_Adapters_ provides three methods for this purpose that work very similar to those for single adapters:

- [`save_adapter_setup()`](adapters.ModelWithHeadsAdaptersMixin.save_adapter_setup) to save an adapter composition along with prediction heads to the local file system.
- [`load_adapter_setup()`](adapters.ModelWithHeadsAdaptersMixin.load_adapter_setup) to load a saved adapter composition from the local file system or the Model Hub.
- [`push_adapter_setup_to_hub()`](adapters.hub_mixin.PushAdapterToHubMixin.push_adapter_setup_to_hub) to upload an adapter setup along with prediction heads to the Model Hub. See our [Hugging Face Model Hub guide](huggingface_hub.md) for more.

As an example, this is how you would save and load an AdapterFusion setup of three adapters with a prediction head:

```python
# Create an AdapterFusion
model = AutoAdapterModel.from_pretrained("bert-base-uncased")
model.load_adapter("sentiment/sst-2@ukp", config=SeqBnConfig(), with_head=False)
model.load_adapter("nli/multinli@ukp", config=SeqBnConfig(), with_head=False)
model.load_adapter("sts/qqp@ukp", config=SeqBnConfig(), with_head=False)
model.add_adapter_fusion(["sst-2", "mnli", "qqp"])
model.add_classification_head("clf_head")
adapter_setup = Fuse("sst-2", "mnli", "qqp")
head_setup = "clf_head"
model.set_active_adapters(adapter_setup)
model.active_head = head_setup

# Train AdapterFusion ...

# Save
model.save_adapter_setup("checkpoint", adapter_setup, head_setup=head_setup)

# Push to Hub
model.push_adapter_setup_to_hub("<user>/fusion_setup", adapter_setup, head_setup=head_setup)

# Re-load
# model.load_adapter_setup("checkpoint", set_active=True)
```
