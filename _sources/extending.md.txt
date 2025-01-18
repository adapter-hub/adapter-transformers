# Extending the Library

## Integrating new Transformer models
Currently, not all model types included in Hugging Face's `transformers` support adapters yet.
However, it is possible to add the existing adapter implementation to new models.
For detailed instructions, see [Adding Adapters to a Model](https://docs.adapterhub.ml/contributing/adding_adapters_to_a_model.html).

## Loading custom module weights

`adapters` provides support for saving and loading adapter and prediction head modules from the local file system or the Hub out of the box.
However, countless additional module integrations into language models are thinkable.
To provide a basis for such new custom model plugins, `adapters` integrates a basic mechanism to save and load custom weights.

All adapter and head module weights are extracted, saved and loaded by implementations of the `WeightsLoader` class, the two preincluded being `AdapterLoader` and `PredictionHeadLoader`. To add basic saving and loading functionalities to your custom module weights, you can implement a new subclass of `WeightsLoader`. The two required abstract methods to be implemented are:

- `filter_func(self, name: str) -> Callable[[str], bool]`: The callable returned by this method is used to extract the module weights to be saved or loaded based on their names.

- `rename_func(self, old_name: str, new_name: str) -> Callable[[str], str]`: The callable returned by this method is used to optionally rename the module weights after loading.

For more advanced functionalities, you may also want to override the `save()` and `load()` method.

Using the custom loader class, weights can now be saved with:
```python
loader = MyCustomWeightsLoader(model)
loader.save("path/to/save/dir", "custom_weights_name")
```

You can also upload these weights to the Hub and then load them from there together with an adapter:
```python
model.load_adapter(
    "adapter_name",
    custom_weights_loaders=[MyCustomWeightsLoader]
)
```
