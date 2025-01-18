# Embeddings

With `adapters`, we support dynamically adding, loading, and deleting of `Embeddings`. This section
will give you an overview of these features. A toy example is illustrated in this [notebook](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/Adapter_With_Embeddings.ipynb).

## Adding and Deleting Embeddings
The methods for handling embeddings are similar to the ones handling adapters. To add new embeddings we call
`add_embeddings`. This adds new embeddings for the vocabulary of the `tokenizer`. 
In some cases, it might be useful to initialize embeddings of tokens to the ones of another embeddings module. If a 
`reference_embedding` and `reference_tokenizer` are provided all embeddings for tokens that are present in both embeddings are initialized to the embedding provided by the `reference_embedding`.  The new embedding will be created and set as the active embedding. If you are unsure which embedding
is currently active, the `active_embeddings` property contains the currently active embedding.

```python
model.add_embeddings('name', tokenizer, reference_embedding='default', reference_tokenizer=reference_tokenizer)
```

The original embedding of the transformers model is always available under the name `"default"`. To set it as the active
embedding simply call the `set_active_embedding('name')` method.
```python
model.set_active_embeddings('name')
```
Similarly, all other embeddings can be set as active by passing their name to the `set_active_embedding` method.

To delete an embedding that is no longer needed, we can call the `delete_embeddings` method with the name of the adapter
we want to delete. However, you cannot delete the default embedding.
```python
model.delete_embeddings('name')
```
Please note, that if the active embedding is deleted the default embedding is set as the active embedding.

## Training Embeddings
Embeddings can only be trained with an adapter. To freeze all weights except for the embedding and the adapter:
```python
model.train_adapter('adapter_name', train_embeddings=True)
```
Except for the `train_embeddings` flag, the training is the same as for just training an adapter (see [Adapter Training](training.md)).

## Saving and Loading Embeddings
You can save the embeddings by calling `save_embeddings('path/to/dir', 'name')` and load them with `load_embeddings('path/to/dir', 'name')`.

```python
model.save_embeddings(path, 'name')
model.load_embeddings(path, 'reloaded_name')
```

The path needs to be to a directory in which the weights of the embedding will be saved. 

You can also save and load the tokenizer
with the embedding by passing the tokenizer to `save_embeddings`.
```python
model.save_embeddings(path, 'name', tokenizer)
loaded_tokenizer = model.load_embeddings(path, 'name')
```
