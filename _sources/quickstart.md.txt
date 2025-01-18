# Quick Start

## Introduction

`adapters` adds adapter functionality to the PyTorch implementations of all Transformer models listed in the [Model Overview](https://docs.adapterhub.ml/model_overview.html).
For working with adapters, a couple of methods, e.g. for creation (`add_adapter()`), loading (`load_adapter()`), 
storing (`save_adapter()`) and deletion (`delete_adapter()`) are added to the model classes.
In the following, we will briefly go through some examples to showcase these methods.

```{eval-rst}
.. note::
    This document focuses on the adapter-related functionalities added by ``adapters``.
    For a more general overview of the *transformers* library, visit
    `the 'Usage' section in Hugging Face's documentation <https://huggingface.co/docs/transformers/main/en/quicktour>`_.
```

## Initialize a Model with Adapters

The `XAdapterModel` is the recommended model for training and inference of adapters:

```
from adapters import AutoAdapterModel

model = AutoAdapterModel.from_pretrained(model_name)
````

This handles the initialization of the adapter-related functionality internally and provides you with the initialized model. The `XAdapterModel` also supports the dynamic adding, loading, and storing of heads for different tasks.


If you want to use adapters in Hugging Face models, the models need to be initialized with the adapters library. This initializes the functionality of adding, loading and storing of adapters within the `transformers` models. 

```
import adapters

adapters.init(model)
```


## Using a Pre-Trained Adapter for Inference

_We also have a Quickstart Colab notebook for adapter inference:_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/02_Adapter_Inference.ipynb)

The following example shows the usage of a basic pre-trained Transformer model with adapters.
Our goal here is to predict the sentiment of a given sentence.

We use BERT in this example, so we first load a pre-trained `BertTokenizer` to encode the input sentence and a pre-trained
`bert-base-uncased` checkpoint from Hugging Face's Model Hub using the [`BertAdapterModel`](adapters.BertAdapterModel) class:

```python
import os

import torch
from transformers import BertTokenizer
from adapters import BertAdapterModel

# Load pre-trained BERT tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# An input sentence
sentence = "It's also, clearly, great fun."

# Tokenize the input sentence and create a PyTorch input tensor
input_data = tokenizer(sentence, return_tensors="pt")

# Load pre-trained BERT model from Hugging Face Hub
# The `BertAdapterModel` class is specifically designed for working with adapters
# It can be used with different prediction heads
model = BertAdapterModel.from_pretrained('bert-base-uncased')
```

Having loaded the model, we now add a pre-trained task adapter that is useful to our task from AdapterHub.
In this case, for sentiment classification, we thus use [an adapter trained on the SST-2 dataset](https://adapterhub.ml/adapters/ukp/bert-base-uncased_sentiment_sst-2_pfeiffer/).
The task prediction head loaded together with the adapter gives us a class label for our sentence:

```python
# Load pre-trained task adapter from Adapter Hub
# This method call will also load a pre-trained classification head for the adapter task
adapter_name = model.load_adapter("sentiment/sst-2@ukp", config='pfeiffer')

# Activate the adapter we just loaded, so that it is used in every forward pass
model.set_active_adapters(adapter_name)

# Predict output tensor
outputs = model(**input_data)

# Retrieve the predicted class label
predicted = torch.argmax(outputs[0]).item()
assert predicted == 1
```

To save our pre-trained model and adapters, we can easily store and reload them as follows:

```python
# For the sake of this demonstration an example path for loading and storing is given below
example_path = os.path.join(os.getcwd(), "adapter-quickstart")

# Save model
model.save_pretrained(example_path)
# Save adapter
model.save_adapter(example_path, adapter_name)

# Load model, similar to Hugging Face's AutoModel class, 
# you can also use AutoAdapterModel instead of BertAdapterModel
model = AutoAdapterModel.from_pretrained(example_path)
model.load_adapter(example_path)
```

Similar to how the weights of the full model are saved, [`save_adapter()`](adapters.ModelWithHeadsAdaptersMixin.save_adapter) will create a file for saving the adapter weights and a file for saving the adapter configuration in the specified directory.

Finally, if we have finished working with adapters, we can restore the base Transformer to its original form by deactivating and deleting the adapter:

```python
# Deactivate all adapters
model.set_active_adapters(None)
# Delete the added adapter
model.delete_adapter(adapter_name)
```

## Adapter training

_We also have a Quickstart Colab notebook for adapter training:_ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adapter-Hub/adapters/blob/main/notebooks/01_Adapter_Training.ipynb)

For more examples of training different adapter setups, refer to the section on [Adapter Training](training.md).
Further information on using adapters with prediction heads can be found in the [Prediction Heads](prediction_heads.md) section.
