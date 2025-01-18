Auto Classes
============

Similar to the ``AutoModel`` classes built-in into HuggingFace Transformers, adapters provides an ``AutoAdapterModel`` class.
As with other auto classes, the correct adapter model class is automatically instantiated based on the pre-trained model passed to the ``from_pretrained()`` method.

.. note::
    If the model loaded with the ``from_pretrained(...)`` function has a head, this head gets loaded as well. However, this only works for non-sharded models. If you want to load a sharded model with a head, you first need to load the model and then the head separately.

AutoAdapterModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.AutoAdapterModel
    :members:
