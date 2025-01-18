Model Mixins
=======================

These classes provide the basis of adapter module integration into model classes such as adapter saving and loading.
Depending on the model, one of these mixins should be implemented by every adapter-supporting model class.

InvertibleAdaptersMixin
----------------------------------

.. autoclass:: adapters.InvertibleAdaptersMixin
    :members:


EmbeddingAdaptersMixin
----------------------------------

.. autoclass:: adapters.EmbeddingAdaptersMixin
    :members:


ModelAdaptersMixin
------------------

.. autoclass:: adapters.ModelAdaptersMixin
    :members:

ModelWithHeadsAdaptersMixin
----------------------------------

.. autoclass:: adapters.ModelWithHeadsAdaptersMixin
    :members:

ModelWithFlexibleHeadsAdaptersMixin
---------------------------------------

.. autoclass:: adapters.ModelWithFlexibleHeadsAdaptersMixin
    :members:

PushAdapterToHubMixin
----------------------

.. autoclass:: adapters.hub_mixin.PushAdapterToHubMixin
    :members:
