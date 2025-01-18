ALBERT
======

.. note::
    Adapter implementation notes for ALBERT:
        - As layers are shared between groups, adapters added to a layer are also shared between groups. Therefore, changing the adapter configuration for a layer affects the behavior of all groups that use this layer.
        - As usual, the ``leave_out`` parameter can be used to specify the layers in which adapters should be added. The layer IDs are counted by putting all layers of the groups into a sequence depending on the group number and their position in the group. I.e., for a ALBERT model with `inner_group_num=2` the first layer of the first group has ID 0, the second layer of the first group has ID 1, the first layer of the second group has ID 2, etc.


The ALBERT model was proposed in `ALBERT: A Lite BERT for Self-supervised Learning of Language Representations <https://arxiv.org/abs/1909.11942>`__
by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
It presents two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT:

- Splitting the embedding matrix into two smaller matrices.
- Using repeating layers split among groups.

AlbertAdapterModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.AlbertAdapterModel
    :members:
    :inherited-members: AlbertPreTrainedModel
