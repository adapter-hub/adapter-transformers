CLIP
=====

.. note::
    Adapter implementation notes:
        - CLIP consists of two separate Transformer encoder models, a ViT-style Transformer for visual features and a language model for textual features. Both encoders can be fitted with adapters. As usual, the ``leave_out`` parameter can be used to specify the layers in which adapters should be added. For CLIP, layer IDs are counted globally across both encoders, starting from the text encoder. I.e., for a CLIP model with 12 layers in each Transformer encoder, the text encoder will have IDs 0-11 and the vision encoder will have IDs 12-23.
        - As CLIP does not come with pre-supported task-specific prediction heads, there is currently no ``CLIPAdapterModel`` class. Use ``CLIPModel`` instead.

The CLIP model was proposed in `Learning Transferable Visual Models From Natural Language Supervision <https://arxiv.org/abs/2103.00020>`_ by Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. CLIP
(Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be
instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing
for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

The abstract from the paper is the following:

*State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This
restricted form of supervision limits their generality and usability since additional labeled data is needed to specify
any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a
much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes
with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400
million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference
learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study
the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks
such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The
model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need
for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot
without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained
model weights at this https URL.*

CLIPTextModel
~~~~~~~~~~~~~

.. autoclass:: transformers.CLIPTextModel
    :members:
    :inherited-members: CLIPPreTrainedModel

CLIPVisionModel
~~~~~~~~~~~~~~~

.. autoclass:: transformers.CLIPVisionModel
    :members:
    :inherited-members: CLIPPreTrainedModel

CLIPModel
~~~~~~~~~

.. autoclass:: transformers.CLIPModel
    :members:
    :inherited-members: CLIPPreTrainedModel
