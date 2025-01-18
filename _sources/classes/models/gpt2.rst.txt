OpenAI GPT2
-----------------------------------------------------------------------------------------------------------------------

OpenAI GPT-2 model was proposed in `Language Models are Unsupervised Multitask Learners
<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_ by Alec
Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever. It's a causal (unidirectional)
transformer pretrained using language modeling on a very large corpus of ~40 GB of text data.

The abstract from the paper is the following:

*GPT-2 is a large transformer-based language model with 1.5 billion parameters, trained on a dataset[1] of 8 million
web pages. GPT-2 is trained with a simple objective: predict the next word, given all of the previous words within some
text. The diversity of the dataset causes this simple goal to contain naturally occurring demonstrations of many tasks
across diverse domains. GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than
10X the amount of data.*


GPT2AdapterModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.GPT2AdapterModel
    :members:
    :inherited-members: GPT2PreTrainedModel
