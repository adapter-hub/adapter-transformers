PLBART
======

The PLBART model was proposed in [Unified Pre-training for Program Understanding and Generation](https://arxiv.org/abs/2103.06333) by Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang.
This is a BART-like model which can be used to perform code-summarization, code-generation, and code-translation tasks. The pre-trained model `plbart-base` has been trained using multilingual denoising task
on Java, Python and English.

According to the abstract,

- PLBART is a sequence-to-sequence model capable of performing a broad spectrum of program and language understanding and generation tasks
- PLBART is pre-trained on an extensive collection of Java and Python functions and associated NL text via denoising autoencoding.
- PLBART learns program syntax, style (e.g., identifier naming convention) and logical flow.


PLBartAdapterModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.PLBartAdapterModel
    :members:
    :inherited-members: PLBartPretrainedModel
