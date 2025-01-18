MT5
=====

The mT5 model was presented in `mT5: A massively multilingual pre-trained text-to-text transformer
<https://arxiv.org/pdf/2010.11934.pdf>`__ by Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou,
Aditya Siddhant, Aditya Barua, Colin Raffel.

The abstract from the paper is the following,


- The recent "Text-to-Text Transfer Transformer" (T5) leveraged a unified text-to-text format and scale to attain
  state-of-the-art results on a wide variety of English-language NLP tasks. In this paper, we introduce mT5, a
  multilingual variant of T5 that was pre-trained on a new Common Crawl-based dataset covering 101 languages. We detail
  the design and modified training of mT5 and demonstrate its state-of-the-art performance on many multilingual
  benchmarks. We also describe a simple technique to prevent "accidental translation" in the zero-shot setting, where a
  generative model chooses to (partially) translate its prediction into the wrong language. All of the code and model
  checkpoints used in this work are publicly available.

MT5AdapterModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.MT5AdapterModel
    :members:
    :inherited-members: MT5PreTrainedModel