MBart
-----------------------------------------------------------------------------------------------------------------------

The MBart model was presented in `Multilingual Denoising Pre-training for Neural Machine Translation
<https://arxiv.org/abs/2001.08210>`_ by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov Marjan
Ghazvininejad, Mike Lewis, Luke Zettlemoyer.

According to the abstract, MBART is a sequence-to-sequence denoising auto-encoder pretrained on large-scale monolingual
corpora in many languages using the BART objective. mBART is one of the first methods for pretraining a complete
sequence-to-sequence model by denoising full texts in multiple languages, while previous approaches have focused only
on the encoder, decoder, or reconstructing parts of the text.


MBartAdapterModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.MBartAdapterModel
    :members:
    :inherited-members: MBartPreTrainedModel
