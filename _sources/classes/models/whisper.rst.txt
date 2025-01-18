Whisper
-----------------------------------------------------------------------------------------------------------------------

The Whisper model was presented in `Robust Speech Recognition via Large-Scale Weak Supervision
<https://arxiv.org/abs/2212.04356>`_ by Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine
McLeavey, Ilya Sutskever. 

Whisper is a state-of-the-art speech recognition model trained on 680,000 hours of multilingual and multitask data, presented by OpenAI.

The abstract from the paper is the following:

*We study the capabilities of speech processing systems trained simply to predict large amounts of
transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask
supervision, the resulting models generalize well to standard benchmarks and are often competitive
with prior fully supervised results but in a zeroshot transfer setting without the need for any finetuning. When compared to humans, the models
approach their accuracy and robustness. We are releasing models and inference code to serve as
a foundation for further work on robust speech processing.*


WhisperAdapterModel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.WhisperAdapterModel
    :members:
    :inherited-members: WhisperPreTrainedModel