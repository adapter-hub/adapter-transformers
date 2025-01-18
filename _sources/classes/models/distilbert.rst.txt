DistilBERT
===========

The DistilBERT model was proposed in the blog post
`Smaller, faster, cheaper, lighter: Introducing DistilBERT, a distilled version of BERT <https://medium.com/huggingface/distilbert-8cf3380435b5>`__,
and the paper `DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter <https://arxiv.org/abs/1910.01108>`__.
DistilBERT is a small, fast, cheap and light Transformer model trained by distilling Bert base. It has 40% less
parameters than `bert-base-uncased`, runs 60% faster while preserving over 95% of Bert's performances as measured on
the GLUE language understanding benchmark.


DistilBertAdapterModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.DistilBertAdapterModel
    :members:
    :inherited-members: DistilBertPreTrainedModel
