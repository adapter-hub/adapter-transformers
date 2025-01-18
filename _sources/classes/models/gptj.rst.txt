EleutherAI GPT-J-6B
-----------------------------------------------------------------------------------------------------------------------

EleutherAI GPT-J-6B is an open source, autoregressive language model created by a group of researchers called
EleutherAI. It's one of the most advanced alternatives to OpenAI's GPT-3 and performs well on a wide array of
natural language tasks such as chat, summarization, and question answering, to name a few.

For a deeper dive, GPT-J is a transformer model trained using Ben Wang's Mesh Transformer JAX `Mesh Transformer JAX
<https://github.com/kingoflolz/mesh-transformer-jax/>`_. "GPT" is short for
generative pre-trained transformer, "J" distinguishes this model from other GPT models, and "6B" represents the 6
billion trainable parameters.

The model consists of 28 layers with a model dimension of 4096, and a feedforward dimension of 16384. The model
dimension is split into 16 heads, each with a dimension of 256. Rotary Position Embedding (RoPE) is applied to
64 dimensions of each head. The model is trained with a tokenization vocabulary of 50257, using the same set of
BPEs as GPT-2/GPT-3.


GPTJAdapterModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: adapters.GPTJAdapterModel
    :members:
    :inherited-members: GPTJPreTrainedModel
