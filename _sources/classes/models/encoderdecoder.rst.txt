.. 
    Copyright 2020 The HuggingFace Team. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
    the License. You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
    an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
    specific language governing permissions and limitations under the License.

Encoder Decoder Models
-----------------------------------------------------------------------------------------------------------------------

.. note::
    Adapter implementation notes:
        - Unlike other models, an explicit EncoderDecoderAdapterModel for the EncoderDecoderModel has not been implemented. This decision was made due to the lack of support for the EncoderDecoderModel in Hugging Face Transformers' ``AutoModel`` class. As a result, our ``AutoAdapterModel`` class would not support the EncoderDecoderAdapterModel either. Thus, to use an EncoderDecoderModel with *Adapters*, follow these steps:

            1. First, create an :class:`~transformers.EncoderDecoderModel` instance, for example, using ``model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")``.
            2. Next, convert this model to an adapter model using the ``adapters.init(model)`` function.

        - Adapters can be added to both the encoder and the decoder. As usual, the ``leave_out`` parameter can be used to specify the layers where adapters are to be added. For the EncoderDecoderModel the layer IDs are counted seperately over the encoder and decoder starting from 0. Thus, specifying ``leave_out=[0,1]`` will leave out the first and second layer of the encoder and the first and second layer of the decoder.

The :class:`~transformers.EncoderDecoderModel` can be used to initialize a sequence-to-sequence model with any
pretrained autoencoding model as the encoder and any pretrained autoregressive model as the decoder.

The effectiveness of initializing sequence-to-sequence models with pretrained checkpoints for sequence generation tasks
was shown in `Leveraging Pre-trained Checkpoints for Sequence Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by
Sascha Rothe, Shashi Narayan, Aliaksei Severyn.

After such an :class:`~transformers.EncoderDecoderModel` has been trained/fine-tuned, it can be saved/loaded just like
any other models (see the examples for more information).

An application of this architecture could be to leverage two pretrained :class:`~transformers.BertModel` as the encoder
and decoder for a summarization model as was shown in: `Text Summarization with Pretrained Encoders
<https://arxiv.org/abs/1908.08345>`__ by Yang Liu and Mirella Lapata.

EncoderDecoderModel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: transformers.EncoderDecoderModel
    :members: forward, from_encoder_decoder_pretrained
