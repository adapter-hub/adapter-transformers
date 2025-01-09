from typing import Iterable, Tuple

import torch.nn as nn

import adapters

from ...model_mixin import (
    EmbeddingAdaptersMixin,
    InvertibleAdaptersMixin,
    ModelAdaptersMixin,
    ModelUsingSubmodelsAdaptersMixin,
)


class EncoderDecoderModelAdaptersMixin(
    EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelUsingSubmodelsAdaptersMixin
):
    """Adds adapters to the EncoderDecoderModel class."""

    support_prompt_tuning = False

    def init_adapters(self, model_config, adapters_config):
        if not isinstance(self.encoder, ModelAdaptersMixin) or not isinstance(self.decoder, ModelAdaptersMixin):
            return

        # Before initializing adapters, forward adding invertible adapters to the encoder
        self.add_invertible_adapter = self.encoder.base_model.add_invertible_adapter

        super().init_adapters(model_config, adapters_config)

        # ensure that encoder and decoder use the same shared parameters
        if hasattr(self.encoder, "set_shared_parameters"):
            self.encoder.set_shared_parameters(self.shared_parameters)
        if hasattr(self.decoder, "set_shared_parameters"):
            self.decoder.set_shared_parameters(self.shared_parameters)

        # Relay all invertible adapter calls to encoder
        self.invertible_adapters = self.encoder.base_model.invertible_adapters
        self.get_invertible_adapter = self.encoder.base_model.get_invertible_adapter
        self.enable_invertible_adapters = self.encoder.base_model.enable_invertible_adapters
        self.invertible_adapters_forward = self.encoder.base_model.invertible_adapters_forward

        # Decoder should use invertible adapters of encoder
        self.decoder.base_model.invertible_adapters = self.encoder.base_model.invertible_adapters
        self.decoder.base_model.add_invertible_adapter = lambda *args: None
        self.decoder.base_model.get_invertible_adapter = self.encoder.base_model.get_invertible_adapter

        # Patched invertible adapters forward for decoder:
        # In the decoder, only the reverse pass in the LM head should be active.
        # Decoder inputs should not be forwarded through the invertible adapter again in its embeddings module.
        def decoder_invertible_adapters_forward(hidden_states, rev=False):
            if rev:
                return self.encoder.base_model.invertible_adapters_forward(hidden_states, rev=True)
            else:
                return hidden_states

        self.decoder.base_model.invertible_adapters_forward = decoder_invertible_adapters_forward

    def init_submodels(self):
        adapters.init(self.encoder, self.adapters_config)
        adapters.init(self.decoder, self.adapters_config)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        # Can't count the layers globally, because then saving and loading pretrained models won't work
        for i, layer in self.encoder.iter_layers():
            yield i, layer

        for i, layer in self.decoder.iter_layers():
            yield i, layer
