from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from ...composition import adjust_tensors_for_parallel
from ...methods.bottleneck import BottleneckLayer
from ...methods.lora import LoRALinear
from ...methods.prefix_tuning import PrefixTuningLayer
from ...model_mixin import (
    EmbeddingAdaptersMixin,
    EmbeddingAdaptersWrapperMixin,
    InvertibleAdaptersMixin,
    InvertibleAdaptersWrapperMixin,
    ModelBaseAdaptersMixin,
)
from ...utils import patch_forward


class M2M100AttentionAdaptersMixin:
    """Adds adapters to the M2M100Attention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.k_proj = LoRALinear.wrap(self.k_proj, "selfattn", model_config, adapters_config, attn_key="k")
        self.v_proj = LoRALinear.wrap(self.v_proj, "selfattn", model_config, adapters_config, attn_key="v")
        self.q_proj = LoRALinear.wrap(self.q_proj, "selfattn", model_config, adapters_config, attn_key="q")

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None, model_config, adapters_config
        )
        patch_forward(self)


class M2M100EncoderLayerAdaptersMixin:
    """Adds adapters to the M2M100EncoderLayer module."""

    def init_adapters(self, model_config, adapters_config):
        self.adapters_config = adapters_config
        # Wrap layers for LoRA
        self.fc1 = LoRALinear.wrap(self.fc1, "intermediate", model_config, adapters_config)
        self.fc2 = LoRALinear.wrap(self.fc2, "output", model_config, adapters_config)

        # Set attention layer location key for prefix tuning
        self.self_attn.location_key = "encoder"
        self.attention_adapters = BottleneckLayer("mh_adapter")
        self.output_adapters = BottleneckLayer("output_adapter")

        patch_forward(self)


class M2M100DecoderLayerAdaptersMixin(M2M100EncoderLayerAdaptersMixin):
    """Adds adapters to the M2M100DecoderLayer module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)
        # Set attention layer location key for prefix tuning
        self.self_attn.location_key = "self"
        self.encoder_attn.location_key = "cross"
        self.cross_attention_adapters = BottleneckLayer("cross_adapter")


class M2M100EncoderAdaptersMixin(InvertibleAdaptersMixin):
    """Adds adapters to the M2M100Encoder module."""

    pass


class M2M100DecoderAdaptersMixin:
    """Adds adapters to the M2M100Decoder module."""

    def init_adapters(self, model_config, adapters_config):
        patch_forward(self)

    def forward(
        self, input_ids: torch.LongTensor = None, encoder_hidden_states: Optional[torch.FloatTensor] = None, **kwargs
    ):
        (input_ids,) = adjust_tensors_for_parallel(encoder_hidden_states, input_ids)
        return super().forward(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, **kwargs)


class M2M100ModelAdaptersMixin(EmbeddingAdaptersMixin, InvertibleAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the M2M100Model class."""

    invertible_adapters_base_name = "encoder"
    support_prompt_tuning = False

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)
        self.encoder.layer_norm.register_forward_hook(self.post_embedding_forward)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        if hasattr(self, "encoder"):
            for i, layer in enumerate(self.encoder.layers):
                yield i, layer
            for i, layer in enumerate(self.decoder.layers, start=len(self.encoder.layers)):
                yield i, layer
        else:
            for i, layer in enumerate(self.decoder.layers):
                yield i, layer

    def post_embedding_forward(self, module, args, embedding_output):
        embedding_output = self.invertible_adapters_forward(embedding_output)
        # Prompt tuning not yet supported
        return embedding_output


class M2M100DecoderWrapperAdaptersMixin(EmbeddingAdaptersWrapperMixin, ModelBaseAdaptersMixin):
    """Adds adapters to the M2M100DecoderWrapper module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.decoder.layers):
            yield i, layer

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()
