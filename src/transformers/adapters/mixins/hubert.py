from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import ModelAdaptersMixin, ModelWithHeadsAdaptersMixin


class HubertEncoderLayerAdaptersMixin:
    """Adds adapters to the Encoder Layer module of Hubert."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class HubertEncoderLayerStableLayerNormAdaptersMixin:
    """Adds adapters to the Encoder Layer Stable Layer Norm module of Hubert."""

    def _init_adapter_modules(self):
        self.attention_adapters = AdapterLayer("mh_adapter", self.config)
        self.output_adapters = AdapterLayer("output_adapter", self.config)
        self.attention_adapters._init_adapter_modules()
        self.output_adapters._init_adapter_modules()


class HubertModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the Hubert module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.transformer.layer):
            yield i, layer


class HubertModelWithHeadsAdaptersMixin(ModelWithHeadsAdaptersMixin):
    pass
