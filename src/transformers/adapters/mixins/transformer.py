import logging
from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


logger = logging.getLogger(__name__)


# For backwards compatibility, TransformerSelfOutput inherits directly from AdapterLayer
class TransformerSelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the TransformerSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter", None)


# For backwards compatibility, TransformerOutput inherits directly from AdapterLayer
class TransformerOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the TransformerOutput module."""

    def __init__(self):
        super().__init__("output_adapter", None)


class TransformerModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the TransformerModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer
