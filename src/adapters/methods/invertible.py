import types
from functools import partial

import torch
import torch.nn as nn

from ..configuration.adapter_config import BnConfig
from ..utils import multigetattr
from .adapter_layer_base import AdapterLayerBase
from .modeling import Adapter, GLOWCouplingBlock, NICECouplingBlock


class InvertibleAdapterLayer(AdapterLayerBase, nn.ModuleDict):
    adapter_modules_name = "_modules"

    def __init__(self, model_config, adapters_config):
        super().__init__()
        self.location_key = "inv_adapter"
        self.model_config = model_config
        self.adapters_config = adapters_config

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        embedding_size = getattr(self.model_config, "embedding_size", self.model_config.hidden_size)
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=BnConfig,
            location_key="inv_adapter",
        )
        if adapter_config is not None and adapter_config["inv_adapter"]:
            if adapter_config["inv_adapter"] == "nice":
                inv_adap = NICECouplingBlock(
                    [[embedding_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            elif adapter_config["inv_adapter"] == "glow":
                inv_adap = GLOWCouplingBlock(
                    [[embedding_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            else:
                raise ValueError(f"Invalid invertible adapter type '{adapter_config['inv_adapter']}'.")
            self[adapter_name] = inv_adap
            self[adapter_name].apply(Adapter.init_bert_weights)
            return True

        return False

    def get_invertible_adapter(self):
        # HACK: returns the first adapter of the currently active setup. for backwards compatibility
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None and len(adapter_setup) > 0:
            first_adapter = adapter_setup.first()
            if first_adapter in self:
                return self[first_adapter]
        return None

    def forward(self, hidden_states: torch.Tensor, rev=False):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None and len(adapter_setup) > 0:
            first_adapter = adapter_setup.first()
            if first_adapter in self:
                hidden_states = self[first_adapter](hidden_states, rev=rev)
        return hidden_states


def hook_fn(model, module, args, embedding_output):
    embedding_output = model.invertible_adapters(embedding_output)
    return embedding_output


def inv_hook_fn(model, module, args):
    inv_output = model.invertible_adapters(args[0], rev=True)
    return (inv_output,) + args[1:]


def init_invertible_adapters(model):
    base_model = model.base_model
    if not hasattr(base_model, "invertible_adapters"):
        base_model.invertible_adapters = InvertibleAdapterLayer(base_model.config, base_model.adapters_config)

        embed_layer = multigetattr(base_model, base_model.adapter_interface.model_embeddings)
        embed_layer.register_forward_hook(partial(hook_fn, base_model))

        # Add methods from original invertible adapter mixin.
        # This is primarily for backwards compatibility and internal use.
        base_model.add_invertible_adapter = types.MethodType(
            lambda self, *args, **kwargs: self.invertible_adapters.add_adapter(*args, **kwargs), base_model
        )
        base_model.delete_invertible_adapter = types.MethodType(
            lambda self, *args, **kwargs: self.invertible_adapters.delete_adapter(*args, **kwargs), base_model
        )
        base_model.get_invertible_adapter = types.MethodType(
            lambda self: self.invertible_adapters.get_invertible_adapter(), base_model
        )
        base_model.invertible_adapters_forward = types.MethodType(
            lambda self, *args, **kwargs: self.invertible_adapters(*args, **kwargs), base_model
        )

        # Register reverse forward pass
        if output_embedding := model.get_output_embeddings():
            output_embedding.register_forward_pre_hook(partial(inv_hook_fn, base_model))
