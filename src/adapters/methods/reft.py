from typing import List, Optional

import torch
import torch.nn as nn

from ..configuration.adapter_config import ReftConfig
from ..context import ForwardContext
from .adapter_layer_base import AdapterLayerBase
from .modeling import Activation_Function_Class


class ReftUnit(nn.Module):
    def __init__(
        self,
        in_dim: int,
        r_dim: int,
        orthogonal: bool = False,
        subtract_projection: bool = True,
        non_linearity: str = None,
        dropout: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.orthogonal = orthogonal
        self.learned_source = nn.Linear(in_dim, r_dim, bias=True, dtype=dtype)

        projection = nn.Linear(in_dim, r_dim, bias=False, dtype=dtype)
        if orthogonal:
            self.projection = nn.utils.parametrizations.orthogonal(projection)
        else:
            self.projection = projection

        self.subtract_projection = subtract_projection
        self.non_linearity = Activation_Function_Class(non_linearity)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        source_states = self.non_linearity(self.learned_source(x))
        if self.subtract_projection:
            projected_states = self.projection(x)
            source_states = source_states - projected_states
        adapted_output = x + torch.matmul(source_states, self.projection.weight)
        adapted_output = self.dropout(adapted_output)
        return adapted_output


class ReftModule(nn.Module):
    def __init__(self, in_features: int, config: ReftConfig):
        super().__init__()
        self.prefix_positions = config.prefix_positions
        self.suffix_positions = config.suffix_positions
        self.tied_weights = config.tied_weights
        n_units = 1 if config.tied_weights else 2
        dtype = getattr(torch, config.dtype) if config.dtype else None
        self.units = nn.ModuleList(
            [
                ReftUnit(
                    in_features,
                    config.r,
                    config.orthogonality,
                    config.subtract_projection,
                    config.non_linearity,
                    config.dropout,
                    dtype,
                )
                for _ in range(n_units)
            ]
        )

    def _gather_adapted_states(self, hidden_states: torch.Tensor):
        context = ForwardContext.get_context()
        bsz, seq_len, ddim = hidden_states.size()

        # if cached indexing matrices are computed for different hidden_states size -> recompute
        cache_invalidated = False
        if hasattr(context, "pref_idx") and hasattr(context, "suff_idx"):
            cache_invalidated = (
                torch.max(context.suff_idx) >= seq_len  # indices out of bounds
                or bsz != context.suff_idx.size(0)  # batch size mismatch
                or ddim != context.suff_idx.size(2)  # hidden size mismatch
            )

        # no cached indexing matrices available -> compute now
        if not hasattr(context, "pref_idx") and not hasattr(context, "suff_idx") or cache_invalidated:
            # read offsets & lengths from context
            if hasattr(context, "seqlens"):
                first_non_padding = context.offsets
                last_non_padding = context.offsets + context.seqlens
            else:
                first_non_padding = torch.tensor([0] * hidden_states.size(0)).to(hidden_states.device)
                last_non_padding = torch.tensor([hidden_states.size(1)] * hidden_states.size(0)).to(
                    hidden_states.device
                )
            # create indexing matrices for prefixes & suffixes
            if self.prefix_positions > 0:
                pref_idx = first_non_padding.view(-1, 1, 1) + (
                    torch.arange(self.prefix_positions)
                    .unsqueeze(-1)
                    .expand(bsz, self.prefix_positions, ddim)
                    .to(hidden_states.device)
                )
                # Cache for next layer
                context.pref_idx = pref_idx
            if self.suffix_positions > 0:
                suff_idx = last_non_padding.view(-1, 1, 1) + (
                    torch.arange(-self.suffix_positions, 0)
                    .unsqueeze(-1)
                    .expand(bsz, self.suffix_positions, ddim)
                    .to(hidden_states.device)
                )
                context.suff_idx = suff_idx

        # gather prefix & suffix states
        if self.prefix_positions > 0:
            prefix = hidden_states.gather(1, context.pref_idx)
        else:
            prefix = torch.zeros(bsz, 0, ddim, device=hidden_states.device)
        if self.suffix_positions > 0:
            suffix = hidden_states.gather(1, context.suff_idx)
        else:
            suffix = torch.zeros(bsz, 0, ddim, device=hidden_states.device)

        if self.tied_weights:
            adapted_states = [torch.cat([prefix, suffix], dim=1)]
        else:
            adapted_states = [prefix, suffix]

        return adapted_states

    def _scatter_adapted_states(self, hidden_states: torch.Tensor, adapted_states: List[torch.Tensor]):
        context = ForwardContext.get_context()

        # merge prefix, suffix and adapted states
        adapted_output = torch.cat(adapted_states, dim=1)

        if self.prefix_positions > 0:
            hidden_states = torch.scatter(
                hidden_states, 1, context.pref_idx, adapted_output[:, : self.prefix_positions, :]
            )
        if self.suffix_positions > 0:
            hidden_states = torch.scatter(
                hidden_states, 1, context.suff_idx, adapted_output[:, -self.suffix_positions :, :]
            )

        return hidden_states

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        adapted_states = self._gather_adapted_states(hidden_states)

        # apply reft
        for i, unit in enumerate(self.units):
            adapted_states[i] = unit(adapted_states[i])

        output = self._scatter_adapted_states(hidden_states, adapted_states)

        return output


class ReftLayer(AdapterLayerBase, nn.Module):
    adapter_modules_name = "refts"

    def __init__(self, location_key: str, model_config, adapters_config):
        super().__init__()
        self.location_key = location_key + "_reft"
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.refts = nn.ModuleDict()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        reft_config = self.adapters_config.match(
            adapter_name,
            config_type=ReftConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if reft_config is not None and (reft_config.layers == "all" or self.layer_idx in reft_config.layers):
            reft = ReftModule(
                self.model_config.hidden_size,
                reft_config,
            )
            reft.train(self.training)
            self.refts[adapter_name] = reft
            return True

        return False

    def forward(self, hidden_states: torch.Tensor):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            first_adapter = adapter_setup.first()
            if first_adapter in self.refts:
                hidden_states = self.refts[first_adapter](hidden_states)

        return hidden_states

    def pre_save_adapters(self):
        # Make sure orthogonal parametrizations are contiguous, otherwise saving with safetensors will fail
        for reft in self.refts.values():
            for unit in reft.units:
                if unit.orthogonal:
                    unit.projection.parametrizations.weight[0].base = unit.projection.parametrizations.weight[
                        0
                    ].base.contiguous()


def hook_fn(module, args, output):
    if isinstance(output, torch.Tensor):
        return module.reft_layer(output)
    else:
        return (module.reft_layer(output[0]),) + output[1:]


def init_reft(model):
    model = model.base_model
    for _, layer in model.iter_layers():
        if not hasattr(layer, "reft_layer"):
            layer.reft_layer = ReftLayer("output", model.config, model.adapters_config)
            layer.register_forward_hook(hook_fn)
