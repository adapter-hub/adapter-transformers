from functools import partial
from typing import List, Mapping, NamedTuple, Optional, Union

import torch
from torch import nn

from ..composition import (
    AdapterCompositionBlock,
    Average,
    BatchSplit,
    Fuse,
    Parallel,
    Split,
    Stack,
    adjust_tensors_for_parallel,
)
from ..configuration import BnConfig
from ..context import ForwardContext
from ..utils import multigetattr
from .adapter_layer_base import ComposableAdapterLayerBase
from .modeling import Adapter, BertFusion, ParallelAdapter


LAYER_HOOK_UNSUPPORTED = [
    ("original_ln_after", False),
]


class BottleneckState(NamedTuple):
    """
    Models the input and output states of a bottleneck adapter layer.

    Args:
        hidden_states (torch.Tensor): The layer input/ output hidden states.
        input_tensor (torch.Tensor): The Transformer sub-block residual connection inputs.
        adapter_residual (torch.Tensor): The adapter residual connection inputs.
        layer_norm (torch.nn.Module, optional): The Transformer layer norm module.
        bottleneck_up (torch.Tensor, optional):
            The up-projected bottleneck MLP output. This is only for Fuse compositions.
        last (str, optional): Name of the last adapter applied in the composition.
    """

    hidden_states: torch.Tensor
    input_tensor: torch.Tensor
    adapter_residual: torch.Tensor
    layer_norm: Optional[torch.nn.Module]
    bottleneck_up: Optional[torch.Tensor] = None
    last: Optional[str] = None


class BottleneckLayer(ComposableAdapterLayerBase, nn.Module):
    adapter_modules_name = "adapters"
    supported_compositions = [Stack, Fuse, Split, Parallel, BatchSplit, Average]

    def __init__(self, location_key: str, is_layer_hooked: bool = False):
        super().__init__()
        self.location_key = location_key
        self.is_layer_hooked = is_layer_hooked

    def init_adapters(self, model_config, adapters_config):
        self._init_mapping()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.adapters = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        if not hasattr(self, "is_layer_hooked"):
            self.is_layer_hooked = False

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=BnConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if adapter_config is not None:
            reduction_factor = adapter_config["reduction_factor"]
            if isinstance(reduction_factor, Mapping):
                if str(self.layer_idx) in reduction_factor:
                    reduction_factor = reduction_factor[str(self.layer_idx)]
                elif "default" in reduction_factor:
                    reduction_factor = reduction_factor["default"]
                else:
                    raise KeyError(
                        "The given reduction factor mapping does not give a default value and does not specify each "
                        "reduction factor individually. You need to provide a default value like this: "
                        '{"1": 16, "default": 16}'
                    )

            # check unsupported configurations for layer hooking mode
            if self.is_layer_hooked:
                for key, value in LAYER_HOOK_UNSUPPORTED:
                    if adapter_config.get(key, None) == value:
                        raise ValueError(
                            f"Unsupported configuration for bottleneck layer hooking mode: {key}={value}. "
                            "Please set this configuration to a supported value."
                        )

            if adapter_config.is_parallel:
                adapter_class = ParallelAdapter
            else:
                adapter_class = Adapter
            adapter = adapter_class(
                adapter_name=adapter_name,
                input_size=self.model_config.hidden_size,
                down_sample=int(self.model_config.hidden_size // reduction_factor),
                config=adapter_config,
            )
            # for adapters hooked via interface:
            # residual & LN are applied by model, so don't apply in adapters
            if self.is_layer_hooked:
                adapter.original_ln_after = False
            adapter.train(self.training)  # make sure training mode is consistent
            self.adapters[adapter_name] = adapter
            return True

        return False

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        """See BertModel.add_fusion_layer"""
        fusion_name = ",".join(adapter_names) if isinstance(adapter_names, list) else adapter_names
        fusion_config, adapter_names = self.adapters_config.get_fusion(fusion_name)
        if self.adapters_config.common_config_value(adapter_names, self.location_key):
            dropout_prob = fusion_config.dropout_prob or getattr(self.model_config, "attention_probs_dropout_prob", 0)
            fusion = BertFusion(
                fusion_config,
                self.model_config.hidden_size,
                dropout_prob,
            )
            fusion.train(self.training)  # make sure training mode is consistent
            self.adapter_fusion_layer[fusion_name] = fusion

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        if adapter_names in self.adapter_fusion_layer:
            del self.adapter_fusion_layer[adapter_names]

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        """
        Unfreezes a given list of adapters, the adapter fusion layer, or both

        Args:
            adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
            unfreeze_adapters: whether the adapter weights should be activated
            unfreeze_fusion: whether the adapter fusion layer for the given adapters should be activated
        """
        if unfreeze_adapters:
            for adapter_name in adapter_setup.flatten():
                if adapter_name in self.adapters:
                    for param in self.adapters[adapter_name].parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_setup, Fuse):
                if adapter_setup.name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[adapter_setup.name].parameters():
                        param.requires_grad = True
            for sub_setup in adapter_setup:
                if isinstance(sub_setup, Fuse):
                    if sub_setup.name in self.adapter_fusion_layer:
                        for param in self.adapter_fusion_layer[sub_setup.name].parameters():
                            param.requires_grad = True

    def get_adapter_fusion(self, adapter_names: Union[List, str]):
        adapter_names = adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        if adapter_names in self.adapter_fusion_layer:
            return self.adapter_fusion_layer[adapter_names]
        else:
            return None

    def pre_block(self, adapter_setup: Union[AdapterCompositionBlock, str], state: BottleneckState) -> BottleneckState:
        if isinstance(adapter_setup, AdapterCompositionBlock):
            adapter_name = adapter_setup.first()
        else:
            adapter_name = adapter_setup
        first_adapter = self.adapters[adapter_name]
        hidden_states, _, residual = first_adapter.pre_forward(
            state.hidden_states, state.input_tensor, state.layer_norm
        )

        return state._replace(hidden_states=hidden_states, adapter_residual=residual)

    def vslice(self, state: BottleneckState, slice_obj: slice) -> BottleneckState:
        return BottleneckState(
            state.hidden_states[slice_obj],
            state.input_tensor[slice_obj],
            state.adapter_residual[slice_obj],
            state.layer_norm,
            state.bottleneck_up[slice_obj] if state.bottleneck_up is not None else None,
            state.last,
        )

    def pad_and_concat(self, states: List[BottleneckState]) -> BottleneckState:
        return BottleneckState(
            torch.cat([state.hidden_states for state in states], dim=0),
            torch.cat([state.input_tensor for state in states], dim=0),
            torch.cat([state.adapter_residual for state in states], dim=0),
            states[0].layer_norm,
            (
                torch.cat([state.bottleneck_up for state in states], dim=0)
                if states[0].bottleneck_up is not None
                else None
            ),
            states[-1].last,
        )

    def repeat(self, state: BottleneckState, channels: int) -> BottleneckState:
        return BottleneckState(
            state.hidden_states.repeat(channels, 1, 1),
            state.input_tensor.repeat(channels, 1, 1),
            state.adapter_residual.repeat(channels, 1, 1),
            state.layer_norm,
            state.bottleneck_up.repeat(channels, 1, 1) if state.bottleneck_up is not None else None,
            state.last,
        )

    def mean(self, states: List[BottleneckState], weights: torch.Tensor) -> BottleneckState:
        return BottleneckState(
            torch.mean(torch.stack([s.hidden_states for s in states], 0) * weights, dim=0),
            states[0].input_tensor,
            states[0].adapter_residual,
            states[0].layer_norm,
            states[0].bottleneck_up,
            states[-1].last,
        )

    def compose_single(self, adapter_setup: str, state: BottleneckState, lvl: int = 0) -> BottleneckState:
        adapter_layer = self.adapters[adapter_setup]
        context = ForwardContext.get_context()
        output_gating = context.output_adapter_gating_scores if context is not None else False
        layer_output = adapter_layer(
            state.hidden_states,
            residual_input=state.adapter_residual,
            output_gating=output_gating,
        )
        hidden_states, up = layer_output[0], layer_output[2]
        if output_gating:
            self._store_gating_score(adapter_setup, layer_output[-1])

        return state._replace(hidden_states=hidden_states, bottleneck_up=up, last=adapter_setup)

    def compose_fuse(self, adapter_setup: Fuse, state: BottleneckState, lvl: int = 0):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        context = ForwardContext.get_context()

        # config of _last_ fused adapter is significant
        fusion_config, _ = self.adapters_config.get_fusion(adapter_setup.name)
        last = adapter_setup.last()
        last_adapter = self.adapters[last]
        hidden_states, query, residual = last_adapter.pre_forward(
            state.hidden_states, state.input_tensor, state.layer_norm, fusion_config=fusion_config
        )
        state = state._replace(hidden_states=hidden_states, adapter_residual=residual)

        children_states = []
        for child in adapter_setup:
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            else:
                pass

        if len(children_states) > 0:
            up_list = torch.stack([state.bottleneck_up for state in children_states])
            up_list = up_list.permute(1, 2, 0, 3)

            output_fusion_attns = context.output_adapter_fusion_attentions if context is not None else False
            fusion_output = self.adapter_fusion_layer[adapter_setup.name](
                query,
                up_list,
                up_list,
                state.adapter_residual,
                output_attentions=output_fusion_attns,
            )
            if output_fusion_attns:
                hidden_states = fusion_output[0]
                self._store_fusion_attentions(adapter_setup.name, fusion_output[-1])
            else:
                hidden_states = fusion_output

        return state._replace(hidden_states=hidden_states, last=last)

    def compose_split(self, adapter_setup: Split, state: BottleneckState, lvl: int = 0):
        """
        Splits the given input between the given adapters.
        """
        if sum(adapter_setup.splits) != state.hidden_states.shape[1]:
            raise IndexError(
                "The given input has sequence length {} which is not equal to the sum of splits {}".format(
                    state.hidden_states.shape[1], adapter_setup.splits
                )
            )

        state = self.pre_block(adapter_setup, state)

        children_states = []
        last = None
        for i, child in enumerate(adapter_setup):
            batch_idx = (
                sum(adapter_setup.splits[:i]),
                sum(adapter_setup.splits[: i + 1]),
            )
            child_state = BottleneckState(
                state.hidden_states[:, batch_idx[0] : batch_idx[1], :],
                state.input_tensor[:, batch_idx[0] : batch_idx[1], :],
                state.adapter_residual[:, batch_idx[0] : batch_idx[1], :],
                state.layer_norm,
                state.bottleneck_up[:, batch_idx[0] : batch_idx[1], :] if state.bottleneck_up is not None else None,
            )
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, child_state, lvl=lvl + 1)
                children_states.append(child_state)
                last = child_state.last or last
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, child_state, lvl=lvl + 1)
                children_states.append(child_state)
                last = child_state.last or last
            else:
                pass

        hidden_states = torch.cat([child.hidden_states for child in children_states], dim=1)
        return state._replace(hidden_states=hidden_states, last=last)

    def bottleneck_layer_forward(self, hidden_states, residual_input, layer_norm):
        """Forward pass through the adapter layer.
        NOTE: This method should only be called if the calling module directly inherits from BottleneckLayer.
        Otherwise, call the regular forward() method.

        Args:
            hidden_states (torch.Tensor): Input hidden states to the adapter layer.
            residual_input (torch.Tensor): Residual input to the adapter layer.
            layer_norm (torch.nn.Module): Transformer layer normalization module to be used by the adapter layer.

        Returns:
            torch.Tensor: Output hidden states of the adapter layer.
        """
        # Batch sizes might be different due to prefix tuning w. Parallel block
        if residual_input is not None:
            (residual_input,) = adjust_tensors_for_parallel(hidden_states, residual_input)
            # Replicate in both directions as residual might be larger (e.g. GPT-J)
            (hidden_states,) = adjust_tensors_for_parallel(residual_input, hidden_states)
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            input_hidden_states = hidden_states

            state = BottleneckState(hidden_states, residual_input, residual_input, layer_norm)
            state = self.compose(adapter_setup, state)
            hidden_states, residual_input, _, _, _, last = state

            last_adapter = self.adapters[last]
            hidden_states = last_adapter.post_forward(hidden_states, input_hidden_states, residual_input, layer_norm)

        elif layer_norm is not None and not self.is_layer_hooked:
            hidden_states = layer_norm(hidden_states + residual_input)
        elif residual_input is not None and not self.is_layer_hooked:
            hidden_states = hidden_states + residual_input

        return hidden_states

    def forward(self, hidden_states, residual_input, layer_norm):
        """Forward pass through the adapter layer.

        Args:
            hidden_states (torch.Tensor): Input hidden states to the adapter layer.
            residual_input (torch.Tensor): Residual input to the adapter layer.
            layer_norm (torch.nn.Module): Transformer layer normalization module to be used by the adapter layer.

        Returns:
            torch.Tensor: Output hidden states of the adapter layer.
        """
        return self.bottleneck_layer_forward(hidden_states, residual_input, layer_norm)


def hook_fn(adapter_layer, ln_get_fn, module, args, output):
    # Retrieve residual from previous hook, if existing
    context = ForwardContext.get_context()
    residual_input = getattr(context, f"{adapter_layer.location_key}_residual_input", None)
    # Retrieve layer norm from getter fn
    if ln_get_fn is not None:
        layer_norm = ln_get_fn()
    else:
        layer_norm = None
    # Call adapter layer
    if isinstance(output, torch.Tensor):
        return adapter_layer(output, residual_input, layer_norm)
    else:
        return (adapter_layer(output[0], residual_input, layer_norm),) + output[1:]


def _residual_hook_fn(location_key, module, args):
    context = ForwardContext.get_context()
    if context is not None:
        setattr(context, f"{location_key}_residual_input", args[0])


def init_bottleneck(model):
    model = model.base_model
    for _, layer in model.iter_layers():
        if self_attn := multigetattr(layer, model.adapter_interface.layer_self_attn, None):
            if o_proj := multigetattr(self_attn, model.adapter_interface.attn_o_proj, None):
                if not hasattr(layer, "attention_adapters"):
                    layer.attention_adapters = BottleneckLayer("mh_adapter", is_layer_hooked=True)
                    ln_1_get_fn = lambda: multigetattr(layer, model.adapter_interface.layer_ln_1, None)
                    o_proj.register_forward_hook(partial(hook_fn, layer.attention_adapters, ln_1_get_fn))
        if layer_output_proj := multigetattr(layer, model.adapter_interface.layer_output_proj, None):
            if not hasattr(layer, "output_adapters"):
                layer.output_adapters = BottleneckLayer("output_adapter", is_layer_hooked=True)
                ln_2_get_fn = lambda: multigetattr(layer, model.adapter_interface.layer_ln_2, None)
                layer_output_proj.register_forward_hook(partial(hook_fn, layer.output_adapters, ln_2_get_fn))
        if cross_attn := multigetattr(layer, model.adapter_interface.layer_cross_attn, None):
            if not hasattr(cross_attn, "cross_attention_adapters"):
                layer.attention_adapters = BottleneckLayer("cross_adapter", is_layer_hooked=True)
                cross_attn.register_forward_hook(partial(hook_fn, layer.attention_adapters, None))

        if model.adapter_interface.layer_pre_self_attn is not None:
            if pre_self_attn := multigetattr(layer, model.adapter_interface.layer_pre_self_attn, None):
                pre_self_attn.register_forward_pre_hook(partial(_residual_hook_fn, "mh_adapter"))
        if model.adapter_interface.layer_pre_cross_attn is not None:
            if pre_cross_attn := multigetattr(layer, model.adapter_interface.layer_pre_cross_attn, None):
                pre_cross_attn.register_forward_pre_hook(partial(_residual_hook_fn, "cross_adapter"))
        if model.adapter_interface.layer_pre_ffn is not None:
            if pre_ffn := multigetattr(layer, model.adapter_interface.layer_pre_ffn, None):
                pre_ffn.register_forward_pre_hook(partial(_residual_hook_fn, "output_adapter"))
