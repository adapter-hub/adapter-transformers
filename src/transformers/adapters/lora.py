# Code adapted from https://github.com/microsoft/LoRA/blob/main/loralib/layers.py.
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .composition import AdapterCompositionBlock
from .configuration import LoRAConfig
from .layer import AdapterLayerBase
from ..configuration_utils import PretrainedConfig


class LoRA(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
    ):
        super().__init__()
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.no_decomposition = config.no_decomposition
        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x

        # Actual trainable parameters
        if self.r > 1 and self.no_decomposition:
            raise ValueError("Can only use 'no_decomposition' when r == 1.")
        if self.r > 0:
            self.lora_A = nn.Parameter(torch.zeros(lora_A_shape))
            if not self.no_decomposition:
                self.lora_B = nn.Parameter(torch.zeros(lora_B_shape))
            self.scaling = self.lora_alpha / self.r

            if config.init_weights == "lora":
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                if not self.no_decomposition:
                    nn.init.zeros_(self.lora_B)
            elif config.init_weights == "bert":
                nn.init.normal_(self.lora_A, std=0.02)
                if not self.no_decomposition:
                    nn.init.normal_(self.lora_B, std=0.02)
            else:
                raise ValueError("Unknown init_weights type: {}".format(config.init_weights))

    def com(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if self.composition_mode == "add":
            return weights + added * self.scaling
        elif self.composition_mode == "scale":
            return weights * (added * self.scaling)
        else:
            raise ValueError("Invalid composition mode.")

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        if self.composition_mode == "add":
            return weights - added * self.scaling
        elif self.composition_mode == "scale":
            return weights / (added * self.scaling)
        else:
            raise ValueError("Invalid composition mode.")


class LoRALayer(AdapterLayerBase):
    def __init__(self, location_key: str, config: PretrainedConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_key = location_key + "_lora"
        self.config = config
        self.loras = nn.ModuleDict(dict())

        self.merged = False

    def _check_lora_location(self, config: LoRAConfig):
        return True

    def _get_lora_shapes(self, config: LoRAConfig):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        lora_config = self.config.adapters.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if lora_config is not None and self._check_lora_location(lora_config):
            lora = LoRA(*self._get_lora_shapes(lora_config), lora_config)
            lora.train(self.training)
            self.loras[adapter_name] = lora
            return True

        return False

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.loras:
            del self.loras[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.loras:
                    for param in self.loras[name].parameters():
                        param.requires_grad = True

    def get_adapter(self, adapter_name: str) -> nn.Module:
        if adapter_name in self.loras:
            return self.loras[adapter_name]
        else:
            return None


class Linear(LoRALayer, nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config: PretrainedConfig,
        attn_key: str = None,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs
    ):
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        self.attn_key = attn_key
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def _check_lora_location(self, config: LoRAConfig):
        return self.attn_key in config.attn_matrices

    def _get_lora_shapes(self, config: LoRAConfig):
        return (config.r, self.in_features), (self.out_features, config.r)

    def reset_lora(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0:
                if lora.no_decomposition:
                    delta_w = lora.lora_A.flatten()
                else:
                    delta_w = T(lora.lora_B @ lora.lora_A)
                self.weight.data = lora.com_inv(self.weight.data, delta_w)
            self.merged = None

    def merge_lora(self, name: str):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                # Merge the weights and mark it
                if lora.r > 0:
                    if lora.no_decomposition:
                        delta_w = lora.lora_A.flatten()
                    else:
                        delta_w = T(lora.lora_B @ lora.lora_A)
                    self.weight.data = lora.com(self.weight.data, delta_w)
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRaLayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        if lora.no_decomposition:
                            delta_w = lora.lora_A.flatten()
                        else:
                            delta_w = lora.lora_dropout(x) @ lora.lora_A.T @ lora.lora_B.T
                        result = lora.com(result, delta_w)
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)


class MergedLinear(LoRALayer, nn.Linear):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        config: PretrainedConfig,
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        LoRALayer.__init__(self, location_key, config, in_features, out_features, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def _get_lora_shapes(self, config: LoRAConfig):
        enable_lora = set(config.attn_matrices)
        return (config.r * len(enable_lora), self.in_features), (
            self.out_features // 3 * len(enable_lora),
            config.r,
        )

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        is_added = super().add_adapter(adapter_name, layer_idx)
        if is_added:
            lora_config = lora_config = self.config.adapters.match(
                adapter_name,
                config_type=LoRAConfig,
                layer_idx=self.layer_idx,
                location_key=self.location_key,
            )
            lora = self.loras[adapter_name]
            lora.enable_lora = [
                "q" in lora_config.attn_matrices,
                "k" in lora_config.attn_matrices,
                "v" in lora_config.attn_matrices,
            ]
            # Actual trainable parameters
            if any(lora.enable_lora):
                # Compute the indices
                lora.lora_ind = self.weight.new_zeros((self.out_features,), dtype=torch.bool).view(
                    len(lora.enable_lora), -1
                )
                lora.lora_ind[lora.enable_lora, :] = True
                lora.lora_ind = lora.lora_ind.view(-1)

    def zero_pad(self, x, lora):
        if lora.composition_mode == "add":
            result = x.new_zeros((*x.shape[:-1], self.out_features))
        else:
            result = x.new_ones((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, lora.lora_ind] = x.reshape(-1, self.out_features // len(lora.enable_lora) * sum(lora.enable_lora))
        return result.view((*x.shape[:-1], self.out_features))

    def reset_lora(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            if lora.r > 0 and any(lora.enable_lora):
                if lora.no_decomposition:
                    delta_w = lora.lora_A.flatten()
                else:
                    delta_w = F.conv1d(
                        lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(lora.enable_lora)
                    ).squeeze(0)
                    delta_w = T(delta_w)
                self.weight.data = lora.com_inv(self.weight.data, self.zero_pad(delta_w, lora))
            self.merged = None

    def merge_lora(self, name: str):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                # Merge the weights and mark it
                if lora.r > 0 and any(lora.enable_lora):
                    if lora.no_decomposition:
                        delta_w = lora.lora_A.flatten()
                    else:
                        delta_w = F.conv1d(
                            lora.lora_A.data.unsqueeze(0), lora.lora_B.data.unsqueeze(-1), groups=sum(lora.enable_lora)
                        ).squeeze(0)
                        delta_w = T(delta_w)
                    self.weight.data = lora.com(self.weight.data, self.zero_pad(delta_w, lora))
                self.merged = name
            elif self.merged != name:
                raise ValueError("LoRaLayer already has a merged LoRA module. Please reset it first.")

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if not self.merged:
            adapter_setup = self.get_active_setup(self.loras)
            if adapter_setup is not None:
                if len(adapter_setup) == 1:
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    lora = self.loras[adapter_setup[0]]
                    if lora.r > 0:
                        if lora.no_decomposition:
                            delta_w = lora.lora_A.flatten()
                        else:
                            after_A = F.linear(lora.lora_dropout(x), lora.lora_A)
                            after_B = F.conv1d(
                                after_A.transpose(-2, -1), lora.lora_B.unsqueeze(-1), groups=sum(lora.enable_lora)
                            ).transpose(-2, -1)
                            delta_w = after_B
                        result = lora.com(result, self.zero_pad(delta_w, lora))
                    return result
                else:
                    raise ValueError(f"Invalid adapter setup. Cannot use {adapter_setup} with LoRA.")

        return F.linear(x, T(self.weight), bias=self.bias)
