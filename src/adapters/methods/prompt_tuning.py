# https://github.com/google-research/prompt-tuning/blob/main/prompt_tuning/train/prompts.py

import math
from functools import partial
from typing import Callable

import numpy as np
import torch
from torch import nn

from transformers import AutoTokenizer
from transformers.configuration_utils import PretrainedConfig

from ..configuration import ModelAdaptersConfig, PromptTuningConfig
from ..context import ForwardContext
from ..utils import multigetattr, prefix_attention_mask
from .adapter_layer_base import AdapterLayerBase


class PromptTuning(nn.Module):
    """Generate a Prompt and concatenate it with the input.

    This is the training time version of prompting a model. Calling the injected `prompt` module will generate your
    unbatched prompt. This model then replicates it for the batched input and concatenates them together.

    Attributes:
        prompt: The module that actually generates the unbatched prompt.
        combine: A function that combines the prompt and the embedded input.
    """

    prompt: nn.Module
    combination_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self,
        adapter_name: str,
        prompt_tuning_config: PromptTuningConfig,
        model_config: PretrainedConfig,
        base_model_embeddings: nn.Module,
    ):
        super().__init__()

        self.name = adapter_name
        self.model_config = model_config
        self.prompt_tuning_config = prompt_tuning_config

        embedding_size = getattr(model_config, "embedding_size", model_config.hidden_size)

        self.prompt_embedding = nn.Embedding(
            num_embeddings=prompt_tuning_config.prompt_length, embedding_dim=embedding_size
        )
        # Initialize prompt tokens
        self.prompt_tokens = torch.arange(prompt_tuning_config.prompt_length).long()

        self._init_prompt_embedding(base_model_embeddings)

        if prompt_tuning_config.combine == "prefix":
            self.combination_fn = lambda prompt, embedded_input: torch.cat([prompt, embedded_input], dim=1)
        elif prompt_tuning_config.combine == "prefix_after_bos":
            self.combination_fn = lambda prompt, embedded_input: torch.cat(
                [embedded_input[:, 0, np.newaxis], prompt, embedded_input[:, 1:]], dim=1
            )
        else:
            raise ValueError(
                f"Unknown combination function: {prompt_tuning_config.combine}. "
                "Must be one of 'prefix' or 'prefix_after_bos'."
            )

    def _init_prompt_embedding(self, base_model_embeddings: nn.Module) -> None:
        if self.prompt_tuning_config.prompt_init == "random_uniform":
            nn.init.uniform_(
                self.prompt_embedding.weight,
                a=-self.prompt_tuning_config.random_uniform_scale,
                b=self.prompt_tuning_config.random_uniform_scale,
            )

        elif self.prompt_tuning_config.prompt_init == "from_string":
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer_name_or_path)
            prompt_length = self.prompt_tuning_config.prompt_length
            prompt_text = self.prompt_tuning_config.prompt_init_text
            if prompt_text is None:
                raise ValueError("Prompt text must be provided when using prompt_init='from_string'.")

            tokenized_prompt_text: list[int] = tokenizer(prompt_text)["input_ids"]  # type: ignore

            # If the prompt text tokens are shorter than the prompt length, we repeat the prompt text tokens until we reach the prompt length
            if len(tokenized_prompt_text) < prompt_length:
                num_reps = math.ceil(prompt_length / len(tokenized_prompt_text))
                tokenized_prompt_text = tokenized_prompt_text * num_reps

            # Adjust length of prompt text tokens to match prompt_length
            tokenized_prompt_text = tokenized_prompt_text[:prompt_length]

            # Initialize prompt embedding with tokenized prompt text
            word_embedding_weights = base_model_embeddings(torch.LongTensor(tokenized_prompt_text)).detach().clone()
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.prompt_embedding.weight = nn.Parameter(word_embedding_weights)

        else:
            raise ValueError(f"Unknown prompt initialization: {self.prompt_tuning_config.prompt_init}")

    def forward(self, embedded_input):
        # Compute prompt embedding
        self.prompt_tokens = self.prompt_tokens.to(embedded_input.device)
        prompt = self.prompt_embedding(self.prompt_tokens)

        # Prompt to batch size
        batch_size = embedded_input.shape[0]
        prompt = torch.tile(torch.unsqueeze(prompt, dim=0), [batch_size] + [1 for _ in prompt.shape])

        # Merge prompt and input
        output = self.combination_fn(prompt, embedded_input)

        # Adapt attention mask
        prefix_attention_mask_length = self.prompt_tuning_config.prompt_length

        return output, prefix_attention_mask_length


class PromptTuningLayer(AdapterLayerBase, nn.Module):
    """
    Prompt Tuning implementation.

    Args:
        model_config: The model configuration.
        adapters_config: The adapter configuration.
        base_model_embeddings:
            The embedding layer of the base model (used to initialize the prompt embedding if
            prompt_init='from_string').
    """

    adapter_modules_name = "prompt_tunings"

    def __init__(
        self,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        base_model_embeddings: nn.Module,
    ):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.base_model_embeddings = base_model_embeddings
        self.prompt_tunings = nn.ModuleDict()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        # ignore layer_idx as prompt tunings are only added after the embedding layer
        prompt_tuning_config = self.adapters_config.match(
            adapter_name,
            config_type=PromptTuningConfig,
        )

        if prompt_tuning_config is not None:
            adapter = PromptTuning(
                adapter_name=adapter_name,
                prompt_tuning_config=prompt_tuning_config,  # type: ignore
                model_config=self.model_config,
                base_model_embeddings=self.base_model_embeddings,
            )
            adapter.train(self.training)  # make sure training mode is consistent
            self.prompt_tunings[adapter_name] = adapter
            return True

        return False

    def forward(self, hidden_states: torch.Tensor):
        prefix_attention_mask_length = None
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None and len(adapter_setup) > 0:
            first_adapter = adapter_setup.first()
            if first_adapter in self.prompt_tunings:
                hidden_states, prefix_attention_mask_length = self.prompt_tunings[first_adapter](hidden_states)

        context = ForwardContext.get_context()
        if context is not None:
            context.prompt_tokens_length = prefix_attention_mask_length

        return hidden_states


def hook_fn(model, module, args, embedding_output):
    embedding_output = model.prompt_tuning.forward(embedding_output)
    return embedding_output


# TODO: this will only work for a limited set of models
def _attn_mask_hook_fn(module, args):
    attn_mask = args[1]
    attn_mask = prefix_attention_mask(attn_mask)
    return (args[0], attn_mask) + args[2:]


def init_prompt_tuning(model):
    model = model.base_model
    if not hasattr(model, "prompt_tuning"):
        model.support_prompt_tuning = True
        model.prompt_tuning = PromptTuningLayer(model.config, model.adapters_config, model.get_input_embeddings())
        embed_layer = multigetattr(model, model.adapter_interface.model_embeddings)
        embed_layer.register_forward_hook(partial(hook_fn, model))

        for _, layer in model.iter_layers():
            layer.register_forward_pre_hook(_attn_mask_hook_fn)
