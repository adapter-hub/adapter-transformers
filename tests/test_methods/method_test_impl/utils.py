import copy
import random

import torch

from adapters import ADAPTER_MODEL_MAPPING, init
from transformers.testing_utils import torch_device


global_rng = random.Random()


def create_twin_models(model_class, config_creator=None):
    if config_creator and model_class.__name__.startswith("Auto"):
        model_config = config_creator()
        model1 = model_class.from_config(model_config)
    elif config_creator:
        model_config = config_creator()
        model1 = model_class(model_config)
    else:
        model_config = model_class.config_class()
        model1 = model_class(model_config)
    init(model1)
    model1.eval()
    # create a twin initialized with the same random weights
    model2 = copy.deepcopy(model1)
    model2.eval()
    return model1, model2


def add_lm_head(config_class, model, adapter_name):
    """Add appropriate language model head based on model type"""
    if "seq2seq_lm" in ADAPTER_MODEL_MAPPING[config_class].head_types:
        model.add_seq2seq_lm_head(adapter_name)
    else:
        model.add_causal_lm_head(adapter_name)


def make_config(config_class, **kwargs):
    return staticmethod(lambda: config_class(**kwargs))


def ids_tensor(shape, vocab_size=5000, dtype=torch.long):
    total_dims = 1
    for dim in shape:
        total_dims *= dim
    values = [global_rng.randint(0, vocab_size - 1) for _ in range(total_dims)]
    return torch.tensor(data=values, dtype=dtype, device=torch_device).view(shape).contiguous()
