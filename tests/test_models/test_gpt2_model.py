# flake8: noqa: F403,F405
from adapters import GPT2AdapterModel
from hf_transformers.tests.models.gpt2.test_modeling_gpt2 import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class GPT2AdapterModelTest(AdapterModelTesterMixin, GPT2ModelTest):
    all_model_classes = (GPT2AdapterModel,)
    fx_compatible = False
