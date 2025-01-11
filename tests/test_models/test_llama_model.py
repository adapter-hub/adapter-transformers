# flake8: noqa: F403,F405
from adapters import LlamaAdapterModel
from hf_transformers.tests.models.llama.test_modeling_llama import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class LlamaAdapterModelTest(AdapterModelTesterMixin, LlamaModelTest):
    all_model_classes = (LlamaAdapterModel,)
    fx_compatible = False
