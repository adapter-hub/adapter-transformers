# flake8: noqa: F403,F405
from adapters import GPTJAdapterModel
from hf_transformers.tests.models.gptj.test_modeling_gptj import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class GPTJAdapterModelTest(AdapterModelTesterMixin, GPTJModelTest):
    all_model_classes = (GPTJAdapterModel,)
    fx_compatible = False
