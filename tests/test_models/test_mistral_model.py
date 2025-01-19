# flake8: noqa: F403,F405
from adapters import MistralAdapterModel
from hf_transformers.tests.models.mistral.test_modeling_mistral import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class MistralAdapterModelTest(AdapterModelTesterMixin, MistralModelTest):
    all_model_classes = (MistralAdapterModel,)
    fx_compatible = False
