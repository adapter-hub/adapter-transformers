# flake8: noqa: F403,F405
from adapters import MllamaAdapterModel
from hf_transformers.tests.models.mllama.test_modeling_mllama import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class MistralAdapterModelTest(AdapterModelTesterMixin, MllamaForConditionalGenerationIntegrationTest):
    all_model_classes = (MllamaAdapterModel,)
    fx_compatible = False
