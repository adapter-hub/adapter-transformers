# flake8: noqa: F403,F405
from adapters import ViTAdapterModel
from hf_transformers.tests.models.vit.test_modeling_vit import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class ViTAdapterModelTest(AdapterModelTesterMixin, ViTModelTest):
    all_model_classes = (ViTAdapterModel,)
    fx_compatible = False
