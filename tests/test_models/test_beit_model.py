# flake8: noqa: F403,F405
from adapters import BeitAdapterModel
from hf_transformers.tests.models.beit.test_modeling_beit import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class BeitAdapterModelTest(AdapterModelTesterMixin, BeitModelTest):
    all_model_classes = (BeitAdapterModel,)
    fx_compatible = False
