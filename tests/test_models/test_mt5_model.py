# flake8: noqa: F403,F405
from adapters import MT5AdapterModel
from hf_transformers.tests.models.mt5.test_modeling_mt5 import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class MT5AdapterModelTest(AdapterModelTesterMixin, MT5IntegrationTest):
    all_model_classes = (MT5AdapterModel,)
    fx_compatible = False
