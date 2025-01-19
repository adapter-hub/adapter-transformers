# flake8: noqa: F403,F405
from adapters import BartAdapterModel
from hf_transformers.tests.models.bart.test_modeling_bart import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class BartAdapterModelTest(AdapterModelTesterMixin, BartModelTest):
    all_model_classes = (BartAdapterModel,)
    fx_compatible = False
