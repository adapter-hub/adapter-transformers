# flake8: noqa: F403,F405
from adapters import MBartAdapterModel
from hf_transformers.tests.models.mbart.test_modeling_mbart import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class MBartAdapterModelTest(AdapterModelTesterMixin, MBartModelTest):
    all_model_classes = (MBartAdapterModel,)
    fx_compatible = False
