# flake8: noqa: F403,F405
from adapters import PLBartAdapterModel
from hf_transformers.tests.models.plbart.test_modeling_plbart import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class PLBartAdapterModelTest(AdapterModelTesterMixin, PLBartModelTest):
    all_model_classes = (PLBartAdapterModel,)
    fx_compatible = False
