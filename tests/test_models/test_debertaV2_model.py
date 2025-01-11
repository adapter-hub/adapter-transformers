# flake8: noqa: F403,F405
from adapters import DebertaV2AdapterModel
from hf_transformers.tests.models.deberta_v2.test_modeling_deberta_v2 import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class DebertaV2AdapterModelTest(AdapterModelTesterMixin, DebertaV2ModelTest):
    all_model_classes = (DebertaV2AdapterModel,)
    fx_compatible = False
