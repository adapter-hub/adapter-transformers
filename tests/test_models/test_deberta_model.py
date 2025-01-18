# flake8: noqa: F403,F405
from adapters import DebertaAdapterModel
from hf_transformers.tests.models.deberta.test_modeling_deberta import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class DebertaAdapterModelTest(AdapterModelTesterMixin, DebertaModelTest):
    all_model_classes = (DebertaAdapterModel,)
    fx_compatible = False
