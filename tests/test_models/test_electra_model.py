# flake8: noqa: F403,F405
from adapters import ElectraAdapterModel
from hf_transformers.tests.models.electra.test_modeling_electra import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class ElectraAdapterModelTest(AdapterModelTesterMixin, ElectraModelTester):
    all_model_classes = (ElectraAdapterModel,)
    fx_compatible = False
