# flake8: noqa: F403,F405
from adapters import RobertaAdapterModel
from hf_transformers.tests.models.roberta.test_modeling_roberta import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class RobertaAdapterModelTest(AdapterModelTesterMixin, RobertaModelTest):
    all_model_classes = (RobertaAdapterModel,)
    fx_compatible = False
