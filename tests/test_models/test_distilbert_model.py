# flake8: noqa: F403,F405
from adapters import DistilBertAdapterModel
from hf_transformers.tests.models.distilbert.test_modeling_distilbert import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class DistilBertAdapterModelTest(AdapterModelTesterMixin, DistilBertModelTest):
    all_model_classes = (DistilBertAdapterModel,)
    fx_compatible = False
