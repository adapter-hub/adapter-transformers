# flake8: noqa: F403,F405
from adapters import AlbertAdapterModel
from hf_transformers.tests.models.albert.test_modeling_albert import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class AlbertAdapterModelTest(AdapterModelTesterMixin, AlbertModelTest):
    all_model_classes = (AlbertAdapterModel,)
    fx_compatible = False
