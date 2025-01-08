# flake8: noqa: F403,F405
from adapters import M2M100AdapterModel
from hf_transformers.tests.models.m2m_100.test_modeling_m2m_100 import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class M2M100AdapterModelTest(AdapterModelTesterMixin, M2M100ModelTest):
    all_model_classes = (M2M100AdapterModel,)
    fx_compatible = False
