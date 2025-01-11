# flake8: noqa: F403,F405
from adapters import XmodAdapterModel
from hf_transformers.tests.models.xmod.test_modeling_xmod import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class XmodAdapterModelTest(AdapterModelTesterMixin, XmodModelTest):
    all_model_classes = (XmodAdapterModel,)
    fx_compatible = False
