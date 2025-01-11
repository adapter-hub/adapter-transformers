# flake8: noqa: F403,F405
from adapters import BertGenerationAdapterModel
from hf_transformers.tests.models.bert_generation.test_modeling_bert_generation import *
from transformers.testing_utils import require_torch

from .base import AdapterModelTesterMixin


@require_torch
class BertGenerationAdapterModelTest(AdapterModelTesterMixin, BertGenerationEncoderTest):
    all_model_classes = (BertGenerationAdapterModel,)
    fx_compatible = False
