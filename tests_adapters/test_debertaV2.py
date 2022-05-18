import unittest

from tests.deberta_v2.test_modeling_deberta_v2 import *
from transformers import DebertaV2AdapterModel
from transformers.testing_utils import require_torch

from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_common import AdapterModelTesterMixin
from .test_adapter_composition import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class DebertaAdapterModelTest(AdapterModelTesterMixin, DebertaV2ModelTest):
    all_model_classes = (
        DebertaV2AdapterModel,
    )


class DebertaAdapterTestBase(AdapterTestBase):
    config_class = DebertaV2Config
    config = make_config(
        DebertaV2Config,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=37,
    )


@require_torch
class DebertaV2AdapterTest(
    AdapterModelTesterMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    DebertaAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class DebertaV2ClassConversionTest(
    ModelClassConversionTestMixin,
    DebertaAdapterTestBase,
    unittest.TestCase,
):
    pass