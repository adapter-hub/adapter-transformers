import unittest

from tests.models.m2m_100.test_modeling_m2m_100 import *
from transformers import M2M100AdapterModel
from transformers.testing_utils import require_torch

from .methods import BottleneckAdapterTestMixin, UniPELTTestMixin, CompacterTestMixin, IA3TestMixin, LoRATestMixin, PrefixTuningTestMixin
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_composition import ParallelAdapterInferenceTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin
from .test_common import AdapterModelTesterMixin


@require_torch
class M2M100AdapterModelTest(AdapterModelTesterMixin, M2M100ModelTest):
    all_model_classes = (
        M2M100AdapterModel,
    )
    fx_compatible = False


class M2M100AdapterTestBase(AdapterTestBase):
    config_class = M2M100Config
    config = make_config(
        M2M100Config,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        vocab_size=128112,
    )
    tokenizer_name = "facebook/m2m100_418M"


@require_torch
class M2M100AdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    M2M100AdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class M2M100ClassConversionTest(
    ModelClassConversionTestMixin,
    M2M100AdapterTestBase,
    unittest.TestCase,
):
    pass
