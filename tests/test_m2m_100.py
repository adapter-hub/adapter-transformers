import unittest

from tests.methods.test_config_union import ConfigUnionAdapterTest
from transformers import M2M100Config
from transformers.testing_utils import require_torch

from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    ReftTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class M2M100AdapterTestBase(AdapterTestBase):
    config_class = M2M100Config
    config = make_config(
        M2M100Config,
        vocab_size=256206,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        scale_embedding=False,
    )
    tokenizer_name = "facebook/nllb-200-distilled-600M"


@require_torch
class M2M100AdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    ReftTestMixin,
    UniPELTTestMixin,
    AdapterFusionModelTestMixin,
    CompabilityTestMixin,
    EmbeddingTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    ConfigUnionAdapterTest,
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
