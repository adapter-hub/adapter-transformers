import unittest

from transformers import GPTNeoXConfig
from transformers.testing_utils import require_torch

from .methods import (
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    PrefixTuningTestMixin,
    UniPELTTestMixin,
)
from .test_adapter import AdapterTestBase, make_config
from .test_adapter_backward_compability import CompabilityTestMixin
from .composition.test_parallel import ParallelAdapterInferenceTestMixin, ParallelTrainingMixin
from .test_adapter_conversion import ModelClassConversionTestMixin
from .test_adapter_embeddings import EmbeddingTestMixin
from .test_adapter_fusion_common import AdapterFusionModelTestMixin
from .test_adapter_heads import PredictionHeadModelTestMixin


class GPTNeoXAdapterTestBase(AdapterTestBase):
    config_class = GPTNeoXConfig
    config = make_config(
        GPTNeoXConfig,
        n_embd=32,
        n_layer=4,
        n_head=4,
        # set pad token to eos token
        pad_token_id=50256,
        resid_pdrop=0.1,
    )
    tokenizer_name = "EleutherAI/gpt-neox-20b"


@require_torch
class GPTNeoXAdapterTest(
    BottleneckAdapterTestMixin,
    CompacterTestMixin,
    IA3TestMixin,
    LoRATestMixin,
    UniPELTTestMixin,
    PrefixTuningTestMixin,
    EmbeddingTestMixin,
    CompabilityTestMixin,
    AdapterFusionModelTestMixin,
    PredictionHeadModelTestMixin,
    ParallelAdapterInferenceTestMixin,
    ParallelTrainingMixin,
    GPTNeoXAdapterTestBase,
    unittest.TestCase,
):
    pass


@require_torch
class GPTNeoXClassConversionTest(
    ModelClassConversionTestMixin,
    GPTNeoXAdapterTestBase,
    unittest.TestCase,
):
    pass
